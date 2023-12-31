# -*- coding: utf-8 -*

import logging
import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# class Attention(torch.nn.Module):
#     def __init__(self, dropout_rate, *args, **kwargs):
#         super(Attention, self).__init__(*args, **kwargs)
#         self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
#
#     def call(self, inputs, training=None):
#         query, key, value, mask = inputs
#         d_k = tf.cast(tf.shape(query)[-1], tf.float32)
#
#         scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(d_k)
#         if mask is not None:
#             #c = tf.constant(-1e9, shape=[1]*len(scores.shape.as_list()), dtype=scores.dtype)
#             c = tf.constant(-1e9, dtype=scores.dtype)
#             scores = tf.where(mask == 0, c, scores)
#
#         p_attn = tf.nn.softmax(scores, axis=-1)
#         p_attn = self.dropout(p_attn, training=training)
#
#         return tf.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_heads=1, dropout_rate=0.1):
        """config contains: h, d_model, dropout_rate"""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_heads == 0, (f"The hidden dimension of the model {d_model} "
                                        f"should be diviede by the number of heads {n_heads} ")

        self.h = n_heads
        self.d_k = d_model // self.h

        self.d_model = d_model
        self.proj_linears = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(4)
        ])
        self.dropout = nn.Dropout(p=dropout_rate)

        # if bias:
        #     self.proj_biases = [nn.Parameter(torch.empty(d_model)) for _ in range(4)]
        # else:
        #     self.proj_biases = None

        self._reset_parameters()

    def _reset_parameters(self):
        for l in self.proj_linears:
            nn.init.xavier_normal_(l.weight)

    def apply_attention(self, q: Tensor, k: Tensor ,v: Tensor, mask=None):
        d_k = q.size(-1)

        scores = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(d_k)
        if mask is not None:
            scores.masked_fill_(mask==0, -1e9)

        p_attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)

        output = torch.matmul(p_attn, v)
        return output, p_attn

    def forward(self, query, key, value, mask):
        B = query.size()[0]   # batch size

        query, key, value = [torch.transpose(torch.reshape(lin(x), [B, -1, self.h, self.d_k]), 1,2)
                             for lin, x in zip(self.proj_linears, (query, key, value))]
        attn_outputs, attn_weights = self.apply_attention(query, key, value, mask)
        attn_weights = torch.mean(attn_outputs, dim=-1)
        attn_outputs = torch.reshape(torch.transpose(attn_outputs, 1,2), [B, -1, self.h*self.d_k])
        attn_outputs = self.proj_linears[-1](attn_outputs)

        return attn_outputs, attn_weights


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate):
        """
        d_ff: dimension feedforward
        """
        super(PositionwiseFeedForward, self).__init__()

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor):
        x = self.ff(x)

        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate):
        """
        config contains: d_model, d_ff, h, dropout_rate, N, d_feats
        """
        super(TransformerEncoderLayer, self).__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)

        self.mhatt = MultiHeadedAttention(d_model=d_model, n_heads=n_heads,
                                          dropout_rate=dropout_rate)

        self.ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate)

    def forward(self, x: Tensor, mask: Tensor=None):
        x = self.norm_1(x)
        attn_output, _ = self.mhatt(x, x, x, mask)
        attn_output = self.dropout_1(attn_output)
        x = x + attn_output

        x = self.norm_2(x)

        ff_output = self.ff(x)
        ff_output = self.dropout_2(ff_output)
        x = x + ff_output

        return x

class PositionEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model, dropout_rate=0.0):
        super(PositionEmbedding, self).__init__()
        self.pos_embedding = nn.Parameter(
            torch.from_numpy(self.make_pos_embedding(max_seq_len, d_model)),
            requires_grad=False)
        self.dropout = nn.Dropout(dropout_rate)

    @classmethod
    def make_pos_embedding(cls, seq_len, d_model):
        d_half = d_model / 2

        f = np.power(10000, -2 * np.arange(d_half) / d_model)[np.newaxis, :]
        p = np.arange(seq_len)[:, np.newaxis]
        angle = np.dot(p, f)

        pe_even = np.sin(angle)
        pe_odd = np.cos(angle)
        embedding = np.dstack((pe_even, pe_odd)).reshape(seq_len, d_model)

        return embedding

    def forward(self, x, training=None):
        seq_len = x.size()[1]
        x = x + self.pos_embedding[None, :seq_len, ...].requires_grad_(False)
        x = self.dropout(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout_rate=0.0):
        super(TransformerEncoder, self).__init__()

        self.n_layers = n_layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout_rate=dropout_rate)
            for _ in range(self.n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor=None):
        for layer in self.encoder_layers:
            x = layer(x, mask)

        x = self.norm(x)

        return x


class LMTransformerDecoderOnly(nn.Module):
    def __init__(self, vocab_size,
                 d_model=512, max_seq_len=1024,
                 n_encoder_layers=12, n_mttn_heads=8, dropout_rate=0.0):
        super(LMTransformerDecoderOnly, self).__init__()

        self.d_model = d_model
        self.d_ff = 4 * d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = PositionEmbedding(max_seq_len=max_seq_len, d_model=d_model)
        self.encoder = TransformerEncoder(n_layers=n_encoder_layers,
                                          n_heads=n_mttn_heads, d_model=d_model,
                                          d_ff=self.d_ff, dropout_rate=dropout_rate)

        self.final_proj = nn.Linear(d_model, vocab_size)

    def make_causal_mask(self, x):
        seq_len =  x.size()[1]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=x.dtype), diagonal=0).to(x.device)
        return mask

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_embedding(x)
        mask = self.make_causal_mask(x)
        x = self.encoder(x, mask=mask)
        x = self.final_proj(x)

        return x




