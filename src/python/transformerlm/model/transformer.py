# -*- coding: utf-8 -*

import os
import sys
from typing import Optional
import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from . import TSUtils


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

        self._reset_parameters()

    def _reset_parameters(self):
        for l in self.proj_linears:
            nn.init.xavier_normal_(l.weight)

    def apply_attention(self, q: Tensor, k: Tensor, v: Tensor, attn_mask=None, key_padding_mask=None):
        d_k = q.size(-1)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask == 0, -1e9)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask[:, None, None, :] # convert  into [B, 1, 1(src_len), tgt_len]
            scores.masked_fill_(key_padding_mask == 0, -1e9)

        p_attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)

        output = torch.matmul(p_attn, v)

        return output, p_attn

    def forward(self, query, key, value,  attn_mask=None, key_padding_mask=None):
        B = query.size()[0]   # batch size

        query, key, value = [lin(x).view(B, -1, self.h, self.d_k).transpose(1, 2)
                             for lin, x in zip(self.proj_linears, (query, key, value))]

        attn_outputs, attn_weights = self.apply_attention(query, key, value, attn_mask, key_padding_mask)
        attn_weights = torch.mean(attn_outputs, dim=-1)
        attn_outputs = attn_outputs.transpose(1, 2).contiguous().view(B, -1, self.h*self.d_k)
        attn_outputs = self.proj_linears[-1](attn_outputs)
        del query, key, value

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
        super(TransformerEncoderLayer, self).__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)

        self.self_attn = MultiHeadedAttention(d_model=d_model, n_heads=n_heads,
                                          dropout_rate=dropout_rate)

        self.ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate)

    def forward(self, src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None):
        x = self.norm_1(src)
        attn_output, _ = self.self_attn(x, x, x, src_mask, src_key_padding_mask)
        attn_output = self.dropout_1(attn_output)
        x = x + attn_output

        x = self.norm_2(x)
        ff_output = self.ff(x)
        ff_output = self.dropout_2(ff_output)
        x = x + ff_output

        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate):
        super(TransformerDecoderLayer, self).__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.dropout_3 = nn.Dropout(dropout_rate)

        self.self_attn = MultiHeadedAttention(d_model=d_model, n_heads=n_heads,
                                          dropout_rate=dropout_rate)
        self.src_attn = MultiHeadedAttention(d_model=d_model, n_heads=n_heads,
                                             dropout_rate=dropout_rate)

        self.ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate)

    def forward(self,
                memory: Tensor,
                tgt: Tensor,
                memory_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None):

        y = self.norm_1(tgt)
        attn_output, _ = self.self_attn(y, y, y, tgt_mask, tgt_key_padding_mask)
        attn_output = self.dropout_1(attn_output)
        y = y + attn_output

        y = self.norm_2(y)
        attn_output, _ = self.src_attn(y, memory, memory, memory_mask, memory_key_padding_mask)
        attn_output = self.dropout_2(attn_output)
        y = y + attn_output

        y = self.norm_3(y)
        ff_output = self.ff(y)
        ff_output = self.dropout_3(ff_output)
        y = y + ff_output

        return y


class TSEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TSEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        x = self.lut(x) * math.sqrt(self.d_model)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=10000, dropout_rate=0.0):
        super(PositionalEncoding, self).__init__()
        self.register_buffer("encoding",
                             torch.from_numpy(self.make_pos_encodng(max_seq_len, d_model))
                             .to(torch.get_default_dtype()))
        self.dropout = nn.Dropout(p=dropout_rate)

    @classmethod
    def make_pos_encodng(cls, seq_len, d_model):
        d_half = d_model / 2

        f = np.power(10000, -2 * np.arange(d_half) / d_model)[np.newaxis, :]
        p = np.arange(seq_len)[:, np.newaxis]
        angle = np.dot(p, f)

        pe_even = np.sin(angle)
        pe_odd = np.cos(angle)
        encoding = np.dstack((pe_even, pe_odd)).reshape(seq_len, d_model)

        return encoding

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.encoding[None, :seq_len]
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

    def forward(self, src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None):
        for layer in self.encoder_layers:
            src = layer(src, src_mask, src_key_padding_mask)

        src = self.norm(src)

        return src

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size, dropout_rate=0.0):
        super(Generator, self).__init__()

        self.proj = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.dropout(self.proj(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout_rate=0.0):
        super(TransformerDecoder, self).__init__()

        self.n_layers = n_layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout_rate=dropout_rate)
            for _ in range(self.n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self,
                memory: Tensor,
                tgt: Tensor,
                memory_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                ):
        for layer in self.decoder_layers:
            tgt = layer(memory, tgt, memory_mask, tgt_mask, memory_key_padding_mask, tgt_key_padding_mask)

        tgt = self.norm(tgt)

        return tgt


class LMTransformerDecoderOnly(nn.Module):
    def __init__(self, vocab_size,
                 d_model=512, max_seq_len=1024,
                 n_encoder_layers=12, n_mttn_heads=8, dropout_rate=0.0):
        super(LMTransformerDecoderOnly, self).__init__()

        self.d_model = d_model
        self.d_ff = 4 * d_model

        self.embedding = nn.Sequential(
            TSEmbedding(vocab_size, d_model),
            PositionalEncoding(d_model=d_model, max_seq_len=max_seq_len)
        )
        self.encoder = TransformerEncoder(n_layers=n_encoder_layers,
                                          n_heads=n_mttn_heads, d_model=d_model,
                                          d_ff=self.d_ff, dropout_rate=dropout_rate)

        self.final_proj = nn.Linear(d_model, vocab_size)

    # def make_causal_mask(self, x):
    #     seq_len = x.size(1)
    #     mask = torch.tril(torch.ones(seq_len, seq_len, dtype=x.dtype), diagonal=0).to(x.device)
    #     return mask

    def forward(self, x):
        x = self.embedding(x)
        mask = TSUtils.make_causal_mask(x)
        x = self.encoder(x, mask=mask)
        x = self.final_proj(x)

        return x

class LMTransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size=None,
                 d_model=512, d_ff=None, max_seq_len=5000,
                 n_encoder_layers=6, n_decoder_layers=None,
                 n_mttn_heads=8, dropout_rate=0.1):
        super(LMTransformerSeq2Seq, self).__init__()

        self.src_vocab_size = src_vocab_size
        if tgt_vocab_size is None:
            tgt_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        self.d_model = d_model
        if d_ff is None:
            d_ff = 4 * d_model
        self.d_ff = d_ff

        self.max_seq_len = max_seq_len

        self.n_encoder_layers = n_encoder_layers
        if n_decoder_layers is None:
            n_decoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers

        self.n_mttn_heads = n_mttn_heads

        self.dropout_rate = dropout_rate

        self.pos_encoder = PositionalEncoding(d_model=self.d_model,
                                              max_seq_len=self.max_seq_len, dropout_rate=dropout_rate)

        self.src_embedding = nn.Sequential(
            TSEmbedding(vocab_size=self.src_vocab_size, d_model=self.d_model),
            self.pos_encoder
        )

        self.encoder = TransformerEncoder(n_layers=self.n_encoder_layers,
                                          n_heads=self.n_mttn_heads, d_model=self.d_model,
                                          d_ff=self.d_ff, dropout_rate=self.dropout_rate)

        self.tgt_embedding = nn.Sequential(
            TSEmbedding(vocab_size=self.tgt_vocab_size, d_model=self.d_model),
            self.pos_encoder
        )
        self.decoder = TransformerDecoder(n_layers=self.n_decoder_layers,
                                          n_heads=self.n_mttn_heads, d_model=self.d_model,
                                          d_ff=self.d_ff, dropout_rate=self.dropout_rate)

        self.generator = Generator(d_model, tgt_vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask, src_key_padding_mask=None):
        src = self.src_embedding(src)
        memory = self.encoder(src, src_mask, src_key_padding_mask)

        return memory

    def decode(self, memory, tgt, memory_mask=None, tgt_mask=None,
               memory_key_padding_mask=None, tgt_key_padding_mask=None):
        tgt = self.tgt_embedding(tgt)
        tgt = self.decoder(memory, tgt, memory_mask, tgt_mask, memory_key_padding_mask, tgt_key_padding_mask)

        return tgt

    def forward(self, src: Tensor, tgt: Tensor,
                src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                final_proj: bool = True):
        memory = self.encode(src, src_mask, src_key_padding_mask)
        tgt = self.decode(memory, tgt, memory_mask, tgt_mask, memory_key_padding_mask, tgt_key_padding_mask)
        if final_proj:
            tgt = self.generator(tgt)

        return tgt





