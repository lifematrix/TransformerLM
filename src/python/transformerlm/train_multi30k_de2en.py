# -*- coding: utf-8 -*-
"""
The program to train q seq2seq model based on Transformer to translate [de]utsch to [en]glish.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import Seq2SeqDataMulti30k
from .models import TSUtils, LMTransformerSeq2Seq

from dotteddict import dotteddict
import yaml
from tqdm import tqdm

config_yaml = """
    batch_size: 128 
    device: "cuda:0"
    n_encoder_layers: 3
    n_decoder_layers: 3  
    dropout: 0.1
    n_epochs: 1 
    torch_seed: 12345
    d_model: 512
    d_ff: 512 
"""

CF = dotteddict(yaml.load(config_yaml, yaml.SafeLoader))


class Trainer:
    def __init__(self, cfg: dotteddict):
        self.cfg = cfg

        print(f"Set torch manual seed {self.cfg.torch_seed}")
        torch.manual_seed(self.cfg.torch_seed)

        self.dp = Seq2SeqDataMulti30k(src_lan="de", tgt_lan="en")  # data provider

        self.model = LMTransformerSeq2Seq(src_vocab_size=self.dp.vocabs[self.dp.src_lan].size,
                                          tgt_vocab_size=self.dp.vocabs[self.dp.tgt_lan].size,
                                          n_encoder_layers=self.cfg.n_encoder_layers,
                                          n_decoder_layers=self.cfg.n_decoder_layers,
                                          d_model=self.cfg.d_model,
                                          d_ff=self.cfg.d_ff,
                                          dropout_rate=self.cfg.dropout).to(self.cfg.device)
        _ = TSUtils.count_parameters(self.model)

        self.train_dataloader = DataLoader(self.dp.datasets['train'], shuffle=True,
                                           batch_size=self.cfg.batch_size, collate_fn=self.dp.pad_batch)

        self.val_dataloader = DataLoader(self.dp.datasets['val'], shuffle=False,
                                         batch_size=self.cfg.batch_size, collate_fn=self.dp.pad_batch)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                                       patience=2, factor=0.2,
                                                                       threshold=0.0001,
                                                                       verbose=True)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.dp.PAD_IDX)

    def train(self):
        self.model.train()
        for epoch in range(self.cfg.n_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.eval_epoch(epoch)
            print(f"Epoch {epoch} | train loss: {train_loss:.5f}, val loss: {val_loss:.5f}")
            self.lr_scheduler.step(val_loss)

    def train_epoch(self, epoch):

        losses = []
        self.model.train()

        pbar = tqdm(self.train_dataloader)
        for i, (src, tgt_orig) in enumerate(pbar):
            src = src.to(self.cfg.device)
            tgt_orig = tgt_orig.to(self.cfg.device)
            tgt = tgt_orig[:, :-1]
            src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask = \
                TSUtils.make_seq2seq_mask(src, tgt, pad_idx=self.dp.PAD_IDX)

            logits = self.model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                                src_key_padding_mask=src_key_padding_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=src_key_padding_mask,
                                final_proj=True)
            tgt_y = tgt_orig[:, 1:]
            self.optimizer.zero_grad()
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_y.reshape(-1))
            loss.backward()
            self.optimizer.step()
            pbar.set_description(f"epoch[{epoch}] step[{i}/{len(self.train_dataloader)}] "
                                 f"| loss: {loss.item():.4f} | lr: {self.optimizer.param_groups[0]['lr']:.6f} | ")
            losses.append(loss.item())

        return TSUtils.mean(losses)

    def eval_epoch(self, epoch=None):

        self.model.eval()
        losses = []

        for i, (src, tgt_orig) in enumerate(self.val_dataloader):
            src = src.to(self.cfg.device)
            tgt_orig = tgt_orig.to(self.cfg.device)
            tgt = tgt_orig[:, :-1]
            src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask = \
                TSUtils.make_seq2seq_mask(src, tgt, pad_idx=self.dp.PAD_IDX)

            logits = self.model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                                src_key_padding_mask=src_key_padding_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=src_key_padding_mask,
                                final_proj=True)

            tgt_y = tgt_orig[:, 1:]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_y.reshape(-1))
            losses.append(loss.item())

        return TSUtils.mean(losses)

    def greedy_decode(self, src, src_mask, src_key_padding_mask, max_len):
        src = src.to(self.cfg.device)
        src_mask = src_mask.to(self.cfg.device)
        src_key_padding_mask = src_key_padding_mask.to(self.cfg.device)

        memory = self.model.encode(src, src_mask,  src_key_padding_mask=src_key_padding_mask)
        ys = torch.ones(1, 1, dtype=torch.int64).fill_(self.dp.BOS_IDX).to(self.cfg.device)
        for i in range(max_len - 1):
            tgt_mask = TSUtils.make_causal_mask(ys)
            output = self.model.decode(memory, ys, memory_mask=src_mask,
                                       tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask)
            logits = self.model.generator(output[:, -1])
            next_pred = torch.argmax(logits, dim=-1, keepdim=True)
            ys = torch.cat([ys, next_pred], dim=1)
            if next_pred[0, 0].item() == self.dp.EOS_IDX:
                break

        return ys[0]

    def translate(self, src: str):
        self.model.eval()
        src = self.dp.preprocess(self.dp.src_lan, src)  # converted into token ids.
        src = src[None, ...]  # add axis 0 as batch dim
        src_mask = torch.ones(src.shape[1]).to(torch.bool)
        src_key_padding_mask = (src != self.dp.PAD_IDX)
        pred_tokens = self.greedy_decode(src, src_mask, src_key_padding_mask, max_len=200)
        pred_tokens = list(pred_tokens.cpu().numpy())

        pred_str = " ".join(self.dp.vocabs[self.dp.tgt_lan][pred_tokens])

        return pred_str


if __name__ == "__main__":
    trainer = Trainer(CF)
    trainer.train()
    print(trainer.translate("Zwei Autos fahren auf einer Rennstrecke."))
