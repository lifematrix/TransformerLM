# -*- coding: utf-8 -*-
"""
The program to train q seq2seq model based on Transformer to translate [de]utsch to [en]glish.
"""

import torch
from torch.utils.data import DataLoader
from .data import Seq2SeqDataMulti30k
from .models import TSUtils, LMTransformerBilateralcoder

from dotteddict import dotteddict
import yaml

CF = dotteddict(
    yaml.load(
        """
        batch_size: 16
        device: "cuda:0"
        n_encoder_layers: 4
        n_decoder_layers: 4
        dropout: 0.1
        n_epochs: 10
        """
        , yaml.SafeLoader
    )
)


class Trainer:
    def __init__(self, cfg: dotteddict):
        self.cfg = cfg
        self.dp = Seq2SeqDataMulti30k(src_lan="de", tgt_lan="en")  # data provider

        self.model = LMTransformerBilateralcoder(src_vocab_size=self.dp.vocabs[self.dp.src_lan].size,
                                                 tgt_vocab_size=self.dp.vocabs[self.dp.tgt_lan].size,
                                                 n_encoder_layers=self.cfg.n_encoder_layers,
                                                 n_decoder_layers=self.cfg.n_decoder_layers,
                                                 dropout_rate=self.cfg.dropout).to(CF.device)
        _ = TSUtils.count_parameters(self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.dp.PAD_IDX)
        self.train_dataloader = DataLoader(self.dp.datasets['train'], shuffle=True,
                                           batch_size=self.cfg.batch_size, collate_fn=self.dp.pad_batch)

    def train(self):
        self.model.train()
        for epoch in range(self.cfg.n_epochs):
            loss = self.train_epoch(epoch)
            print(f"Epoch {epoch} loss: {loss:.5f}")

    def train_epoch(self, epoch):

        sum_loss = 0.0
        self.model.train()

        for i, (src, tgt_orig) in enumerate(self.train_dataloader):
            src = src.to(self.cfg.device)
            tgt_orig = tgt_orig.to(self.cfg.device)
            tgt = tgt_orig[:, :-1]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = \
                TSUtils.make_seq2seq_mask(src, tgt, pad_idx=self.dp.PAD_IDX)

            logits = self.model(src, tgt, src_mask, tgt_mask, final_proj=True)
            tgt_y = tgt_orig[:, 1:]
            self.optimizer.zero_grad()
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_y.reshape(-1))
            loss.backward()
            self.optimizer.step()
            print(f"epoch[{epoch}] step[{i}/{len(self.train_dataloader)}] | loss: {loss.item():.4f}")
            sum_loss += loss.item()

        return sum_loss/len(self.train_dataloader)


if __name__ == "__main__":
    trainer = Trainer(CF)
    trainer.train()


