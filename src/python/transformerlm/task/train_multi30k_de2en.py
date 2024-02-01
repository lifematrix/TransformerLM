# -*- coding: utf-8 -*-
"""
The program to train q seq2seq model based on Transformer to translate [de]utsch to [en]glish.
"""
import os.path

import numpy as np
import torch
from torch.utils.data import DataLoader
from dotteddict import dotteddict
import yaml
from tqdm import tqdm
import argparse
from collections import OrderedDict

from ..data import Seq2SeqDataMulti30k
from ..model import TSUtils, LMTransformerBilateralcoder
from ..utils import LogUtils, CommUtils
from ..metric import BLEU
from . import Translator
import logging


class Trainer:
    def __init__(self, cfg: dotteddict, lanmgr=None, ckpt_file=None):
        self.cfg = cfg

        print(f"Set torch manual seed {self.cfg.torch_seed}")
        torch.manual_seed(self.cfg.torch_seed)

        self.ckpt_file = ckpt_file
        self.dp = Seq2SeqDataMulti30k(src_lan="de", tgt_lan="en", lanmgr=lanmgr)  # data provider
        self.lanmgr = self.dp.lanmgr if lanmgr is None else lanmgr

        self.model = LMTransformerBilateralcoder(src_vocab_size=self.lanmgr.vocabs[self.dp.src_lan].size,
                                                 tgt_vocab_size=self.lanmgr.vocabs[self.dp.tgt_lan].size,
                                                 n_encoder_layers=self.cfg.model.n_encoder_layers,
                                                 n_decoder_layers=self.cfg.model.n_decoder_layers,
                                                 d_model=self.cfg.model.d_model,
                                                 d_ff=self.cfg.model.d_ff,
                                                 dropout_rate=self.cfg.model.dropout).to(self.cfg.train.device)
        _ = TSUtils.count_parameters(self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.lanmgr.PAD_IDX)

        self.train_dataloader = DataLoader(self.dp.datasets['train'], shuffle=True,
                                           batch_size=self.cfg.train.batch_size, collate_fn=self.lanmgr.pad_batch)

        self.val_dataloader = DataLoader(self.dp.datasets['val'], shuffle=False,
                                         batch_size=self.cfg.train.batch_size, collate_fn=self.lanmgr.pad_batch)

        self.translator = Translator(src_lan=self.dp.src_lan, tgt_lan=self.dp.tgt_lan,
                                     model=self.model, lanmgr=self.lanmgr).to(self.cfg.train.device)

    def train(self):
        self.model.train()
        for epoch in range(self.cfg.train.n_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.eval_epoch(epoch)
            print(f"Epoch {epoch} | train loss: {train_loss:.5f}, val loss: {val_loss:.5f}")
            self.save_checkpoint(epoch, {'train_loss': train_loss, 'val_loss': val_loss})

        self.save_checkpoint("final")

    def save_checkpoint(self, num, extra_state=None):
        if self.ckpt_file is None or len(self.ckpt_file) == 0:
            print("The checkpoint file is not specified!")
            return

        fname, ext = os.path.splitext(self.ckpt_file)
        cur_ckpt_fname = f"{fname}_{num}{ext}"

        state_dict = OrderedDict({
            'model_config': self.cfg.model,
            'model': self.model.state_dict(),
            'lanmgr': self.lanmgr.state_dict(),
            'lans': self.dp.lans,
            'num': num,
            'datetime': CommUtils.now_str()
        })

        if extra_state is not None:
            state_dict.update(extra_state)

        torch.save(state_dict, cur_ckpt_fname)

        print(f"save checkpoints to {cur_ckpt_fname} OK")

    def train_epoch(self, epoch):

        self.model.train()
        losses = []

        pbar = tqdm(self.train_dataloader)
        for i, (src, tgt_orig) in enumerate(pbar):
            src = src.to(self.cfg.train.device)
            tgt_orig = tgt_orig.to(self.cfg.train.device)
            tgt = tgt_orig[:, :-1]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = \
                TSUtils.make_seq2seq_mask(src, tgt, pad_idx=self.lanmgr.PAD_IDX)

            logits = self.model(src, tgt, src_mask, tgt_mask, final_proj=True)
            tgt_y = tgt_orig[:, 1:]
            self.optimizer.zero_grad()
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_y.reshape(-1))
            loss.backward()
            self.optimizer.step()
            pbar.set_description(f"epoch[{epoch}] step[{i}/{len(self.train_dataloader)}] | loss: {loss.item():.4f}")
            losses.append(loss.item())

        return TSUtils.mean(losses)

    def eval_epoch(self, epoch=None):
        losses = []

        for i, (src, tgt_orig) in enumerate(self.val_dataloader):
            src = src.to(self.cfg.train.device)
            tgt_orig = tgt_orig.to(self.cfg.train.device)
            tgt = tgt_orig[:, :-1]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = \
                TSUtils.make_seq2seq_mask(src, tgt, pad_idx=self.lanmgr.PAD_IDX)

            logits = self.model(src, tgt, src_mask, tgt_mask, final_proj=True)
            tgt_y = tgt_orig[:, 1:]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_y.reshape(-1))
            losses.append(loss.item())

        return TSUtils.mean(losses)


def inner_test():
    config_yaml = """
        torch_seed: 12345
        train:
            batch_size: 128 
            n_epochs: 15 
            device: "cuda:0"
        model:
            n_encoder_layers: 3
            n_decoder_layers: 3  
            dropout: 0.1
            d_model: 512
            d_ff: 512 
    """

    CF = dotteddict(yaml.load(config_yaml, yaml.SafeLoader))

    trainer = Trainer(CF, ckpt_file="checkpoints/20240118.pt")
    trainer.train()
    translator = Translator(src_lan=trainer.dp.src_lan, tgt_lan=trainer.dp.tgt_lan,
                            model=trainer.model, lanmgr=trainer.lanmgr).to(CF.train.device)
    print(translator.translate("Zwei Autos fahren auf einer Rennstrecke."))


def train(args):
    config, _ = CommUtils.load_yaml_config(args.config_file)
    trainer = Trainer(dotteddict(config), ckpt_file=args.ckpt_file)
    trainer.train()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training seq2seq Transformer model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(help='task', dest="task")
    train_p = subparsers.add_parser('train', help='Train the model')
    train_p.add_argument('--config', help="The configuration file (YAML format)",
                         required=True, type=str, dest="config_file")
    train_p.add_argument('--ckpt', help='The path name of checkpoint to save during training',
                         required=False, dest="ckpt_file")

    inner_test_p = subparsers.add_parser('test_train', help='Test the training function for debugging')
    args = parser.parse_args()
    logging.info(args)

    return args


if __name__ == "__main__":
    LogUtils.initlog()
    args = parse_args()
    if args.task == "test_train":
        inner_test()
    elif args.task == "train":
        train(args)



