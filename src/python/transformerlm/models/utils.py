# -*- coding: utf-8 -*-

import torch
from torch import Tensor
from prettytable import PrettyTable


class TSUtils:
    """Utility class for transformer models"""
    @classmethod
    def make_causal_mask(cls, x: Tensor):
        seq_len = x.size()[1]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=x.dtype), diagonal=0).to(torch.bool).to(x.device)
        return mask

    @classmethod
    def make_seq2seq_mask(cls, src: Tensor, tgt: Tensor, pad_idx: int=1):
        src_len = src.shape[1]

        src_mask = torch.ones(src_len, device=src.device).to(torch.bool)
        tgt_mask = cls.make_causal_mask(tgt)

        src_padding_mask = (src == pad_idx)
        tgt_padding_mask = (tgt == pad_idx)

        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    @classmethod
    def count_parameters(cls, model):
        table = PrettyTable(["Modules", "size", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, list(parameter.shape), params])
            total_params += params
        print(table)
        print(f"Total Trainable Params: {total_params}, or  {total_params/(1024*1024):.3f}M")
        return total_params

    @classmethod
    def mean(cls, data):
        return sum(data)/len(data)
