# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class TSUtils:
    """Utility class for transformer models"""
    @classmethod
    def make_causal_mask(cls, x):
        seq_len =  x.size()[1]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=x.dtype), diagonal=0).to(x.device)
        return mask
