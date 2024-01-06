# -*- coding: utf-8 -*-

import torch

def data_gen(V, batch_size, n_batches):
    """Generate random data for a src-tgt copy task."""

    for i in range(n_batches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).colone().detach()
        tgt = data.requires_grad_(False).colone().detach()
        yield Batches(src, tgt, 0)
