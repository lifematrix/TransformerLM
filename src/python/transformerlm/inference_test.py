# -*- coding: utf-8 -*-

import sys
import numpy as np
import torch
from .models import LMTransformerBilateralcoder
def inference_test():
    test_model = LMTransformerBilateralcoder(src_vocab_size=11,
                                             tgt_vocab_size=11, n_encoder_layers=2).to("mps")
    test_model.eval()

    src = torch.LongTensor(torch.LongTensor(np.arange(1, 11)[np.newaxis, ...])).to("mps")
    print(src)
    src_mask = torch.ones(1, 1, 10).to("mps")
    print(src_mask)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1,1).type_as(src).to("mps")

    for i in range(9):
        tgt_mask = test_model.make_causal_mask(ys).to("mps")
        out = test_model.decode(memory, src_mask, ys, tgt_mask)
        prob = test_model.generator(out[:, -1])
        next_token = torch.argmax(prob, dim=1)
        #print("tgt_mask, out, prob, next_token: ", tgt_mask.shape, out.shape, prob.shape, next_token.shape)

        ys = torch.cat([ys, next_token[None, ...]], dim=1)

    print("Example untrained model prediction: ", ys)


def run_test():
    for _ in range(9):
        inference_test()


if __name__ == "__main__":
    run_test()

