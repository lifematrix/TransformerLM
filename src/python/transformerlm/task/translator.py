# -*- coding: utf-8 -*-

import torch
from ..lan import LanguageSetManager
from ..model import TSUtils, LMTransformerBilateralcoder
import re


class Translator:
    def __init__(self, src_lan: str = None, tgt_lan: str = None, lanmgr=None, model=None):
        super(Translator, self).__init__()
        self.src_lan = src_lan
        self.tgt_lan = tgt_lan
        self.lanmgr = lanmgr
        self.model = model
        self.device = "cpu"

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)

        return self

    def greedy_decode(self, src, src_mask, max_len):
        src = src.to(self.device)
        src_mask = src_mask.to(self.device)

        memory = self.model.encode(src, src_mask)
        ys = torch.ones(1, 1, dtype=torch.int64).fill_(self.lanmgr.BOS_IDX).to(self.device)
        output_prev = None
        for i in range(max_len-1):
            tgt_mask = TSUtils.make_causal_mask(ys)
            output = self.model.decode(memory, src_mask, ys, tgt_mask)
            # print("output.shape", output.shape)
            # if output_prev is not None:
            #     print(i, torch.all(output[:, :-1] == output_prev))

            logits = self.model.generator(output[:, -1])
            next_pred = torch.argmax(logits, dim=-1, keepdim=True)
            ys = torch.cat([ys, next_pred], dim=1)
            if next_pred[0, 0].item() == self.lanmgr.EOS_IDX:
                break
            output_prev = output

        return ys[0]


    def strip_specials_tokens(self, text: str):
        text = text.replace(self.lanmgr.PAD_TKN, "")
        text = text.replace(self.lanmgr.BOS_TKN, "")
        text = text.replace(self.lanmgr.EOS_TKN, "")
        text = re.sub("\s+.\s+$", ".", text)
        text = re.sub("^\s+", "", text)

        return text



    def translate(self, src):
        self.model.eval()
        src = self.lanmgr.text2id(self.src_lan, src, to_tensor=True)   # converted into token ids.
        src = src[None, ...]  # add axis 0 as batch dim
        src_mask = torch.ones(src.shape[1]).to(torch.bool)
        pred_tokens = self.greedy_decode(src, src_mask, max_len=200)
        pred_tokens = list(pred_tokens.cpu().numpy())
        pred_str = " ".join(self.lanmgr.vocabs[self.tgt_lan][pred_tokens])

        return pred_str


    @classmethod
    def create_from_state(cls, state_dict):
        inst = Translator()
        inst.src_lan, inst.tgt_lan = state_dict['lans']
        
        mcfg = state_dict['model_config']
        inst.lanmgr = LanguageSetManager().load_state_dict(state_dict['lanmgr'])
        inst.model = LMTransformerBilateralcoder(src_vocab_size=inst.lanmgr.vocabs[inst.src_lan].size,
                                                 tgt_vocab_size=inst.lanmgr.vocabs[inst.tgt_lan].size,
                                                 n_encoder_layers=mcfg.n_encoder_layers,
                                                 n_decoder_layers=mcfg.n_decoder_layers,
                                                 d_model=mcfg.d_model,
                                                 d_ff=mcfg.d_ff,
                                                 dropout_rate=mcfg.dropout)
        inst.model.load_state_dict(state_dict['model'])
        inst.model.eval()

        return inst


if __name__ == "__main__":
    state_dict = torch.load("checkpoints/seq2seq-transformer_20240119-155222_final.pt")
    print(state_dict.keys())

    translator = Translator.create_from_state(state_dict).to("cuda:0")
    #print(translator.translate("Zwei Autos fahren auf einer Rennstrecke."))
    print(translator.translate("Une fille au bord d'une plage avec une montagne au loin."))

