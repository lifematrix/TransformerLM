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
        for i in range(max_len-1):
            tgt_mask = TSUtils.make_causal_mask(ys)
            output = self.model.decode(memory, src_mask, ys, tgt_mask)

            logits = self.model.generator(output[:, -1])
            next_pred = torch.argmax(logits, dim=-1, keepdim=True)
            ys = torch.cat([ys, next_pred], dim=1)
            if next_pred[0, 0].item() == self.lanmgr.EOS_IDX:
                break

        return ys[0]

    def beam_search(self, src, src_mask, max_len, beam_size=5):
        def topk_indices(d, k):
            top_vals, top_indices_flat = torch.topk(d.flatten(), k)
            L = d.size(1)
            top_indices = torch.stack((top_indices_flat // L, top_indices_flat % L), dim=1)
            top_vals = top_vals.unsqueeze(1)
            return top_vals, top_indices

        src = src.to(self.device)
        src_mask = src_mask.to(self.device)

        memory = self.model.encode(src, src_mask)
        ys = torch.ones(1, 1, dtype=torch.int64).fill_(self.lanmgr.BOS_IDX).to(self.device)
        cum_logprobs = torch.zeros_like(ys, dtype=torch.get_default_dtype())   # cumulative logprobs

        for i in range(max_len-1):
            if ys.size(0) != src.size(0):
                src = src.repeat((ys.size(0), 1))
                memory = memory.repeat((ys.size(0), 1, 1))

            tgt_mask = TSUtils.make_causal_mask(ys)
            output = self.model.decode(memory, src_mask, ys, tgt_mask)
            logits = self.model.generator(output[:, -1])
            logprobs = torch.log_softmax(logits, dim=1)
            logprobs_all = cum_logprobs + logprobs
            cum_logprobs, top_indices = topk_indices(logprobs_all, beam_size)
            next_yss = []
            for j in range(beam_size):
                b1, b2 = top_indices[j]
                next_token = torch.tensor([b2], device=ys.device)
                next_seq = torch.cat((ys[b1], next_token))
                next_yss.append(next_seq)

            ys = torch.stack(next_yss, dim=0)

        return ys

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
    def translate_beamsearch(self, src):
        self.model.eval()
        src = self.lanmgr.text2id(self.src_lan, src, to_tensor=True)   # converted into token ids.
        src = src[None, ...]  # add axis 0 as batch dim
        src_mask = torch.ones(src.shape[1]).to(torch.bool)
        pred_tokens_topn = self.beam_search(src, src_mask, max_len=50)

        pred_str = []
        for i in range(pred_tokens_topn.size(0)):
            pred_tokens = list(pred_tokens_topn[i].cpu().numpy())
            pred_str.append(" ".join(self.lanmgr.vocabs[self.tgt_lan][pred_tokens]))

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
    state_dict = torch.load("checkpoints/20240118_final.pt")

    translator = Translator.create_from_state(state_dict).to("cuda:0")
    print("\n".join(translator.translate_beamsearch("Zwei Autos fahren auf einer Rennstrecke.")))
    #print(translator.translate_beamsearch("Une fille au bord d'une plage avec une montagne au loin."))
    #print(translator.translate("Une fille au bord d'une plage avec une montagne au loin."))

