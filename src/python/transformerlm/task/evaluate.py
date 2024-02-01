#-*- coding: utf-8 -*-

"""
Define the evaluator to compute the bleu and other metric on a trained language model.
"""

import torch
from torch.utils.data import DataLoader
from ..utils import LogUtils
from ..metric import BLEU
from ..lan import LanguageSetManager
from ..data import Seq2SeqDataMulti30k
from ..model import TSUtils
from . import Translator
import logging

class Evaluator:
    def __init__(self, model_ckpt_fname, batch_size=16, device=None):
        self.device = device if device is not None else self.determine_device()
        logging.info(f"Use device: {self.device}")

        self.batch_size = batch_size
        state_dict = torch.load(model_ckpt_fname)
        print(state_dict.keys())
        self.translator = Translator.create_from_state(state_dict).to(self.device)

        self.dp = Seq2SeqDataMulti30k(src_lan="de", tgt_lan="en", lanmgr=self.translator.lanmgr)
        self.dataloader = DataLoader(self.dp.datasets['test'], shuffle=False,
                                     batch_size=self.batch_size, collate_fn=self.translator.lanmgr.pad_batch)

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.translator.lanmgr.PAD_IDX)

    @classmethod
    def determine_device(cls):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

        return device

    def compute_batch_bleu(self, src_batch, tgt_batch):
        bleu = BLEU()
        scores = []
        for i, (src, tgt) in enumerate(zip(src_batch, tgt_batch)):

            src_str = " ".join(self.translator.lanmgr.vocabs[self.dp.src_lan][src.cpu().numpy()])
            hyp_str = self.translator.translate(src_str)
            ref_str = " ".join(self.translator.lanmgr.vocabs[self.dp.tgt_lan][tgt.cpu().numpy()])

            hyp_str = self.translator.strip_specials_tokens(hyp_str)
            ref_str = self.translator.strip_specials_tokens(ref_str)
            sc = bleu.sentence_score(hyp_str, [ref_str])
            logging.info(f"eval bleu: {i}\nHyp: {hyp_str}\nRef: {ref_str}\nScore: {sc}")
            scores.append(sc)

        return scores

    def run(self):
        self.translator.model.eval()

        losses = []
        bleu_scores = []

        for i, (src, tgt_orig) in enumerate(self.dataloader):
            src = src.to(self.device)
            tgt_orig = tgt_orig.to(self.device)
            tgt = tgt_orig[:, :-1]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = \
                TSUtils.make_seq2seq_mask(src, tgt, pad_idx=self.translator.lanmgr.PAD_IDX)

            logits = self.translator.model(src, tgt, src_mask, tgt_mask, final_proj=True)
            tgt_y = tgt_orig[:, 1:]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_y.reshape(-1))
            losses.append(loss.item())
            bleu_scores.extend(self.compute_batch_bleu(src, tgt_orig))

        m_loss = TSUtils.mean(losses)
        m_bleu = TSUtils.mean([bs.sentence_score for bs in bleu_scores])
        logging.info(f"mean loss: {m_loss:.4f}, mean bleu: {m_bleu:.4f}")


if __name__ == "__main__":
    LogUtils.initlog()
    evaluator = Evaluator("checkpoints/20240118_final.pt")
    evaluator.run()
    print(evaluator.translator.translate("Zwei Autos fahren auf einer Rennstrecke."))
