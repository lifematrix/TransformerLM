# -*- coding: utf-8 -*-

from typing import Mapping, Any, List
from torch import nn
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from . import Vocabulary


_tokenizers = {
    'de': get_tokenizer('spacy', language='de_core_news_sm'),
    'en': get_tokenizer('spacy', language='en_core_web_sm')
}


class LanguageSetManager(nn.Module):
    """Manage the set of multiple languages."""

    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    UNK_TKN, PAD_TKN, BOS_TKN, EOS_TKN = "<unk>", "<pad>", "<bos>", "<eos>"

    def __init__(self):
        super(LanguageSetManager, self).__init__()
        self.tokenizers = _tokenizers
        self.vocabs = dict()

    @property
    def lans(self):
        return list(self.vocabs.keys())

    def build_vocabs(self, lans_text):

        special_tokens = [self.UNK_TKN, self.PAD_TKN, self.BOS_TKN, self.EOS_TKN]
        vocabs = {
            lan: Vocabulary.create_from(map(lambda x: self.tokenizers[lan](x), lans_text[lan]),
                                        specials=special_tokens)
            for lan in self.lans
        }

        for vocab in vocabs.values():
            vocab.default_index = self.UNK_IDX

        return vocabs

    def state_dict(self, *args, **kwargs):
        d = super(LanguageSetManager, self).state_dict()
        d["vacabs"] = self.vocabs

        return d

    def load_state_dict(self, state_dict: dict, strict: bool = True, assign: bool = False):

        self.vocabs = state_dict.pop("vocabs", dict())
        super(LanguageSetManager, self).load_state_dict(state_dict, strict, assign)

    def add_head_tail(self, token_ids: List[int]):
        return [self.BOS_IDX] + token_ids + [self.EOS_IDX]

    def text2id(self, lan: str, text: str, bos_eos=True):
        tokens = self.tokenizers[lan](text)
        token_ids = self.vocabs[lan][tokens]
        if bos_eos:
            self.add_head_tail(token_ids)

        return token_ids

    def id2text(self, lan: str, token_ids: str):

        return self.vocabs[lan][token_ids]

    def pad_batch(self, batch):
        src_batch, tgt_batch = [], []
        for x in batch:
            src_batch.append(x[0])
            tgt_batch.append(x[1])

        src_batch = pad_sequence(src_batch, batch_first=True, padding_value=self.PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=self.PAD_IDX)

        return src_batch, tgt_batch



