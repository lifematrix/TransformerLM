# -*- coding: utf-8 -*-

from typing import Mapping, Any, List
from collections import OrderedDict
import logging

import torch
from torch import nn
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from . import Vocabulary



_tokenizers = OrderedDict(**{
    'de': get_tokenizer('spacy', language='de_core_news_sm'),
    'en': get_tokenizer('spacy', language='en_core_web_sm'),
    'fr': get_tokenizer('spacy', language='fr_core_news_sm'),

})


class LanguageSetManager:
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
            for lan in lans_text.keys()
        }

        for vocab in vocabs.values():
            vocab.default_index = self.UNK_IDX

        logging.info("Build vocabs: %s" % ' '.join(["%s: %s" % (k, len(v)) for k, v in vocabs.items()]))

        self.vocabs = vocabs

    def state_dict(self):
        return OrderedDict({'vocabs': {k: v.state_dict() for k, v in self.vocabs.items()}})

    def load_state_dict(self, state_dict: dict, strict: bool = True, assign: bool = False):

        vocabs = state_dict["vocabs"]
        for k, v in vocabs.items():
            self.vocabs.update({k: Vocabulary()})
            self.vocabs[k].load_state_dict(v)

        return self

    def add_head_tail(self, token_ids: List[int]):
        return [self.BOS_IDX] + token_ids + [self.EOS_IDX]

    def text2id(self, lan: str, text: str, bos_eos=True, to_tensor=False):
        tokens = self.tokenizers[lan](text)
        token_ids = self.vocabs[lan][tokens]
        if bos_eos:
            token_ids = self.add_head_tail(token_ids)

        if to_tensor:
            token_ids = torch.tensor(token_ids, dtype=torch.int64)

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



