# -*- coding: utf-8 -*-
"""
Define data provider for seq2seq model.
"""

from collections import OrderedDict
from torchtext.utils import download_from_url, extract_archive
from torchtext.data.utils import get_tokenizer
from . import Vocabulary

class Seq2SeqDataMulti30k:
    URL_BASE = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"

    DATASET_URLS = OrderedDict([
        ("train", "train.[lan].gz"),
        ("val", "val.[lan].gz"),
        ("test", "test_2016_flickr.[lan].gz")
    ])

    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    UNK_TKN, PAD_TKN, BOS_TKN, EOS_TKN = "<unk>", "<pad>", "<bos>", "<eos>"

    def __init__(self, src_lan: str, tgt_lan: str):
        self.src_lan = src_lan
        self.tgt_lan = tgt_lan
        self.lans = (self.src_lan, self.tgt_lan)
        self.file_pairs = self.download()
        self.data = self.read_data(self.file_pairs)
        self.tokeninzers = {
            'de': get_tokenizer('spacy', language='de_core_news_sm'),
            'en': get_tokenizer('spacy', language='en_core_web_sm')
        }
        self.vocabs = self.build_vocabs()

    def download(self):

        url_pairs = OrderedDict(
            [
                (
                    k,
                    [self.URL_BASE+url.replace("[lan]", lan) for lan in self.lans]
                )
                for k, url in self.DATASET_URLS.items()
            ]
        )
        print(url_pairs)

        fname_pairs = OrderedDict(
            [
                (
                    k,
                    [extract_archive(download_from_url(url))[0] for url in urls]
                )
                for k, urls in url_pairs.items()
            ]
        )
        print(fname_pairs)

        return fname_pairs

    def read_data(self, fname_pairs):
        def read_pair(fname_pair):
            with open(fname_pair[0], encoding="utf8") as f0, open(fname_pair[1], encoding="utf8") as f1:
                lines = [(line0, line1) for line0, line1 in zip(f0, f1)]
            print(f"read {fname_pairs} The number of lines: {len(lines)}")

            return {
                self.src_lan: [x[0] for x in lines],
                self.tgt_lan: [x[1] for x in lines]
            }

        return OrderedDict(
            [
                (k, read_pair(fname_pair)) for k, fname_pair
                in fname_pairs.items()
            ]
        )


    def build_vocabs(self):
        data_to_voc = {
            lan: self.data['train'][lan] + self.data['val'][lan] for lan in self.lans
        }

        special_tokens = [self.UNK_TKN, self.PAD_TKN, self.BOS_TKN, self.EOS_TKN]
        vocabs = {
            lan: Vocabulary.create_from(map(lambda x: self.tokeninzers[lan](x), data_to_voc[lan]),
                                        specials=special_tokens)
            for lan in self.lans
        }

        for vocab in vocabs.values():
            vocab.default_index = self.UNK_IDX

        return vocabs


if __name__ == "__main__":
    d = Seq2SeqDataMulti30k(src_lan="de", tgt_lan="en")
    print({lan: len(d.vocabs[lan]) for lan in d.lans})



