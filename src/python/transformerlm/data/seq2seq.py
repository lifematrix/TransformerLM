# -*- coding: utf-8 -*-
"""
Define dataset and data provider class for seq2seq models.
"""

from typing import List, Optional, Callable
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import OrderedDict
from torchtext.utils import download_from_url, extract_archive
from torchtext.data.utils import get_tokenizer
from ..lan import LanguageSetManager


class Seq2SeqDataset(Dataset):
    def __init__(self, src_list: List[str], tgt_list: List[str],
                 src_transform: Optional[Callable] = None, tgt_transform: Optional[Callable] = None):
        if len(src_list) != len(tgt_list):
            raise ValueError(f"The length of source list {len(src_list)} is not equal to {len(tgt_list)}")

        self.src_list = src_list
        self.tgt_list = tgt_list
        self.src_transform = src_transform
        self.tgt_transform = tgt_transform

    def __getitem__(self, idx):
        src_item = self.src_list[idx].rstrip("\n")
        if self.src_transform is not None:
            src_item = self.src_transform(text=src_item)

        tgt_item = self.tgt_list[idx].rstrip("\n")
        if self.tgt_transform is not None:
            tgt_item = self.tgt_transform(text=tgt_item)

        return src_item, tgt_item

    def __len__(self):
        return len(self.src_list)

    def size(self):
        return self.__len__()

class Seq2SeqDataMulti30k:
    URL_BASE = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"

    DATASET_URLS = OrderedDict([
        ("train", "train.[lan].gz"),
        ("val", "val.[lan].gz"),
        ("test", "test_2016_flickr.[lan].gz")
    ])

    def __init__(self, src_lan: str, tgt_lan: str):
        self.src_lan = src_lan
        self.tgt_lan = tgt_lan
        self.lans = (self.src_lan, self.tgt_lan)
        self.data = self.read_data()

        self.lanmgr = self.create_lanmgr()

        self.datasets = {
            k: Seq2SeqDataset(src_list=v[self.src_lan], tgt_list=v[self.tgt_lan],
                              src_transform=partial(self.preprocess, lan=self.src_lan),
                              tgt_transform=partial(self.preprocess, lan=self.tgt_lan))
            for k, v in self.data.items()
        }

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

    def read_data(self):
        fname_pairs = self.download()

        def read_pair(fname_pair):
            with open(fname_pair[0], encoding="utf8") as f0, open(fname_pair[1], encoding="utf8") as f1:
                lines = [(line0.rstrip("\n"), line1.rstrip("\n")) for line0, line1 in zip(f0, f1)]
            print(f"read {fname_pair} The number of lines: {len(lines)}")

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

    def create_lanmgr(self):
        lan_text = {
            #lan: self.data['train'][lan] + self.data['val'][lan] for lan in self.lans
            lan: self.data['train'][lan] for lan in self.lans
        }

        lanmgr = LanguageSetManager()
        lanmgr.build_vocabs(lan_text)

        return lanmgr



def test_main():
    d = Seq2SeqDataMulti30k(src_lan="de", tgt_lan="en")
    print(d.lanmgr.vocabs['en']["<eos>"])
    print(d.lanmgr.vocabs['en'][100])
    print({lan: len(d.lanmgr.vocabs[lan]) for lan in d.lans})
    print(d.preprocess("de", "Drei Leute sitzen an einem Picknicktisch vor einem Gebäude, das wie der Union Jack bemalt ist."))
    print(d.preprocess("en", "Three people sit at a picnic table outside of a building painted like a union jack."))
    # for k, ds in d.datasets.items():
    #     print(k)
    #     for i, item in enumerate(ds):
    #         if i > 5:
    #             break
    #         print(f"##{i}", item)
    #         print(d.vocabs[d.src_lan][item[0]], d.vocabs[d.tgt_lan][item[1]])
    #         print(ds.src_list[i], ds.tgt_list[i])

    train_dataloader = DataLoader(d.datasets['train'], batch_size=4, collate_fn=d.lanmgr.pad_batch)
    for i, batch in enumerate(train_dataloader):
        if i > 3:
            break
        print(i, batch)


if __name__ == "__main__":
    test_main()



