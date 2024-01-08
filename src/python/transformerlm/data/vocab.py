# -*- coding: utf-8 -*-
"""
Implement a custom vocabulary class
"""
from typing import List


class Vocabulary:
    """A custom class"""
    def __init__(self):
        self.voc_list = None
        self.idx_map = dict()
        self._default_index = None

    def __len__(self):
        return len(self.voc_list)

    def __getitem__(self, item: int or str or List[int] or List[str]):
        if isinstance(item, list):
            return [self.__getitem__(x) for x in item]
        elif isinstance(item, int):
            return self.voc_list[item]
        elif isinstance(item, str):
            return self.voc_dict[item]
        else:
            raise ValueError(f"The type of arg 'item' (value: {item}) is wrong!")

    def __contains__(self, item: str) -> bool:
        return self.voc_dict.__contains__(item)

    @property
    def size(self):
        return len(self.voc_list)

    @property
    def default_index(self):
        return self.set_default_index

    @default_index.setter
    def default_index(self, idx: int):
        self.default_index = idx

    @property
    def default_token(self):
        return self.voc_list[self._default_index]

    def build_map(self):

    def insert(self, token: str, index: int):
        self.voc_list.insert(index, str)