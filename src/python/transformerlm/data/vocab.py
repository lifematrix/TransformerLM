# -*- coding: utf-8 -*-
"""
Implement a custom vocabulary class
"""
from typing import List, Iterable, Optional
from collections import Counter, OrderedDict


class Vocabulary:
    """A custom class"""
    def __init__(self):
        self.voc_list = []
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
        return self._default_index

    @default_index.setter
    def default_index(self, idx: int):
        self._default_index = idx

    @property
    def default_token(self):
        return self.voc_list[self._default_index] if self._default_index is not None else None

    def build_map(self, start: int = 0):
        if self.idx_map is None:
            self.idx_map = dict()
        self.idx_map.update({token: (i+start) for i, token in enumerate(self.voc_list[start:])})

    def insert_token(self, token: str or List[str], index: int):
        if index < 0:
            index += len(self.voc_list)
        self.voc_list[index:index] = token
        self.build_map(index)
    
    def append_token(self, token: str or List[str]):
        index = len(self.voc_list)
        self.insert_token(token, index)

    @classmethod
    def create_from(cls, iterator: Iterable, min_freq: int = 1,
                    specials: Optional[str or List[str]] = None, special_first: bool = True,
                    max_tokens: Optional[int] = None):
        def select_tokens(iterator, min_freq):
            counter = Counter()
            for tokens in iterator:
                counter.update(tokens)

            tokens_sorted = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
            if max_tokens is not None and max_tokens > 1:
                tokens_sorted = tokens_sorted[:max_tokens]

            return [x[0] for x in tokens_sorted]

        new_voc = Vocabulary()
        new_voc.voc_list = select_tokens(iterator, min_freq)

        if specials is not None:
            pos = 0 if special_first else len(new_voc.voc_list)
            new_voc.voc_list[pos:pos] = specials

        new_voc.build_map()

        return new_voc


