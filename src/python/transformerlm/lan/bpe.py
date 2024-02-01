# -*- coding: utf-8 -*-
"""
Implements the BPE (Byte-pair Encoding Algorithm)
   according to the paper "Neural Machine Translation of Rare Words with Subword Units" (Rico sennrich et al., 2016)
"""
import io
from collections import OrderedDict, defaultdict, Counter


class VocabEntry:
    def __init__(self):
        self.units = []
        self.freq = 0


class BPE:
    EOW = "</w>"

    def __init__(self, max_types=30000):
        self.vocab = defaultdict(lambda: VocabEntry())
        self.subwords = OrderedDict()

    def build_vocab_from_corpus(self, corpus_or_file, first_n):
        if isinstance(corpus_or_file, list):
            data = corpus_or_file
        else:
            data = open(fname, "r", encoding="utf-8")

        for i, line in enumerate(data):
            if 0 <= first_n <= i:
                break
            words = line.strip().split()
            for w in words:
                self.vocab[w].freq += 1

        if isinstance(data, io.IOBase):
            data.close()

        self.init_vocab_units()

    def init_vocab_units(self):
        for w, v in self.vocab.items():
            v.units = list(w)
            v.units.append(self.EOW)

    def build_vocab_from_dict(self, dict_or_file):
        if isinstance(dict_or_file, dict):
            for w, freq in dict_or_file.items():
                self.vocab[w].freq = freq
        else:
            with open(dict_or_file, "r", encoding="utf-8") as f:
                for line in f:
                    w, freq = f.strip().split()
                    self.vocab[w].freq = freq

        self.init_vocab_units()

    def add_alphabet(self):
        alphabet = defaultdict(int)
        for v in self.vocab.values():
            for u in v.units:
                alphabet[u] += v.freq

        self.subwords.update(sorted(alphabet.items(), key=lambda x: x[1], reverse=True))

    def merge_vocab(self, vocab, subword, pair):
        n_matches = 0
        for v in vocab.values():
            i = 0
            while i < len(v.units) - 1:
                if v.units[i] == pair[0] and v.units[i+1] == pair[1]:
                    v.units[i] = subword
                    v.units.pop(i+1)
                    n_matches += 1
                i += 1

    def learn(self, max_size=30000):
        def get_stats(vocab):
            pairs = defaultdict(int)
            for v in vocab.values():
                for k, pair in enumerate(zip(v.units[:-1], v.units[1:])):
                    pairs[pair] += v.freq

            return pairs

        self.subwords = OrderedDict()
        self.add_alphabet()
        n_rounds = max_size - len(self.subwords)

        for i in range(n_rounds):
            cur_pairs = get_stats(self.vocab)
            if len(cur_pairs) == 0:
                break
            best_pair = max(cur_pairs, key=cur_pairs.get)
            new_subword = "".join(best_pair)
            self.subwords[new_subword] = cur_pairs[best_pair]
            self.merge_vocab(self.vocab, new_subword, best_pair)
            print(f"{new_subword}, {best_pair}, {cur_pairs[best_pair]}")


if __name__ == "__main__":
    bpe = BPE()
    vocab = {'low': 5, 'lower': 2, 'newest': 6, 'widest': 3}
    #fname = "data/nlp/WMT-14_en-de/train.en"
    #bpe.build_vocab(fname, first_n=100)
    bpe.build_vocab_from_dict(vocab)
    bpe.learn()


