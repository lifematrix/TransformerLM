# -*- coding: utf-8 -*-
"""
Implements the BPE (Byte-pair Encoding Algorithm)
   according to the paper "Neural Machine Translation of Rare Words with Subword Units" (Rico sennrich et al., 2016)
"""
import os
import io
import argparse
from collections import OrderedDict, defaultdict
from tqdm import tqdm
from ..utils import TimeU


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
            data = open(corpus_or_file, "r", encoding="utf-8")

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
                    w, freq = line.strip().split()
                    self.vocab[w].freq = int(freq)

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
        def get_pairs_stat(vocab):
            pairs = defaultdict(int)
            for v in vocab.values():
                for k, pair in enumerate(zip(v.units[:-1], v.units[1:])):
                    pairs[pair] += v.freq

            return pairs

        self.subwords = OrderedDict()
        with TimeU.trace_time("add alphabet"):
            self.add_alphabet()
        n_rounds = max_size - len(self.subwords)

        for i in tqdm(range(n_rounds)):
            with TimeU.trace_time("get_pairs_stat"):
                cur_pairs = get_pairs_stat(self.vocab)
            if len(cur_pairs) == 0:
                break
            best_pair = max(cur_pairs, key=cur_pairs.get)
            new_subword = "".join(best_pair)
            self.subwords[new_subword] = cur_pairs[best_pair]
            with TimeU.trace_time("merge_vocab"):
                self.merge_vocab(self.vocab, new_subword, best_pair)
            print(f"\n{new_subword}, {best_pair}, {cur_pairs[best_pair]}")

    def save(self, bpe_learn_fname):
        with open(bpe_learn_fname, "w") as f:
            f.write("%d\n" % len(self.vocab))
            for k, v in self.vocab.items():
                f.write("%s %s %s\n" % (k, v.freq, " ".join(v.units)))

            f.write("%d\n" % len(self.subwords))
            for k, v in self.subwords.items():
                f.write("%s %s\n" % (k, v))

def test_main_simple():
    bpe = BPE()
    vocab = {'low': 5, 'lower': 2, 'newest': 6, 'widest': 3}
    #fname = "data/nlp/WMT-14_en-de/train.en"
    #bpe.build_vocab(fname, first_n=100)
    bpe.build_vocab_from_dict(vocab)
    bpe.learn()


def test_main_vocab():
    bpe = BPE()
    vocab_file = "data/generated/WMT-14/vocab.en"
    bpe_learn_file = "data/generated/WMT-14/bpe_learn_vocab.en"
    bpe.build_vocab_from_dict(vocab_file)
    bpe.learn(max_size=3000)
    bpe.save(bpe_learn_file)


def parse_args():
    parser = argparse.ArgumentParser(description="BPE Learn")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()


if __name__ == "__main__":
    if os.environ.get("INNER_TEST", "0") == "1":
        test_main_vocab()
    else:
        main()

