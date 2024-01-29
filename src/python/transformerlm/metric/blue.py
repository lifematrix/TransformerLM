# -*- coding: utf-8 -*-
"""
Compute the BLEU score to evaluation of Neural Machine Learning Model.
"""

import math
from collections import Counter
import sacrebleu

class BLEUScore:
    def __init__(self):
        self.max_ngram = None
        self.hyp_len = None
        self.refs_len = None
        self.effective_ref_len = None
        self.BP = None
        self.counts = None
        self.matches = None
        self.precisions = None
        self.weights = None
        self.sentence_score = None


class BLEU:
    """
    Implement blue score according to the paper "BLEU: a Method for Automatic Evaluation of Machine Translation"
    """
    SMOOTH_METHODS = {
        "mteval",
    }
    def __init__(self, smooth_method="mteval", invcnt=1):

        if smooth_method not in self.SMOOTH_METHODS:
            raise ValueError(f"Smoothing method {smooth_method} is not supported")

        self.smooth_method = smooth_method
        self.invcnt = 1

        self.tokenizer = sacrebleu.tokenizers.tokenizer_13a.Tokenizer13a()

    def tokenize(self, text):
        text = self.tokenizer(text)
        tokens = text.lower().split()

        return tokens


    def count_ngrams(self, text, max_ngrams):
        text = self.tokenize(text)
        tokens = text.lower().split()
        cnr = Counter()
        for ngrams in range(1, max_ngrams+1):
            ngram_segments = [tuple(tokens[i:i+ngrams]) for i in range(len(tokens)-ngrams+1)]
            cnr.update(ngram_segments)

        return cnr, len(tokens)

    def count_refs_ngram(self, refs_text, max_ngrams=1):
        refs_len = []
        refs_cnr = None

        for ref_text in refs_text:
            cnr, l = self.count_ngrams(ref_text, max_ngrams)
            refs_len.append(l)
            if refs_cnr is None:
                ref_cnr = cnr
            else:
                for k, c in cnr.items():
                    refs_cnr[k] = max(c, refs_cnr.get(k, 0))

        return refs_cnr, refs_len

    def BP(self, hyp_len, refs_len):
        """calculate brevity penalty
        """
        refs_len = list(set(refs_len))  # de-duplicate
        sorted_refs_len = sorted(refs_len, key=lambda x: (abs(hyp_len - x), x))
        r = sorted_refs_len[0]

        c = hyp_len
        bp = 1 if c >= r else math.exp(1 - r / c)

        return bp, r

    def compute_stats(self, hyp_text, refs_text, max_ngrams=1):
        """hyp_sen: candidate text/sentence
        refs_text: reference text/sentence, or ground truth
        """

        bs = BLEUScore()

        bs.max_ngram = max_ngrams
        hyp_cnr, bs.hyp_len = self.count_ngrams(hyp_text, max_ngrams=max_ngrams)
        refs_cnr, bs.refs_len = self.count_refs_ngram(refs_text, max_ngrams=max_ngrams)

        bs.counts = [0]*max_ngrams
        bs.matches = [0]*max_ngrams

        for k, c in hyp_cnr.items():
            n = len(k)
            n_matches = min(c, refs_cnr.get(k, 0))
            bs.counts[n] = n
            bs.matches[n] = n_matches

        return bs

    def sentence_score(self, hyp_text, refs_text, max_ngrams=4, weights=None):
        if weights is None:
            weights = [1.0/max_ngrams] * max_ngrams

        bs = self.compute_stats(hyp_text, refs_text, max_ngrams)
        bs.BP, bs.effective_ref_len = self.BP(bs.hyp_len, bs.refs_len)

        bs.precisions = [0.0] * max_ngrams
        invcnt = self.invcnt
        for i in range(1, max_ngrams+1):
            if bs.matches[i-1] > 0:
                n_matches = bs.matches[i-1]
            else:
                invcnt *= 2
                n_matches = 1/invcnt
            bs.precisions[i] = n_matches/bs.counts[i]

        bs.weights = weights
        wsum_log_ps = sum([math.log(p) * w for p, w in zip(bs.precisions, bs.weights)])
        bs.sentence_score = bs.BP * math.exp(wsum_log_ps) * 100

        return bs

