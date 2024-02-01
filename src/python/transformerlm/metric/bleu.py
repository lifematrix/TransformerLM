# -*- coding: utf-8 -*-
"""
Compute the BLEU score to evaluation of Neural Machine Learning Model.
"""

import math
from collections import Counter
import sacrebleu
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a

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

    def __str__(self):
        return (f"BLEU Score: {self.sentence_score} (Precisions: {self.lst2str(self.precisions)}) |"
                f" Counts: {self.lst2str(self.counts)}, Matches: {self.lst2str(self.matches)} |"
                f" BP: {self.BP:.5f}, Effective Reference Length: {self.lst2str(self.effective_ref_len)}, Hypothesis Length: {self.hyp_len},"
                f" References Length: {self.lst2str(self.refs_len)} | Max ngrams: {self.max_ngram}, "
                f" Weights: {self.lst2str(self.weights)})")

    def lst2str(self, lst, n_decimals=4):
        if lst is None:
            return ""

        if isinstance(lst, list):
            if len(lst) == 0:
                return ""

            if isinstance(lst[0], str):
                return " / ".join(lst)

            if isinstance(lst[0], float):
                return " / ".join([f"{x:.{n_decimals}f}" for x in lst])

            return " / ".join([str(x) for x in lst] )

        return f"{lst}"


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

        self.tokenizer = Tokenizer13a()

    def tokenize(self, text):
        text = self.tokenizer(text)
        tokens = text.lower().split()

        return tokens


    def count_ngrams(self, text, max_ngrams):
        tokens = self.tokenize(text.lower())
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
                refs_cnr = cnr
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
        bp = 1.0 if c >= r else math.exp(1 - r / c)

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
            bs.counts[n-1] += c
            bs.matches[n-1] += n_matches

        return bs

    def sentence_score(self, hyp_text, refs_text, max_ngrams=4, weights=None):
        if weights is None:
            weights = [1.0/max_ngrams] * max_ngrams

        bs = self.compute_stats(hyp_text, refs_text, max_ngrams)
        bs.BP, bs.effective_ref_len = self.BP(bs.hyp_len, bs.refs_len)

        bs.precisions = [0.0] * max_ngrams
        invcnt = self.invcnt
        for i in range(max_ngrams):
            if bs.counts[i] == 0:
                continue
            if bs.matches[i] > 0:
                n_matches = bs.matches[i]
            else:
                invcnt *= 2
                n_matches = 1/invcnt
            bs.precisions[i] = n_matches/bs.counts[i]

        bs.weights = weights
        wsum_log_ps = sum([math.log(p) * w for p, w in zip(bs.precisions, bs.weights) if p > 0.0])
        bs.sentence_score = bs.BP * math.exp(wsum_log_ps) * 100

        return bs

def test_sentence_score():
    example_1 = {
        "Candidate":
            [
                "It is a guide to action which ensures that the military always obeys the commands of the party.",
                "It is to insure the troops forever hearing the activity guidebook that party direct."
            ],
        "Reference":
            ["It is a guide to action that ensures that the military will forever heed Party commands.",
             "It is the guiding principle which guarantees the military forces always being under the command of the Party."
             ]
    }
    example_2 = {
        "Candidate":
            [
                "the the the the the the the."
            ],
        "Reference":
            [
                "The cat is on the mat.",
                "There is a cat on the mat."
            ]
    }

    example_3 = {
        "Candidate":
            [
                "of the"
            ],
        "Reference":
            [
                "It is a guide to action that ensures that the military will forever heed Party commands.",
                "It is the guiding principle which guarantees the military forces always being under the command of the Party.",
                "It is the practical guide for the army always to heed the directions of the party"
            ]
    }

    blue = BLEU()
    print(blue.sentence_score(example_1['Candidate'][0], example_1['Reference'], max_ngrams=4))
    print(blue.sentence_score(example_1['Candidate'][1], example_1['Reference'], max_ngrams=4))
    print(blue.sentence_score(example_2['Candidate'][0], example_2['Reference'], max_ngrams=4))
    print(blue.sentence_score(example_3['Candidate'][0], example_3['Reference'], max_ngrams=4))


if __name__ == "__main__":
    test_sentence_score()
