#!/usr/bin/env python
# coding: utf-8



corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]


from transformers import AutoTokenizer
from collections import defaultdict


def compute_pairs_scores(word_freqs, word_splits, tokens_freqs):
    pair_freqs = defaultdict(int)
#    tokens_freqs = defaultdict(int)
    for word, clist in word_splits.items():
        wfreq = word_freqs[word]
        # for c in clist:
        #     tokens_freqs[c] += wfreq

        for pair in zip(clist[:-1], clist[1:]):
            pair_freqs[pair] += wfreq

    scores = {
        p: (
                freq / (tokens_freqs[p[0]] * tokens_freqs[p[1]]),
                (freq, tokens_freqs[p[0]],  tokens_freqs[p[1]])
            )
        for p, freq in pair_freqs.items()
    }

    return scores


def merge_pair(pair, splits, token_freqs, word_freqs):
    a, b = pair
    new_token = a + b[2:] if b.startswith("##") else a + b
    for word, clist in splits.items():
        i = 0
        while i < len(clist) - 1:
            if clist[i] == a and clist[i + 1] == b:
                clist[i] = new_token
                clist.pop(i + 1)
                wfreq = word_freqs[word]
                token_freqs[a] -= wfreq
                token_freqs[b] -= wfreq
            i += 1

    return splits, new_token


def learn():

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    word_freqs = defaultdict(int)
    for text in corpus:
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        for word, _ in words_with_offsets:
            word_freqs[word] += 1

    alphabet = set()
    tokens_freqs = defaultdict(int)
    word_splits = dict()

    for word, freq in word_freqs.items():
        clist = [c if i==0 else "##" + c for i, c in enumerate(word)]
        for c in clist:
            if c not in alphabet:
                alphabet.add(c)
            tokens_freqs[c] += freq
        word_splits[word] = clist

    alphabet = sorted(alphabet)

    print(alphabet)
    print(tokens_freqs)
    print(word_splits)

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet
    print(vocab)


    import copy

    max_vocab_size = 70
    cur_word_splits = copy.deepcopy(word_splits)
    cur_vocab = copy.deepcopy(vocab)
    cur_token_freqs = copy.deepcopy(tokens_freqs)

    while len(cur_vocab) < max_vocab_size:
        print("-"*80)
        pair_scores = compute_pairs_scores(word_freqs, cur_word_splits, cur_token_freqs)
        if len(pair_scores) == 0:
            break
        best_pair, (best_score, (best_freq, freq_a, freq_b)) = max(pair_scores.items(), key=lambda x: x[1][0])

        cur_word_splits, new_token = merge_pair(best_pair, cur_word_splits, cur_token_freqs, word_freqs)
    #    new_token = best_pair[0] + best_pair[1][2:]
    #    new_token = best_pair[0] + best_pair[1][2:] if best_pair[1].startswith("##") else best_pair[0] + best_pair[1]
        print(best_pair, best_score, best_freq, new_token, freq_a, freq_b)
        # if new_token == "##ful":
        #     print(pair_scores[('##c', '##t')])
        #     print(sorted(pair_scores.items(), key=lambda x:x[1][0], reverse=True))
        cur_vocab.append(new_token)
        cur_token_freqs[new_token] = best_freq

    print(cur_word_splits)
    print(cur_vocab)



if __name__ == "__main__":
    learn()


