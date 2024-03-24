# -*- coding: utf-8 -*-

"""
read multi-corpus that has been normalized and pre-tokenized line by line,
count the frequency of words, and output a dictionary.
"""
import os
import sys
import io
from collections import defaultdict
import argparse
from ..utils import CommUtils
from tqdm import tqdm
import re


def log(msg):
    sys.stderr.write(f"{msg}\n")

class LinePreprocessor:
    def __init__(self):
        # Define a configuration list of substitution patterns
        self.subs_patterns = [
            (r"(\w);(\w)", r"\1 ##ISC## \2"),
            (r"(\w);", r"\1 ##SSC##"),
            (r";(\w)", r"##PSC## \1"),
            (r"-(\w)", r"##PHP## \1"),
            (r"(\w)-", r"\1 ##SHP##"),
            (r"\.(\w)", r"##PDT## \1"),
        ]

        # Compile the regular expressions and associate them with their replacements
        self.compiled_patterns = [(re.compile(pattern), replacement) for pattern, replacement in self.subs_patterns]

    def __call__(self, line):
        # Apply each substitution pattern
        for pattern, repl in self.compiled_patterns:
            line = pattern.sub(repl, line)
        return line


# Example usage
line_preprocessor = LinePreprocessor()


def process_data(read_iter, first_n=-1):
    vocab = defaultdict(int)

    pbar = tqdm(read_iter, desc="Read lines")
    for i, line in enumerate(pbar):
        if 0 <= first_n <= i:
            break
        # words = line.strip().split()
        line = line_preprocessor(line)
        for w in line.strip().split():
            vocab[w] += 1

        # if (i+1) % 10000 == 0:
        #    sys.stderr.write(f"[{CommUtils.now_str()}] {i+1}\n")

    return vocab


def parse_args():
    parser = argparse.ArgumentParser(description="Word count")
    parser.add_argument("corpus_files", nargs="*", type=str)
    parser.add_argument("-o", nargs="?", type=str, dest="vocab_file")
    parser.add_argument("--first", type=int, dest="first_n", default=-1)

    args = parser.parse_args()

    return args


def gen_corpus_iter(corpus_files):
    for fname in corpus_files:
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                yield line


def corpus_stats(source=None, output=None, first_n=-1):
    if isinstance(source, io.IOBase):
        read_iter = source
    elif isinstance(source, list) and len(source) > 0:
        read_iter = gen_corpus_iter(source)
    else:
        read_iter = sys.stdin

    vocab = process_data(read_iter, first_n=first_n)
    words_freq = sorted(vocab.items(), key=lambda t: (t[1], t[0]))
    
    if isinstance(output, io.IOBase):
        f_out = output
    elif isinstance(output, str):    # is file name
        f_out = open(output, "w", encoding="utf-8")
    else:
        f_out = sys.stdout
        
    for i, x in enumerate(words_freq):
        f_out.write(f"{x[0]} {x[1]}\n")

    # output is filename and should close it after open and write dict to it
    if isinstance(output, str):
        f_out.close()


def main():
    args = parse_args()
    log(args)
    corpus_stats(args.corpus_files, args.vocab_file, args.first_n)


def test_main():
    corpus = """
    hello world .  
    love and wisdom .
    hello world .  
    love and wisdom .
    """
    source = io.StringIO(corpus)
    corpus_stats(source, None)


if __name__ == "__main__":
    if os.environ.get("INNER_TEST", "0") == "1":
        test_main()
    else:
        main()
