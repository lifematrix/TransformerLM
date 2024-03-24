#!/usr/bin/env bash

export PYTHONPATH=src/python:$PYTHONPATH

lan=en
src_dir="data/nlp/WMT-14_en-de"
tgt_dir="data/generated/WMT-14"

python -m transformerlm.lan.gen_vocab $src_dir/train.en -o $tgt_dir/vocab.en

