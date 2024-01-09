#!/usr/bin/env bash

export PYTHONPATH=src/python:$PYTHONPATH

python -m transformerlm.data.seq2seq
