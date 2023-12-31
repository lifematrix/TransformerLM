#!/usr/bin/env bash

PYTHONPATH=src/python
python -m transformerlm.train_ptb --batch_size 2 --gpu
