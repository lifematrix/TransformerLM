#!/usr/bin/env bash

export PYTHONPATH=src/python:$PYTHONPATH

python -m transformerlm.train_ptb --device cuda --num_iters 100000  \
                    --steps_per_eval 20000 \
                    --batch_size 6  \
                    --eval_test     \
                    --save_checkpoint checkpoint/lm_20240101
