#!/usr/bin/env bash

export PYTHONPATH=src/python:$PYTHONPATH

python -m transformerlm.train_ptb --device mps --num_iters 100000  \
                    --steps_per_eval 20000 \
                    --batch_size 2  \
                    --eval_test     \
                    --save_checkpoint checkpoint/lm_20240101
