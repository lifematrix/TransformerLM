#!/usr/bin/env bash

export PYTHONPATH=src/python:$PYTHONPATH

datetime=`date +"%Y%m%d-%H%M%S"`
task_dir=$(dirname $0)
model_name="seq2seq-transformer"

ckpt_file=checkpoints/${model_name}_${datetime}.pt

python -m transformerlm.task.train_multi30k_de2en train \
      --config $task_dir/config.yaml             \
      --ckpt $ckpt_file

