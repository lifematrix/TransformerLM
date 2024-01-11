#!/usr/bin/env bash

export PYTHONPATH=src/python:$PYTHONPATH

python -m transformerlm.train_multi30k_de2en
