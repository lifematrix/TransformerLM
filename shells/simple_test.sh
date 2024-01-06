#!/usr/bin/env bash

export PYTHONPATH=src/python:$PYTHONPATH

python -m transformerlm.inference_test
