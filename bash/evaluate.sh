#!/bin/bash

# rm -rf logs/evaluate.log
exec >> logs/evaluate.log 2>&1

python evaluate.py \
    --config configs/evaluate.yaml
    