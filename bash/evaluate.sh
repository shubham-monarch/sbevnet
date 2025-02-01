#!/bin/bash

rm -rf logs/evaluate.log
exec >> logs/evaluate.log 2>&1

python evaluate_model.py \
    --config configs/evaluate.yaml \
    --color_map configs/Mavis.yaml 