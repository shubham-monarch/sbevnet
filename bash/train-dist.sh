#!/bin/bash

rm -rf logs/train-dist.log
exec >> logs/train-dist.log 2>&1

python train-dist.py --config configs/train-dist.yaml

# python train-dist.py --config configs/train-dist.yaml --resume checkpoints/best_model.pth