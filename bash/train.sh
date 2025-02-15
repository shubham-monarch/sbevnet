#!/bin/bash

# rm -rf logs/train.log
exec >> logs/train.log 2>&1

python train.py --config configs/train.yaml

# Uncomment the following line to resume training from the best checkpoint:
# python train.py --config configs/train.yaml --resume checkpoints/best_model.pth 