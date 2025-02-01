#!/bin/bash
exec >> logs/main.log 2>&1

python train-dist.py --config configs/train-dist.yaml

# python train-dist.py --config configs/train-dist.yaml --resume checkpoints/best_model.pth