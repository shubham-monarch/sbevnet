#!/bin/bash

python3 compare_checkpoints.py --config configs/compare_checkpoints.yaml 
if [ $? -eq 0 ]; then
  echo "Checkpoint comparison completed successfully"
else
  echo "Error: Comparison failed"
  exit 1
fi 