#!/bin/bash

# Run the SVO evaluation Python script with the config parameter using the fourth GPU (index 3)
python3 svo_eval.py --config configs/svo-eval.yaml

if [ $? -eq 0 ]; then
    echo "SVO evaluation completed successfully"
else
    echo "Error: SVO evaluation failed"
    exit 1
fi 