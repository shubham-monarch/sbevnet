#!/bin/bash

# Run the SVO evaluation Python script with the config parameter
python3 svo-eval.py --config configs/svo-eval.yaml

if [ $? -eq 0 ]; then
    echo "SVO evaluation completed successfully"
else
    echo "Error: SVO evaluation failed"
    exit 1
fi 