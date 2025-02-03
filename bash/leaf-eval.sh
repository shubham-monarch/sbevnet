#!/bin/bash

# Run the Python script with config file
python3 -m leaf-eval --config configs/leaf-eval.yaml

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "Dataset generation completed successfully"
else
    echo "Error: Dataset generation failed"
    exit 1
fi
