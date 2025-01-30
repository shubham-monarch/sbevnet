#!/bin/bash

# Run the Python script with config file
python3 -m data-handler --config configs/generate-model-data.yaml

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "Dataset generation completed successfully"
else
    echo "Error: Dataset generation failed"
    exit 1
fi
