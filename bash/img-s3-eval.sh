#!/bin/bash

# Run the Python script with config file
python3 -m img-s3-eval --config configs/img-s3-eval.yaml

if [ $? -eq 0 ]; then
    echo "S3 data download completed successfully"
else
    echo "Error: S3 data download failed"
    exit 1
fi 