#!/usr/bin/env python3
import argparse
import yaml
from svo_eval import EvalSVO  # Assuming your EvalSVO class is in svo_eval.py

def main():
    parser = argparse.ArgumentParser(description="Run SVO extraction in an isolated environment")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    gpu_id = config.get("gpu_id", 0)
    svo_file = config.get('svo_file')
    output_dir = config.get('output_dir')
    sampling_freq = config.get('sampling_freq')
    frame_cnt = config.get('frame_cnt', -1)

    EvalSVO.generate_svo_train_data(svo_file, output_dir, sampling_freq, frame_cnt, gpu_id)

if __name__ == "__main__":
    main() 