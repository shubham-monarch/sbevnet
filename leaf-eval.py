import os
import yaml 
import argparse

from data_handler import ModelDataHandler
from helpers import get_logger
from evaluate import evaluate_sbevnet

class Evaluation: 

    @staticmethod
    def evaluate_leaf_folder(config_path: str): 
        # generate model-dataset for the leaf-folder
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # download leaf-folder and generate model-dataset
        ModelDataHandler.generate_model_dataset(config_path)

        # evaluate model-dataset
        evaluate_sbevnet(config_path)

        


def main():
    parser = argparse.ArgumentParser(description='Generate model dataset from GT dataset stored in S3')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    args = parser.parse_args()
    
    Evaluation.evaluate_leaf_folder(args.config)

if __name__ == "__main__":
    main()