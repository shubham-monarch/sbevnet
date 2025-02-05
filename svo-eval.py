#! /usr/bin/env python3

import argparse
import os
import cv2
import pyzed.sl as sl
from tqdm import tqdm
import yaml
from pathlib import Path

from helpers import get_logger
from data_handler import ModelDataHandler
from evaluate import evaluate_sbevnet

class EvalSVO: 

    @staticmethod
    def evaluate_svo_folder(config_path: str): 
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        svo_file = config.get('svo_file')
        output_dir = config.get('output_dir')
        sampling_freq = config.get('sampling_freq')
        
        EvalSVO.generate_svo_train_data(svo_file, output_dir, sampling_freq)

        # download leaf-folder and generate model-dataset
        ModelDataHandler.generate_model_dataset(config_path)

        # evaluate model-dataset
        evaluate_sbevnet(config_path)

    @staticmethod
    def generate_svo_train_data(svo_file: str, output_dir: str, sampling_freq: int = 1) -> None:
        logger = get_logger("generate_svo_train_data")
        
        # assert not (Path(output_dir).exists() and any(Path(output_dir).iterdir())), \
        #     f"Output directory {output_dir} is not empty"
        
        if Path(output_dir).exists() and any(Path(output_dir).iterdir()):
            logger.warning(f"───────────────────────────────")
            logger.warning(f"Output directory {output_dir} already exists")
            logger.warning(f"Skipping SVO processing")
            logger.warning(f"───────────────────────────────")
            return

        zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(svo_file)
        if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            logger.error(f"Failed to open SVO file: {svo_file}")
            return

        runtime_parameters = sl.RuntimeParameters()
        image_left = sl.Mat()
        image_right = sl.Mat()
        frame_idx = 0

        total_frames = zed.get_svo_number_of_frames()
        with tqdm(total=total_frames, desc="Processing SVO", unit="frame") as pbar:
            while True:
                if zed.grab(runtime_parameters) != sl.ERROR_CODE.SUCCESS:
                    break
                if frame_idx % sampling_freq == 0:
                    zed.retrieve_image(image_left, sl.VIEW.LEFT)
                    zed.retrieve_image(image_right, sl.VIEW.RIGHT)
                    frame_folder = os.path.join(output_dir, f"frame-{frame_idx}")
                    os.makedirs(frame_folder, exist_ok=True)
                    left_img = image_left.get_data()
                    right_img = image_right.get_data()
                    left_img_resized = cv2.resize(left_img, (640, 480))
                    right_img_resized = cv2.resize(right_img, (640, 480))
                    cv2.imwrite(os.path.join(frame_folder, "left.jpg"), left_img_resized)
                    cv2.imwrite(os.path.join(frame_folder, "right.jpg"), right_img_resized)
                frame_idx += 1
                pbar.update(1)

        zed.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate SVO dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    args = parser.parse_args()
    EvalSVO.evaluate_svo_folder(args.config)

if __name__ == "__main__":
    main()
