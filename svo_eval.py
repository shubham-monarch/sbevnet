#! /usr/bin/env python3

import argparse
import os
import cv2
from tqdm import tqdm
import yaml
from pathlib import Path
import torch
import subprocess

from helpers import get_logger
from data_handler import ModelDataHandler
from evaluate import evaluate_sbevnet

class EvalSVO: 

    @staticmethod
    def evaluate_svo_folder(config_path: str): 
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Retrieve the GPU id from the config; this is the physical GPU id.
        gpu_id = config.get("gpu_id", 0)
        
        # -- SVO extraction in isolated subprocess ---------------
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        ret = subprocess.run(
            ["python3", "svo-eval-extract.py", "--config", config_path],
            env=env
        )
        if ret.returncode != 0:
            raise RuntimeError("SVO extraction subprocess failed")
        
        # -- Continue with dataset generation & evaluation ---------------
        ModelDataHandler.generate_model_dataset(config_path)
        evaluate_sbevnet(config_path)

    @staticmethod
    def generate_svo_train_data(svo_file: str, 
                            output_dir: str, 
                            sampling_freq: int = 1, 
                            frame_cnt: int = -1,
                            gpu_id: int = 0) -> None:
        # Save the current environment variable value
        old_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        # Set CUDA_VISIBLE_DEVICES for this block
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        try:
            # Import the ZED SDK; it will read the current CUDA_VISIBLE_DEVICES
            import pyzed.sl as sl

            logger = get_logger("generate_svo_train_data")

            if Path(output_dir).exists() and any(Path(output_dir).iterdir()):
                logger.warning("───────────────────────────────")
                logger.warning(f"Output directory {output_dir} already exists")
                logger.warning("Skipping SVO processing")
                logger.warning("───────────────────────────────")
                return

            zed = sl.Camera()
            init_params = sl.InitParameters()
            init_params.set_from_svo_file(svo_file)
            # Rely on CUDA_VISIBLE_DEVICES rather than setting a property directly

            if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
                logger.error(f"Failed to open SVO file: {svo_file}")
                return

            runtime_parameters = sl.RuntimeParameters()
            image_left = sl.Mat()
            image_right = sl.Mat()
            frame_idx = 0

            total_frames = zed.get_svo_number_of_frames()
            if frame_cnt > 0:
                total_frames = min(total_frames, frame_cnt)
            with tqdm(total=total_frames, desc="Processing SVO", unit="frame") as pbar:
                while True:
                    if frame_cnt > 0 and frame_idx >= frame_cnt:
                        break
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
        finally:
            # Restore the original environment variable so other functions remain unaffected
            if old_cuda_visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible

def main():
    parser = argparse.ArgumentParser(description='Evaluate SVO dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    args = parser.parse_args()
    EvalSVO.evaluate_svo_folder(args.config)

if __name__ == "__main__":
    main()
