#! /usr/bin/env python3

import argparse
import os
import cv2
from tqdm import tqdm
import yaml
from pathlib import Path
import torch
import subprocess
import numpy as np

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
    def project_point_to_image(pt: np.ndarray, K: np.ndarray, RT: np.ndarray) -> np.ndarray:
        """
        Project a 3D point from world coordinates to pixel coordinates in the image plane.

        This function takes a 3D point, a camera intrinsics matrix (K), and a camera 
        extrinsics matrix (RT) to project the 3D point onto the 2D image plane.  
        It returns the (u, v) pixel coordinates of the projected point.

        Args:
            pt (np.ndarray): A 3D point in world coordinates (x, y, z).
            K (np.ndarray): The 3x3 camera intrinsics matrix.
            RT (np.ndarray): The 3x4 camera extrinsics matrix (rotation and translation).

        Returns:
            np.ndarray: The 2D pixel coordinates (u, v) of the projected point.
        
        Raises:
            AssertionError: if the input matrices have incorrect shapes.
        """
        assert pt.shape == (3,), "Point must be a 3D coordinate (x, y, z)."
        assert K.shape == (3, 3), "Camera intrinsics matrix K must be 3x3."
        assert RT.shape == (3, 4), "Camera extrinsics matrix RT must be 3x4."

        # Convert 3D point to homogeneous coordinates
        X = np.array([pt[0], pt[1], pt[2], 1.0], dtype=np.float32).reshape(4, 1)
        
        # Project the 3D point to the image plane
        x = K @ (RT @ X)  # 3x1 vector
        x = x.flatten()
        
        # Normalize the homogeneous coordinates
        x = x / x[2]
        
        return x[:2]

    @staticmethod
    def H_img_to_bev(K: np.ndarray, bev_region: dict, bev_size: int, ground_height: float) -> np.ndarray:
        """
        Computes the inverse homography matrix that maps the camera image to a bird's-eye view (BEV).
        
        This function calculates the transformation by first defining ground points based on a given BEV 
        region and then projecting these points onto the image using the updated camera extrinsics. The 
        computed homography is inverted to obtain the transformation from image coordinates to BEV coordinates.
        The homography matrix is normalized before being returned.
        
        Args:
            K (np.ndarray): The 3x3 camera intrinsics matrix.
            bev_region (dict): A dictionary defining the region of interest in ground coordinates (meters),
                               with keys 'x_min', 'x_max' (lateral range) and 'z_min', 'z_max' (forward range).
            bev_size (int): The desired size of the BEV image (bev_size x bev_size).
        
        Returns:
            np.ndarray: The inverse homography matrix mapping image coordinates to BEV coordinates.
        """
        # calculate the ground height using the class method
        h = ground_height

        # extract bev region boundaries from the dictionary
        x_min, x_max = bev_region['x_min'], bev_region['x_max']
        z_min, z_max = bev_region['z_min'], bev_region['z_max']
        Y_ground = h
        
        # define four 3D ground points (X, Y, Z); these correspond to the corners of the region of interest
        pts_ground = np.array([
            [x_min, Y_ground, z_min],  # top-left ground point
            [x_max, Y_ground, z_min],  # top-right ground point
            [x_min, Y_ground, z_max],  # bottom-left ground point
            [x_max, Y_ground, z_max]   # bottom-right ground point
        ], dtype=np.float32)
        
        # project the ground points into image coordinates using the updated camera extrinsics
        # axis_angles: np.ndarray = np.array([np.deg2rad(25), 0, 0])
        axis_angles: np.ndarray = np.array([0.436, 0, 0])
        R, _ = cv2.Rodrigues(axis_angles)
        t = np.array([0, 0, 0])
        RT = np.concatenate([R, t.reshape(-1, 1)], axis=1)
        
        # RT = self.get_updated_camera_extrinsics()[:3, :4]
        pts_img = np.array([EvalSVO.project_point_to_image(pt, K, RT) for pt in pts_ground], dtype=np.float32)
        
        # define the BEV image coordinates corresponding to the ground points
        pts_bev = np.array([
            [0, 0],                      # corresponds to (x_min, z_min)
            [bev_size, 0],               # corresponds to (x_max, z_min)
            [0, bev_size],               # corresponds to (x_min, z_max)
            [bev_size, bev_size]         # corresponds to (x_max, z_max)
        ], dtype=np.float32)
        
        # compute the homography matrix that maps BEV coordinates to image coordinates
        H_bev_to_img = cv2.getPerspectiveTransform(pts_bev, pts_img)
        
        # invert the homography to get the transformation from image coordinates to BEV coordinates
        H_img_to_bev = np.linalg.inv(H_bev_to_img)
        
        # Normalize the homography matrix
        if H_img_to_bev[2, 2] != 0:
            H_img_to_bev = H_img_to_bev / H_img_to_bev[2, 2]
        
        return H_img_to_bev

    @staticmethod
    def generate_ipm_image(input_image: np.ndarray, 
                           K: np.ndarray, 
                           bev_region: dict, 
                           bev_size: int, 
                           ground_height: float) -> np.ndarray:
        """
        Generates an Inverse Perspective Mapping (IPM) or bird's-eye view image from a camera image.
        
        This function transforms a given camera image into an IPM image, which simulates a view
        from directly above the scene. It uses the camera's intrinsic parameters, the ground plane
        height, and a specified region of interest to perform the transformation.
        
        Args:
            input_image (np.ndarray): The input camera image (H x W x C, e.g., 1080x1920x3).
            K (np.ndarray): The 3x3 camera intrinsics matrix.
            bev_region (dict): A dictionary defining the region of interest in ground coordinates (meters)
                                with keys 'x_min', 'x_max' (lateral range) and 'z_min', 'z_max' (forward range).
                                E.g., {'x_min': -2, 'x_max': 3, 'z_min': 4, 'z_max': 9}.
            bev_size (int): The desired size of the output IPM image (bev_size x bev_size).
        
        Returns:
            np.ndarray: The warped IPM image of size bev_size x bev_size.
        """
        # obtain the inverse homography matrix by calling the helper function
        H_img_to_bev = EvalSVO.H_img_to_bev(K, bev_region, bev_size, ground_height)

        # warp the input image to get the BEV image using the computed transform
        bev_image = cv2.warpPerspective(
            input_image, 
            H_img_to_bev, 
            (bev_size, bev_size),
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT
        )
        bev_image = np.flip(bev_image, axis=0)
        return bev_image

    @staticmethod
    def generate_svo_train_data(config_path: str) -> None:

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        gpu_id = config.get("gpu_id", 3)
        svo_file = config.get('svo_file')
        output_dir = config.get('output_dir')
        sampling_freq = config.get('sampling_freq')
        frame_cnt = config.get('frame_cnt', -1)
        start_frame_idx = config.get('start_frame_idx', 0)
        ground_height = config.get('ground_height')
        K = np.array(config['camera_matrix'], dtype=np.float32).reshape(3, 3)
        
        bev_region = {
            'z_min': config['xmin'],
            'z_max': config['xmax'],
            'x_min': config['ymin'],
            'x_max': config['ymax']
        }

        ipm_bev_size = config['ipm_bev_size']
        
        
        # Save the current environment variable value
        old_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        # Set CUDA_VISIBLE_DEVICES for this block
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        try:
            # Import the ZED SDK; it will read the current CUDA_VISIBLE_DEVICES
            import pyzed.sl as sl

            logger = get_logger("generate_svo_train_data")

            logger.warning  (f"───────────────────────────────")
            logger.warning(f"Start frame index: {start_frame_idx}")
            logger.warning(f"───────────────────────────────")

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
                total_frames = min(total_frames, start_frame_idx + frame_cnt)
            
            # Use the correct attribute naming (CamelCase)
            cam_info = zed.get_camera_information()
            calib = cam_info.camera_configuration.calibration_parameters

            # Print the intrinsic parameters of the left camera
            print("fx:", calib.left_cam.fx)
            print("fy:", calib.left_cam.fy)
            print("cx:", calib.left_cam.cx)
            print("cy:", calib.left_cam.cy)

            # Skip to the start frame if needed
            if start_frame_idx > 0:
                logger.info(f"Skipping to frame {start_frame_idx}")
                zed.set_svo_position(start_frame_idx)
                frame_idx = start_frame_idx

            with tqdm(total=total_frames - start_frame_idx, desc="Processing SVO", unit="frame") as pbar:
                while True:
                    if frame_cnt > 0 and frame_idx >= start_frame_idx + frame_cnt:
                        break
                    if zed.grab(runtime_parameters) != sl.ERROR_CODE.SUCCESS:
                        break
                    if frame_idx % sampling_freq == 0:
                        zed.retrieve_image(image_left, sl.VIEW.LEFT)
                        zed.retrieve_image(image_right, sl.VIEW.RIGHT)
                        frame_folder = os.path.join(output_dir, f"frame-{frame_idx}")
                        os.makedirs(frame_folder, exist_ok=True)
                        
                        # generate left / right images
                        left_img = image_left.get_data()
                        right_img = image_right.get_data()
                        left_img_resized = cv2.resize(left_img, (640, 480))
                        right_img_resized = cv2.resize(right_img, (640, 480))
                        
                        cv2.imwrite(os.path.join(frame_folder, "left.jpg"), left_img_resized)
                        cv2.imwrite(os.path.join(frame_folder, "right.jpg"), right_img_resized)

                        
                        # generate ipm images
                        left_ipm = EvalSVO.generate_ipm_image(left_img, K, bev_region, ipm_bev_size, ground_height)
                        cv2.imwrite(os.path.join(frame_folder, "ipm-left.png"), left_ipm)
                        
                        # generate H_img_to_bev matrix
                        (ref_height, ref_width) = (1080, 1920)
                        H_img_to_bev = EvalSVO.H_img_to_bev(K, bev_region, ipm_bev_size, ground_height)
                        ipm_m = np.concatenate([H_img_to_bev.flatten(), [ref_height, ref_width]])
                        
                        np.save(os.path.join(frame_folder, "H_img_to_bev.npy"), ipm_m)

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
