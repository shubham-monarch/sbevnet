"""Debug test cases implementation"""

import logging
import os
import random
import shutil
import traceback
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import inspect
import fire

from helpers import get_logger


logger = get_logger("debug_cases")


def test_labels_in_seg_masks():
    """Case 1: Test labels in seg masks"""
    
    logger.info("───────────────────────────────")
    logger.info(f"Case 1: test_labels_in_seg_masks")
    logger.info("───────────────────────────────")

    masks_dir = Path("data/model-dataset/train/seg-masks-mono")
    mask_files = list(masks_dir.glob("*.png"))
    
    if not mask_files:
        logger.error(f"No mask files found in {masks_dir}")
        return

    num_samples = min(20, len(mask_files))
    sampled_files = random.sample(mask_files, num_samples)

    for mask_file in sampled_files:
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.error(f"Could not read mask file: {mask_file}")
            continue
        unique_labels, counts = np.unique(mask, return_counts=True)
        
        logger.warning("───────────────────────────────")   
        logger.warning(f"Mask: {mask_file.name}")
        logger.warning("───────────────────────────────")   
        for label, count in zip(unique_labels, counts):
            logger.info(f"  Label {label}: {count} pixels")

def get_cam_extrinsics_distribution(folder_path: str):
    """Plot the distribution of rotation vector components extracted from camera extrinsics (.npy files).
    
    For each .npy file in folder_path, loads a 4x4 matrix, extracts its top-left 3x3 rotation matrix, converts it 
    to axis-angle representation using cv2.Rodrigues, and then plots histograms for the x, y, and z components separately.
    Also logs any outliers in each component (using the IQR and z-score methods) along with the filenames corresponding 
    to these outliers. The filenames for the outliers are sorted alphabetically based on the numeric prefix before the '__' 
    separator. Additionally, logs the percentage count of outliers for the x, y, and z axes.
    """
    import math
    logger = get_logger("debug_cases")
    folder = Path(folder_path)
    npy_files = list(folder.glob("*.npy"))
    if not npy_files:
        logger.error(f"No .npy files found in folder: {folder_path}")
        return
    rvec_x, rvec_y, rvec_z = [], [], []
    valid_files = []
    for npy_file in npy_files:
        matrix = np.load(npy_file)
        if matrix.shape != (4, 4):
            logger.warning(f"File {npy_file} does not contain a 4x4 matrix. Skipping.")
            continue
        rotation = matrix[:3, :3]
        rotation_vector, _ = cv2.Rodrigues(rotation)
        rotation_vector = rotation_vector.flatten()
        rvec_x.append(rotation_vector[0])
        rvec_y.append(rotation_vector[1])
        rvec_z.append(rotation_vector[2])
        valid_files.append(npy_file.name)
    
    if not rvec_x:
        logger.error("───────────────────────────────")
        logger.error("No valid rotation vectors found in provided folder.")
        logger.error("───────────────────────────────")
        return

    # Convert lists to numpy arrays for vectorized operations.
    rvec_x = np.array(rvec_x)
    rvec_y = np.array(rvec_y)
    rvec_z = np.array(rvec_z)
    valid_files = np.array(valid_files)
    
    # Create boolean masks using vectorized comparisons.
    mask_inliers_x = (rvec_x >= 0.3) & (rvec_x <= 0.525)
    mask_inliers_y = (rvec_y >= -0.01) & (rvec_y <= 0.01)
    mask_inliers_z = (rvec_z >= -0.075) & (rvec_z <= 0.1)
    
    files_outliers_x = sorted(valid_files[~mask_inliers_x], key=lambda x: int(x.split('__')[0]))
    files_outliers_y = sorted(valid_files[~mask_inliers_y], key=lambda x: int(x.split('__')[0]))
    files_outliers_z = sorted(valid_files[~mask_inliers_z], key=lambda x: int(x.split('__')[0]))
    
    logger.info(f"len(x_outliers): {len(files_outliers_x)}, {len(files_outliers_x) / len(valid_files) * 100:.2f}%")
    logger.info(f"len(y_outliers): {len(files_outliers_y)}, {len(files_outliers_y) / len(valid_files) * 100:.2f}%")
    logger.info(f"len(z_outliers): {len(files_outliers_z)}, {len(files_outliers_z) / len(valid_files) * 100:.2f}%")
    
    
    logger.info("───────────────────────────────"); 
    logger.info("x-axis outliers:"); 
    [logger.info(f"x_outlier: {file}") for file in files_outliers_x]; 
    logger.info("───────────────────────────────")
    
    logger.info("───────────────────────────────"); 
    logger.info("y-axis outliers:"); 
    [logger.info(f"y_outlier: {file}") for file in files_outliers_y]; 
    logger.info("───────────────────────────────")
    
    logger.info("───────────────────────────────"); 
    logger.info("z-axis outliers:"); 
    [logger.info(f"z_outlier: {file}") for file in files_outliers_z]; 
    logger.info("───────────────────────────────")
    

    indices = list(range(len(rvec_x)))
    assets_dir = Path("assets")
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Plot X Component
    plt.figure()
    plt.plot(indices, rvec_x, "ro", label="X Component")
    plt.xlabel("File Index")
    plt.ylabel("Rotation Vector Component")
    plt.title("X Component of Rotation Vector per File")
    plt.legend()
    y_min, y_max = plt.ylim()
    for y in np.arange(math.floor(y_min * 10) / 10, math.ceil(y_max * 10) / 10, 0.05):
        plt.axhline(y, color='gray', linestyle='--', linewidth=0.5)
    save_path = assets_dir / "cam_extrinsics_x_values.png"
    plt.savefig(str(save_path))
    plt.close()
    logger.info(f"Camera extrinsics X rotation vector plot saved to: {save_path}")

    # Plot Y Component
    plt.figure()
    plt.plot(indices, rvec_y, "go", label="Y Component")
    plt.xlabel("File Index")
    plt.ylabel("Rotation Vector Component")
    plt.title("Y Component of Rotation Vector per File")
    plt.legend()
    y_min, y_max = plt.ylim()
    for y in np.arange(math.floor(y_min * 10) / 10, math.ceil(y_max * 10) / 10, 0.05):
        plt.axhline(y, color='gray', linestyle='--', linewidth=0.5)
    save_path = assets_dir / "cam_extrinsics_y_values.png"
    plt.savefig(str(save_path))
    plt.close()
    logger.info(f"Camera extrinsics Y rotation vector plot saved to: {save_path}")

    # Plot Z Component
    plt.figure()
    plt.plot(indices, rvec_z, "bo", label="Z Component")
    plt.xlabel("File Index")
    plt.ylabel("Rotation Vector Component")
    plt.title("Z Component of Rotation Vector per File")
    plt.legend()
    y_min, y_max = plt.ylim()
    for y in np.arange(math.floor(y_min * 10) / 10, math.ceil(y_max * 10) / 10, 0.025):
        if abs(y) < 1e-6:  # Check if y is approximately zero
            plt.axhline(y, color='yellow', linestyle='-', linewidth=2)
        else:
            plt.axhline(y, color='gray', linestyle='--', linewidth=0.5)
    save_path = assets_dir / "cam_extrinsics_z_values.png"
    plt.savefig(str(save_path))
    plt.close()
    logger.info(f"Camera extrinsics Z rotation vector plot saved to: {save_path}")

    #logger.info(f"Camera extrinsics rotation vector plot saved to: {save_path}")


def main():
    
    logger = get_logger('debug_cases')
    
    current_module = sys.modules[__name__]
    # Get all functions defined in the current module.
    all_functions = inspect.getmembers(current_module, inspect.isfunction)
    # Filter out functions that shouldn't be exposed as CLI commands.
    commands = {
        name: func 
        for name, func in all_functions 
        if not name.startswith('_') and name != "main"
    }
    
    fire.Fire(commands)

if __name__ == "__main__":
    main()