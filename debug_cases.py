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

def calculate_iqr_outlier_bounds(data):
    """
    Calculate the IQR and outlier bounds for a given dataset.
    
    Args:
        data (np.array): A numpy array of numerical data.
        
    Returns:
        tuple: A tuple containing the lower bound and upper bound for outlier detection.
    """
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

def get_cam_extrinsics_distribution(folder_path: str = "data/model-dataset/train/cam-extrinsics"):
    """Plot the distribution of rotation vector components extracted from camera extrinsics (.npy files).
    
    For each .npy file in folder_path, loads a 4x4 matrix, extracts its top-left 3x3 rotation matrix, converts it 
    to axis-angle representation using cv2.Rodrigues, and then plots histograms for the x, y, and z components separately.
    Also logs any outliers in each component (using the modified z-score method) along with the filenames corresponding 
    to these outliers. The filenames for the outliers are sorted alphabetically based on the numeric prefix before the '__' 
    separator. Additionally, logs the percentage count of outliers for the x, y, and z axes.
    Now including the magnitude (i.e. the rotation angle) of the rotation vector.
    """
    import math
    from helpers import calculate_modified_z_outlier_bounds
    logger = get_logger("debug_cases")
    folder = Path(folder_path)
    npy_files = list(folder.glob("*.npy"))
    if not npy_files:
        logger.error(f"No .npy files found in folder: {folder_path}")
        return
    # Initialize lists for normalized rotation vector components and magnitude value
    rvec_x, rvec_y, rvec_z, rvec_mag = [], [], [], []
    valid_files = []
    for npy_file in npy_files:
        matrix = np.load(npy_file)
        if matrix.shape != (4, 4):
            logger.warning(f"File {npy_file} does not contain a 4x4 matrix. Skipping.")
            continue
        rotation = matrix[:3, :3]
        rotation_vector, _ = cv2.Rodrigues(rotation)
        rotation_vector = rotation_vector.flatten()
        norm = np.linalg.norm(rotation_vector)
        rvec_mag.append(norm)
        if norm > 1e-6:
            rotation_vector = rotation_vector / norm
        rvec_x.append(rotation_vector[0])
        rvec_y.append(rotation_vector[1])
        rvec_z.append(rotation_vector[2])
        valid_files.append(npy_file.name)
    
    if not rvec_x:
        logger.error("───────────────────────────────")
        logger.error("No valid rotation vectors found in provided folder.")
        logger.error("───────────────────────────────")
        return

    rvec_x = np.array(rvec_x)
    rvec_y = np.array(rvec_y)
    rvec_z = np.array(rvec_z)
    rvec_mag = np.array(rvec_mag)
    valid_files = np.array(valid_files)
    
    lower_bound_x, upper_bound_x = calculate_modified_z_outlier_bounds(rvec_x)
    lower_bound_y, upper_bound_y = calculate_modified_z_outlier_bounds(rvec_y)
    lower_bound_z, upper_bound_z = calculate_modified_z_outlier_bounds(rvec_z)
    lower_bound_mag, upper_bound_mag = calculate_modified_z_outlier_bounds(rvec_mag)

    logger.warning("───────────────────────────────")
    logger.warning(f"Modified Z-score bounds for X: lower_bound: {lower_bound_x}, upper_bound: {upper_bound_x}")
    logger.warning(f"Modified Z-score bounds for Y: lower_bound: {lower_bound_y}, upper_bound: {upper_bound_y}")
    logger.warning(f"Modified Z-score bounds for Z: lower_bound: {lower_bound_z}, upper_bound: {upper_bound_z}")
    logger.warning(f"Modified Z-score bounds for magnitude: lower_bound: {lower_bound_mag}, upper_bound: {upper_bound_mag}")
    logger.warning("───────────────────────────────")

    mask_inliers_x = (rvec_x >= lower_bound_x) & (rvec_x <= upper_bound_x)
    mask_inliers_y = (rvec_y >= lower_bound_y) & (rvec_y <= upper_bound_y)
    mask_inliers_z = (rvec_z >= lower_bound_z) & (rvec_z <= upper_bound_z)
    mask_inliers_mag = (rvec_mag >= lower_bound_mag) & (rvec_mag <= upper_bound_mag)
    
    files_outliers_x = sorted(valid_files[~mask_inliers_x], key=lambda x: int(x.split('__')[0]))
    files_outliers_y = sorted(valid_files[~mask_inliers_y], key=lambda x: int(x.split('__')[0]))
    files_outliers_z = sorted(valid_files[~mask_inliers_z], key=lambda x: int(x.split('__')[0]))
    files_outliers_mag = sorted(valid_files[~mask_inliers_mag], key=lambda x: int(x.split('__')[0]))
    
    logger.info("───────────────────────────────")  
    logger.info(f"len(x_outliers): {len(files_outliers_x)}, {len(files_outliers_x) / len(valid_files) * 100:.2f}%")
    logger.info(f"len(y_outliers): {len(files_outliers_y)}, {len(files_outliers_y) / len(valid_files) * 100:.2f}%")
    logger.info(f"len(z_outliers): {len(files_outliers_z)}, {len(files_outliers_z) / len(valid_files) * 100:.2f}%")
    logger.info(f"len(magnitude outliers): {len(files_outliers_mag)}, {len(files_outliers_mag) / len(valid_files) * 100:.2f}%")
    logger.info("───────────────────────────────")  
    
    # Use magnitude outliers to compute the unique outlier count.
    unique_outliers = set(files_outliers_mag) | set(files_outliers_x) | set(files_outliers_z)
    unique_count = len(unique_outliers)
    
    logger.warning("───────────────────────────────")
    logger.warning(f"Unique outliers: {unique_count} [{unique_count / len(valid_files) * 100:.2f}%]")
    logger.warning("───────────────────────────────")

    indices = list(range(len(rvec_x)))
    plot_cam_extrinsics_components(indices, rvec_x, rvec_y, rvec_z, rvec_mag,
                                   mask_inliers_x, mask_inliers_y, mask_inliers_z, mask_inliers_mag,
                                   lower_bound_x, upper_bound_x,
                                   lower_bound_y, upper_bound_y,
                                   lower_bound_z, upper_bound_z,
                                   lower_bound_mag, upper_bound_mag, logger)

def plot_cam_extrinsics_components(indices, rvec_x, rvec_y, rvec_z, rvec_mag,
                                   mask_inliers_x, mask_inliers_y, mask_inliers_z, mask_inliers_mag,
                                   lower_bound_x, upper_bound_x,
                                   lower_bound_y, upper_bound_y,
                                   lower_bound_z, upper_bound_z,
                                   lower_bound_mag, upper_bound_mag, logger):
    assets_dir = Path("assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    indices_np = np.array(indices)

    # Plot X Component with inliers and outliers
    plt.figure()
    inlier_idx_x = indices_np[mask_inliers_x]
    outlier_idx_x = indices_np[~mask_inliers_x]
    plt.plot(inlier_idx_x, rvec_x[mask_inliers_x], "ro", label="X Inliers")
    plt.plot(outlier_idx_x, rvec_x[~mask_inliers_x], "ko", label="X Outliers")
    plt.axhline(lower_bound_x, color='blue', linestyle='--', linewidth=1, label="Lower Bound")
    plt.axhline(upper_bound_x, color='green', linestyle='--', linewidth=1, label="Upper Bound")
    plt.xlabel("File Index")
    plt.ylabel("Rotation Vector Component")
    plt.title("X Component of Rotation Vector per File")
    plt.legend()
    save_path = assets_dir / "cam_extrinsics_x_values.png"
    plt.savefig(str(save_path))
    plt.close()
    logger.info(f"Camera extrinsics X rotation vector plot saved to: {save_path}")
    
    # Plot Y Component with inliers and outliers
    plt.figure()
    inlier_idx_y = indices_np[mask_inliers_y]
    outlier_idx_y = indices_np[~mask_inliers_y]
    plt.plot(inlier_idx_y, rvec_y[mask_inliers_y], "go", label="Y Inliers")
    plt.plot(outlier_idx_y, rvec_y[~mask_inliers_y], "ko", label="Y Outliers")
    plt.axhline(lower_bound_y, color='blue', linestyle='--', linewidth=1, label="Lower Bound")
    plt.axhline(upper_bound_y, color='green', linestyle='--', linewidth=1, label="Upper Bound")
    plt.xlabel("File Index")
    plt.ylabel("Rotation Vector Component")
    plt.title("Y Component of Rotation Vector per File")
    plt.legend()
    save_path = assets_dir / "cam_extrinsics_y_values.png"
    plt.savefig(str(save_path))
    plt.close()
    logger.info(f"Camera extrinsics Y rotation vector plot saved to: {save_path}")
    
    # Plot Z Component with inliers and outliers
    plt.figure()
    inlier_idx_z = indices_np[mask_inliers_z]
    outlier_idx_z = indices_np[~mask_inliers_z]
    plt.plot(inlier_idx_z, rvec_z[mask_inliers_z], "bo", label="Z Inliers")
    plt.plot(outlier_idx_z, rvec_z[~mask_inliers_z], "ko", label="Z Outliers")
    plt.axhline(lower_bound_z, color='blue', linestyle='--', linewidth=1, label="Lower Bound")
    plt.axhline(upper_bound_z, color='green', linestyle='--', linewidth=1, label="Upper Bound")
    plt.xlabel("File Index")
    plt.ylabel("Rotation Vector Component")
    plt.title("Z Component of Rotation Vector per File")
    plt.legend()
    save_path = assets_dir / "cam_extrinsics_z_values.png"
    plt.savefig(str(save_path))
    plt.close()
    logger.info(f"Camera extrinsics Z rotation vector plot saved to: {save_path}")

    # Plot Magnitude of rotation vector
    plt.figure()
    inlier_idx_mag = indices_np[mask_inliers_mag]
    outlier_idx_mag = indices_np[~mask_inliers_mag]
    plt.plot(inlier_idx_mag, rvec_mag[mask_inliers_mag], "mo", label="Magnitude Inliers")
    plt.plot(outlier_idx_mag, rvec_mag[~mask_inliers_mag], "ko", label="Magnitude Outliers")
    plt.axhline(lower_bound_mag, color='blue', linestyle='--', linewidth=1, label="Lower Bound")
    plt.axhline(upper_bound_mag, color='green', linestyle='--', linewidth=1, label="Upper Bound")
    plt.xlabel("File Index")
    plt.ylabel("Rotation Angle Magnitude")
    plt.title("Magnitude (Angle) of Rotation Vector per File")
    plt.legend()
    save_path = assets_dir / "cam_extrinsics_magnitude_values.png"
    plt.savefig(str(save_path))
    plt.close()
    logger.info(f"Camera extrinsics rotation magnitude plot saved to: {save_path}")

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