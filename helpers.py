#! /usr/bin/env python3

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
import json
import os
import shutil
from tqdm import tqdm
import glob
import json
import logging
import sys
import coloredlogs 
import numpy as np
import cv2
import yaml
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torch
import random
import shutil
from typing import Tuple
from tqdm import tqdm
import fire
import sys, inspect
import matplotlib.pyplot as plt

def get_logger(name: str, rank: int = 0) -> logging.Logger:
    """Create a logger that only logs on rank 0 by default."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d,%H:%M:%S'
        )
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # # File handler
        # fh = logging.FileHandler(f'training_{rank}.log')
        # fh.setFormatter(formatter)
        # logger.addHandler(fh)
        
        logger.setLevel(logging.INFO)
        coloredlogs.install(level=logging.INFO, logger=logger, force=True)
        
    return logger

def print_available_gpus():
    """Print information about available CUDA GPUs"""
    logger = get_logger('print_available_gpus')
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device_ids = [f"GPU {i}" for i in range(num_gpus)]
        
        logger.info(f"===============")
        logger.info(f"Available GPUs and their IDs: {device_ids}")
        for device_id in device_ids:
            logger.info(device_id)
        logger.info(f"===============\n")
    
    else:
        logger.info("No GPUs available.")

def get_files_from_folder(folder, extensions):
    '''Get all files with the given extensions from the folder.'''
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, f'*{ext}')))
    return sorted(files)

def populate_json(json_path, dataset_path, split="train"):
    '''Populate the json file with the file paths of the images in the dataset.'''
    
    # Remove the json at the json path if it exists
    if os.path.exists(json_path):
        os.remove(json_path)
    
    IMG_EXTENSIONS = ['.jpg', '.png']
    
    def get_relative_files(folder, extensions):
        '''Get all files with the given extensions from the folder and make their paths relative to dataset_path.'''
        files = get_files_from_folder(folder, extensions)
        return [os.path.relpath(file, dataset_path) for file in files]
    
    data = {
        
        "train": {
            "rgb_left": get_relative_files(os.path.join(dataset_path, 'sample-svo/left'), IMG_EXTENSIONS),
            "rgb_right": get_relative_files(os.path.join(dataset_path, 'sample-svo/right'), IMG_EXTENSIONS),
            "top_seg": get_relative_files(os.path.join(dataset_path, 'sample-svo/seg-masks-mono-cropped'), ['.png']),
        },
        "test": {
            "rgb_left": get_relative_files(os.path.join(dataset_path, 'sample-svo/left'), IMG_EXTENSIONS),
            "rgb_right": get_relative_files(os.path.join(dataset_path, 'sample-svo/right'), IMG_EXTENSIONS),
            "top_seg": get_relative_files(os.path.join(dataset_path, 'sample-svo/seg-masks-mono-cropped'), ['.png'])
        }
    }

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

def flip_mask(mask_path: str) -> np.ndarray:
    '''Flip the single channel segmentation mask vertically.'''
    mask_i_mono = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    mask_f_mono = cv2.flip(mask_i_mono, 0)
    return mask_f_mono

def crop_resize_mask(mask_path: str) -> np.ndarray:
    '''Crop and resize the mask from 400x400 to 256x256.'''
    
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    assert mask.shape == (400, 400), f"Expected mask shape to be (400, 400), but got {mask.shape}"
    
    h, w = mask.shape
    mask_cropped = mask[144:400, 72:328]
    
    assert mask_cropped.shape == (256,256), f"Expected mask shape to be (256, 256), but got {mask_cropped.shape}"
    return mask_cropped

def show_help():
    """Show help and available commands including parameter hints."""
    logger = get_logger('show_help')
    current_module = sys.modules[__name__]
    all_functions = inspect.getmembers(current_module, inspect.isfunction)
    commands = {name: func for name, func in all_functions if not name.startswith('_') and name != "main"}
    help_str = "Available commands:\n"
    
    for cmd, func in commands.items():
        sig = inspect.signature(func)
        doc = inspect.getdoc(func)
        summary = doc.splitlines()[0] if doc else "No description provided."
        help_str += f"  {cmd}{sig}: {summary}\n"
    
    logger.info("───────────────────────────────")
    logger.info(help_str)
    logger.info("───────────────────────────────")

def get_label_distribution(folder_path: str = "data/model-dataset/train/seg-masks-mono") -> Dict[int, float]:
    """Get the label distribution of the dataset including per-image statistics and histograms."""
    logger = get_logger('get_label_distribution')
    folder_path = Path(folder_path)

    if not folder_path.exists():
        logger.error(f"Segmentation mask folder not found: {folder_path}")
        return

    mask_files = list(folder_path.glob('*.png'))

    # logger.info("───────────────────────────────")
    # logger.info(f"mask_files: {mask_files[:10]}")
    # logger.info("───────────────────────────────")

    if not mask_files:
        logger.warning(f"No mask files found in: {folder_path}")
        return

    total_label_counts = {}
    total_pixels = 0

    per_mask_percentages = []  # List of dicts mapping label: percentage for each image
    observed_labels = set()

    for mask_file in mask_files:
        mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
        if mask is None:
            logger.warning(f"Could not read mask file: {mask_file}")
            continue

        mask_size = mask.size
        per_image = {}
        unique_labels = np.unique(mask)
        for label in unique_labels:
            count = np.sum(mask == label)
            total_label_counts[label] = total_label_counts.get(label, 0) + count
            per_image[label] = (count / mask_size) * 100
        per_mask_percentages.append(per_image)
        observed_labels.update(unique_labels.tolist())
        total_pixels += mask_size

    if total_pixels == 0:
        logger.warning("No pixels found in any mask.")
        return

    # global_label_percentages = {label: (total_label_counts[label] / total_pixels) * 100
    #                             for label in total_label_counts}
    
    # Build a dict mapping each label to a list of its percentage across images.
    label_percentages_across_images = {label: [] for label in observed_labels}
    for per_image in per_mask_percentages:
        for label in observed_labels:
            label_percentages_across_images[label].append(per_image.get(label, 0))

    for label, percentages in label_percentages_across_images.items():
        percentages_np = np.array(percentages)
        mean_val = np.mean(percentages_np)
        median_val = np.median(percentages_np)
        std_val = np.std(percentages_np)
        logger.info(f"Label {label}: mean: {mean_val:.2f}%, median: {median_val:.2f}%, std: {std_val:.2f}%")
        
        plt.figure()
        plt.hist(percentages, bins=20, edgecolor='black')
        plt.title(f"Histogram for Label {label}")
        plt.xlabel("Percentage of pixels per image (%)")
        plt.ylabel("Frequency")
        plt.savefig(f"assets/label_{label}_histogram.png")
        plt.close()

    # return global_label_percentages

def main():
    logger = get_logger('helpers')
    
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