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

    # return cv2.flip(mask, 0)
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
    import inspect
    available_commands = {
        "print_available_gpus": print_available_gpus,
        "get_files_from_folder": get_files_from_folder,
        "populate_json": populate_json,
        "flip_mask": flip_mask,
        "crop_resize_mask": crop_resize_mask,
        "help": show_help,
    }
    help_str = "Available commands:\n"
    for cmd, func in available_commands.items():
        sig = inspect.signature(func)
        doc = inspect.getdoc(func)
        summary = doc.splitlines()[0] if doc else "No description provided."
        help_str += f"  {cmd}{sig}: {summary}\n"
    print(help_str)

def main():
    logger = get_logger('helpers')
    logger.info(f"Available commands: {commands}")

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