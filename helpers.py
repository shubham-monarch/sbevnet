#! /usr/bin/env python3

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


class ComposeDatasetDict(data.Dataset):

    "TO take a dictionary of datasets and return a dataset which produces elements as a dictionalry "
    
    def __init__(self , data_loaders ,ret_double=False ):
        self.data_loaders = data_loaders
        self.logger = get_logger('ComposeDatasetDict')
        
        self.logger.warning(f"=================")
        for k in self.data_loaders:
            self.logger.warning(f"{k} --> {len(self.data_loaders[k])}")
        self.logger.warning(f"=================\n") 
        
        # make sure all the datasets are of the same size!! 
        for k in self.data_loaders:
            l = len( self.data_loaders[k])
            break 
        for k in self.data_loaders:
            assert l == len( self.data_loaders[k] ) , "The sizes of the datasets do not match! "+k  
            # print( l , k , )
        
        self.ret_double = ret_double 
        
    def __getitem__(self, index):
        ret = {}
        for k in self.data_loaders:
            ret[k] = self.data_loaders[k].__getitem__(index)

        if self.ret_double:
            return ret , ret 
        else:
            return ret 

    def __len__(self):
        for k in self.data_loaders:
            return len(self.data_loaders[k]) 



def get_label_colors_from_yaml(yaml_path=None):
    """Read label colors from Mavis.yaml config file."""
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Get BGR colors directly from yaml color_map
    label_colors_bgr = config['color_map']
    
    # Convert BGR to RGB by reversing color channels
    label_colors_rgb = {
        label: color[::-1] 
        for label, color in label_colors_bgr.items()
    }
    
    return label_colors_bgr, label_colors_rgb
        

def mono_to_rgb_mask(mono_mask: np.ndarray, yaml_path: str = "Mavis.yaml") -> np.ndarray:
    """Convert single channel segmentation mask to RGB using label mapping from a YAML file."""
    
    label_colors_bgr, _ = get_label_colors_from_yaml(yaml_path)
    
    H, W = mono_mask.shape
    rgb_mask = np.zeros((H, W, 3), dtype=np.uint8)
    
    for label_id, rgb_value in label_colors_bgr.items():
        mask = mono_mask == label_id
        rgb_mask[mask] = rgb_value
        
    return rgb_mask


def resize_segmentation_masks(
    input_folder: str, 
    output_folder: str, 
    new_size: tuple, 
    labels=[0, 1, 2, 3, 4, 5]
):
    '''Resize segmentation masks in the input folder and save them to the output folder.'''
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all mask files from the input folder
    mask_files = get_files_from_folder(input_folder, ['.png', '.jpg', '.bmp'])
    
    for mask_file in tqdm(mask_files, desc="Resizing masks"):
        # Read the mask
        mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
        
        if mask is None:
            raise ValueError(f"Failed to read mask file: {mask_file}")
        
        # Resize the mask
        resized_mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
        
        # Ensure the resized mask contains only the specified label values
        unique_values = np.unique(resized_mask)
        min_label, max_label = min(labels), max(labels)
        resized_mask = np.clip(np.round(resized_mask), min_label, max_label)
        
        # Assert that the number of unique labels is less than the number of specified labels
        assert len(unique_values) <= len(labels), f"Number of unique labels {len(unique_values)} is not less than {len(labels)}"

        # Save the resized mask to the output folder
        output_path = os.path.join(output_folder, os.path.basename(mask_file))
        cv2.imwrite(output_path, resized_mask)


def convert_rgb_to_single_channel(segmentation_mask: str) -> np.ndarray:
    ''' Convert an RGB segmentation mask to a single channel image.'''

    segmentation_mask = cv2.imread(segmentation_mask, cv2.IMREAD_UNCHANGED)

    if segmentation_mask.ndim != 3 or segmentation_mask.shape[2] != 3:
        raise ValueError("Input segmentation mask must be an RGB image with 3 channels.")

    # Convert the RGB image to a single channel image by encoding the segmentation mask
    single_channel_mask = segmentation_mask[:, :, 0] * 256 * 256 + segmentation_mask[:, :, 1] * 256 + segmentation_mask[:, :, 2]

    # Calculate the number of unique classes in the RGB and single channel masks
    num_classes_rgb = len(np.unique(segmentation_mask.reshape(-1, 3), axis=0))
    num_classes_single_channel = len(np.unique(single_channel_mask))

    print(f"Number of segmentation classes in RGB image: {num_classes_rgb}")
    print(f"Number of segmentation classes in single channel image: {num_classes_single_channel}")

    return single_channel_mask


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
            "rgb_left": get_relative_files(os.path.join(dataset_path, 'train-640x480/left'), IMG_EXTENSIONS),
            "rgb_right": get_relative_files(os.path.join(dataset_path, 'train-640x480/right'), IMG_EXTENSIONS),
            "top_seg": get_relative_files(os.path.join(dataset_path, 'train-640x480/seg-masks-mono'), ['.png']),
        },
        "test": {
            "rgb_left": get_relative_files(os.path.join(dataset_path, 'train-640x480/left'), IMG_EXTENSIONS),
            "rgb_right": get_relative_files(os.path.join(dataset_path, 'train-640x480/right'), IMG_EXTENSIONS),
            "top_seg": get_relative_files(os.path.join(dataset_path, 'train-640x480/seg-masks-mono'), ['.png'])
        }
    }

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)


def print_unique_ids_in_mask(mask_path: str) -> None:
    '''Print the number of unique IDs in a single channel segmentation mask.'''
    
    logger = get_logger('print_unique_ids_in_mask')
    
    # Read mask
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Failed to read mask: {mask_path}")
        
    # Get unique values
    unique_ids = np.unique(mask)
    
    # logger.info(f"===============")
    # logger.info(f"Found {len(unique_ids)} unique IDs in mask:")
    # logger.info(f"IDs: {unique_ids}")
    # logger.info(f"===============\n")

    return len(unique_ids), unique_ids, mask.shape    
if __name__ == "__main__":
    # pass
    
    logger = get_logger('main')


    # CASE => 1
    # Restructure the train-dataset into left, right, and bev-segmented folders.
    # src_folder = 'train-data'
    # target_folder = 'train-data-organized'
    # restructure_dataset(src_folder, target_folder)

    # CASE => 2
    # Populate the json file with the file paths of the images in the dataset.
    # json_path = 'datasets/dataset.json'
    # dataset_path = 'datasets'
    # populate_json(json_path, dataset_path)

    # # CASE => 3
    # # Convert an RGB segmentation mask to a single channel image.
    # segmentation_mask = 'datasets/train/bev-segmented/1__left.disp.png'
    # single_channel_mask = convert_rgb_to_single_channel(segmentation_mask)
    # # print(f"single_channel_mask.shape: {single_channel_mask.shape}")

    # CASE => 4
    # Get the number of unique labels from an RGB segmentation mask.
    # segmentation_mask = 'datasets/train/bev-segmented/1__left.disp.png'
    # num_labels = get_unique_labels_from_rgb_mask(segmentation_mask)
    # print(f"Number of unique labels: {num_labels}")

    # CASE => 5
    # Resize segmentation masks in the input folder and save them to the output folder.
    # input_folder = 'datasets/train/cropped-seg-masks-mono'
    # output_folder = 'datasets/train/cropped-seg-masks-mono'
    # new_size = (480,480)
    # resize_segmentation_masks(input_folder, output_folder, new_size, labels=[0, 1, 2, 3, 4, 5])

    # CASE => 6
    # Convert all mono segmentation masks to RGB masks.
    # src_folder = 'debug/cropped-seg-masks-mono'
    # dst_folder = 'debug/cropped-seg-masks-rgb'
    # convert_mono_to_rgb_masks(src_folder, dst_folder)

    # CASE => 7
    # Print information about available CUDA GPUs.
    # print_available_gpus()

    # CASE => 8
    # Print the number of unique IDs in a single channel segmentation mask.
    
    seg_mask_mono_folder = 'datasets/test-640x480/seg-masks-mono'
    
    # Get a random mask path from the folder
    mask_path = random.choice(glob.glob(os.path.join(seg_mask_mono_folder, '*.png')))
    num_labels, unique_ids, mask_shape = print_unique_ids_in_mask(mask_path)

    logger.info(f"===============")
    logger.info(f"Mask file name: {os.path.basename(mask_path)}")
    logger.info(f"Mask shape: {mask_shape}")
    logger.info(f"Number of unique labels: {num_labels}")
    logger.info(f"Unique IDs: {unique_ids}")
    logger.info(f"===============\n")

    # mask_path = 'predictions/pred_0_0.png'
    # num_labels, unique_ids = print_unique_ids_in_mask(mask_path)
    # logger.info(f"===============")
    # logger.info(f"Number of unique labels: {num_labels}")
    # logger.info(f"Unique IDs: {unique_ids}")
    # logger.info(f"===============\n")