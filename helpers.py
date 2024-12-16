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
import shutil
from typing import Tuple
import pyzed.sl as sl



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
            "rgb_left": get_relative_files(os.path.join(dataset_path, 'sample-svo/left'), IMG_EXTENSIONS),
            "rgb_right": get_relative_files(os.path.join(dataset_path, 'sample-svo/right'), IMG_EXTENSIONS),
            "top_seg": get_relative_files(os.path.join(dataset_path, 'sample-svo/seg-masks-mono-cropped'), ['.png']),
        },
        "test": {
            "rgb_left": get_relative_files(os.path.join(dataset_path, 'sample-svo/left'), IMG_EXTENSIONS),
            "rgb_right": get_relative_files(os.path.join(dataset_path, 'sample-svo/right'), IMG_EXTENSIONS),
            "top_seg": get_relative_files(os.path.join(dataset_path, 'sample-svo/seg-masks-mono-cropped'), ['.png'])
        }
        
        # "train": {
        #     "rgb_left": get_relative_files(os.path.join(dataset_path, 'train-640x480/left'), IMG_EXTENSIONS),
        #     "rgb_right": get_relative_files(os.path.join(dataset_path, 'train-640x480/right'), IMG_EXTENSIONS),
        #     "top_seg": get_relative_files(os.path.join(dataset_path, 'train-640x480/seg-masks-mono-cropped'), ['.png']),
        # },
        # "test": {
        #     "rgb_left": get_relative_files(os.path.join(dataset_path, 'train-640x480/left'), IMG_EXTENSIONS),
        #     "rgb_right": get_relative_files(os.path.join(dataset_path, 'train-640x480/right'), IMG_EXTENSIONS),
        #     "top_seg": get_relative_files(os.path.join(dataset_path, 'train-640x480/seg-masks-mono-cropped'), ['.png'])
        # }
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

def flip_mask(mask_path: str) -> np.ndarray:
    
    '''Flip the single channel segmentation mask vertically.'''

    # return cv2.flip(mask, 0)
    mask_i_mono = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    mask_f_mono = cv2.flip(mask_i_mono, 0)

    return mask_f_mono

def flip_masks(src_mono: str, dest_mono: str) -> None:
    '''Flip the mono masks in the source folder and save them to the destination folder.'''
    
    # assert not (os.path.exists(dest_mono) and os.listdir(dest_mono)), "Destination folder for mono masks is not empty"
    # os.makedirs(dest_mono, exist_ok=True)
    
    masks = get_files_from_folder(src_mono, ['.png'])
    for mask_path in tqdm(masks):
        mask_flipped_mono = flip_mask(mask_path)
        
        cv2.imwrite(os.path.join(dest_mono, os.path.basename(mask_path)), mask_flipped_mono)


def crop_resize_mask(mask_path: str) -> np.ndarray:
    '''Crop and resize the mask from 400x400 to 256x256.'''
    
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    assert mask.shape == (400, 400), f"Expected mask shape to be (400, 400), but got {mask.shape}"
    
    h, w = mask.shape
    mask_cropped = mask[144:400, 72:328]
    
    assert mask_cropped.shape == (256,256), f"Expected mask shape to be (256, 256), but got {mask_cropped.shape}"
    return mask_cropped

def crop_resize_masks(src_mono: str, dest_mono: str, dest_rgb: str) -> None:
    '''Crop and resize the mono / rgb masks in the source folder and save them to the destination folder.'''
    
    assert not (os.path.exists(dest_mono) and os.listdir(dest_mono)), "Destination folder for mono masks is not empty"
    assert not (os.path.exists(dest_rgb) and os.listdir(dest_rgb)), "Destination folder for RGB masks is not empty"

    os.makedirs(dest_mono, exist_ok=True)
    os.makedirs(dest_rgb, exist_ok=True)

    masks = get_files_from_folder(src_mono, ['.png'])
    
    for mask_path in tqdm(masks):
        mask_cropped_mono = crop_resize_mask(mask_path)
        mask_cropped_rgb = mono_to_rgb_mask(mask_cropped_mono)
        
        cv2.imwrite(os.path.join(dest_mono, os.path.basename(mask_path)), mask_cropped_mono)
        cv2.imwrite(os.path.join(dest_rgb, os.path.basename(mask_path).replace('-mono.png', '-rgb.png')), mask_cropped_rgb)

def generate_dataset_from_svo(svo_path: str, dataset_path: str, size: Tuple[int, int]) -> None:
    '''Generate a dataset from an SVO file.'''
    
    logger = get_logger('generate_dataset_from_svo')

    assert size is not None, "Size must be provided"
    assert not (os.path.exists(dataset_path) and os.listdir(dataset_path)), "Destination folder for dataset must be empty"

    filepath = os.path.abspath(svo_path)
    dir_path = os.path.abspath(dataset_path)

    logger.info(f"===============")
    logger.info(f"svo_file: {filepath}")
    logger.info(f"===============\n")

    input_type = sl.InputType()
    input_type.set_from_svo_file(filepath)

    init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
    init.coordinate_units = sl.UNIT.METER   

    zed = sl.Camera()
    status = zed.open(init)
    
    if status != sl.ERROR_CODE.SUCCESS:
        logger.error(f"===============")
        logger.error(f"Failed to open the camera")
        logger.error(f"===============\n")
        exit()

    runtime_parameters = sl.RuntimeParameters()
    image_l = sl.Mat()
    image_r = sl.Mat()

    logger.info(f"===============") 
    logger.info(f"Trying to delete the {dir_path} directory")
    logger.info(f"===============\n")
    
    try:
        shutil.rmtree(dir_path)
        logger.info(f"Cleared the {dir_path} directory!")
    except OSError as e:
        logger.warning("Warning: %s : %s" % (dir_path, e.strerror))
    
    left_folder = os.path.join(dir_path, 'left')
    right_folder = os.path.join(dir_path, 'right')
    
    os.makedirs(left_folder, exist_ok=True)
    os.makedirs(right_folder, exist_ok=True)

    for frame_idx in tqdm(range(0, 400, 2)):   
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:  
            zed.set_svo_position(frame_idx)    
           
            zed.retrieve_image(image_l, sl.VIEW.LEFT)
            zed.retrieve_image(image_r, sl.VIEW.RIGHT)

            # resize images before writing to disk
            image_l_resized = cv2.resize(image_l.get_data(), size)
            image_r_resized = cv2.resize(image_r.get_data(), size)

            cv2.imwrite(os.path.join(left_folder, f'{frame_idx:04d}.jpg'), image_l_resized)
            cv2.imwrite(os.path.join(right_folder, f'{frame_idx:04d}.jpg'), image_r_resized)
        
        else:    
            logger.error(f"===============")
            logger.error(f"Failed to grab frame")
            logger.error(f"===============\n")
            
            exit(1)
    
    zed.close()
    
    logger.info(f"===============")
    logger.info(f"Extracted images from {svo_path} to {dataset_path}")
    logger.info(f"===============\n")

import numpy as np
import matplotlib.pyplot as plt

def plot_segmentation_classes(mask: np.ndarray) -> None:
    """
    Reads a single channel segmentation mask and plots (x,y) coordinates of each unique class.
    """
    # get unique classes
    unique_classes = np.unique(mask)
    
    # create a plot
    plt.figure(figsize=(10, 6))
    
    for class_id in unique_classes:
        # get coordinates of the pixels belonging to the class
        y_coords, x_coords = np.where(mask == class_id)
        
        # plot the points
        plt.scatter(x_coords, y_coords, label=f'Class {class_id}', alpha=0.5)
    
    plt.title('Segmentation Classes')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.gca().invert_yaxis()  # invert y axis to match image coordinates
    plt.savefig('segmentation_classes_plot.png')  # save plot to disk
    plt.close()  # close the plot to free memory


if __name__ == "__main__":
    # pass
    
    logger = get_logger('main')

    # CASE 14
    # plot segmentation classes
    mask_path = 'datasets/train-640x480/seg-masks-mono-cropped/58__seg-mask-mono.png'
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    # flip the mask in the y-direction
    mask_ = np.flip(mask, axis=0)
    plot_segmentation_classes(mask_)


    # # CASE 13
    # # generate a dataset from an SVO file
    # svo_path = "front_2024-06-04-10-39-57.svo"
    # dataset_path = "datasets/sample-svo"
    # generate_dataset_from_svo(svo_path, dataset_path, size=(640, 480))
    # populate_json('datasets/dataset.json', 'datasets')

    # # CASE 12
    # # crop + flip mono / rgb masks
    # src_mono = 'datasets/train-640x480/seg-masks-mono'
    # dest_mono = 'datasets/train-640x480/seg-masks-mono-cropped'
    # dest_rgb = 'datasets/train-640x480/seg-masks-rgb-cropped'
    
    # # resize mono / rgb masks
    # crop_resize_masks(src_mono, dest_mono, dest_rgb)

    # # flip mono / rgb masks
    # flip_masks(dest_mono, dest_mono)
    # flip_masks(dest_rgb, dest_rgb)

    # # CASE 11
    # src_mono = 'datasets/train-640x480/seg-masks-mono'
    # dest_mono = 'datasets/train-640x480/seg-masks-mono-cropped'
    # dest_rgb = 'datasets/train-640x480/seg-masks-rgb-cropped'
    # crop_resize_masks(src_mono, dest_mono, dest_rgb)


    # # CASE 10
    # mask_path = '/home/ubuntu/sbevnet/datasets/train-640x480/seg-masks-mono/19__seg-mask-mono.png'
    # mask_i = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    # mask_f = crop_resize_mask(mask_path)
    # mask_f_rgb = mono_to_rgb_mask(mask_f)
    # cv2.imwrite('debug/mask_cropped.png', mask_f_rgb)
    
    # # CASE 9
    # mask_path = '/home/ubuntu/sbevnet/datasets/train-640x480/seg-masks-mono/19__seg-mask-mono.png'
    # mask_f = flip_mask(mask_path)
    # # cv2.imwrite('debug/mask_f.png', mask_f)

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

    # # CASE => 8
    # # Print the number of unique IDs in a single channel segmentation mask.
    
    # seg_mask_mono_folder = 'datasets/test-640x480/seg-masks-mono'
    
    # # Get a random mask path from the folder
    # mask_path = random.choice(glob.glob(os.path.join(seg_mask_mono_folder, '*.png')))
    # num_labels, unique_ids, mask_shape = print_unique_ids_in_mask(mask_path)

    # logger.info(f"===============")
    # logger.info(f"Mask file name: {os.path.basename(mask_path)}")
    # logger.info(f"Mask shape: {mask_shape}")
    # logger.info(f"Number of unique labels: {num_labels}")
    # logger.info(f"Unique IDs: {unique_ids}")
    # logger.info(f"===============\n")

    # mask_path = 'predictions/pred_0_0.png'
    # num_labels, unique_ids = print_unique_ids_in_mask(mask_path)
    # logger.info(f"===============")
    # logger.info(f"Number of unique labels: {num_labels}")
    # logger.info(f"Unique IDs: {unique_ids}")
    # logger.info(f"===============\n")