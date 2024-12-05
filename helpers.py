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


def get_logger(name, level=logging.INFO):
    '''Get a logger with colored output'''
    
    logging.basicConfig(level=level)
    logger = logging.getLogger(name)
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
    )
    consoleHandler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="\x1b[32m%(asctime)s\x1b[0m %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
    )
    consoleHandler.setFormatter(formatter)
    logger.handlers = [consoleHandler]
    coloredlogs.install(level=level, logger=logger, force=True)

    return logger    


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
            "rgb_left": get_relative_files(os.path.join(dataset_path, 'train/left'), IMG_EXTENSIONS),
            "rgb_right": get_relative_files(os.path.join(dataset_path, 'train/right'), IMG_EXTENSIONS),
            # "top_seg": get_relative_files(os.path.join(dataset_path, 'train/seg-masks-mono'), ['.png']),
            "top_seg": get_relative_files(os.path.join(dataset_path, 'train/cropped-seg-masks-mono'), ['.png'])
        },
        "test": {
            "rgb_left": [],
            "rgb_right": [],
            "top_seg": []
        }
    }

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)



def restructure_dataset(src_folder, target_folder):
    '''Restructure the train-dataset into left, right, and bev-segmented folders.'''
    
    # Create the target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)

    # Define the target subfolders
    left_folder = os.path.join(target_folder, 'left')
    right_folder = os.path.join(target_folder, 'right')
    seg_masks_mono_folder = os.path.join(target_folder, 'seg-masks-mono')
    seg_masks_rgb_folder = os.path.join(target_folder, 'seg-masks-rgb')


    # Create the target subfolders if they don't exist
    os.makedirs(left_folder, exist_ok=True)
    os.makedirs(right_folder, exist_ok=True)
    os.makedirs(seg_masks_mono_folder, exist_ok=True)
    os.makedirs(seg_masks_rgb_folder, exist_ok=True)


    # Count total files for progress bar
    total_files = sum(len(files) for _, _, files in os.walk(src_folder))

    with tqdm(total=total_files, desc="Organizing Images") as pbar:
        for root, dirs, files in os.walk(src_folder):
            for file in files:
                if file.endswith('_left.jpg'):
                    # Use the directory name as a prefix to prevent overwriting
                    prefix = os.path.basename(root)
                    new_filename = f"{prefix}_{file}"
                    shutil.copy(os.path.join(root, file), os.path.join(left_folder, new_filename))
                elif file.endswith('_right.jpg'):
                    # Use the directory name as a prefix to prevent overwriting
                    prefix = os.path.basename(root)
                    new_filename = f"{prefix}_{file}"
                    shutil.copy(os.path.join(root, file), os.path.join(right_folder, new_filename))
                elif file.endswith('-mono.png'):
                    # Use the directory name as a prefix to prevent overwriting
                    prefix = os.path.basename(root)
                    new_filename = f"{prefix}_{file}"
                    shutil.copy(os.path.join(root, file), os.path.join(seg_masks_mono_folder, new_filename))
                elif file.endswith('-rgb.png'):
                    # Use the directory name as a prefix to prevent overwriting
                    prefix = os.path.basename(root)
                    new_filename = f"{prefix}_{file}"
                    shutil.copy(os.path.join(root, file), os.path.join(seg_masks_rgb_folder, new_filename))
                pbar.update(1)

if __name__ == "__main__":
    pass    
    # CASE => 1
    # Restructure the train-dataset into left, right, and bev-segmented folders.
    # src_folder = 'train-data'
    # target_folder = 'train-data-organized'
    # restructure_dataset(src_folder, target_folder)

    # CASE => 2
    # Populate the json file with the file paths of the images in the dataset.
    json_path = 'datasets/dataset.json'
    dataset_path = 'datasets'
    populate_json(json_path, dataset_path)

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


