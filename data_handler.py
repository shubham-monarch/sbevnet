#! /usr/bin/env python3

import os
import random
import shutil
from tqdm import tqdm
from typing import List, Dict
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import boto3
from urllib.parse import urlparse
import argparse
import yaml
from typing import Tuple

from helpers import get_logger, flip_mask, get_files_from_folder


class S3_DataHandler:
    def __init__(self, src_dir: str, dst_dir: str, required_keys: List[str]) -> None:
        """Initialize GTDataHandler with source directory, destination directory, and required files
        
        Args:
            src_dir (str): Path to GT-dataset
            dst_dir (str): Path to save gt-train/gt-test 
            required_keys (List[str]): List of required keys in each GT folder
        """
        self.logger = get_logger("GTHandler")
        
        self.src_dir = src_dir
        self.dst_dir = dst_dir

        # files required in each valid GT-dataset folder
        self.key_files = required_keys

        # [GT-train / GT-test] folders
        self.GT_train = os.path.join(self.dst_dir, "GT-train")
        self.GT_test = os.path.join(self.dst_dir, "GT-test")
        
        self.logger.info("───────────────────────────────")
        self.logger.info("Generating [GT-train / GT-test] from GT-dataset...")
        self.logger.info("───────────────────────────────\n")

    @staticmethod
    def _get_leaf_folders(src_dir: str) -> List[str]:
        """Get all leaf folders inside the given folder recursively and return their relative positions as a list"""
        leaf_folders = []
        for root, dirs, files in os.walk(src_dir):
            if not dirs: 
                leaf_folders.append(os.path.relpath(root, src_dir))  
        return leaf_folders


    def _copy_folders(self, folder_list: List[str], src_dir: str, dst_dir: str) -> None:
        """Copy folders to destination directory"""
        
        assert not (os.path.exists(dst_dir) and os.listdir(dst_dir)), f"Destination directory must be empty!"
        
        for idx, folder in enumerate(tqdm(folder_list, desc="Copying folders"), 1):
            src = os.path.join(src_dir, folder)
            dst = os.path.join(dst_dir, str(idx))
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)

    def _remove_incomplete_GT_folders(self):
        """Remove folders with missing left.jpg / right.jpg / seg-mask-mono.png / seg-mask-rgb.png"""

        self.logger.info("───────────────────────────────")
        self.logger.info("Checking for incomplete GT folders...")
        self.logger.info("───────────────────────────────\n")
        
        leaf_folders = S3_DataHandler._get_leaf_folders(self.src_dir)

        
        incomplete_folders = []
        for folder in tqdm(leaf_folders, desc="Checking folders"):
            folder_files = os.listdir(os.path.join(self.src_dir, folder))
            
            # check if folder has exactly the required files
            if not all(f in folder_files for f in self.key_files):
                incomplete_folders.append(folder)
                
        # Remove incomplete folders
        for folder in incomplete_folders:
            shutil.rmtree(os.path.join(self.src_dir, folder))
        
        self.logger.error("───────────────────────────────")
        self.logger.error(f"Removed {len(incomplete_folders)} incomplete folders")
        self.logger.error(f"Remaining folders: {len(leaf_folders) - len(incomplete_folders)}")
        self.logger.error("───────────────────────────────\n")
    
    @staticmethod
    def download_s3_folder(s3_uri, local_dir):
        """Recursively download an S3 folder to a local directory
        
        Args:
            s3_uri (str): S3 URI in format s3://bucket-name/path/to/folder
            local_dir (str): Local directory to download files to
        """
        
        logger = get_logger("DataHandler")
    
        parsed_uri = urlparse(s3_uri)
        bucket_name = parsed_uri.netloc
        s3_folder = parsed_uri.path.lstrip('/')
        
        os.makedirs(local_dir, exist_ok=True)
        
        logger.info("───────────────────────────────")
        logger.info(f"Downloading files from {s3_uri} to {local_dir}")
        logger.info("───────────────────────────────")
        
        s3_client = boto3.client('s3')
        paginator = s3_client.get_paginator('list_objects_v2')
        total_files = 0
        objects = []
        for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
            objects.extend(page.get('Contents', []))
        total_files = len(objects)
        
        with tqdm(total=total_files, unit='file', desc='Downloading') as pbar:
            for obj in objects:
                relative_path = obj['Key'][len(s3_folder):].lstrip('/')
                local_file_path = os.path.join(local_dir, relative_path)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                s3_client.download_file(bucket_name, obj['Key'], local_file_path)
                pbar.update(1)

    def generate_GT_train_test(self, n_train: int, n_test: int) -> None:
        """generate GT-train / GT-test folders from GT-aws folders
        
        gt-train/ gt-test
        ├── 1
        │   ├── left.jpg
        │   ├── right.jpg
        │   ├── seg-mask-mono.png
        │   └── seg-mask-rgb.png
        ├── 2
        │   ├── left.jpg
        │   ├── right.jpg
        │   ├── seg-mask-mono.png
        │   └── seg-mask-rgb.png
        ├── ...
        """
        
        # remove incomplete GT folders
        self._remove_incomplete_GT_folders()

        # assert [gt-train / gt-test] folders are empty
        assert not (os.path.exists(self.GT_train) and os.listdir(self.GT_train)), f"GT-train must be empty!"
        assert not (os.path.exists(self.GT_test) and os.listdir(self.GT_test)), f"GT-test must be empty!"

        leaf_folders = S3_DataHandler._get_leaf_folders(self.src_dir)
        random.shuffle(leaf_folders)
        
        assert len(leaf_folders) > n_train + n_test, f'Expected at least {n_train + n_test} leaf folders, but got {len(leaf_folders)}'

        os.makedirs(self.GT_train, exist_ok=True)
        os.makedirs(self.GT_test, exist_ok=True)
                
        train_folders = leaf_folders[:n_train]
        test_folders = leaf_folders[n_train:n_train + n_test]
        
        # copy folders to GT_train / GT_test 
        self._copy_folders(train_folders, self.src_dir, self.GT_train)
        self._copy_folders(test_folders, self.src_dir, self.GT_test)

        # assert number of folders in gt-train / gt-test
        assert len(S3_DataHandler._get_leaf_folders(self.GT_train)) == n_train, f'Expected {n_train} training files, but got {len(S3_DataHandler._get_leaf_folders(self.GT_train))}'
        assert len(S3_DataHandler._get_leaf_folders(self.GT_test)) == n_test, f'Expected {n_test} test files, but got {len(S3_DataHandler._get_leaf_folders(self.GT_test))}'
      
class ModelDataHandler:
    
    def __init__(self, model_dir: str, GT_train: str, GT_test: str) -> None:
        """Initialize ModelDataHandler with model directory, GT-train directory, and GT-test directory
        
        Args:
            model_dir (str): Path to model dataset
            GT_train (str): Path to GT-train
            GT_test (str): Path to GT-test
        """
        self.logger = get_logger("DataHandlerModel")
        
        # assert model_dir is empty
        assert not (os.path.exists(model_dir) and os.listdir(model_dir)), f"Model directory {model_dir} must be empty!"
        
        # GT folders
        self.GT_train = GT_train
        self.GT_test = GT_test
        
        # model-dataset folders
        self.model_dir = model_dir
        self.model_train_dir = os.path.join(self.model_dir, "train")
        self.model_test_dir = os.path.join(self.model_dir, "test")

        self.logger.info("───────────────────────────────")
        self.logger.info("Generating [MODEL-train / MODEL-test] from [GT-train / GT-test]...")
        self.logger.info("───────────────────────────────\n")

    def _restructure_GT_folder(self, GT_dir: str, MODEL_dir: str):
        """Restructure GT folder into model-train / model-test folders"""
        
        # [model-train / model-test] folders should be empty
        assert not (os.path.exists(MODEL_dir) and os.listdir(MODEL_dir)), "model_train must be empty"
        
        # create [model-train / model-test] folders
        os.makedirs(MODEL_dir, exist_ok=True)

        # model-train folders
        left_folder = os.path.join(MODEL_dir, 'left')
        right_folder = os.path.join(MODEL_dir, 'right')
        seg_masks_mono_folder = os.path.join(MODEL_dir, 'seg-masks-mono')
        seg_masks_rgb_folder = os.path.join(MODEL_dir, 'seg-masks-rgb')
        cam_extrinsics_folder = os.path.join(MODEL_dir, 'cam-extrinsics')
        
        # Create the target subfolders if they don't exist
        os.makedirs(left_folder, exist_ok=True)
        os.makedirs(right_folder, exist_ok=True)
        os.makedirs(seg_masks_mono_folder, exist_ok=True)
        os.makedirs(seg_masks_rgb_folder, exist_ok=True)
        os.makedirs(cam_extrinsics_folder, exist_ok=True)

        # Count total files for progress bar
        total_files = 0
        for root, dirs, files in os.walk(GT_dir):
            for file in files:
                if file.endswith('left.jpg') or \
                   file.endswith('right.jpg') or \
                   file.endswith('-mono.png') or \
                   file.endswith('-rgb.png') or \
                   file.endswith('cam-extrinsics.npy'):
                    total_files += 1


        with tqdm(total=total_files, desc="Organizing Images") as pbar:
            for root, dirs, files in os.walk(GT_dir):
                # Get folder number from root path
                folder_num = os.path.basename(root)
                if not folder_num.isdigit():
                    continue

                for file in files:
                    if file.endswith('left.jpg'):
                        new_filename = f"{folder_num}__{file}"
                        shutil.copy(os.path.join(root, file), os.path.join(left_folder, new_filename))
                        pbar.update(1)
                    elif file.endswith('right.jpg'):
                        new_filename = f"{folder_num}__right.jpg"
                        shutil.copy(os.path.join(root, file), os.path.join(right_folder, new_filename))
                        pbar.update(1)
                    elif file.endswith('-mono.png'):
                        new_filename = f"{folder_num}__seg-mask-mono.png"
                        shutil.copy(os.path.join(root, file), os.path.join(seg_masks_mono_folder, new_filename))
                        pbar.update(1)
                    elif file.endswith('-rgb.png'):
                        new_filename = f"{folder_num}__seg-mask-rgb.png"
                        shutil.copy(os.path.join(root, file), os.path.join(seg_masks_rgb_folder, new_filename))
                        pbar.update(1)
                    elif file.endswith('cam-extrinsics.npy'):
                        new_filename = f"{folder_num}__cam-extrinsics.npy"
                        shutil.copy(os.path.join(root, file), os.path.join(cam_extrinsics_folder, new_filename))
                        pbar.update(1)

    def _flip_masks(self, src_dir: str, dest_dir: str) -> None:
        '''Flip the masks in the source folder and save them to the destination folder.'''
        
        masks = get_files_from_folder(src_dir, ['.png'])
        for mask_path in tqdm(masks, desc="Flipping masks"):
            mask_flipped_mono = flip_mask(mask_path)
            cv2.imwrite(os.path.join(dest_dir, os.path.basename(mask_path)), mask_flipped_mono)

    def _populate_json(self, json_path, dataset_path):
        '''Populate the json file with the file paths of the images in the dataset.'''
        
        if os.path.exists(json_path):
            os.remove(json_path)
        
        IMG_EXTENSIONS = ['.jpg', '.png']
        
        def get_relative_files(folder, extensions):
            '''Get all files with the given extensions from the folder and make their paths relative to dataset_path.'''
            files = get_files_from_folder(folder, extensions)
            return [os.path.relpath(file, dataset_path) for file in files]

        data = {
            
            "train": {
                "rgb_left": get_relative_files(os.path.join(self.model_train_dir, 'left'), IMG_EXTENSIONS),
                "rgb_right": get_relative_files(os.path.join(self.model_train_dir, 'right'), IMG_EXTENSIONS),
                "top_seg": get_relative_files(os.path.join(self.model_train_dir, 'seg-masks-mono'), ['.png']),
                "confs": get_relative_files(os.path.join(self.model_train_dir, 'cam-extrinsics'), ['.npy']),
            },
            "test": {
                "rgb_left": get_relative_files(os.path.join(self.model_test_dir, 'left'), IMG_EXTENSIONS),
                "rgb_right": get_relative_files(os.path.join(self.model_test_dir, 'right'), IMG_EXTENSIONS),
                "top_seg": get_relative_files(os.path.join(self.model_test_dir, 'seg-masks-mono'), ['.png']),
                "confs": get_relative_files(os.path.join(self.model_test_dir, 'cam-extrinsics'), ['.npy']),
            }
        }

        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)

    def _remap_mask_labels(self, mask_dir: str) -> None:
        '''Changes all 255 labels to 0'''
        masks = get_files_from_folder(mask_dir, ['.png'])
        for mask_path in tqdm(masks, desc="Remapping mask labels"):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask[mask == 255] = 0
            cv2.imwrite(mask_path, mask)

    @staticmethod
    def _remove_mask_from_model_dataset(seg_mask_mono_path: str) -> None:
        """Remove mask from the model-dataset folders 
        [left / right / seg-masks-mono / seg-masks-rgb/ ..]"""
        
        logger = get_logger("DataHandlerModel")
        parent_dir = os.path.dirname(os.path.dirname(seg_mask_mono_path))
        try:
            file_index = os.path.basename(seg_mask_mono_path).split('_')[0]
        except IndexError:
            logger.error(f"Could not extract index from {seg_mask_mono_path}")
            return
        
        for folder in os.listdir(parent_dir):
            for file in os.listdir(os.path.join(parent_dir, folder)):
                if file.split('_')[0] == file_index:
                    os.remove(os.path.join(parent_dir, folder, file))
                    break

    @staticmethod
    def _remove_outliers(mask_dir: str, target_label: int, threshold: float) -> Tuple[int, List[str]]:
        '''Remove masks with more than threshold % of the target label'''
        logger = get_logger("DataHandlerModel")
        masks = get_files_from_folder(mask_dir, ['.png'])
        cnt = 0
        files = []
        for mask_path in tqdm(masks, desc="Removing label outliers"):
            seg_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            label_mask_cnt = np.sum(seg_mask == target_label)
            if label_mask_cnt / seg_mask.size >= threshold:
                cnt += 1
                files.append(os.path.basename(mask_path))
                # ModelDataHandler._remove_mask_from_model_dataset(mask_path)
        
        return cnt, files

    def generate_MODEL_train_test(self):
        logger = get_logger("DataHandlerModel")

        self._restructure_GT_folder(self.GT_train, self.model_train_dir)
        self._restructure_GT_folder(self.GT_test, self.model_test_dir)

        # flip mono / rgb masks in model-train
        self._flip_masks(os.path.join(self.model_train_dir, 'seg-masks-mono'),\
                           os.path.join(self.model_train_dir, 'seg-masks-mono'))
        self._flip_masks(os.path.join(self.model_test_dir, 'seg-masks-rgb'),\
                           os.path.join(self.model_test_dir, 'seg-masks-rgb'))
        
        # remap mask labels
        self._remap_mask_labels(os.path.join(self.model_train_dir, 'seg-masks-mono'))
        self._remap_mask_labels(os.path.join(self.model_test_dir, 'seg-masks-mono'))

        # remove label 0 outliers
        cnt, _ = self._remove_outliers(os.path.join(self.model_train_dir, 'seg-masks-mono'), 0, 0.8)
        
        logger.info("───────────────────────────────")
        logger.info(f"Removed {cnt} masks with label outliers")
        logger.info("───────────────────────────────\n")

        # populate json file
        self._populate_json(os.path.join(self.model_dir, 'dataset.json'), self.model_dir)


def generate_model_dataset(config):
    """
    Generate model-dataset by downloading from S3 and processing the data
    
    Args:
        config (dict): Configuration dictionary from YAML file
    """

    logger = get_logger("DataHandler")
    
    s3_uri = config['s3_uri']
    n_train = config['n_train']
    n_test = config['n_test']
    aws_dir = config['aws_dir']
    keys = config['keys']

    # assert not (os.path.exists(aws_dir) and os.listdir(aws_dir)), \
    #     f"Directory {aws_dir} is not empty. Please provide an empty directory."

    if not os.path.exists(aws_dir) or not os.listdir(aws_dir):
        os.makedirs(aws_dir, exist_ok=True)
        S3_DataHandler.download_s3_folder(s3_uri, aws_dir)

    else: 
        logger.info("───────────────────────────────")
        logger.info("aws-data already exist. Skipping download...")
        logger.info("───────────────────────────────\n")

    # generate GT-train / GT-test folders
    gt_handler = S3_DataHandler(
        src_dir=aws_dir,
        dst_dir="data",
        required_keys = keys
    )
    gt_handler.generate_GT_train_test(n_train=n_train, n_test=n_test)
    

    # generate model-train / model-test folders
    model_handler = ModelDataHandler(
        GT_train=config['output_dirs']['gt_train'],
        GT_test=config['output_dirs']['gt_test'],
        model_dir=config['output_dirs']['model_dataset']
    )
    model_handler.generate_MODEL_train_test()


def main():
    parser = argparse.ArgumentParser(description='Generate model dataset from GT dataset stored in S3')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    generate_model_dataset(config)

if __name__ == "__main__":
    main()