#! /usr/bin/env python3

import os
import random
import shutil
from tqdm import tqdm
from typing import List, Dict, Union, Any
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import boto3
from urllib.parse import urlparse
import argparse
import yaml
from typing import Tuple

from helpers import get_logger, flip_mask, get_files_from_folder, calculate_modified_z_outlier_bounds


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
        
        # self.logger.info("───────────────────────────────")
        # self.logger.info("Generating [GT-train / GT-test] from GT-dataset...")
        # self.logger.info("───────────────────────────────\n")

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
             # Create a file that stores the relative path of the folder from src
            with open(os.path.join(dst, "file_name.txt"), "w") as f:
                f.write(folder)

    def _remove_incomplete_GT_folders(self):
        """Remove folders with missing left.jpg / right.jpg / seg-mask-mono.png / seg-mask-rgb.png"""

        self.logger.info("───────────────────────────────")
        self.logger.info("Checking for incomplete GT folders...")
        self.logger.info("───────────────────────────────\n")
        
        leaf_folders = S3_DataHandler._get_leaf_folders(self.src_dir)

        
        incomplete_folders = []
        for folder in tqdm(leaf_folders, desc="Removing incomplete folders"):
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
    def download_s3_folder(s3_uri: Union[str, List[str]], local_dir: str) -> None:
        """Recursively download an S3 folder or a list of S3 folders to a local directory.
        
        Args:
            s3_uri (Union[str, List[str]]): S3 URI in format s3://bucket-name/path/to/folder, or a list of such URIs.
            local_dir (str): Local directory to download files to.
        """
        if isinstance(s3_uri, list):
            for uri in s3_uri:
                S3_DataHandler.download_s3_folder(uri, local_dir)
            return

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
        │   ├── seg-mask-rgb.png
        │   ├── cam-extrinsics.npy
        │   └── file_name.txt
        ├── 2
        │   ├── left.jpg
        │   ├── right.jpg
        │   ├── seg-mask-mono.png
        │   ├── seg-mask-rgb.png
        │   ├── cam-extrinsics.npy
        │   └── file_name.txt
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
        assert not (n_train == -1 and n_test == -1), "Both n_train and n_test cannot be -1."

        os.makedirs(self.GT_train, exist_ok=True)
        os.makedirs(self.GT_test, exist_ok=True)
        
        if n_train >= 0:
            train_folders = leaf_folders[:n_train]
        else: 
            train_folders = leaf_folders[:]
        
        if n_test >= 0:
            test_folders = leaf_folders[n_train:n_train + n_test]
        else: 
            test_folders = leaf_folders[n_train:]
        
        # copy folders to GT_train / GT_test 
        self._copy_folders(train_folders, self.src_dir, self.GT_train)
        self._copy_folders(test_folders, self.src_dir, self.GT_test)

        # assert number of folders in gt-train / gt-test
        # assert len(S3_DataHandler._get_leaf_folders(self.GT_train)) == len(train_folders), f'Expected {len(train_folders)} training files, but got {len(S3_DataHandler._get_leaf_folders(self.GT_train))}'
        # assert len(S3_DataHandler._get_leaf_folders(self.GT_test)) == len(test_folders), f'Expected {len(test_folders)} test files, but got {len(S3_DataHandler._get_leaf_folders(self.GT_test))}'
      
class ModelDataHandler:
    
    @staticmethod
    def _restructure_GT_folder(GT_dir: str, MODEL_dir: str) -> None:
        """Restructure GT folder into model-train / model-test folders and generate a filenames folder"""
        
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
        filenames_folder = os.path.join(MODEL_dir, 'filenames')
        ipm_left_folder = os.path.join(MODEL_dir, 'ipm-left')
        H_img_to_bev_folder = os.path.join(MODEL_dir, 'H_img_to_bev')

        # Create the target subfolders if they don't exist
        os.makedirs(left_folder, exist_ok=True)
        os.makedirs(right_folder, exist_ok=True)
        os.makedirs(seg_masks_mono_folder, exist_ok=True)
        os.makedirs(seg_masks_rgb_folder, exist_ok=True)
        os.makedirs(cam_extrinsics_folder, exist_ok=True)
        os.makedirs(filenames_folder, exist_ok=True)
        os.makedirs(ipm_left_folder, exist_ok=True)
        os.makedirs(H_img_to_bev_folder, exist_ok=True)

        # Count total files for progress bar (including file_name.txt files)
        total_files = 0
        for root, dirs, files in os.walk(GT_dir):
            for file in files:
                if file.endswith('left.jpg') or \
                   file.endswith('right.jpg') or \
                   file.endswith('-mono.png') or \
                   file.endswith('-rgb.png') or \
                   file.endswith('cam-extrinsics.npy') or \
                   file == "file_name.txt" or \
                   file.endswith('ipm-left.png') or \
                   file.endswith('H_img_to_bev.npy'):
                    total_files += 1

        with tqdm(total=total_files, desc="Organizing Images") as pbar:
            for root, dirs, files in os.walk(GT_dir):
                # Only process numbered folders
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
                    elif file == "file_name.txt":
                        new_filename = f"{folder_num}__filename.txt"
                        shutil.copy(os.path.join(root, file), os.path.join(filenames_folder, new_filename))
                        pbar.update(1)
                    elif file.endswith('ipm-left.png'):
                        new_filename = f"{folder_num}__ipm-left.png"
                        shutil.copy(os.path.join(root, file), os.path.join(ipm_left_folder, new_filename))
                        pbar.update(1)
                    elif file.endswith('H_img_to_bev.npy'):
                        new_filename = f"{folder_num}__H_img_to_bev.npy"
                        shutil.copy(os.path.join(root, file), os.path.join(H_img_to_bev_folder, new_filename))
                        pbar.update(1)

    @staticmethod
    def _flip_masks(src_dir: str, dest_dir: str) -> None:
        '''Flip the masks in the source folder and save them to the destination folder.'''
        
        masks = get_files_from_folder(src_dir, ['.png'])
        for mask_path in tqdm(masks, desc="Flipping masks"):
            mask_flipped_mono = flip_mask(mask_path)
            cv2.imwrite(os.path.join(dest_dir, os.path.basename(mask_path)), mask_flipped_mono)

    @staticmethod
    def _populate_json(json_path: str, dataset_path: str, model_train_dir: str, model_test_dir: str) -> None:
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
                "rgb_left": get_relative_files(os.path.join(model_train_dir, 'left'), IMG_EXTENSIONS),
                "rgb_right": get_relative_files(os.path.join(model_train_dir, 'right'), IMG_EXTENSIONS),
                "top_seg": get_relative_files(os.path.join(model_train_dir, 'seg-masks-mono'), ['.png']),
                "ipm_rgb": get_relative_files(os.path.join(model_train_dir, 'ipm-left'), ['.png']),
                "top_ipm_m": get_relative_files(os.path.join(model_train_dir, 'H_img_to_bev'), ['.npy']),
                #"confs": get_relative_files(os.path.join(model_train_dir, 'cam-extrinsics'), ['.npy']),
            },
            "test": {
                "rgb_left": get_relative_files(os.path.join(model_test_dir, 'left'), IMG_EXTENSIONS),
                "rgb_right": get_relative_files(os.path.join(model_test_dir, 'right'), IMG_EXTENSIONS),
                "top_seg": get_relative_files(os.path.join(model_test_dir, 'seg-masks-mono'), ['.png']),
                "ipm_rgb": get_relative_files(os.path.join(model_test_dir, 'ipm-left'), ['.png']),
                "top_ipm_m": get_relative_files(os.path.join(model_test_dir, 'H_img_to_bev'), ['.npy']),
                #"confs": get_relative_files(os.path.join(model_test_dir, 'cam-extrinsics'), ['.npy']),
            }
        }

        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def _remap_mask_labels(mask_dir: str) -> None:
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
    def _remove_label_outliers(mask_dir: str, labels_to_remove: List[int], threshold: float) -> Tuple[int, List[str]]:
        '''Remove masks with more than threshold % of any of the target labels'''
        logger = get_logger("DataHandlerModel")
        masks = get_files_from_folder(mask_dir, ['.png'])
        cnt = 0
        files = []
        for mask_path in tqdm(masks, desc="Removing label outliers"):
            seg_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            for label in labels_to_remove:
                label_mask_cnt = np.sum(seg_mask == label)
                if label_mask_cnt / seg_mask.size >= threshold:
                    cnt += 1
                    files.append(os.path.basename(mask_path))
                    ModelDataHandler._remove_mask_from_model_dataset(mask_path)
                    break
        return cnt, files

    @staticmethod
    def _remove_axis_angle_outliers(cam_extrinsics_dir: str) -> Tuple[int, List[str]]:
        """Remove files with axis angle outliers in the given cam_extrinsics folder using modified z-score bounds."""
        logger = get_logger("DataHandlerModel")
        npy_files = get_files_from_folder(cam_extrinsics_dir, ['.npy'])
        valid_files = []
        rvec_x_list, rvec_y_list, rvec_z_list, rvec_mag_list = [], [], [], []
        for npy_path in tqdm(npy_files, desc="Removing axis angle outliers"):
            matrix = np.load(npy_path)
            if matrix.shape != (4, 4):
                logger.warning(f"File {npy_path} does not contain a 4x4 matrix. Skipping.")
                continue
            rotation = matrix[:3, :3]
            rotation_vector, _ = cv2.Rodrigues(rotation)
            rotation_vector = rotation_vector.flatten()
            norm = np.linalg.norm(rotation_vector)
            rvec_mag_list.append(norm)
            if norm > 1e-6:
                rotation_vector = rotation_vector / norm
            rvec_x_list.append(rotation_vector[0])
            rvec_y_list.append(rotation_vector[1])
            rvec_z_list.append(rotation_vector[2])
            valid_files.append(npy_path)
        
        if not valid_files:
            logger.error("No valid rotation vectors found in provided folder.")
            return 0, []
        
        valid_files = np.array(valid_files)
        rvec_x = np.array(rvec_x_list)
        rvec_y = np.array(rvec_y_list)
        rvec_z = np.array(rvec_z_list)
        rvec_mag = np.array(rvec_mag_list)
        
        lb_x, ub_x = calculate_modified_z_outlier_bounds(rvec_x)
        lb_y, ub_y = calculate_modified_z_outlier_bounds(rvec_y)
        lb_z, ub_z = calculate_modified_z_outlier_bounds(rvec_z)
        lb_mag, ub_mag = calculate_modified_z_outlier_bounds(rvec_mag)
        
        mask_x = (rvec_x >= lb_x) & (rvec_x <= ub_x)
        mask_y = (rvec_y >= lb_y) & (rvec_y <= ub_y)
        mask_z = (rvec_z >= lb_z) & (rvec_z <= ub_z)
        mask_mag = (rvec_mag >= lb_mag) & (rvec_mag <= ub_mag)
        
        # Identify files as outliers if any of the x, z, or magnitude components is outside its bounds
        # outlier_indices = np.where(~mask_x | ~mask_z | ~mask_mag)[0]
        outlier_indices = np.where(~mask_z)[0]
        
        cnt = 0
        outlier_files = []
        for idx in sorted(outlier_indices, key=lambda i: int(os.path.basename(valid_files[i]).split('__')[0])):
            file = valid_files[idx]
            ModelDataHandler._remove_mask_from_model_dataset(file)
            cnt += 1
            outlier_files.append(os.path.basename(file))
    
        return cnt, outlier_files

    @staticmethod
    def generate_MODEL_train_test(GT_train: str, GT_test: str, 
                                  model_train_dir: str, model_test_dir: str, model_dir: str, 
                                  labels_to_remove: List[int]) -> None:
        """Generate model train and test datasets."""
        
        logger = get_logger("DataHandlerModel")

        ModelDataHandler._restructure_GT_folder(GT_train, model_train_dir)
        ModelDataHandler._restructure_GT_folder(GT_test, model_test_dir)

        # flip seg-masks-mono in model-train / model-test
        ModelDataHandler._flip_masks(os.path.join(model_train_dir, 'seg-masks-mono'),\
                           os.path.join(model_train_dir, 'seg-masks-mono'))
        ModelDataHandler._flip_masks(os.path.join(model_test_dir, 'seg-masks-mono'),\
                           os.path.join(model_test_dir, 'seg-masks-mono'))
        
        # flip seg-masks-rgb in model-train / model-test
        ModelDataHandler._flip_masks(os.path.join(model_train_dir, 'seg-masks-rgb'),\
                           os.path.join(model_train_dir, 'seg-masks-rgb'))
        ModelDataHandler._flip_masks(os.path.join(model_test_dir, 'seg-masks-rgb'),\
                           os.path.join(model_test_dir, 'seg-masks-rgb'))

        # flip ipm-left in model-train / model-test
        ModelDataHandler._flip_masks(os.path.join(model_train_dir, 'ipm-left'),\
                           os.path.join(model_train_dir, 'ipm-left'))
        ModelDataHandler._flip_masks(os.path.join(model_test_dir, 'ipm-left'),\
                           os.path.join(model_test_dir, 'ipm-left'))
        
        # remap label 255 to 0
        ModelDataHandler._remap_mask_labels(os.path.join(model_train_dir, 'seg-masks-mono'))
        ModelDataHandler._remap_mask_labels(os.path.join(model_test_dir, 'seg-masks-mono'))

        # # remove outlier masks for the specified labels from train and test dirs
        # total_train_cnt, _ = ModelDataHandler._remove_label_outliers(os.path.join(model_train_dir, 'seg-masks-mono'), labels_to_remove, 0.8)
        # total_test_cnt, _ = ModelDataHandler._remove_label_outliers(os.path.join(model_test_dir, 'seg-masks-mono'), labels_to_remove, 0.8)
        # total_cnt = total_train_cnt + total_test_cnt
        
        # logger.warning("───────────────────────────────")
        # logger.warning(f"Removed {total_cnt} masks with label outliers")
        # logger.warning(f"Removed {total_train_cnt} masks with label outliers in train")
        # logger.warning(f"Removed {total_test_cnt} masks with label outliers in test")
        # logger.warning("───────────────────────────────\n  ")

        # remove axis angle outliers from cam-extrinsics
        train_axis_cnt, _ = ModelDataHandler._remove_axis_angle_outliers(
            os.path.join(model_train_dir, 'cam-extrinsics'))
        test_axis_cnt, _ = ModelDataHandler._remove_axis_angle_outliers(
            os.path.join(model_test_dir, 'cam-extrinsics'))
        
        logger.warning("───────────────────────────────")
        logger.warning(f"Removed {train_axis_cnt + test_axis_cnt} axis angle outliers from cam-extrinsics")
        logger.warning(f"Removed {train_axis_cnt} axis angle outliers in train")
        logger.warning(f"Removed {test_axis_cnt} axis angle outliers in test")
        logger.warning("───────────────────────────────\n  ")

        ModelDataHandler._populate_json(os.path.join(model_dir, 'dataset.json'), model_dir, model_train_dir, model_test_dir)

    @staticmethod
    def generate_model_dataset(config_path: str) -> None:
        """
        Generate model-dataset by downloading from S3 and processing the data
        using the updated configuration structure.

        Args:
            config_path (str): Path to the YAML config file.
        """
        logger = get_logger("DataHandler")
        with open(config_path, 'r') as f:
            config: Dict[str, Any] = yaml.safe_load(f)
        
        # Extract S3 configuration from 's3_data_handler' section
        s3_config: Dict[str, Any] = config.get('s3_data_handler', {})
        s3_uri: Union[str, List[str]] = s3_config.get('s3_uri')  # type: either str or list of strings
        base_dir_s3: str = s3_config.get('base_dir')
        aws_dir: str = os.path.join(base_dir_s3, "GT-aws")
        s3_dest_dir: str = base_dir_s3
        required_keys: Any = s3_config.get('required_keys')
        n_train: int = s3_config.get('n_train')
        n_test: int = s3_config.get('n_test')
        
        # Extract output directories from 'model_data_handler' section
        model_config: Dict[str, Any] = config.get('model_data_handler', {})
        base_dir_model: str = model_config.get('base_dir')
        gt_train: str = os.path.join(base_dir_model, "GT-train")
        gt_test: str = os.path.join(base_dir_model, "GT-test")
        model_dataset: str = os.path.join(base_dir_model, "model-dataset")
        labels_to_remove: List = model_config.get('labels_to_remove', [0])

        if not os.path.exists(aws_dir) or not os.listdir(aws_dir):
            os.makedirs(aws_dir, exist_ok=True)
            S3_DataHandler.download_s3_folder(s3_uri, aws_dir)
        else: 
            logger.info("───────────────────────────────")
            logger.info("aws-data already exist. Skipping download...")
            logger.info("───────────────────────────────\n")

        gt_handler = S3_DataHandler(
            src_dir=aws_dir,
            dst_dir=s3_dest_dir,
            required_keys=required_keys
        )
        gt_handler.generate_GT_train_test(n_train=n_train, n_test=n_test)

        ModelDataHandler.generate_MODEL_train_test(
            GT_train=gt_train,
            GT_test=gt_test,
            model_train_dir=os.path.join(model_dataset, "train"),
            model_test_dir=os.path.join(model_dataset, "test"),
            model_dir=model_dataset,
            labels_to_remove=labels_to_remove
        )


def main():
    parser = argparse.ArgumentParser(description='Generate model dataset from GT dataset stored in S3')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    args = parser.parse_args()
    
    ModelDataHandler.generate_model_dataset(args.config)

if __name__ == "__main__":
    main()