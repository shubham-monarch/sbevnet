#! /usr/bin/env python3

import os
import random
import shutil
from tqdm import tqdm
from typing import List

from helpers import get_logger

def _copy_files(files: List[str], src_dir: str, dest_dir: str, desc: str):
    '''Copy files to destination'''
    
    for f in tqdm(files, desc=desc):
        src = os.path.join(src_dir, f)
        dst = os.path.join(dest_dir, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

def _get_all_files(src_dir: str) -> List[str]:
            all_files = []
            for root, dirs, files in os.walk(src_dir):
                for f in files:
                    rel_path = os.path.relpath(os.path.join(root, f), src_dir)
                    all_files.append(rel_path)
            return all_files

        
        
class DataHandlerGT:
    def __init__(self, src_dir: str, dst_dir: str):
        self.logger = get_logger("GTHandler")
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        
    def split_into_train_val_test(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15):
        '''gt-dataset -->  gt-train / gt-val / gt-test'''
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) <= 1e-9, 'Train, validation and test ratios must sum to 1'

        dir_train = os.path.join(self.dst_dir, "gt-train")
        dir_val = os.path.join(self.dst_dir, "gt-val")
        dir_test = os.path.join(self.dst_dir, "gt-test")

        if os.path.exists(dir_train): assert not os.listdir(dir_train), f'Expected {dir_train} to be empty, but it is not.'
        if os.path.exists(dir_val): assert not os.listdir(dir_val), f'Expected {dir_val} to be empty, but it is not.'
        if os.path.exists(dir_test): assert not os.listdir(dir_test), f'Expected {dir_test} to be empty, but it is not.'
        
        os.makedirs(dir_train, exist_ok=True)
        os.makedirs(dir_val, exist_ok=True) 
        os.makedirs(dir_test, exist_ok=True)
        
        all_files = _get_all_files(self.src_dir)
        random.shuffle(all_files)

        n_files = len(all_files)
        n_train = int(train_ratio * n_files)
        n_val = int(val_ratio * n_files)
        
        train_files = all_files[:n_train]
        val_files = all_files[n_train:n_train+n_val]
        test_files = all_files[n_train+n_val:]

        _copy_files(train_files, self.src_dir, dir_train, 'Copying training files')
        _copy_files(val_files, self.src_dir, dir_val, 'Copying validation files')
        _copy_files(test_files, self.src_dir, dir_test, 'Copying test files')

        assert len(train_files) == n_train, f'Expected {n_train} training files, but got {len(train_files)}'
        assert len(val_files) == n_val, f'Expected {n_val} validation files, but got {len(val_files)}'
        assert len(test_files) == n_files - n_train - n_val, f'Expected {n_files - n_train - n_val} test files, but got {len(test_files)}'


class DataHandlerModel:
    
    def __init__(self,\
                gt_train: str, gt_val: str, gt_test: str,\
                n_train: int, n_val: int, n_test: int,\
                model: str, 
                ):
        '''
        :param n_train: number of model training samples
        :param n_val: number of model validation samples
        :param n_test: number of model test samples
        :param model: path to model dataset folder
        :param gt_train: path to gt-train folder
        :param gt_val: path to gt-val folder
        :param gt_test: path to gt-test folder
        '''
        self.logger = get_logger("DataHandlerModel")        
        
        # num-train / num-val / num-test samples
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        
        # gt folders
        self.gt_train = gt_train
        self.gt_val = gt_val
        self.gt_test = gt_test

        # model-dataset folders
        self.model = model
        self.model_train = os.path.join(self.model, "train")
        self.model_val = os.path.join(self.model, "val")
        self.model_test = os.path.join(self.model, "test")

        
    def sample_gt_data(self):
        '''Sample n_train, n_val, and n_test samples from gt-train, gt-val, and gt-test folders'''
        
        # model-train / model-val / model-test must be empty
        assert not (os.path.exists(self.model_train) and os.listdir(self.model_train)), "model_train must be empty"
        assert not (os.path.exists(self.model_val) and os.listdir(self.model_val)), "model_val must be empty"
        assert not (os.path.exists(self.model_test) and os.listdir(self.model_test)), "model_test must be empty"

        gt_train_files = _get_all_files(self.gt_train)
        gt_val_files = _get_all_files(self.gt_val)
        gt_test_files = _get_all_files(self.gt_test)

        sampled_train_files = random.sample(gt_train_files, self.n_train)
        sampled_val_files = random.sample(gt_val_files, self.n_val)
        sampled_test_files = random.sample(gt_test_files, self.n_test)

        _copy_files(sampled_train_files, self.gt_train, self.model_train, 'Sampling training files')
        _copy_files(sampled_val_files, self.gt_val, self.model_val, 'Sampling validation files')
        _copy_files(sampled_test_files, self.gt_test, self.model_test, 'Sampling test files')
    

    def generate_model_dataset(self, raw_data_dir = None, model_data_dir = None):
        '''Generate [model-train / model-val / model-test] from [gt-train / gt-val / gt-test]'''

        assert raw_data_dir is not None, "raw_data_dir is required"
        assert model_data_dir is not None, "model_data_dir is required"        
        assert not (os.path.exists(model_data_dir) and os.listdir(model_data_dir)), "model_data_dir must be empty"

        self.logger.info(f"=========================")
        self.logger.info(f"Processing raw data from {raw_data_dir} to {model_data_dir}")
        self.logger.info(f"=========================\n")

        # Create the target folder if it doesn't exist
        os.makedirs(model_data_dir, exist_ok=True)

        # Define the target subfolders
        left_folder = os.path.join(model_data_dir, 'left')
        right_folder = os.path.join(model_data_dir, 'right')
        seg_masks_mono_folder = os.path.join(model_data_dir, 'seg-masks-mono')
        seg_masks_rgb_folder = os.path.join(model_data_dir, 'seg-masks-rgb')
        cam_extrinsics_folder = os.path.join(model_data_dir, 'cam-extrinsics')


        # Create the target subfolders if they don't exist
        os.makedirs(left_folder, exist_ok=True)
        os.makedirs(right_folder, exist_ok=True)
        os.makedirs(seg_masks_mono_folder, exist_ok=True)
        os.makedirs(seg_masks_rgb_folder, exist_ok=True)
        os.makedirs(cam_extrinsics_folder, exist_ok=True)

        # Count total files for progress bar
        total_files = 0
        for root, dirs, files in os.walk(raw_data_dir):
            for file in files:
                if file.endswith('left.jpg') or \
                   file.endswith('right.jpg') or \
                   file.endswith('-mono.png') or \
                   file.endswith('-rgb.png') or \
                   file.endswith('cam-extrinsics.npy'):
                    total_files += 1
        
        self.logger.info(f"=========================")
        self.logger.info(f"Total files: {total_files}") 
        self.logger.info(f"=========================\n")

        with tqdm(total=total_files, desc="Organizing Images") as pbar:
            for root, dirs, files in os.walk(raw_data_dir):
                # Get folder number from root path
                folder_num = os.path.basename(root)
                if not folder_num.isdigit():
                    continue

                for file in files:
                    if file == 'left.jpg':
                        new_filename = f"{folder_num}__left.jpg"
                        shutil.copy(os.path.join(root, file), os.path.join(left_folder, new_filename))
                        pbar.update(1)
                    elif file == 'right.jpg':
                        new_filename = f"{folder_num}__right.jpg"
                        shutil.copy(os.path.join(root, file), os.path.join(right_folder, new_filename))
                        pbar.update(1)
                    elif file == 'seg_mask_mono.png':
                        new_filename = f"{folder_num}__seg-mask-mono.png"
                        shutil.copy(os.path.join(root, file), os.path.join(seg_masks_mono_folder, new_filename))
                        pbar.update(1)
                    elif file == 'seg_mask_rgb.png':
                        new_filename = f"{folder_num}__seg-mask-rgb.png"
                        shutil.copy(os.path.join(root, file), os.path.join(seg_masks_rgb_folder, new_filename))
                        pbar.update(1)
                    elif file == 'cam_extrinsics.npy':
                        new_filename = f"{folder_num}__cam-extrinsics.npy"
                        shutil.copy(os.path.join(root, file), os.path.join(cam_extrinsics_folder, new_filename))
                        pbar.update(1)


if __name__ == "__main__":
    # gt_handler = DataHandlerGT(src_dir="data/dataset-gt", dst_dir="data")
    # gt_handler.split_into_train_val_test()

    model_handler = DataHandlerModel(gt_train="data/gt-train", 
                                     gt_val="data/gt-val", 
                                     gt_test="data/gt-test", 
                                     n_train=700, 
                                     n_val=150, 
                                     n_test=150, 
                                     model="data/model-dataset")
    model_handler.sample_gt_data()