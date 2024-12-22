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

        
        
class GTDataHandler:
    def __init__(self, src_dir: str, dst_dir: str,\
                n_train: int, n_val: int, n_test: int):
        
        ''''
        :param src_dir: path to gt-dataset
        :param dst_dir: path to save split-train / split-val / split-test
        :param n_train: number of training samples
        :param n_val: number of validation samples
        :param n_test: number of test samples
        '''
        
        self.logger = get_logger("GTHandler")
        
        self.src_dir = src_dir
        self.dst_dir = dst_dir

        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
          
        # [GT-train / GT-val / GT-test] folders
        self.gt_train = os.path.join(self.dst_dir, "GT-train")
        self.gt_val = os.path.join(self.dst_dir, "GT-val")
        self.gt_test = os.path.join(self.dst_dir, "GT-test")
        

    def split_into_train_val_test(self):
        '''gt-dataset -->  gt-train / gt-val / gt-test'''
        
        if os.path.exists(self.gt_train): assert not os.listdir(self.gt_train), f'Expected {self.gt_train} to be empty, but it is not.'
        if os.path.exists(self.gt_val): assert not os.listdir(self.gt_val), f'Expected {self.gt_val} to be empty, but it is not.'
        if os.path.exists(self.gt_test): assert not os.listdir(self.gt_test), f'Expected {self.gt_test} to be empty, but it is not.'

        os.makedirs(self.gt_train, exist_ok=True)
        os.makedirs(self.gt_val, exist_ok=True) 
        os.makedirs(self.gt_test, exist_ok=True)
        
        all_files = _get_all_files(self.src_dir)
        random.shuffle(all_files)

        train_files = all_files[:self.n_train]
        val_files = all_files[self.n_train:self.n_train + self.n_val]
        test_files = all_files[self.n_train + self.n_val:self.n_train + self.n_val + self.n_test]

        # copy files to split-train / split-val / split-test
        _copy_files(train_files, self.src_dir, self.gt_train, 'Copying training files')
        _copy_files(val_files, self.src_dir, self.gt_val, 'Copying validation files')
        _copy_files(test_files, self.src_dir, self.gt_test, 'Copying test files')

        assert len(train_files) == self.n_train, f'Expected {self.n_train} training files, but got {len(train_files)}'
        assert len(val_files) == self.n_val, f'Expected {self.n_val} validation files, but got {len(val_files)}'
        assert len(test_files) == self.n_test, f'Expected {self.n_test} test files, but got {len(test_files)}'
      

class ModelDataHandler:
    
    def __init__(self,\
                model: str,\
                gt_train: str,gt_val: str, gt_test: str):
        '''
        :param model: path to model dataset folder
        :param gt_train: path to gt-train folder
        :param gt_val: path to gt-val folder
        :param gt_test: path to gt-test folder
        '''
        self.logger = get_logger("DataHandlerModel")        
        
        # gt folders
        self.gt_train = gt_train
        self.gt_val = gt_val
        self.gt_test = gt_test

        # model-dataset folders
        self.model = model
        self.model_train = os.path.join(self.model, "train")
        self.model_val = os.path.join(self.model, "val")
        self.model_test = os.path.join(self.model, "test")


    def generate_model_dataset(self):
        
        # [model-train / model-val / model-test] folders should be empty
        assert not (os.path.exists(self.model_train) and os.listdir(self.model_train)), "model_train must be empty"
        assert not (os.path.exists(self.model_val) and os.listdir(self.model_val)), "model_val must be empty"
        assert not (os.path.exists(self.model_test) and os.listdir(self.model_test)), "model_test must be empty"

        
        # create [model-train / model-val / model-test] folders
        os.makedirs(self.model_train, exist_ok=True)
        os.makedirs(self.model_val, exist_ok=True)
        os.makedirs(self.model_test, exist_ok=True)

        # model-train folders
        left_folder = os.path.join(self.model_train, 'left')
        right_folder = os.path.join(self.model_train, 'right')
        seg_masks_mono_folder = os.path.join(self.model_train, 'seg-masks-mono')
        seg_masks_rgb_folder = os.path.join(self.model_train, 'seg-masks-rgb')
        
        # Create the target subfolders if they don't exist
        os.makedirs(left_folder, exist_ok=True)
        os.makedirs(right_folder, exist_ok=True)
        os.makedirs(seg_masks_mono_folder, exist_ok=True)
        os.makedirs(seg_masks_rgb_folder, exist_ok=True)

        # Count total files for progress bar
        total_files = 0
        for root, dirs, files in os.walk(self.gt_train):
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
            for root, dirs, files in os.walk(self.gt_train):
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
                    # elif file == 'cam_extrinsics.npy':
                    #     new_filename = f"{folder_num}__cam-extrinsics.npy"
                    #     shutil.copy(os.path.join(root, file), os.path.join(cam_extrinsics_folder, new_filename))
                    #     pbar.update(1)


if __name__ == "__main__":
    gt_handler = GTDataHandler(src_dir="data/GT", dst_dir="data",\
                               n_train=700, n_val=150, n_test=150)
    
    gt_handler.split_into_train_val_test()

    # model_handler = ModelDataHandler(gt_train="data/dataset-gt/split-train", 
    #                                  gt_val="data/dataset-gt/split-val", 
    #                                  gt_test="data/dataset-gt/split-test", 
    #                                  model="data/model-dataset")
    # model_handler.generate_model_dataset()

    