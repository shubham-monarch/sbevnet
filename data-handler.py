#! /usr/bin/env python3

import os
import random
import shutil
from tqdm import tqdm
from typing import List
import boto3

from helpers import get_logger


def _get_leaf_folders(src_dir: str) -> List[str]:
    """Get all leaf folders inside the given folder recursively and return their relative positions as a list"""
    leaf_folders = []
    for root, dirs, files in os.walk(src_dir):
        if not dirs: 
            leaf_folders.append(os.path.relpath(root, src_dir))  
    return leaf_folders

# def _copy_files(files: List[str], src_dir: str, dest_dir: str, desc: str):
#     '''Copy files to destination'''
    
#     for f in tqdm(files, desc=desc):
#         src = os.path.join(src_dir, f)
#         dst = os.path.join(dest_dir, f)
#         os.makedirs(os.path.dirname(dst), exist_ok=True)
#         shutil.copy2(src, dst)

def _copy_folders(folder_list: List[str], dst_dir: str):
    '''Copy folders to destination directory'''
    
    for folder in tqdm(folder_list, desc="Copying folders"):
        src = folder
        dst = os.path.join(dst_dir, os.path.basename(folder))
        shutil.copytree(src, dst, dirs_exist_ok=True)
   


def _get_all_files(src_dir: str) -> List[str]:
            all_files = []
            for root, dirs, files in os.walk(src_dir):
                for f in files:
                    rel_path = os.path.relpath(os.path.join(root, f), src_dir)
                    all_files.append(rel_path)
            return all_files

        
        
class GTDataHandler:
    def __init__(self, src_dir: str, dst_dir: str) -> None:
        '''
        :param src_dir: path to GT-dataset
        :param dst_dir: path to save gt-train /  gt-test
        '''
        
        self.logger = get_logger("GTHandler")
        
        self.src_dir = src_dir
        self.dst_dir = dst_dir

        # [GT-train / GT-test] folders
        self.GT_train = os.path.join(self.dst_dir, "GT-train")
        self.GT_test = os.path.join(self.dst_dir, "GT-test")
        

    def _copy_folders(self, folder_list: List[str], src_dir: str, dst_dir: str) -> None:
        '''Copy folders to destination directory'''
        logger = get_logger("_copy_folders")

        assert not (os.path.exists(dst_dir) and os.listdir(dst_dir)), f"Destination directory must be empty!"
        
        for folder in tqdm(folder_list, desc="Copying folders"):
            src = os.path.join(src_dir, folder)
            dst = os.path.join(dst_dir, folder)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)
        
      
    def generate_sample_train_test(self, n_train: int, n_test: int) -> None:
        '''generate GT-train / GT-test folders'''
        
        # assert [gt-train / gt-test] folders are empty
        assert not (os.path.exists(self.GT_train) and os.listdir(self.GT_train)), f"GT-train must be empty!"
        assert not (os.path.exists(self.GT_test) and os.listdir(self.GT_test)), f"GT-test must be empty!"

        leaf_folders = _get_leaf_folders(self.src_dir)
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
        assert len(_get_leaf_folders(self.GT_train)) == n_train, f'Expected {n_train} training files, but got {len(_get_leaf_folders(self.GT_train))}'
        assert len(_get_leaf_folders(self.GT_test)) == n_test, f'Expected {n_test} test files, but got {len(_get_leaf_folders(self.GT_test))}'
      

class ModelDataHandler:
    
    def __init__(self,\
                model_dir: str,\
                gt_train: str,gt_val: str, gt_test: str):
        '''
        :param model_dir: path to model dataset folder
        :param gt_train: path to gt-train folder
        :param gt_val: path to gt-val folder
        :param gt_test: path to gt-test folder
        '''
        self.logger = get_logger("DataHandlerModel")        
        
        # GT folders
        self.gt_train = gt_train
        self.gt_val = gt_val
        self.gt_test = gt_test

        # model-dataset folders
        self.model_dir = model_dir
        self.model_train = os.path.join(self.model_dir, "train")
        self.model_val = os.path.join(self.model_dir, "val")
        self.model_test = os.path.join(self.model_dir, "test")


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
                    # elif file == 'cam_extrinsics.npy':
                    #     new_filename = f"{folder_num}__cam-extrinsics.npy"
                    #     shutil.copy(os.path.join(root, file), os.path.join(cam_extrinsics_folder, new_filename))
                    #     pbar.update(1)


if __name__ == "__main__":
    gt_handler = GTDataHandler(src_dir="data/GT", dst_dir="data")
    gt_handler.generate_sample_train_test(n_train=800, n_test=200)

    # model_handler = ModelDataHandler(gt_train="data/GT-train", 
    #                                  gt_val="data/GT-val", 
    #                                  gt_test="data/GT-test", 
    #                                  model_dir="data/model-dataset")
    # model_handler.generate_model_dataset()

    