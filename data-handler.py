#! /usr/bin/env python3

import os
import random
import shutil
from tqdm import tqdm
from typing import List

from helpers import get_logger


class GTDataHandler:
    def __init__(self, src_dir: str, dst_dir: str) -> None:
        '''
        :param src_dir: path to GT-dataset
        :param dst_dir: path to save gt-train /  gt-test
        '''
        
        self.logger = get_logger("GTHandler")
        
        self.src_dir = src_dir
        self.dst_dir = dst_dir

        # files required in each valid GT-dataset folder
        self.key_files = ['left.jpg', 'right.jpg', 'seg-mask-mono.png', 'seg-mask-rgb.png']

        # [GT-train / GT-test] folders
        self.GT_train = os.path.join(self.dst_dir, "GT-train")
        self.GT_test = os.path.join(self.dst_dir, "GT-test")
        
        self.logger.info(f"=========================")
        self.logger.info("Generating [GT-train / GT-test] from GT-dataset...")
        self.logger.info(f"=========================\n")

    def __get_leaf_folders(self, src_dir: str) -> List[str]:
        '''Get all leaf folders inside the given folder recursively and return their relative positions as a list'''
        leaf_folders = []
        for root, dirs, files in os.walk(src_dir):
            if not dirs: 
                leaf_folders.append(os.path.relpath(root, src_dir))  
        return leaf_folders


    def __copy_folders(self, folder_list: List[str], src_dir: str, dst_dir: str) -> None:
        '''Copy folders to destination directory'''
        
        assert not (os.path.exists(dst_dir) and os.listdir(dst_dir)), f"Destination directory must be empty!"
        
        for idx, folder in enumerate(tqdm(folder_list, desc="Copying folders"), 1):
            src = os.path.join(src_dir, folder)
            dst = os.path.join(dst_dir, str(idx))
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)

    def __remove_incomplete_GT_folders(self):
        '''Remove incomplete GT folders'''
        
        self.logger.info(f"=========================")
        self.logger.info("Removing incomplete GT folders...")
        self.logger.info(f"=========================\n")
        
        leaf_folders = self.__get_leaf_folders(self.src_dir)

        
        incomplete_folders = []
        for folder in tqdm(leaf_folders, desc="Checking folders"):
            folder_files = os.listdir(os.path.join(self.src_dir, folder))
            
            # check if folder has exactly the required files
            if not all(f in folder_files for f in self.key_files):
                incomplete_folders.append(folder)
                
        # Remove incomplete folders
        for folder in incomplete_folders:
            shutil.rmtree(os.path.join(self.src_dir, folder))
        
        self.logger.error(f"=========================")
        self.logger.error(f"Removed {len(incomplete_folders)} incomplete folders")
        self.logger.error(f"Remaining folders: {len(leaf_folders) - len(incomplete_folders)}")
        self.logger.error(f"=========================\n")
        
      
    def generate_GT_train_test(self, n_train: int, n_test: int) -> None:
        '''generate GT-train / GT-test folders'''
        
        # remove incomplete GT folders
        self.__remove_incomplete_GT_folders()

        # assert [gt-train / gt-test] folders are empty
        assert not (os.path.exists(self.GT_train) and os.listdir(self.GT_train)), f"GT-train must be empty!"
        assert not (os.path.exists(self.GT_test) and os.listdir(self.GT_test)), f"GT-test must be empty!"

        leaf_folders = self.__get_leaf_folders(self.src_dir)
        random.shuffle(leaf_folders)
        
        assert len(leaf_folders) > n_train + n_test, f'Expected at least {n_train + n_test} leaf folders, but got {len(leaf_folders)}'

        os.makedirs(self.GT_train, exist_ok=True)
        os.makedirs(self.GT_test, exist_ok=True)
                
        train_folders = leaf_folders[:n_train]
        test_folders = leaf_folders[n_train:n_train + n_test]
        
        # copy folders to GT_train / GT_test 
        self.__copy_folders(train_folders, self.src_dir, self.GT_train)
        self.__copy_folders(test_folders, self.src_dir, self.GT_test)

        # assert number of folders in gt-train / gt-test
        assert len(self.__get_leaf_folders(self.GT_train)) == n_train, f'Expected {n_train} training files, but got {len(self.__get_leaf_folders(self.GT_train))}'
        assert len(self.__get_leaf_folders(self.GT_test)) == n_test, f'Expected {n_test} test files, but got {len(self.__get_leaf_folders(self.GT_test))}'
      

class ModelDataHandler:
    
    def __init__(self,\
                model_dir: str,\
                GT_train: str, GT_test: str):
        '''
        :param model_dir: path to model dataset folder
        :param gt_train: path to gt-train folder
        :param gt_test: path to gt-test folder
        '''
        self.logger = get_logger("DataHandlerModel")        
        
        # GT folders
        self.GT_train = GT_train
        self.GT_test = GT_test
        
        # model-dataset folders
        self.model_dir = model_dir
        self.model_train = os.path.join(self.model_dir, "train")
        self.model_test = os.path.join(self.model_dir, "test")

        self.logger.info(f"=========================")
        self.logger.info("Generating [MODEL-train / MODEL-test] from [GT-train / GT-test]...")
        self.logger.info(f"=========================\n")

    def __restructure_GT_folder(self, GT_dir: str, MODEL_dir: str):
        # [model-train / model-test] folders should be empty
        assert not (os.path.exists(MODEL_dir) and os.listdir(MODEL_dir)), "model_train must be empty"

        # create [model-train / model-test] folders
        os.makedirs(MODEL_dir, exist_ok=True)

        # model-train folders
        left_folder = os.path.join(MODEL_dir, 'left')
        right_folder = os.path.join(MODEL_dir, 'right')
        seg_masks_mono_folder = os.path.join(MODEL_dir, 'seg-masks-mono')
        seg_masks_rgb_folder = os.path.join(MODEL_dir, 'seg-masks-rgb')
        
        # Create the target subfolders if they don't exist
        os.makedirs(left_folder, exist_ok=True)
        os.makedirs(right_folder, exist_ok=True)
        os.makedirs(seg_masks_mono_folder, exist_ok=True)
        os.makedirs(seg_masks_rgb_folder, exist_ok=True)

        # Count total files for progress bar
        total_files = 0
        for root, dirs, files in os.walk(GT_dir):
            for file in files:
                if file.endswith('left.jpg') or \
                   file.endswith('right.jpg') or \
                   file.endswith('-mono.png') or \
                   file.endswith('-rgb.png'):
                    total_files += 1

        # self.logger.info(f"=========================")
        # self.logger.info(f"Total files: {total_files}") 
        # self.logger.info(f"=========================\n")


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


    def generate_MODEL_train_test(self):
        self.__restructure_GT_folder(self.GT_train, self.model_train)
        self.__restructure_GT_folder(self.GT_test, self.model_test)
      
if __name__ == "__main__":
    gt_handler = GTDataHandler(src_dir="data/GT", dst_dir="data")
    gt_handler.generate_GT_train_test(n_train=800, n_test=200)

    model_handler = ModelDataHandler(GT_train="data/GT-train", 
                                     GT_test="data/GT-test", 
                                     model_dir="data/model-dataset")
    model_handler.generate_MODEL_train_test()

    