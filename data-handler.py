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

from helpers import get_logger, flip_mask, get_files_from_folder


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

    @staticmethod
    def log_class_distribution(src_dir: str) -> List[Dict[int, float]]:
        '''Log the % distribution of each class in all segmentation masks in GT-aws'''
        
        leaf_folders = GTDataHandler.__get_leaf_folders(src_dir)
        class_distributions = []

        for folder in tqdm(leaf_folders, desc="Logging class % distribution"):
            seg_mask_mono_path = os.path.join(src_dir, folder, 'seg-mask-mono.png')
            if os.path.exists(seg_mask_mono_path):
                mask = cv2.imread(seg_mask_mono_path, cv2.IMREAD_GRAYSCALE)
                unique, counts = np.unique(mask, return_counts=True)
                total_pixels = mask.size
                distribution = {val: count / total_pixels * 100 for val, count in zip(unique, counts)}
                class_distributions.append(distribution)

        return class_distributions
        # GTDataHandler.__plot_class_distribution(class_distributions)

    @staticmethod
    def plot_class_distribution(class_distributions: List[Dict[int, float]]) -> None:
        '''Plot the % distribution for each class in all segmentation masks and save to disk.'''
        
        from scipy.stats import norm
        import matplotlib.pyplot as plt
        import numpy as np

        # Aggregate data for class 0
        class_0_data = [dist.get(0, 0) for dist in class_distributions]

        # Fit a normal distribution to the data
        mu, std = norm.fit(class_0_data)

        # Plot the histogram with counts instead of density
        plt.figure(figsize=(10, 6))
        plt.hist(class_0_data, bins=20, alpha=0.6, color='blue', edgecolor='black')

        # Remove density-based PDF plot since we're now showing counts
        plt.title(f"Distribution of Class 0 Percentages\nTotal masks: {len(class_0_data)}")
        plt.xlabel('Percentage of Class 0')
        plt.ylabel('Number of Masks')
        plt.grid(True)

        # Save the plot to disk
        plot_path = 'assets/class_0_distribution.png'
        plt.savefig(plot_path)
        plt.close()

        print(f"Distribution plot for class 0 saved to {plot_path}")

    @staticmethod
    def __get_leaf_folders(src_dir: str) -> List[str]:
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
        '''remove folders with missing left.jpg / right.jpg / seg-mask-mono.png / seg-mask-rgb.png'''

        self.logger.info(f"=========================")
        self.logger.info("Checking for incomplete GT folders...")
        self.logger.info(f"=========================\n")
        
        leaf_folders = GTDataHandler.__get_leaf_folders(self.src_dir)

        
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
        '''generate GT-train / GT-test folders from GT-aws folders
        
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
        '''
        
        # remove incomplete GT folders
        self.__remove_incomplete_GT_folders()

        # assert [gt-train / gt-test] folders are empty
        assert not (os.path.exists(self.GT_train) and os.listdir(self.GT_train)), f"GT-train must be empty!"
        assert not (os.path.exists(self.GT_test) and os.listdir(self.GT_test)), f"GT-test must be empty!"

        leaf_folders = GTDataHandler.__get_leaf_folders(self.src_dir)
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
        assert len(GTDataHandler.__get_leaf_folders(self.GT_train)) == n_train, f'Expected {n_train} training files, but got {len(GTDataHandler.__get_leaf_folders(self.GT_train))}'
        assert len(GTDataHandler.__get_leaf_folders(self.GT_test)) == n_test, f'Expected {n_test} test files, but got {len(GTDataHandler.__get_leaf_folders(self.GT_test))}'
      

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
        self.model_train_dir = os.path.join(self.model_dir, "train")
        self.model_test_dir = os.path.join(self.model_dir, "test")

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

    def __flip_masks(self, src_dir: str, dest_dir: str) -> None:
        '''Flip the masks in the source folder and save them to the destination folder.'''
        
        masks = get_files_from_folder(src_dir, ['.png'])
        for mask_path in tqdm(masks, desc="Flipping masks"):
            mask_flipped_mono = flip_mask(mask_path)
            cv2.imwrite(os.path.join(dest_dir, os.path.basename(mask_path)), mask_flipped_mono)

    def __populate_json(self, json_path, dataset_path):
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

    def generate_MODEL_train_test(self):
        self.__restructure_GT_folder(self.GT_train, self.model_train_dir)
        self.__restructure_GT_folder(self.GT_test, self.model_test_dir)

        # flip mono / rgb masks in model-train
        self.__flip_masks(os.path.join(self.model_train_dir, 'seg-masks-mono'),\
                           os.path.join(self.model_train_dir, 'seg-masks-mono'))
        self.__flip_masks(os.path.join(self.model_train_dir, 'seg-masks-rgb'),\
                           os.path.join(self.model_train_dir, 'seg-masks-rgb'))

        # populate json file
        self.__populate_json(os.path.join(self.model_dir, 'dataset.json'), self.model_dir)

def generate_sample_model_dataset():
    gt_handler = GTDataHandler(src_dir="data/GT-aws", dst_dir="data")
    gt_handler.generate_GT_train_test(n_train=480, n_test=60)

    model_handler = ModelDataHandler(GT_train="data/GT-train", 
                                     GT_test="data/GT-test", 
                                     model_dir="data/model-dataset")
    model_handler.generate_MODEL_train_test()

    

if __name__ == "__main__":
    generate_sample_model_dataset()

    # class_distribution = GTDataHandler.log_class_distribution("data/GT-train")
    # GTDataHandler.plot_class_distribution(class_distribution)
