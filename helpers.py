#! /usr/bin/env python3

import os
import shutil
from tqdm import tqdm



def restructure_dataset(src_folder, target_folder):
    '''Restructure the dataset into left, right, and segmented-GT folders.'''
    
    # Create the target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)

    # Define the target subfolders
    left_folder = os.path.join(target_folder, 'left')
    right_folder = os.path.join(target_folder, 'right')
    segmented_gt_folder = os.path.join(target_folder, 'segmented-GT')

    # Create the target subfolders if they don't exist
    os.makedirs(left_folder, exist_ok=True)
    os.makedirs(right_folder, exist_ok=True)
    os.makedirs(segmented_gt_folder, exist_ok=True)


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
                elif file.endswith('_left.disp.png'):
                    # Use the directory name as a prefix to prevent overwriting
                    prefix = os.path.basename(root)
                    new_filename = f"{prefix}_{file}"
                    shutil.copy(os.path.join(root, file), os.path.join(segmented_gt_folder, new_filename))
                pbar.update(1)

if __name__ == "__main__":
    # Example usage
    src_folder = 'train-data'
    target_folder = 'train-data-organized'
    restructure_dataset(src_folder, target_folder)


