#! /usr/bin/env python3

import os
import shutil
from tqdm import tqdm
import glob
import json


def get_files_from_folder(folder, extensions):
    '''Get all files with the given extensions from the folder.'''
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, f'*{ext}')))
    return sorted(files)


def populate_json(json_path, dataset_path, split="train"):
    '''Populate the json file with the file paths of the images in the dataset.'''
    IMG_EXTENSIONS = ['.jpg', '.png']
    
    def get_relative_files(folder, extensions):
        '''Get all files with the given extensions from the folder and make their paths relative to dataset_path.'''
        files = get_files_from_folder(folder, extensions)
        return [os.path.relpath(file, dataset_path) for file in files]
    
    data = {
        "train": {
            "rgb_left": get_relative_files(os.path.join(dataset_path, 'train/left'), IMG_EXTENSIONS),
            "rgb_right": get_relative_files(os.path.join(dataset_path, 'train/right'), IMG_EXTENSIONS),
            "top_seg": get_relative_files(os.path.join(dataset_path, 'train/bev-segmented'), ['.png'])
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
    bev_segmented_folder = os.path.join(target_folder, 'bev-segmented')

    # Create the target subfolders if they don't exist
    os.makedirs(left_folder, exist_ok=True)
    os.makedirs(right_folder, exist_ok=True)
    os.makedirs(bev_segmented_folder, exist_ok=True)


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
                    shutil.copy(os.path.join(root, file), os.path.join(bev_segmented_folder, new_filename))
                pbar.update(1)

if __name__ == "__main__":
    
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

