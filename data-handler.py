#! /usr/bin/env python3

import os
import random
import shutil
from tqdm import tqdm

from helpers import get_logger

class GTHandler:
    def __init__(self, gt_path: str):
        self.path = gt_path
        self.logger = get_logger("GTHandler")

    def __copy_files(self, files: list[str], dest_dir: str, desc: str):
        '''Copy files to destination'''
        
        for f in tqdm(files, desc=desc):
            src = os.path.join(self.path, f)
            dst = os.path.join(dest_dir, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            self.logger.info(f'Copied {src} to {dst}')

    def split_into_train_val_test(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15):
        '''gt-dataset -->  gt-train / gt-val / gt-test'''
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) <= 1e-9, 'Train, validation and test ratios must sum to 1'

        os.makedirs("gt-train", exist_ok=True)
        os.makedirs("gt-val", exist_ok=True) 
        os.makedirs("gt-test", exist_ok=True)

        all_files = []
        for root, dirs, files in os.walk(self.path):
            for f in files:
                rel_path = os.path.relpath(os.path.join(root, f), self.path)
                all_files.append(rel_path)

        random.shuffle(all_files)

        n_files = len(all_files)
        n_train = int(train_ratio * n_files)
        n_val = int(val_ratio * n_files)
        
        train_files, val_files, test_files = all_files[:n_train], all_files[n_train:n_train+n_val], all_files[n_train+n_val:]

        self.__copy_files(train_files, 'gt-train', 'Copying training files')
        self.__copy_files(val_files, 'gt-val', 'Copying validation files')
        self.__copy_files(test_files, 'gt-test', 'Copying test files')

        assert len(train_files) == n_train, f'Expected {n_train} training files, but got {len(train_files)}'
        assert len(val_files) == n_val, f'Expected {n_val} validation files, but got {len(val_files)}'
        assert len(test_files) == n_files - n_train - n_val, f'Expected {n_files - n_train - n_val} test files, but got {len(test_files)}'

    

class DataHandler:
    '''
    Sample [N1 / N2 / N3] samples from [gt-train / gt-val / gt-test] 
    for the model on each [train / val / test] run
    '''
    
    def __init__(self):
        pass

    def load_data(self):
        pass

    def preprocess_data(self):
        pass


if __name__ == "__main__":
    gt_handler = GTHandler(gt_path="data/gt-dataset")
    gt_handler.split_into_train_val_test()
