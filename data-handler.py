#! /usr/bin/env python3

import os
import random
import shutil
from tqdm import tqdm
from typing import List

from helpers import get_logger

class DataHandlerGT:
    def __init__(self, src_dir: str, dst_dir: str):
        self.logger = get_logger("GTHandler")
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        
    def __copy_files(self, files: List[str], dest_dir: str, desc: str):
        '''Copy files to destination'''
        
        for f in tqdm(files, desc=desc):
            src = os.path.join(self.src_dir, f)
            dst = os.path.join(dest_dir, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            # self.logger.info(f'Copied {src} to {dst}')

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
    
        all_files = []
        for root, dirs, files in os.walk(self.src_dir):
            for f in files:
                rel_path = os.path.relpath(os.path.join(root, f), self.src_dir)
                all_files.append(rel_path)

        random.shuffle(all_files)

        n_files = len(all_files)
        n_train = int(train_ratio * n_files)
        n_val = int(val_ratio * n_files)
        
        train_files = all_files[:n_train]
        val_files = all_files[n_train:n_train+n_val]
        test_files = all_files[n_train+n_val:]

        self.__copy_files(train_files, dir_train, 'Copying training files')
        self.__copy_files(val_files, dir_val, 'Copying validation files')
        self.__copy_files(test_files, dir_test, 'Copying test files')

        assert len(train_files) == n_train, f'Expected {n_train} training files, but got {len(train_files)}'
        assert len(val_files) == n_val, f'Expected {n_val} validation files, but got {len(val_files)}'
        assert len(test_files) == n_files - n_train - n_val, f'Expected {n_files - n_train - n_val} test files, but got {len(test_files)}'


class DataHandlerModel:
    
    def __init__(self, model_path: str):
        self.path = model_path
        self.logger = get_logger("ModelHandler")

    def load_data(self):
        pass

    def preprocess_data(self):
        pass


if __name__ == "__main__":
    gt_handler = DataHandlerGT(src_dir="data/gt-dataset", dst_dir="data")
    gt_handler.split_into_train_val_test()
