#! /usr/bin/env python3



import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2

from evaluate_model import get_colored_segmentation_image
from helpers import get_logger

class ComposeDatasetDict(data.Dataset):

    "TO take a dictionary of datasets and return a dataset which produces elements as a dictionalry "
    
    def __init__(self , data_loaders ,ret_double=False ):
        self.data_loaders = data_loaders
        self.logger = get_logger('ComposeDatasetDict')
        
        self.logger.warning(f"=================")
        for k in self.data_loaders:
            self.logger.warning(f"len(self.data_loaders[k]): {len(self.data_loaders[k])}")
        self.logger.warning(f"=================\n") 
        
        # make sure all the datasets are of the same size!! 
        for k in self.data_loaders:
            l = len( self.data_loaders[k])
            break 
        for k in self.data_loaders:
            assert l == len( self.data_loaders[k] ) , "The sizes of the datasets do not match! "+k  
            # print( l , k , )
        
        self.ret_double = ret_double 
        
    def __getitem__(self, index):
        ret = {}
        for k in self.data_loaders:
            ret[k] = self.data_loaders[k].__getitem__(index)

        if self.ret_double:
            return ret , ret 
        else:
            return ret 

    def __len__(self):
        for k in self.data_loaders:
            return len(self.data_loaders[k]) 

 

if __name__ == '__main__':
    
    logger = get_logger('debug')
    
    seg_mask_mono_path = 'datasets/test-640x480/seg-masks-mono/4__seg-mask-mono.png'
    seg_mask_mono = cv2.imread(seg_mask_mono_path, cv2.IMREAD_GRAYSCALE)

    logger.info(f"=================")
    logger.info(f"seg_mask_mono.shape: {seg_mask_mono.shape}")
    logger.info(f"=================\n")

    seg_mask_rgb = get_colored_segmentation_image(seg_mask_mono, 'Mavis.yaml')
    
    cv2.imwrite('seg_mask_rgb.png', seg_mask_rgb)
