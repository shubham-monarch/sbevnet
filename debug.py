#! /usr/bin/env python3

from sbevnet.data_utils.bev_dataset import sbevnet_dataset
from helpers import get_logger
# from pytorch_propane.data_utils import ComposeDatasetDict

import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
    
class ComposeDatasetDict(data.Dataset):

    "TO take a dictionary of datasets and return a dataset which produces elements as a dictionalry "
    
    def __init__(self , data_loaders ,ret_double=False ):
        self.data_loaders = data_loaders

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
    params = {
        'image_w': 640,
        'image_h': 480,
        'max_disp': 64,
        'n_hmap': 100,
        'xmin': 1,
        'xmax': 39,
        'ymin': -19,
        'ymax': 19,
        'cx': 256,
        'cy': 144,
        'f': 179.2531,
        'tx': 0.2,
        'camera_ext_x': 0.9,
        'camera_ext_y': -0.1,
        'batch_size': 1,
        'num_epochs': 20,
        'learning_rate': 0.001
    }
    
    # type(composed_dataset_) = <class 'ComposeDatasetDict'>
    composed_dataset_ = sbevnet_dataset(
        json_path='datasets/dataset.json',
        dataset_split='train',
        do_ipm_rgb=False,
        do_ipm_feats=False,
        fixed_cam_confs=True,
        do_mask=False,
        do_top_seg=True,
        zero_mask=False,
        image_w=params['image_w'],
        image_h=params['image_h']
    )



    logger.info(f"=================")
    logger.info(f"type(composed_dataset_): {type(composed_dataset_)}")
    logger.info(f"=================\n")
    
    data_loader = DataLoader(
        composed_dataset_,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    for batch in data_loader:
        logger.info(f"type(batch): {type(batch)}")
        # logger.info(f"batch: {batch}")
        break