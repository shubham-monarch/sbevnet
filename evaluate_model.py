#! /usr/bin/env python3

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import numpy as np
import cv2
import yaml

from sbevnet.models.network_sbevnet import SBEVNet
from sbevnet.data_utils.bev_dataset import sbevnet_dataset
from helpers import get_logger

def get_colored_segmentation_image(seg_arr: np.ndarray, config_path: str) -> np.ndarray:
    """Convert seg array to colored image"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    color_map = config['color_map']
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]
    
    seg_img = np.zeros((output_height, output_width, 3))
    
    for c, color in color_map.items():
        seg_arr_c = seg_arr == int(c)
        seg_img[:, :, 0] += ((seg_arr_c) * color[0]).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c) * color[1]).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c) * color[2]).astype('uint8')
    
    return seg_img

def calculate_iou(pred, target, n_classes):
    """Calculate IoU for each class"""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        iou = intersection / (union + 1e-10)
        ious.append(iou)
    
    return ious

def evaluate_sbevnet():
    """evaluate sbevnet model and save visualization results"""
    from typing import Dict, Any, Optional
    
    logger = get_logger("evaluate")
    
    # load config
    try:
        with open('configs/evaluate_model.yaml', 'r') as file:
            params = yaml.safe_load(file)
    except Exception as e:
        logger.error(f'failed to load config: {str(e)}')
        return

    # setup output directory
    pred_dir = 'predictions'
    assert not os.path.exists(pred_dir) or not os.listdir(pred_dir), 'predictions directory must be empty or not exist'
    os.makedirs(pred_dir)

    # initialize model
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'using device: {device}')
        
        network = SBEVNet(
            image_w=params['image_w'],
            image_h=params['image_h'],
            max_disp=params['max_disp'],
            n_classes_seg=params['n_classes_seg'],
            n_hmap=params['n_hmap'],
            xmin=params['xmin'],
            xmax=params['xmax'],
            ymin=params['ymin'],
            ymax=params['ymax'],
            cx=params['cx'] * float(640/1920),  # scale for input resolution
            cy=params['cy'] * float(480/1080),
            f=params['f'] * float(640/1920),
            tx=params['tx'],
            camera_ext_x=params['camera_ext_x'],
            camera_ext_y=params['camera_ext_y'],
            do_ipm_rgb=params['do_ipm_rgb'],
            do_ipm_feats=params['do_ipm_feats'],
            fixed_cam_confs=params['fixed_cam_confs']
        ).to(device)
    except Exception as e:
        logger.error(f'failed to initialize network: {str(e)}')
        return

    # load checkpoint
    try:
        checkpoint_path = 'checkpoints/best_model.pth'
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'checkpoint not found at {checkpoint_path}')
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        network.load_state_dict(checkpoint['model_state_dict'])
        network.eval()
        logger.info(f"loaded checkpoint from epoch {checkpoint['epoch']}")
    except Exception as e:
        logger.error(f'failed to load checkpoint: {str(e)}')
        return

    # setup dataset
    try:
        test_dataset = sbevnet_dataset(
            json_path='data/model-dataset/dataset.json',
            dataset_split='test',
            do_ipm_rgb=params['do_ipm_rgb'],
            do_ipm_feats=params['do_ipm_feats'],
            fixed_cam_confs=params['fixed_cam_confs'],
            do_mask=params['do_mask'],
            do_top_seg=False,  # override param
            zero_mask=params['zero_mask'],
            image_w=params['image_w'],
            image_h=params['image_h']
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # force batch size 1 for visualization
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        logger.info(f'test dataset size: {len(test_dataset)}')
    except Exception as e:
        logger.error(f'failed to setup dataset: {str(e)}')
        return

    # evaluation loop
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader, desc="evaluating")):
            # prepare input
            img_paths = data.pop('img_path')  # remove img_path before moving to device
            for key in data:
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(device)
                elif isinstance(data[key], list):
                    data[key] = [item.to(device) if isinstance(item, torch.Tensor) else item 
                               for item in data[key]]

            # get model prediction
            output = network(data)
            pred = output['top_seg'].argmax(1)  # [b, h, w]
            
            # process each sample in batch
            for i in range(pred.size(0)):
                # get prediction visualization
                pred_np: np.ndarray = pred[i].cpu().numpy()
                colored_pred: np.ndarray = get_colored_segmentation_image(
                    pred_np, 
                    config_path='configs/Mavis.yaml'
                )
                colored_pred = cv2.flip(colored_pred, 0)  # flip to match camera view

                # load and resize input image using path from dataset
                img_path: str = img_paths[i]
                input_img: np.ndarray = cv2.imread(img_path)
                if input_img is None:
                    raise ValueError(f'failed to read image at {img_path}')
                    
                input_img = cv2.resize(input_img, (256, 256), 
                                     interpolation=cv2.INTER_LINEAR)

                # create and save visualization
                # ensure same size
                if colored_pred.shape[:2] != (256, 256):
                    colored_pred = cv2.resize(colored_pred, (256, 256),
                                            interpolation=cv2.INTER_LINEAR)
                    
                vis_img: np.ndarray = np.hstack((input_img, colored_pred))
                
                save_path: str = os.path.join(pred_dir, f'pred_{batch_idx * pred.size(0) + i + 1:04d}.png')
                cv2.imwrite(save_path, vis_img)

    logger.info('evaluation complete')

if __name__ == '__main__':
    evaluate_sbevnet() 