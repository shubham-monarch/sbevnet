#! /usr/bin/env python3

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import numpy as np
import cv2
import argparse
import sys

from sbevnet.models.network_sbevnet import SBEVNet
from sbevnet.data_utils.bev_dataset import sbevnet_dataset
from helpers import get_logger

def get_colored_segmentation_image(seg_arr: np.ndarray, config_path: str) -> np.ndarray:
    """Convert seg array to colored image"""
    import yaml

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

def evaluate_sbevnet(config_path: str, color_map_path: str):
    """Evaluate SBEVNet model using provided configuration files.
    
    Args:
        config_path: Path to the evaluation configuration YAML file
        color_map_path: Path to the color map configuration YAML file
    """
    logger = get_logger("evaluate")
    
    import yaml

    with open(config_path, 'r') as file:
        params = yaml.safe_load(file)

    scale_x = float(640 / 1920)
    scale_y = float(480 / 1080)

    params['cx'] *= scale_x
    params['cy'] *= scale_y
    params['f'] *= scale_x
    
    # params['checkpoint_path'] = 'checkpoints/best_model.pth'
    # params['checkpoint_path'] = 'checkpoints/best_model.pth'
    # params['batch_size'] = 1
    # params['do_top_seg'] = False

    # mkdir predictions
    pred_dir = params['predictions_dir']

    # predictions directory must be empty
    assert not (os.path.exists(pred_dir) and os.listdir(pred_dir)), "Predictions directory must be empty"
    os.makedirs(pred_dir, exist_ok=True)
    
    # initialize network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    
    logger.warning(f'==============')
    logger.warning(f'Using device: {device}')
    logger.warning(f'==============\n')
    
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
        cx=params['cx'],
        cy=params['cy'],
        f=params['f'],
        tx=params['tx'],
        camera_ext_x=params['camera_ext_x'],
        camera_ext_y=params['camera_ext_y'],
        do_ipm_rgb=params['do_ipm_rgb'],
        do_ipm_feats=params['do_ipm_feats'],
        fixed_cam_confs=params['fixed_cam_confs']
    ).to(device)
    
    # load checkpoint
    if os.path.exists(params['checkpoint_path']):
        checkpoint = torch.load(params['checkpoint_path'], map_location=device)
        network.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        logger.error(f"No checkpoint found at {params['checkpoint_path']}")
        return
    
    # set network to evaluation mode
    network.eval()
    
    # load test dataset
    test_dataset = sbevnet_dataset(
        json_path='data/model-dataset/dataset.json',
        dataset_split='test',
        do_ipm_rgb=params['do_ipm_rgb'],
        do_ipm_feats=params['do_ipm_feats'],
        fixed_cam_confs=params['fixed_cam_confs'],
        do_mask=params['do_mask'],
        do_top_seg=params['do_top_seg'],
        zero_mask=params['zero_mask'],
        image_w=params['image_w'],
        image_h=params['image_h']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.warning(f'=================')    
    logger.warning(f'Test dataset size: {len(test_dataset)}')
    logger.warning(f'=================\n')
    
    
    # initialize metrics storage
    total_ious = [0] * params['n_classes_seg']
    total_samples = 0
    
    # evaluation loop
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader, desc="Evaluating")):
            # if batch_idx > 0:
            #     break
            
            try:
                # move data to device
                for key in data:
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].to(device)
                    elif isinstance(data[key], list):
                        data[key] = [item.to(device) if isinstance(item, torch.Tensor) else item 
                                   for item in data[key]]

                # forward pass
                output = network(data)
                
                # get predictions
                pred = output['top_seg']  # [B, H, W]                
                pred = output['top_seg'].argmax(1)  # [B, H, W]

                # Save predictions
                for i in range(pred.size(0)):
                    pred_np = pred[i].cpu().numpy()
                        
                    # Save colored visualization
                    colored_pred = get_colored_segmentation_image(pred_np, config_path=color_map_path)
                    colored_pred = cv2.flip(colored_pred, 0)
                    
                    # Load and resize left image
                    left_idx = batch_idx * params['batch_size'] + i + 1
                    left_img_path = os.path.join('data/model-dataset/test/left', f'{left_idx}__left.jpg')
                    
                    logger.info(f'=================')
                    logger.info(f'{left_img_path}')
                    logger.info(f'=================\n')
                    
                    left_img = cv2.imread(left_img_path)
                    if left_img is None:
                        logger.error(f'Failed to read image at {left_img_path}')
                        continue
                    left_img_resized = cv2.resize(left_img, (256, 256), interpolation=cv2.INTER_LINEAR)
                    
                    # Combine and save
                    combined_image = np.hstack((left_img_resized, cv2.flip(colored_pred, 0)))
                    combined_dir = os.path.join(pred_dir, f'combined')
                    os.makedirs(combined_dir, exist_ok=True)
                    combined_path = os.path.join(combined_dir, f'pred_{left_idx:04d}.png')
                    cv2.imwrite(combined_path, combined_image)
            
            except Exception as e:
                logger.error(f'Error in batch {batch_idx}: {str(e)}')
                continue

def main():
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate SBEVNet model')
    parser.add_argument('--config', type=str, default='configs/evaluate.yaml', 
                       help='Path to evaluation config file')
    parser.add_argument('--color_map', type=str, default='configs/Mavis.yaml', 
                       help='Path to color map config file')
    args = parser.parse_args()
    
    # Validate config files exist
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found")
        sys.exit(1)
    if not os.path.exists(args.color_map):
        print(f"Error: Color map file {args.color_map} not found")
        sys.exit(1)
        
    evaluate_sbevnet(args.config, args.color_map)

if __name__ == '__main__':
    main() 