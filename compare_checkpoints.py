#! /usr/bin/env python3
import os
import sys
import torch
from torch.utils.data import DataLoader
import yaml
import json
import cv2
import numpy as np
import argparse
from tqdm import tqdm

from sbevnet.models.network_sbevnet import SBEVNet
from sbevnet.data_utils.bev_dataset import sbevnet_dataset
from helpers import get_logger
from evaluate import get_colored_segmentation_image

def main():
    parser = argparse.ArgumentParser(description="Compare checkpoints predictions")
    parser.add_argument('--config', type=str, default='configs/compare_checkpoints.yaml', help="Path to evaluation config file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        params = yaml.safe_load(f)
    logger = get_logger("compare_checkpoints")
    color_map_path = params.get('color_map', 'configs/Mavis.yaml')

    scale_x = float(640 / 1920)
    scale_y = float(480 / 1080)
    params['cx'] *= scale_x
    params['cy'] *= scale_y
    params['f'] *= scale_x

    output_dir = "compare_checkpoints"
    combined_dir = os.path.join(output_dir, "combined")
    os.makedirs(combined_dir, exist_ok=True)

    gpu_id = params.get('gpu_id', 0)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(gpu_id)
    logger.info(f"Using device: {device}")

    networks = []
    ckpt_names = []
    checkpoint_dirs = params.get("checkpoint_dirs", [])
    if not checkpoint_dirs:
        logger.error("No checkpoint_dirs provided in the config file. Exiting.")
        sys.exit(1)
    for ckpt_dir in checkpoint_dirs:
        ckpt_path = os.path.join(ckpt_dir, "best_val_model.pth")
        if not os.path.exists(ckpt_path):
            logger.error(f"Checkpoint not found at {ckpt_path}")
            continue
        net = SBEVNet(
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
        ckpt = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(ckpt['model_state_dict'])
        net.eval()
        networks.append(net)
        ckpt_names.append(os.path.basename(os.path.normpath(ckpt_dir)))
        logger.info(f"Loaded checkpoint from {ckpt_dir}")
    if len(networks) == 0:
        logger.error("No valid checkpoints loaded. Exiting.")
        sys.exit(1)

    test_dataset = sbevnet_dataset(
        json_path=params['json_path'],
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

    with open(params['json_path'], 'r') as f:
        dataset_json = json.load(f)
    left_img_list = dataset_json['test']['rgb_left']
    if params.get("enable_IPM", False):
        ipm_left_list = dataset_json['test']['ipm_rgb']
    else:
        ipm_left_list = [None] * len(left_img_list)

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader, desc="Comparing checkpoints")):
            for key in data:
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(device)
                elif isinstance(data[key], list):
                    data[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in data[key]]
            
            preds_per_ckpt = []
            for net in networks:
                output = net(data)
                preds = output['top_seg'].argmax(1)
                preds_per_ckpt.append(preds)
            
            batch_size = next(iter(data.values())).shape[0]
            for i in range(batch_size):
                img_idx = batch_idx * params['batch_size'] + i
                left_img_path = os.path.join(params['s3_data_handler']['base_dir'], "model-dataset", left_img_list[img_idx])
                left_img = cv2.imread(left_img_path)
                if left_img is None:
                    logger.error(f"Failed to read left image at {left_img_path}")
                    continue
                left_img_resized = cv2.resize(left_img, (256,256), interpolation=cv2.INTER_LINEAR)

                if ipm_left_list[img_idx] is not None:
                    ipm_left_path = os.path.join(params['s3_data_handler']['base_dir'], "model-dataset", ipm_left_list[img_idx])
                    ipm_img = cv2.imread(ipm_left_path)
                    if ipm_img is None:
                        logger.error(f"Failed to read ipm image at {ipm_left_path}")
                        ipm_img_resized = np.zeros((256,256,3), dtype=np.uint8)
                    else:
                        ipm_img_resized = cv2.resize(ipm_img, (256,256), interpolation=cv2.INTER_LINEAR)
                        ipm_img_resized = cv2.flip(ipm_img_resized, 0)
                else:
                    ipm_img_resized = np.zeros((256,256,3), dtype=np.uint8)
                
                pred_columns = []
                for pred in preds_per_ckpt:
                    pred_np = pred[i].cpu().numpy()
                    colored_pred = get_colored_segmentation_image(pred_np, config_path=color_map_path)
                    colored_pred = cv2.flip(colored_pred, 0)
                    pred_columns.append(colored_pred)
                
                columns = [left_img_resized, ipm_img_resized] + pred_columns
                combined_image = np.hstack(columns)
                out_path = os.path.join(combined_dir, left_img_list[img_idx])
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                cv2.imwrite(out_path, combined_image)

if __name__ == '__main__':
    main() 