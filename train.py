#! /usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from typing import Dict, Any
import yaml

from sbevnet.models.network_sbevnet import SBEVNet
from sbevnet.data_utils.bev_dataset import sbevnet_dataset
from helpers import get_logger


def train_sbevnet():
    logger = get_logger("train")
    
    scale_x = float(640/1920)
    scale_y = float(480/1080)

    
    with open('configs/train.yaml', 'r') as file:
        params = yaml.safe_load(file)

    scale_x = float(640 / 1920)
    scale_y = float(480 / 1080)

    params['cx'] *= scale_x
    params['cy'] *= scale_y
    params['f'] *= scale_x
    
    # Create save directory
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    network = SBEVNet(
        
        # image dimensions
        image_w=params['image_w'],
        image_h=params['image_h'],
        max_disp=params['max_disp'],

        # segmentation and heatmap parameters
        n_classes_seg=params['n_classes_seg'],
        n_hmap=params['n_hmap'],
        xmin=params['xmin'],
        xmax=params['xmax'],
        ymin=params['ymin'],
        ymax=params['ymax'],

        # camera parameters
        cx=params['cx'],
        cy=params['cy'],
        f=params['f'],
        tx=params['tx'],
        camera_ext_x=params['camera_ext_x'],
        camera_ext_y=params['camera_ext_y'],

        # additional parameters for SBEVNet
        do_ipm_rgb=params['do_ipm_rgb'],
        do_ipm_feats=params['do_ipm_feats'],
        fixed_cam_confs=params['fixed_cam_confs']
    
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.NLLLoss2d(ignore_index=-100)
    optimizer = optim.Adam(network.parameters(), lr=params['learning_rate'])
    
    # Load datasets
    train_dataset = sbevnet_dataset(
        json_path='data/model-dataset/dataset.json',
        dataset_split='train',
        do_ipm_rgb=params['do_ipm_rgb'],
        do_ipm_feats=params['do_ipm_feats'],
        fixed_cam_confs=params['fixed_cam_confs'],
        do_mask=params['do_mask'],
        do_top_seg=params['do_top_seg'],
        zero_mask=params['zero_mask'],
        image_w=params['image_w'],
        image_h=params['image_h']
    )

    val_dataset = sbevnet_dataset(
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


    train_loader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f'==================')
    logger.info(f'type(train_dataset): {type(train_dataset)}')
    logger.info(f'type(val_dataset): {type(val_dataset)}')
    logger.info(f'Training dataset size: {len(train_dataset)}')
    logger.info(f'Validation dataset size: {len(val_dataset)}')
    logger.info(f'==================\n')
    
    
    # Training loop
    best_val_loss = float('inf')
    losses = []
    val_losses = []
    

    # for epoch in range(params['num_epochs']):
    for epoch in range(1):
        network.train()
        epoch_loss = 0
        epoch_val_loss = 0

        logger.warning(f'==================')
        logger.warning(f'epoch: {epoch}')
        logger.warning(f'==================\n')
        
        for batch_idx, data in enumerate(train_loader):
            if batch_idx > 0:
                break
            
            logger.info(f'==================')
            logger.info(f'batch_idx: {batch_idx}')
            logger.info(f'==================\n')
            
            try:
                # Move data to device
                for key in data:
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].to(device)
                    elif isinstance(data[key], list):
                        data[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in data[key]]

                # Forward pass
                optimizer.zero_grad()

                
                if not isinstance(data, dict):
                    raise TypeError("Expected 'data' to be a dictionary")
                
                output = network(data)

                # Ensure target is on the same device
                target = data['top_seg'].to(device)

                loss = criterion(output['top_seg'], target)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Log loss for each batch
                logger.info(f'Epoch {epoch+1}, Batch {batch_idx+1} - Loss: {loss.item():.4f}')
                
            except Exception as e:
                logger.error(f'Error in batch {batch_idx}: {str(e)}')
                continue
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f'Epoch {epoch+1} - Average Training Loss: {avg_epoch_loss:.4f}')
        
        # Validation loop
        network.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                try:
                    # Move data to device
                    for key in data:
                        if isinstance(data[key], torch.Tensor):
                            data[key] = data[key].to(device)
                        elif isinstance(data[key], list):
                            data[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in data[key]]

                    # Forward pass
                    if not isinstance(data, dict):
                        raise TypeError("Expected 'data' to be a dictionary")
                    
                    output = network(data)
                    target = data['top_seg'].to(device)
                    loss = criterion(output['top_seg'], target)
                    
                    epoch_val_loss += loss.item()
                
                except Exception as e:
                    logger.error(f'Error in validation batch {batch_idx}: {str(e)}')
                    continue
        
        avg_epoch_val_loss = epoch_val_loss / len(val_loader)
        logger.info(f'Epoch {epoch+1} - Average Validation Loss: {avg_epoch_val_loss:.4f}\n')
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
            'val_loss': avg_epoch_val_loss,
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(save_dir, 'latest_checkpoint.pth'))
        


        # Save best model
        if avg_epoch_val_loss < best_val_loss:
            best_val_loss = avg_epoch_val_loss
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            logger.info(f'New best model saved with validation loss: {best_val_loss:.4f}')

        
        # Plot loss vs epoch graph  
        losses.append(avg_epoch_loss)
        val_losses.append(avg_epoch_val_loss)

        # Create the plot
        plt.figure(figsize=(10,6))
        plt.plot(range(1, len(losses) + 1), losses, 'b-', label='Training Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss vs Epoch')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
        plt.close()

if __name__ == '__main__':
    train_sbevnet() 