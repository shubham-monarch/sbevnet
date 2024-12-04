#! /usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm

from sbevnet.models.network_sbevnet import SBEVNet
from sbevnet.data_utils.bev_dataset import sbevnet_dataset
from helpers import get_logger


def train_sbevnet():
    logger = get_logger("train")
    
    # Training parameters
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
    
    # Create save directory
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    network = SBEVNet(
        image_w=params['image_w'],
        image_h=params['image_h'],
        xmin=params['xmin'],
        xmax=params['xmax'],
        ymin=params['ymin'],
        ymax=params['ymax'],
        n_hmap=params['n_hmap'],
        max_disp=params['max_disp'],
        cx=params['cx'],
        cy=params['cy'],
        f=params['f'],
        tx=params['tx'],
        camera_ext_x=params['camera_ext_x'],
        camera_ext_y=params['camera_ext_y'],
        do_ipm_rgb=True,
        do_ipm_feats=True,
        fixed_cam_confs=True
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.NLLLoss2d(ignore_index=-100)
    optimizer = optim.Adam(network.parameters(), lr=params['learning_rate'])
    
    # Load datasets
    train_dataset = sbevnet_dataset(
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


    train_loader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f'==================')
    logger.info(f'type(train_dataset): {type(train_dataset)}')
    logger.info(f'Training dataset size: {len(train_dataset)}')
    logger.info(f'==================\n')
    
    
    # Training loop
    best_loss = float('inf')
    

    # for epoch in range(params['num_epochs']):
    for epoch in range(1):
        network.train()
        epoch_loss = 0
        
        for batch_idx, data in enumerate(train_loader):
            
            if batch_idx > 2:
                break

            # train_logger.warning(f'==================')
            # train_logger.warning(f'batch_idx: {batch_idx}')
            # train_logger.warning(f'==================\n')

            # train_logger.info(f'==================')
            # train_logger.info(f'type(data): {type(data)}')
            # train_logger.info(f'len(data): {len(data)}')
            # train_logger.info(f'data: {data}')
            # # train_logger.info(f'data keys: {data.keys()}')
            # train_logger.info(f'==================\n')

            try:
                # Move data to device
                for key in data:
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].to(device)
                    elif isinstance(data[key], list):
                        data[key] = [item.to(device) if isinstance(item, torch.Tensor) else item 
                                   for item in data[key]]
                
                # Forward pass
                optimizer.zero_grad()

                logger.warning(f'==================')
                logger.warning(f'CKPT-1')
                logger.warning(f'==================\n')
                
                output = network(data)
                
                logger.warning(f'==================')
                logger.warning(f'CKPT-2')
                logger.warning(f'==================\n')

                # Compute loss
                loss = criterion(output['top_seg'], data['top_seg'])
                
                logger.warning(f'==================')
                logger.warning(f'CKPT-3')
                logger.warning(f'==================\n')

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
        logger.info(f'Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}')
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(save_dir, 'latest_checkpoint.pth'))
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            logger.info(f'New best model saved with loss: {best_loss:.4f}')

if __name__ == '__main__':
    train_sbevnet() 