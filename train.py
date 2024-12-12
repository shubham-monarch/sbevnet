#! /usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

from sbevnet.models.network_sbevnet import SBEVNet
from sbevnet.data_utils.bev_dataset import sbevnet_dataset
from helpers import get_logger


def train_sbevnet():
    logger = get_logger("train")
    
    # Training parameters
    params = {
        
        # image dimensions
        'image_w': 640,
        'image_h': 480,
        'max_disp': 64,

        # segmentation and heatmap parameters
        'n_classes_seg': 6,
        'n_hmap': 400,
        
        # depth range (in meters)
        'xmin': 0,
        'xmax': 10,
        
        # horizontal range (in meters)
        'ymin': -5,
        'ymax': 5,
        
        # camera parameters
        'cx': 964.989,
        'cy': 569.276,
        'f': 1093.2768,
        'tx': 0.13,
        'camera_ext_x': 0.0,
        'camera_ext_y': 0.0,

        # additional parameters for SBEVNet
        'do_ipm_rgb': False,
        'do_ipm_feats': False,
        'fixed_cam_confs': True,
        
        # training parameters
        'batch_size': 1,
        'num_epochs': 20,
        'learning_rate': 0.001,

        # dataset parameters
        'do_mask': False,
        'do_top_seg': True,
        'zero_mask': False
    }
    
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
        json_path='datasets/dataset.json',
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
    

    for epoch in range(params['num_epochs']):
    # for epoch in range(1):
        network.train()
        epoch_loss = 0

        logger.warning(f'==================')
        logger.warning(f'epoch: {epoch}')
        logger.warning(f'==================\n')
        
        for batch_idx, data in enumerate(train_loader):
            # if batch_idx > 00:
            #     break
            
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

                # logger.warning(f'==================')
                # logger.warning(f'CKPT-1')
                # logger.warning(f'==================\n')
                
                if not isinstance(data, dict):
                    raise TypeError("Expected 'data' to be a dictionary")
                
                output = network(data)

                # Ensure target is on the same device
                target = data['top_seg'].to(device)

                # logger.warning(f"=================")
                # logger.warning(f"[train_sbevnet] --> before loss")
                # logger.warning(f"=================\n")
                
                # logger.info(f"=================")
                # logger.info(f"output['top_seg'].shape: {output['top_seg'].shape}")
                # logger.info(f"target.shape: {target.shape}")
                # logger.info(f"=================\n")

                loss = criterion(output['top_seg'], target)

                # logger.warning(f"=================")
                # logger.warning(f"[train_sbevnet] --> after loss")
                # logger.warning(f"=================\n")


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

        
        # Plot loss vs epoch graph
        if not hasattr(train_sbevnet, 'losses'):
            train_sbevnet.losses = []
        train_sbevnet.losses.append(avg_epoch_loss)

        # Create the plot
        plt.figure(figsize=(10,6))
        plt.plot(range(1, len(train_sbevnet.losses) + 1), train_sbevnet.losses, 'b-', label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss vs Epoch')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
        plt.close()

if __name__ == '__main__':
    train_sbevnet() 