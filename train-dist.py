#! /usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

from sbevnet.models.network_sbevnet import SBEVNet
from sbevnet.data_utils.bev_dataset import sbevnet_dataset
from helpers import get_logger


def setup(rank: int, world_size: int) -> None:
    """Initialize distributed training process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup() -> None:
    """Clean up distributed training process group."""
    dist.destroy_process_group()


def train(rank: int, world_size: int, params: dict) -> None:
    """Training function for each process."""
    try:
        setup(rank, world_size)
        logger = get_logger(f"train_rank_{rank}")
        
        # Set the device for the current process
        torch.cuda.set_device(rank)
        
        # Create save directory
        save_dir = 'checkpoints'
        if rank == 0:
            os.makedirs(save_dir, exist_ok=True)
        
        # Initialize network
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
        ).to(rank)
        
        # Wrap model with DDP
        network = DDP(network, device_ids=[rank])
        
        # Define loss function and optimizer
        criterion = nn.NLLLoss2d(ignore_index=-100).to(rank)
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

        # Use DistributedSampler
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=False,  # Sampler handles shuffling
            num_workers=4,
            pin_memory=True,
            sampler=train_sampler
        )
        
        if rank == 0:
            logger.info(f'Training dataset size: {len(train_dataset)}')
        
        # Training loop
        best_loss = float('inf')
        losses = []

        for epoch in range(params['num_epochs']):
            train_sampler.set_epoch(epoch)  # Important for proper shuffling
            network.train()
            epoch_loss = 0.0
            
            if rank == 0:
                logger.warning(f'Epoch: {epoch}')
            
            for batch_idx, data in enumerate(train_loader):
                try:
                    # Move data to device
                    for key in data:
                        if isinstance(data[key], torch.Tensor):
                            data[key] = data[key].to(rank)
                        elif isinstance(data[key], list):
                            data[key] = [item.to(rank) if isinstance(item, torch.Tensor) else item for item in data[key]]

                    # Forward pass
                    optimizer.zero_grad()
                    
                    if not isinstance(data, dict):
                        raise TypeError("Expected 'data' to be a dictionary")
                    
                    output = network(data)
                    target = data['top_seg'].to(rank)
                    loss = criterion(output['top_seg'], target)

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    if rank == 0 and batch_idx % 10 == 0:
                        logger.info(f'Epoch {epoch+1}, Batch {batch_idx+1} - Loss: {loss.item():.4f}')
                    
                except Exception as e:
                    logger.error(f'Error in batch {batch_idx}: {str(e)}')
                    continue
            
            # Calculate average epoch loss across all processes
            epoch_loss = torch.tensor(epoch_loss / len(train_loader), device=rank)
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            avg_epoch_loss = epoch_loss.item() / world_size
            
            if rank == 0:
                logger.info(f'Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}')
                losses.append(avg_epoch_loss)
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': network.module.state_dict(),
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
                
                # Plot loss curve
                plt.figure(figsize=(10,6))
                plt.plot(range(1, len(losses) + 1), losses, 'b-', label='Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training Loss vs Epoch')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
                plt.close()

        if rank == 0:
            logger.info("Cleaned up distributed training")
    except KeyboardInterrupt:
        logger.info("Caught keyboard interrupt, cleaning up...")
    finally:
        cleanup()
        if rank == 0:
            logger.info("Cleaned up distributed training")


def train_sbevnet_distributed() -> None:
    """Main function to initialize distributed training."""
    scale_x = float(640/1920)
    scale_y = float(480/1080)

    params = {
        'image_w': 640,
        'image_h': 480,
        'max_disp': 64,
        'n_classes_seg': 6,
        'n_hmap': 400,
        'xmin': 0,
        'xmax': 10,
        'ymin': -5,
        'ymax': 5,
        'cx': 964.989 * scale_x,
        'cy': 569.276 * scale_y,
        'f': 1093.2768 * scale_x,
        'tx': 0.13,
        'camera_ext_x': 0.0,
        'camera_ext_y': 0.0,
        'do_ipm_rgb': False,
        'do_ipm_feats': False,
        'fixed_cam_confs': True,
        'batch_size': 2,  # Per GPU batch size
        'num_epochs': 20,
        'learning_rate': 0.001,
        'do_mask': False,
        'do_top_seg': True,
        'zero_mask': False
    }
    
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices available")
    
    mp.spawn(
        train,
        args=(world_size, params),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    train_sbevnet_distributed() 