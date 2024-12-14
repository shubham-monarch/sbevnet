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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml

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


def compute_class_weights(dataset: DataLoader, params: dict) -> torch.Tensor:
    """compute inverse frequency class weights"""
    logger = logging.getLogger("compute_class_weights")
    
    # count frequencies of each class
    class_counts = torch.zeros(params['n_classes_seg'])
    for data in dataset:
        labels = data['top_seg']
        for i in range(params['n_classes_seg']):
            class_counts[i] += (labels == i).sum()
    
    # print counts for each class and total sum
    # total_samples = class_counts.sum()
    
    # logger.warning(f"=================")
    # logger.warning(f"Class counts: {class_counts}, Total sum: {total_samples.item()}")
    # logger.warning(f"=================\n")
    
    # # compute inverse frequency weights
    # class_weights = total_samples / (class_counts * params['n_classes_seg'])
    
    # # normalize weights to have median of 1
    # class_weights = class_weights / class_weights.median()
    
    class_weights = torch.tensor([10.0, 0.1, 0.1, 0.1, 10.0, 10.0])

    logger.warning(f"=================")
    logger.warning(f"computed class weights: {class_weights}")
    logger.warning(f"=================\n")
    
    return class_weights


def train(rank: int, world_size: int, params: dict) -> None:
    """Training function for each process."""
    try:
        setup(rank, world_size)
        logger = get_logger("train", rank)
        
        # Only log on rank 0 for most messages
        is_main_process = rank == 0
        
        torch.cuda.set_device(rank)
        
        # Create save directory
        save_dir = 'checkpoints'
        if is_main_process:
            os.makedirs(save_dir, exist_ok=True)
        
        # Initialize network and move to device
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
        
        # Wait for all processes to sync up
        dist.barrier()
        
        # Wrap model with DDP
        network = DDP(network, device_ids=[rank])
        
        class_weights = torch.tensor([0.1, 0.1, 0.1, 1.0, 10.0, 10.0]).to(rank)
        # normalize
        class_weights = class_weights / class_weights.sum()
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100).to(rank)
        
        # Fixed learning rate of 0.0001
        optimizer = optim.Adam(network.parameters(), lr=0.0001)

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

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            sampler=train_sampler
        )
        
        if is_main_process:
            logger.info(f'Training dataset size: {len(train_dataset)}')
        
        losses = []
        learning_rates = []
        best_loss = float('inf')

        for epoch in range(params['num_epochs']):
            train_sampler.set_epoch(epoch)
            network.train()
            epoch_loss = 0.0
            
            if is_main_process:
                pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}')
            
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
                    
                    # Update progress bar only on main process
                    if is_main_process:
                        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                        pbar.update()
                    
                except Exception as e:
                    logger.error(f'Error in batch {batch_idx}: {str(e)}')
                    continue
            
            if is_main_process:
                pbar.close()
            
            # Synchronize processes before computing metrics
            dist.barrier()
            
            # Calculate average epoch loss across all processes
            epoch_loss = torch.tensor(epoch_loss / len(train_loader), device=rank)
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            avg_epoch_loss = epoch_loss.item() / world_size
            
            if is_main_process:
                logger.info(f'Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}')
                losses.append(avg_epoch_loss)
                
                # Plot loss
                plt.figure(figsize=(10,6))
                plt.plot(range(1, len(losses) + 1), losses, 'b-', label='Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training Loss vs Epoch')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'training_plot.png'))
                plt.close()

                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': network.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_epoch_loss,
                    'losses': losses,
                }
                
                torch.save(checkpoint, os.path.join(save_dir, 'latest_checkpoint.pth'))
                
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
                    logger.info(f'New best model saved with loss: {best_loss:.4f}')

    except KeyboardInterrupt:
        logger.info("Caught keyboard interrupt, cleaning up...")
    finally:
        cleanup()
        if is_main_process:
            logger.info("Cleaned up distributed training")


def train_sbevnet_distributed() -> None:
    """Main function to initialize distributed training."""
   

    with open('configs/train.yaml', 'r') as file:
        params = yaml.safe_load(file)

    scale_x = float(640 / 1920)
    scale_y = float(480 / 1080)

    params['cx'] *= scale_x
    params['cy'] *= scale_y
    params['f'] *= scale_x
    
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