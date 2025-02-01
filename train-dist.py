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
from torch.utils.tensorboard import SummaryWriter
import time
import torch.nn.functional as F
import argparse
import random
import numpy as np

from sbevnet.models.network_sbevnet import SBEVNet
from sbevnet.data_utils.bev_dataset import sbevnet_dataset
from helpers import get_logger, populate_json


def set_seed(seed: int):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Enable deterministic CuDNN algorithms (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup(rank: int, world_size: int) -> None:
    """Initialize distributed training process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For deterministic CUDA ops
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.use_deterministic_algorithms(True, warn_only=True)  # Force deterministic algorithms


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
    
    # compute inverse frequency weights
    total_samples = class_counts.sum()
    class_weights = total_samples / (class_counts * params['n_classes_seg'])
    
    # normalize weights to have median of 1
    class_weights = class_weights / class_weights.median()

    logger.warning(f"=================")
    logger.warning(f"computed class weights: {class_weights}")
    logger.warning(f"=================\n")
    
    return class_weights


class FocalLoss(nn.Module):
    '''implementation of focal loss from "Focal Loss for Dense Object Detection"'''
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None, reduction: str = 'mean') -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # compute cross entropy loss
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        
        # compute pt (probability of true class)
        pt = torch.exp(-ce_loss)
        
        # compute focal loss
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def seed_worker(worker_id):
    """Global worker seeding function"""
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def train(rank: int, world_size: int, params: dict) -> None:
    """Training function for each process."""
    try:
        # Get seed from config with default 420 if not specified
        seed = params.get('random_seed', 420)
        set_seed(seed + rank)
        
        setup(rank, world_size)
        logger = get_logger("train", rank)
        
        is_main_process = rank == 0
        
        torch.cuda.set_device(rank)

        # Create save directory and subdirectories
        save_dir = 'checkpoints'
        if is_main_process:
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(os.path.join(save_dir, 'epochs'), exist_ok=True)
            writer = SummaryWriter(log_dir=os.path.join(save_dir, 'runs'))
        
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
        
        # class_weights = torch.tensor([0.1, 0.1, 0.1, 1.0, 10.0, 10.0]).to(rank)
        # class_weights = torch.tensor([0.1, 10.0, 0.1, 0.1, 10.0, 10.0]).to(rank)
        
        train_dataset = sbevnet_dataset(
            json_path=params['dataset_path'],
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
        
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            seed=seed  # Add seed for deterministic shuffling
        )
        
        # Modified DataLoader configurations
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            sampler=train_sampler,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(seed + rank),
            persistent_workers=True
        )
        
        # class_weights = compute_class_weights(train_loader, params).to(rank)
        # class_weights = torch.tensor([0.1, 10.0, 1.0, 1.0, 20.0, 20.0]).to(rank)
        # class_weights = torch.tensor([0.1, 10.0, 0.1, 0.5, 5.0, 5.0]).to(rank)
        class_weights = torch.tensor(params['class_weights'], dtype=torch.float32).to(rank)
        
        logger.warning(f"=================")
        logger.warning(f"computed class weights: {class_weights}")
        logger.warning(f"=================\n")

        # Criterion selection based on config (loss_type: "focal" or "cross_entropy")
        loss_type = params.get('loss_type', 'cross_entropy')
        if loss_type.lower() == 'focal':
            gamma = params.get('focal_gamma', 2.0)
            criterion = FocalLoss(gamma=gamma, weight=class_weights).to(rank)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100).to(rank)
        
        # Fixed learning rate of 0.0001
        optimizer = optim.Adam(network.parameters(), lr=0.0001)
        # optimizer = optim.Adam(network.parameters(), lr=0.00001)

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
        
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        val_loader = DataLoader(
            val_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            sampler=val_sampler,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(seed + rank),
            persistent_workers=True
        )
        
        if is_main_process:
            logger.info(f'Training dataset size: {len(train_dataset)}')
            logger.info(f'Validation dataset size: {len(val_dataset)}')
        
        losses = []
        val_losses = []
        learning_rates = []
        best_val_loss = float('inf')
        best_train_loss = float('inf')
        patience = 20
        epochs_no_improve = 0
        
        for epoch in range(params['num_epochs']):
            train_sampler.set_epoch(epoch + seed)  # Seed based on initial seed + epoch
            network.train()
            epoch_loss = 0.0
            
            if is_main_process:
                pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch+1} Training')
            
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
            
            # Validation loop
            network.eval()
            epoch_val_loss = 0.0
            
            if is_main_process:
                pbar_val = tqdm(total=len(val_loader), desc=f'Epoch {epoch+1} Validation')
            
            with torch.no_grad():
                for batch_idx, data in enumerate(val_loader):
                    try:
                        # Move data to device
                        for key in data:
                            if isinstance(data[key], torch.Tensor):
                                data[key] = data[key].to(rank)
                            elif isinstance(data[key], list):
                                data[key] = [item.to(rank) if isinstance(item, torch.Tensor) else item for item in data[key]]

                        # Forward pass
                        if not isinstance(data, dict):
                            raise TypeError("Expected 'data' to be a dictionary")
                        
                        output = network(data)
                        target = data['top_seg'].to(rank)
                        loss = criterion(output['top_seg'], target)
                        
                        epoch_val_loss += loss.item()
                        
                        if is_main_process:
                            pbar_val.set_postfix({'loss': f'{loss.item():.4f}'})
                            pbar_val.update()
                    
                    except Exception as e:
                        logger.error(f'Error in validation batch {batch_idx}: {str(e)}')
                        continue
            
            if is_main_process:
                pbar_val.close()
            
            # Synchronize processes before computing metrics
            dist.barrier()
            
            # Calculate average epoch loss across all processes
            epoch_val_loss = torch.tensor(epoch_val_loss / len(val_loader), device=rank)
            dist.all_reduce(epoch_val_loss, op=dist.ReduceOp.SUM)
            avg_epoch_val_loss = epoch_val_loss.item() / world_size
            
            if is_main_process:
                logger.info(f'Epoch {epoch+1} - Average Training Loss: {avg_epoch_loss:.4f}, Average Validation Loss: {avg_epoch_val_loss:.4f}')
                losses.append(avg_epoch_loss)
                val_losses.append(avg_epoch_val_loss)
                
                # Plot loss
                plt.figure(figsize=(10,6))
                plt.plot(range(1, len(losses) + 1), losses, 'b-', label='Training Loss')
                plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training and Validation Loss vs Epoch')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'training_plot.png'))
                plt.close()

                # Save checkpoint for current epoch
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': network.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_epoch_loss,
                    'val_loss': avg_epoch_val_loss,
                    'losses': losses,
                    'val_losses': val_losses,
                }
                
                # Save checkpoint for current epoch
                # torch.save(checkpoint, os.path.join(save_dir, 'epochs', f'checkpoint_epoch_{epoch+1}.pth'))
                torch.save(checkpoint, os.path.join(save_dir, 'latest_checkpoint.pth'))
                
                # Save best validation model
                if avg_epoch_val_loss < best_val_loss:
                    best_val_loss = avg_epoch_val_loss
                    torch.save(checkpoint, os.path.join(save_dir, 'best_val_model.pth'))
                    logger.info(f'New best validation model saved with loss: {best_val_loss:.4f}')
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                # Save best training model
                if avg_epoch_loss < best_train_loss:
                    best_train_loss = avg_epoch_loss
                    torch.save(checkpoint, os.path.join(save_dir, 'best_train_model.pth'))
                    logger.info(f'New best training model saved with loss: {best_train_loss:.4f}')

        if is_main_process:
            writer.close()

    except KeyboardInterrupt:
        logger.info("Caught keyboard interrupt, cleaning up...")
    finally:
        cleanup()
        if is_main_process:
            logger.info("Cleaned up distributed training")


def train_sbevnet_distributed(config_path: str) -> None:
    """Main function to initialize distributed training.
    
    Args:
        config_path: Path to the YAML config file
    """
    with open(config_path, 'r') as file:
        params = yaml.safe_load(file)

    scale_x = float(640 / 1920)
    scale_y = float(480 / 1080)

    params['cx'] *= scale_x
    params['cy'] *= scale_y
    params['f'] *= scale_x
    
    for key, value in params.items():
        print(f"{key}: {value}")
    
    time.sleep(2)

    world_size = params.get('num_gpus', torch.cuda.device_count())
    world_size = min(world_size, torch.cuda.device_count())  # Don't exceed available GPUs
    if world_size < 1:
        raise RuntimeError("No CUDA devices available")
    
    mp.spawn(
        train,
        args=(world_size, params),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()
    
    train_sbevnet_distributed(args.config) 