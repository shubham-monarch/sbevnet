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
import sys
from torch.cuda.amp import GradScaler, autocast

from sbevnet.models.network_sbevnet import SBEVNet
from sbevnet.data_utils.bev_dataset import sbevnet_dataset
from helpers import get_logger, populate_json


def set_seed(seed: int, deterministic: bool = True):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        # Enable deterministic CuDNN algorithms (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup(rank: int, world_size: int, deterministic: bool = True) -> None:
    """Initialize distributed training process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For deterministic CUDA ops
        torch.use_deterministic_algorithms(True, warn_only=True)  # Force deterministic algorithms
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup() -> None:
    """Clean up distributed training process group."""
    dist.destroy_process_group()



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


def move_data_to_device(data, device):
    """
    Recursively moves tensors in nested data structures (dict, list, tuple) to the specified device.
    """
    if isinstance(data, dict):
        return {key: move_data_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_data_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_data_to_device(item, device) for item in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    return data


def train_one_epoch(network, train_loader, optimizer, criterion, scaler, device, epoch, logger, is_main_process):
    """
    Executes one training epoch.
    """
    epoch_loss = 0.0
    if is_main_process:
        pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch+1} Training')
    for batch_idx, data in enumerate(train_loader):
        try:
            data = move_data_to_device(data, device)
            optimizer.zero_grad()
            if not isinstance(data, dict):
                raise TypeError("Expected 'data' to be a dictionary")
            with autocast():
                output = network(data)
                target = move_data_to_device(data['top_seg'], device)
                loss = criterion(output['top_seg'], target)
            scaler.scale(loss).backward()
            
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            # Clip gradients to a maximum norm (adjust the value as needed)
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            if is_main_process:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                pbar.update()
        except Exception as e:
            logger.error(f'Error in batch {batch_idx}: {str(e)}')
            continue
    if is_main_process:
        pbar.close()
    return epoch_loss


def validate_one_epoch(network, val_loader, criterion, device, logger, is_main_process, epoch, num_classes, ignore_index=-100):
    """
    Executes one validation epoch and computes the mIoU while ignoring the background class (label 0).

    Returns:
        epoch_val_loss: accumulated loss over the epoch
        miou: mean Intersection over Union across classes (ignoring label 0)
    """
    epoch_val_loss = 0.0
    total_intersections = torch.zeros(num_classes, device=device, dtype=torch.float64)
    total_unions = torch.zeros(num_classes, device=device, dtype=torch.float64)
    if is_main_process:
        pbar_val = tqdm(total=len(val_loader), desc=f'Epoch {epoch+1} Validation')
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            try:
                data = move_data_to_device(data, device)
                if not isinstance(data, dict):
                    raise TypeError("Expected 'data' to be a dictionary")
                with autocast():
                    output = network(data)
                    target = move_data_to_device(data['top_seg'], device)
                    loss = criterion(output['top_seg'], target)
                epoch_val_loss += loss.item()
                pred = torch.argmax(output['top_seg'], dim=1)
                valid_mask = target != ignore_index
                # Skip label 0 (background) by iterating from 1 to num_classes-1
                for c in range(1, num_classes):
                    pred_c = (pred == c)
                    target_c = (target == c)
                    intersection = ((pred_c & target_c) & valid_mask).sum().to(torch.float64)
                    union = ((pred_c | target_c) & valid_mask).sum().to(torch.float64)
                    total_intersections[c] += intersection
                    total_unions[c] += union
                if is_main_process:
                    pbar_val.set_postfix({'loss': f'{loss.item():.4f}'})
                    pbar_val.update()
            except Exception as e:
                logger.error(f'Error in validation batch {batch_idx}: {str(e)}')
                continue
    if is_main_process:
        pbar_val.close()
    dist.all_reduce(total_intersections, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_unions, op=dist.ReduceOp.SUM)
    # Compute per-class IoU for classes 1 to N-1 (ignoring the background class)
    iou_per_class = torch.where(
        total_unions > 0,
        total_intersections / total_unions,
        torch.zeros_like(total_intersections)
    )
    miou = iou_per_class[1:].mean().item()
    return epoch_val_loss, miou


def train(rank: int, world_size: int, params: dict) -> None:
    """Training function for each process."""
    try:
        # Get seed and deterministic settings from config
        seed = params.get('random_seed', 420)
        deterministic = params.get('deterministic', True)
        
        set_seed(seed + rank, deterministic)
        setup(rank, world_size, deterministic)
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
        
        logger.warning("───────────────────────────────")
        logger.warning(f"original class_weights: {class_weights}")
        logger.warning("───────────────────────────────")

        total_weight = class_weights.sum().item()
        if total_weight > 0:
            class_weights = class_weights / total_weight
        
        logger.warning("───────────────────────────────")
        logger.warning(f"normalized_class_weights: {class_weights}")
        logger.warning("───────────────────────────────")

        # Criterion selection based on config (loss_type: "focal" or "cross_entropy")
        loss_type = params.get('loss_type', 'cross_entropy')
        if loss_type.lower() == 'focal':
            gamma = params.get('focal_gamma', 2.0)
            criterion = FocalLoss(gamma=gamma, weight=class_weights).to(rank)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100).to(rank)
        
        # Initialize the optimizer and scheduler
        base_lr = params.get("initial_learning_rate", 0.001)  # Use a configurable base learning rate
        optimizer = optim.Adam(network.parameters(), lr=base_lr, weight_decay=1e-4, betas=(0.9, 0.999))
        lr_scheduler = ReduceLROnPlateau(optimizer, 
                                         mode='min', 
                                         patience=6, 
                                         factor=0.8, 
                                         min_lr=1e-6,
                                         verbose=True)

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
        lrs = []
        miou_values = []
        best_val_loss = float('inf')
        best_train_loss = float('inf')
        patience = 20
        epochs_no_improve = 0
        
        scaler = GradScaler()
        
        for epoch in range(params['num_epochs']):
            train_sampler.set_epoch(epoch + seed)
            network.train()
            epoch_loss = train_one_epoch(network, train_loader, optimizer, criterion, scaler, rank, epoch, logger, is_main_process)
            dist.barrier()
            epoch_loss_tensor = torch.tensor(epoch_loss / len(train_loader), device=rank)
            dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
            avg_epoch_loss = epoch_loss_tensor.item() / world_size
            
            network.eval()
            epoch_val_loss, epoch_miou = validate_one_epoch(network, val_loader, criterion, rank, logger, is_main_process, epoch, params['n_classes_seg'])
            dist.barrier()
            epoch_val_loss_tensor = torch.tensor(epoch_val_loss / len(val_loader), device=rank)
            dist.all_reduce(epoch_val_loss_tensor, op=dist.ReduceOp.SUM)
            avg_epoch_val_loss = epoch_val_loss_tensor.item() / world_size
            
            # Update learning rate with scheduler adjustments, synchronizing across processes
            if is_main_process:
                lr_scheduler.step(avg_epoch_val_loss)
                new_lr = optimizer.param_groups[0]['lr']
                new_lr_tensor = torch.tensor(new_lr, device=rank)
            else:
                new_lr_tensor = torch.tensor(0.0, device=rank)
            dist.broadcast(new_lr_tensor, src=0)
            new_lr = new_lr_tensor.item()
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr
            
            if is_main_process:
                logger.info(f'Epoch {epoch+1} - Average Training Loss: {avg_epoch_loss:.4f}, Average Validation Loss: {avg_epoch_val_loss:.4f}, mIoU: {epoch_miou:.4f}')
                losses.append(avg_epoch_loss)
                val_losses.append(avg_epoch_val_loss)
                lrs.append(optimizer.param_groups[0]['lr'])
                miou_values.append(epoch_miou)
                
                # Plot training loss, validation loss, and learning rate in the same plot
                plt.figure(figsize=(12,10))
                ax = plt.subplot(111)
                ax.plot(range(1, len(losses) + 1), losses, 'b-', label='Training Loss')
                ax.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Training/Validation Loss & Learning Rate')
                ax.grid(True, alpha=0.3)

                ax2 = ax.twinx()
                ax2.plot(range(1, len(lrs) + 1), lrs, 'g-', label='Learning Rate')
                ax2.set_ylabel('Learning Rate', color='g')
                ax2.tick_params(axis='y', labelcolor='g')

                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
                plt.tight_layout()

                save_path = os.path.join(save_dir, 'training_plot.png')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                plt.close()
                
                # Plot mIoU in a separate plot
                plt.figure(figsize=(8,6))
                plt.plot(range(1, len(miou_values) + 1), miou_values, 'm-', label='mIoU')
                plt.xlabel('Epoch')
                plt.ylabel('mIoU')
                plt.title('Mean Intersection over Union (mIoU) Over Epochs')
                plt.grid(True, alpha=0.3)
                plt.legend(loc='best')
                plt.tight_layout()
                
                miou_save_path = os.path.join(save_dir, 'miou_plot.png')
                os.makedirs(os.path.dirname(miou_save_path), exist_ok=True)
                plt.savefig(miou_save_path)
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
                    'learning_rates': lrs,
                    'miou': miou_values,
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
                writer.add_scalar("Loss/train", avg_epoch_loss, epoch)
                writer.add_scalar("Loss/val", avg_epoch_val_loss, epoch)
                writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar("mIoU", epoch_miou, epoch)

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


def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()
    
    logger = get_logger("train-dist", logging.INFO)
    
    try:
        with open(args.config, 'r') as file:
            params = yaml.safe_load(file)
        
        train_sbevnet_distributed(args.config)
        
    except Exception as e:
        logger.error(f"Error: {e.__class__.__name__}: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main() 