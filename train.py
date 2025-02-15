#! /usr/bin/env python3

import os
import sys
import time
import random
import logging
import yaml
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast

from sbevnet.models.network_sbevnet import SBEVNet
from sbevnet.data_utils.bev_dataset import sbevnet_dataset
from helpers import get_logger, populate_json


def set_seed(seed: int, deterministic: bool = True):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Worker initialization function for DataLoader."""
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def move_data_to_device(data, device):
    """
    Recursively moves tensors in nested data structures (dict, list, tuple)
    to the specified device.
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


class FocalLoss(nn.Module):
    """
    Implementation of the focal loss from "Focal Loss for Dense Object Detection".
    """
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def train_one_epoch(network, train_loader, optimizer, criterion, scaler, device, epoch, logger):
    """
    Executes one training epoch using AMP and gradient clipping.
    """
    epoch_loss = 0.0
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
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            pbar.update()
        except Exception as e:
            logger.error(f'Error in batch {batch_idx}: {str(e)}')
            continue
    pbar.close()
    return epoch_loss


def validate_one_epoch(network, val_loader, criterion, device, logger, epoch, num_classes, ignore_index=-100):
    """
    Executes one validation epoch and computes the mIoU (excluding label 0).
    """
    epoch_val_loss = 0.0
    total_intersections = torch.zeros(num_classes, device=device, dtype=torch.float64)
    total_unions = torch.zeros(num_classes, device=device, dtype=torch.float64)
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
                # Compute IoU (ignoring class 0)
                for c in range(1, num_classes):
                    pred_c = (pred == c)
                    target_c = (target == c)
                    intersection = ((pred_c & target_c) & valid_mask).sum().to(torch.float64)
                    union = ((pred_c | target_c) & valid_mask).sum().to(torch.float64)
                    total_intersections[c] += intersection
                    total_unions[c] += union
                pbar_val.set_postfix({'loss': f'{loss.item():.4f}'})
                pbar_val.update()
            except Exception as e:
                logger.error(f'Error in validation batch {batch_idx}: {str(e)}')
                continue
    pbar_val.close()
    iou_per_class = torch.where(
        total_unions > 0,
        total_intersections / total_unions,
        torch.zeros_like(total_intersections)
    )
    miou = iou_per_class[1:].mean().item()
    return epoch_val_loss, miou


def train_sbevnet():
    logger = get_logger("train")
    
    # Load configuration
    with open('configs/train.yaml', 'r') as file:
        params = yaml.safe_load(file)
    
    # Scale camera parameters using fixed scale factors
    scale_x = float(640/1920)
    scale_y = float(480/1080)
    params['cx'] *= scale_x
    params['cy'] *= scale_y
    params['f'] *= scale_x
    
    save_dir = params.get('save_dir', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Set seed for reproducibility
    seed = params.get('random_seed', 420)
    deterministic = params.get('deterministic', True)
    set_seed(seed, deterministic)
    
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
    ).to(device)
    
    # Initialize datasets and DataLoaders (no distributed sampler)
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
    
    val_dataset = sbevnet_dataset(
        json_path=params['dataset_path'],
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
        num_workers=params.get('num_workers', 4),
        pin_memory=True,
        worker_init_fn=seed_worker
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=params.get('num_workers', 4),
        pin_memory=True,
        worker_init_fn=seed_worker
    )
    
    logger.info(f'Training dataset size: {len(train_dataset)}')
    logger.info(f'Validation dataset size: {len(val_dataset)}')
    
    # Setup class weights if provided; otherwise use None
    class_weights = None
    if 'class_weights' in params:
        class_weights = torch.tensor(params['class_weights'], dtype=torch.float32).to(device)
        total_weight = class_weights.sum().item()
        if total_weight > 0:
            class_weights = class_weights / total_weight
    
    # Select criterion based on config (either focal loss or cross entropy)
    loss_type = params.get('loss_type', 'cross_entropy')
    if loss_type.lower() == 'focal':
        gamma = params.get('focal_gamma', 2.0)
        criterion = FocalLoss(gamma=gamma, weight=class_weights).to(device)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100).to(device)
    
    # Initialize optimizer and learning rate scheduler
    base_lr = params.get("learning_rate", 0.001)
    optimizer = optim.Adam(network.parameters(), lr=base_lr, weight_decay=1e-4, betas=(0.9, 0.999))
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=6, factor=0.8, min_lr=1e-6, verbose=True)
    
    scaler = GradScaler()
    
    # Setup TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'runs'))
    
    losses = []
    val_losses = []
    lrs = []
    miou_values = []
    
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    
    epochs_no_improve = 0
    for epoch in range(params['num_epochs']):
        network.train()
        epoch_loss = train_one_epoch(network, train_loader, optimizer, criterion, scaler, device, epoch, logger)
        avg_epoch_loss = epoch_loss / len(train_loader)
        
        network.eval()
        epoch_val_loss, epoch_miou = validate_one_epoch(network, val_loader, criterion, device, logger, epoch, params['n_classes_seg'])
        avg_epoch_val_loss = epoch_val_loss / len(val_loader)
        
        lr_scheduler.step(avg_epoch_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        for pg in optimizer.param_groups:
            pg['lr'] = new_lr
        
        logger.info(f'Epoch {epoch+1} - Avg Train Loss: {avg_epoch_loss:.4f}, '
                    f'Avg Val Loss: {avg_epoch_val_loss:.4f}, mIoU: {epoch_miou:.4f}')
        losses.append(avg_epoch_loss)
        val_losses.append(avg_epoch_val_loss)
        lrs.append(new_lr)
        miou_values.append(epoch_miou)
        
        writer.add_scalar("Loss/train", avg_epoch_loss, epoch)
        writer.add_scalar("Loss/val", avg_epoch_val_loss, epoch)
        writer.add_scalar("Learning Rate", new_lr, epoch)
        writer.add_scalar("mIoU", epoch_miou, epoch)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_epoch_loss,
            'val_loss': avg_epoch_val_loss,
            'losses': losses,
            'val_losses': val_losses,
            'learning_rates': lrs,
            'miou': miou_values,
        }
        torch.save(checkpoint, os.path.join(save_dir, 'latest_checkpoint.pth'))

        if avg_epoch_val_loss < best_val_loss:
            best_val_loss = avg_epoch_val_loss
            torch.save(checkpoint, os.path.join(save_dir, 'best_val_model.pth'))
            logger.info(f'New best validation model saved with loss: {best_val_loss:.4f}')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if avg_epoch_loss < best_train_loss:
            best_train_loss = avg_epoch_loss
            torch.save(checkpoint, os.path.join(save_dir, 'best_train_model.pth'))
            logger.info(f'New best training model saved with loss: {best_train_loss:.4f}')

        # Optional early stopping based on validation loss
        if epochs_no_improve >= params.get('early_stopping_patience', 20):
            logger.info("Early stopping triggered. Exiting training loop.")
            break

        # Plot training progress
        plt.figure(figsize=(12, 10))
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
        plt.savefig(os.path.join(save_dir, 'training_plot.png'))
        plt.close()
        
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(miou_values) + 1), miou_values, 'm-', label='mIoU')
        plt.xlabel('Epoch')
        plt.ylabel('mIoU')
        plt.title('Mean Intersection over Union (mIoU) Over Epochs')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'miou_plot.png'))
        plt.close()
    
    writer.close()
    logger.info("Training completed.")


if __name__ == '__main__':
    train_sbevnet() 