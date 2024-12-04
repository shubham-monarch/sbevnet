import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm

from sbevnet.models.network_sbevnet import SBEVNet
from sbevnet.data_utils.bev_dataset import sbevnet_dataset

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )

def train_sbevnet():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Training parameters
    params = {
        'image_w': 512,
        'image_h': 288,
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
        'batch_size': 3,
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
        datapath='dataset.json',
        split='train',
        th=params['image_h'],
        tw=params['image_w']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f'Training dataset size: {len(train_dataset)}')
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(params['num_epochs']):
        network.train()
        epoch_loss = 0
        
        # Progress bar for batches
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{params["num_epochs"]}')
        
        for batch_idx, data in enumerate(progress_bar):
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
                output = network(data)
                
                # Compute loss
                loss = criterion(output['top_seg'], data['top_seg'])
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
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