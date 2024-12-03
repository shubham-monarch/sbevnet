import os
import argparse
import torch
from torch.utils.data import DataLoader

from pytorch_propane.models import Model
from pytorch_propane.registry import registry

from sbevnet.models.network_sbevnet import SBEVNet
from sbevnet.models.model_sbevnet import sbevnet_model
from sbevnet.data_utils.bev_dataset import ImgsLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Train SBEVNet')
    
    # Dataset parameters
    parser.add_argument('--datapath', required=True, help='Path to dataset JSON file')
    parser.add_argument('--batch_size', type=int, default=3, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=1, help='Evaluation batch size')
    
    # Model parameters
    parser.add_argument('--image_w', type=int, default=512, help='Input image width')
    parser.add_argument('--image_h', type=int, default=288, help='Input image height')
    parser.add_argument('--max_disp', type=int, default=64, help='Maximum disparity')
    parser.add_argument('--n_hmap', type=int, default=100, help='Height map size')
    parser.add_argument('--xmin', type=float, default=1, help='Minimum x coordinate')
    parser.add_argument('--xmax', type=float, default=39, help='Maximum x coordinate')
    parser.add_argument('--ymin', type=float, default=-19, help='Minimum y coordinate')
    parser.add_argument('--ymax', type=float, default=19, help='Maximum y coordinate')
    
    # Camera parameters
    parser.add_argument('--cx', type=float, default=256, help='Principal point x')
    parser.add_argument('--cy', type=float, default=144, help='Principal point y')
    parser.add_argument('--f', type=float, default=179.2531, help='Focal length')
    parser.add_argument('--tx', type=float, default=0.2, help='Baseline')
    parser.add_argument('--camera_ext_x', type=float, default=0.9, help='Camera external parameter x')
    parser.add_argument('--camera_ext_y', type=float, default=-0.1, help='Camera external parameter y')
    
    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--save_path', required=True, help='Path to save model checkpoints')
    parser.add_argument('--do_ipm_rgb', type=bool, default=True, help='Use IPM RGB')
    parser.add_argument('--do_ipm_feats', type=bool, default=True, help='Use IPM features')
    parser.add_argument('--fixed_cam_confs', type=bool, default=True, help='Use fixed camera configuration')
    parser.add_argument('--check_degenerate', type=bool, default=True, help='Check for degenerate predictions')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create network
    network = SBEVNet(
        image_w=args.image_w,
        image_h=args.image_h,
        xmin=args.xmin,
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
        n_hmap=args.n_hmap,
        max_disp=args.max_disp,
        cx=args.cx,
        cy=args.cy,
        f=args.f,
        tx=args.tx,
        camera_ext_x=args.camera_ext_x,
        camera_ext_y=args.camera_ext_y,
        do_ipm_rgb=args.do_ipm_rgb,
        do_ipm_feats=args.do_ipm_feats,
        fixed_cam_confs=args.fixed_cam_confs
    )
    
    # Create model wrapper
    model = sbevnet_model(
        network=network,
        check_degenerate=args.check_degenerate
    )
    
    # Create data loaders
    # Note: You'll need to implement your dataset class based on your data format
    train_dataset = ImgsLoader(
        rgb_imgs=None,  # Load your training data here
        th=args.image_h,
        tw=args.image_w
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)
    
    # Training loop
    for epoch in range(args.n_epochs):
        model.train()
        
        for batch_idx, data in enumerate(train_loader):
            # Your training step here
            # The model expects input in the format:
            # data = {
            #     'input_imgs': [left_img, right_img],
            #     'ipm_rgb': ipm_rgb,  # if do_ipm_rgb
            #     'ipm_feats_m': ipm_feats_m,  # if do_ipm_feats
            #     'cam_confs': cam_confs  # if not fixed_cam_confs
            # }
            
            loss = model.train_on_batch(data)
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss}')
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.save_path, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
        }, checkpoint_path)

if __name__ == '__main__':
    main() 