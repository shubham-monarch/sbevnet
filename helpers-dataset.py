import os
import shutil
import json
import random
from pathlib import Path
import numpy as np

def create_directory_structure(output_root):
    """Create the required directory structure"""
    splits = ['train', 'test']
    subdirs = ['rgb_left', 'rgb_right', 'top_seg', 'mask', 'top_ipm', 'top_ipm_m', 'confs']
    
    for split in splits:
        for subdir in subdirs:
            os.makedirs(os.path.join(output_root, split, subdir), exist_ok=True)

def process_data(input_dir, output_root, train_ratio=0.8, seed=42):
    """Process and reorganize the data"""
    random.seed(seed)
    
    # Create directory structure
    create_directory_structure(output_root)
    
    # Get all scene directories
    scene_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d)) and d.isdigit()]
    scene_dirs.sort(key=int)
    
    # Split into train/test
    random.shuffle(scene_dirs)
    split_idx = int(len(scene_dirs) * train_ratio)
    train_scenes = scene_dirs[:split_idx]
    test_scenes = scene_dirs[split_idx:]
    
    dataset = {"train": {k: [] for k in ['rgb_left', 'rgb_right', 'top_seg', 'mask', 'top_ipm', 'top_ipm_m', 'confs']},
               "test": {k: [] for k in ['rgb_left', 'rgb_right', 'top_seg', 'mask', 'top_ipm', 'top_ipm_m', 'confs']}}
    
    def process_split(scenes, split_name):
        for scene in scenes:
            scene_path = os.path.join(input_dir, scene)
            
            # Find the required files
            left_img = None
            right_img = None
            
            # Check for different naming patterns
            if os.path.exists(os.path.join(scene_path, '_left.jpg')):
                left_img = '_left.jpg'
                right_img = '_right.jpg'
            elif os.path.exists(os.path.join(scene_path, 'left.jpg')):
                left_img = 'left.jpg'
                right_img = 'right.jpg'
            
            if left_img is None or right_img is None:
                print(f"Warning: Missing stereo pair in scene {scene}")
                continue
                
            # Create output filenames
            scene_id = f"{int(scene):06d}"
            
            # Copy and rename files
            # RGB Left
            src_path = os.path.join(scene_path, left_img)
            dst_path = os.path.join(output_root, split_name, 'rgb_left', f'{scene_id}.png')
            shutil.copy2(src_path, dst_path)
            dataset[split_name]['rgb_left'].append(f"{split_name}/rgb_left/{scene_id}.png")
            
            # RGB Right
            src_path = os.path.join(scene_path, right_img)
            dst_path = os.path.join(output_root, split_name, 'rgb_right', f'{scene_id}.png')
            shutil.copy2(src_path, dst_path)
            dataset[split_name]['rgb_right'].append(f"{split_name}/rgb_right/{scene_id}.png")
            
            # BEV image (top_ipm)
            bev_path = os.path.join(scene_path, 'bev_image.png')
            if os.path.exists(bev_path):
                dst_path = os.path.join(output_root, split_name, 'top_ipm', f'{scene_id}.png')
                shutil.copy2(bev_path, dst_path)
                dataset[split_name]['top_ipm'].append(f"{split_name}/top_ipm/{scene_id}.png")
            
            # Generate placeholder files for missing data
            # You'll need to implement actual data generation for these files
            
            # # top_seg (from left-segmented-labelled.ply)
            # dst_path = os.path.join(output_root, split_name, 'top_seg', f'{scene_id}.png')
            # # TODO: Convert PLY to top-view segmentation
            # np.zeros((100, 100), dtype=np.uint8).tofile(dst_path)
            # dataset[split_name]['top_seg'].append(f"{split_name}/top_seg/{scene_id}.png")
            
            # mask
            # dst_path = os.path.join(output_root, split_name, 'mask', f'{scene_id}.png')
            # np.ones((100, 100), dtype=np.uint8).tofile(dst_path)
            # dataset[split_name]['mask'].append(f"{split_name}/mask/{scene_id}.png")
            
            # # top_ipm_m (IPM transformation matrices)
            # dst_path = os.path.join(output_root, split_name, 'top_ipm_m', f'{scene_id}.npy')
            # np.save(dst_path, np.eye(3))  # Placeholder transformation matrix
            # dataset[split_name]['top_ipm_m'].append(f"{split_name}/top_ipm_m/{scene_id}.npy")
            
            # # camera configurations
            # dst_path = os.path.join(output_root, split_name, 'confs', f'{scene_id}.npy')
            # # Default camera configuration [f, cx, cy, tx]
            # np.save(dst_path, np.array([179.2531, 256, 144, 0.2]))
            # dataset[split_name]['confs'].append(f"{split_name}/confs/{scene_id}.npy")
    
    # Process train and test splits
    process_split(train_scenes, 'train')
    process_split(test_scenes, 'test')
    
    # Save dataset.json
    with open('dataset.json', 'w') as f:
        json.dump(dataset, f, indent=4)

if __name__ == "__main__":
    # Usage example
    input_dir = "train-data"  # Your original data directory
    output_root = "processed-data"  # Where to store the reorganized data
    process_data(input_dir, output_root) 