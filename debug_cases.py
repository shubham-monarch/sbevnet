"""Debug test cases implementation"""

import logging
import os
import random
import shutil
import traceback
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm

from helpers import get_logger


logger = get_logger("debug_cases")


def test_labels_in_seg_masks():
    """Case 1: Test labels in seg masks"""
    
    logger.info("───────────────────────────────")
    logger.info(f"Case 1: test_labels_in_seg_masks")
    logger.info("───────────────────────────────")

    masks_dir = Path("data/model-dataset/train/seg-masks-mono")
    mask_files = list(masks_dir.glob("*.png"))
    
    if not mask_files:
        logger.error(f"No mask files found in {masks_dir}")
        return

    num_samples = min(20, len(mask_files))
    sampled_files = random.sample(mask_files, num_samples)

    for mask_file in sampled_files:
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.error(f"Could not read mask file: {mask_file}")
            continue
        unique_labels, counts = np.unique(mask, return_counts=True)
        
        logger.warning("───────────────────────────────")   
        logger.warning(f"Mask: {mask_file.name}")
        logger.warning("───────────────────────────────")   
        for label, count in zip(unique_labels, counts):
            logger.info(f"  Label {label}: {count} pixels")