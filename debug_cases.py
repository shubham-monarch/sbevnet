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
    logger.info(f"test_labels_in_seg_masks")
    logger.info("───────────────────────────────")
