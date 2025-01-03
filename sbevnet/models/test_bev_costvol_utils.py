import pytest
import numpy as np
import torch
from .bev_costvol_utils import get_grid_one

def test_get_grid_one_basic():
    # Basic test case with simple parameters
    cam_conf = [100.0, 320.0, 240.0, 0.5]  # f, cx, cy, tx
    img_h = 480
    img_w = 640
    n_hmap = 128
    xmax = 50
    xmin = -50
    ymax = 50
    ymin = -50
    max_disp = 64
    camera_ext_x = 0
    camera_ext_y = 0
    rotation_matrix = np.eye(3)
    
    grid = get_grid_one(
        cam_conf, img_h, img_w, n_hmap, 
        xmax, xmin, ymax, ymin, 
        max_disp, camera_ext_x, camera_ext_y, 
        rotation_matrix
    )
    
    # Check output shape and type
    assert isinstance(grid, torch.Tensor)
    assert grid.shape == (1, n_hmap, n_hmap, 2)
    assert grid.dtype == torch.float32

def test_get_grid_one_cache():
    # Test that caching works
    cam_conf = [100.0, 320.0, 240.0, 0.5]
    img_h = 480
    img_w = 640
    n_hmap = 128
    xmax = 50
    xmin = -50
    ymax = 50
    ymin = -50
    max_disp = 64
    camera_ext_x = 0
    camera_ext_y = 0
    rotation_matrix = np.eye(3)
    
    # First call
    grid1 = get_grid_one(
        cam_conf, img_h, img_w, n_hmap, 
        xmax, xmin, ymax, ymin, 
        max_disp, camera_ext_x, camera_ext_y, 
        rotation_matrix
    )
    
    # Second call with same parameters
    grid2 = get_grid_one(
        cam_conf, img_h, img_w, n_hmap, 
        xmax, xmin, ymax, ymin, 
        max_disp, camera_ext_x, camera_ext_y, 
        rotation_matrix
    )
    
    # Use allclose instead of exact equality
    assert torch.allclose(grid1, grid2, rtol=1e-5, atol=1e-8)

def test_get_grid_one_rotation():
    # Test with different rotation matrices
    cam_conf = [100.0, 320.0, 240.0, 0.5]
    img_h = 480
    img_w = 640
    n_hmap = 128
    xmax = 50
    xmin = -50
    ymax = 50
    ymin = -50
    max_disp = 64
    camera_ext_x = 0
    camera_ext_y = 0
    
    # Create rotation matrix for 90 degrees around Y axis
    rotation_matrix = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ])
    
    grid = get_grid_one(
        cam_conf, img_h, img_w, n_hmap, 
        xmax, xmin, ymax, ymin, 
        max_disp, camera_ext_x, camera_ext_y, 
        rotation_matrix
    )
    
    assert isinstance(grid, torch.Tensor)
    assert grid.shape == (1, n_hmap, n_hmap, 2)

def test_get_grid_one_invalid_inputs():
    # Test invalid inputs
    cam_conf = [100.0, 320.0, 240.0]  # Missing tx
    img_h = 480
    img_w = 640
    n_hmap = 128
    xmax = 50
    xmin = -50
    ymax = 50
    ymin = -50
    max_disp = 64
    camera_ext_x = 0
    camera_ext_y = 0
    rotation_matrix = np.eye(3)
    
    # Should raise assertion error due to invalid cam_conf length
    with pytest.raises(AssertionError):
        get_grid_one(
            cam_conf, img_h, img_w, n_hmap, 
            xmax, xmin, ymax, ymin, 
            max_disp, camera_ext_x, camera_ext_y, 
            rotation_matrix
        )

def test_get_grid_one_none_rotation():
    # Test that None rotation matrix raises assertion error
    cam_conf = [100.0, 320.0, 240.0, 0.5]
    img_h = 480
    img_w = 640
    n_hmap = 128
    xmax = 50
    xmin = -50
    ymax = 50
    ymin = -50
    max_disp = 64
    camera_ext_x = 0
    camera_ext_y = 0
    rotation_matrix = None
    
    with pytest.raises(AssertionError, match="rotation_matrix is None"):
        get_grid_one(
            cam_conf, img_h, img_w, n_hmap, 
            xmax, xmin, ymax, ymin, 
            max_disp, camera_ext_x, camera_ext_y, 
            rotation_matrix
        )

def test_get_grid_one_output_range():
    # Test that output values are in expected range (-1 to 1 for grid_sample)
    cam_conf = [100.0, 320.0, 240.0, 0.5]
    img_h = 480
    img_w = 640
    n_hmap = 128
    xmax = 50
    xmin = -50
    ymax = 50
    ymin = -50
    max_disp = 64
    camera_ext_x = 0
    camera_ext_y = 0
    rotation_matrix = np.eye(3)
    
    grid = get_grid_one(
        cam_conf, img_h, img_w, n_hmap, 
        xmax, xmin, ymax, ymin, 
        max_disp, camera_ext_x, camera_ext_y, 
        rotation_matrix
    )
    
    # Check that all values are in the valid range for grid_sample (-1 to 1)
    assert torch.all(grid >= -1)
    assert torch.all(grid <= 1) 