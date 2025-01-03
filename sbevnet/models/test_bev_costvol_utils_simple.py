import pytest
import numpy as np
import torch
from .bev_costvol_utils import get_grid_one

# def test_get_grid_one_basic():
#     # Basic test case with simple parameters
#     cam_conf = [100.0, 320.0, 240.0, 0.5]  # f, cx, cy, tx
#     img_h = 480
#     img_w = 640
#     n_hmap = 128
#     xmax = 51
#     xmin = -50
#     ymax = 50
#     ymin = -50
#     max_disp = 64
#     camera_ext_x = 0
#     camera_ext_y = 0
    
#     # Add input validation assertions
#     assert len(cam_conf) == 4, "cam_conf must have exactly 4 elements"
#     assert all(isinstance(x, float) for x in cam_conf), "all cam_conf values must be floats"
#     assert xmax > xmin, "xmax must be greater than xmin"
#     assert ymax > ymin, "ymax must be greater than ymin"
    
#     try:
#         grid = get_grid_one(
#             cam_conf, img_h, img_w, n_hmap, 
#             xmax, xmin, ymax, ymin, 
#             max_disp, camera_ext_x, camera_ext_y
#         )
#     except Exception as e:
#         pytest.fail(f"get_grid_one failed with error: {str(e)}")
    
#     # Enhanced output checks
#     assert isinstance(grid, torch.Tensor), "Output must be a torch.Tensor"
#     assert grid.shape == (1, n_hmap, n_hmap, 2), f"Expected shape (1, {n_hmap}, {n_hmap}, 2), got {grid.shape}"
#     assert grid.dtype == torch.float32, f"Expected dtype torch.float32, got {grid.dtype}"
#     assert not torch.isnan(grid).any(), "Output contains NaN values"
#     assert not torch.isinf(grid).any(), "Output contains infinite values"

# def test_get_grid_one_cache():
#     # Test that caching works
#     cam_conf = [100.0, 320.0, 240.0, 0.5]
#     img_h = 480
#     img_w = 640
#     n_hmap = 128
#     xmax = 50
#     xmin = -50
#     ymax = 50
#     ymin = -50
#     max_disp = 64
#     camera_ext_x = 0
#     camera_ext_y = 0
    
#     # First call
#     grid1 = get_grid_one(
#         cam_conf, img_h, img_w, n_hmap, 
#         xmax, xmin, ymax, ymin, 
#         max_disp, camera_ext_x, camera_ext_y
#     )
    
#     # Second call with same parameters
#     grid2 = get_grid_one(
#         cam_conf, img_h, img_w, n_hmap, 
#         xmax, xmin, ymax, ymin, 
#         max_disp, camera_ext_x, camera_ext_y
#     )
    
#     # Use allclose instead of exact equality
#     assert torch.allclose(grid1, grid2, rtol=1e-5, atol=1e-8)

# def test_get_grid_one_invalid_inputs():
#     # Test invalid inputs
#     cam_conf = [100.0, 320.0, 240.0]  # Missing tx
#     img_h = 480
#     img_w = 640
#     n_hmap = 128
#     xmax = 50
#     xmin = -50
#     ymax = 50
#     ymin = -50
#     max_disp = 64
#     camera_ext_x = 0
#     camera_ext_y = 0
    
#     # Should raise assertion error due to invalid cam_conf length
#     with pytest.raises(AssertionError):
#         get_grid_one(
#             cam_conf, img_h, img_w, n_hmap, 
#             xmax, xmin, ymax, ymin, 
#             max_disp, camera_ext_x, camera_ext_y
#        )

def test_get_grid_one_output_range():
    # Test that output values are in expected range (-1 to 1 for grid_sample)
    cam_conf = [1093.2768, 964.989, 569.276, 0.12]
    img_h = 480
    img_w = 640
    n_hmap = 256
    xmax = 7.0
    xmin = 2.0
    ymax = 2.5
    ymin = -2.5
    max_disp = 64
    camera_ext_x = 0.0
    camera_ext_y = 0.0
    
    grid = get_grid_one(
        cam_conf, img_h, img_w, n_hmap, 
        xmax, xmin, ymax, ymin, 
        max_disp, camera_ext_x, camera_ext_y
    )
    
    # Check that all values are in the valid range for grid_sample (-1 to 1)
    assert torch.all(grid >= -1)
    assert torch.all(grid <= 1)