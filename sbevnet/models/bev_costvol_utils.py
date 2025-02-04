import torch
import torch.nn as nn
import torchgeometry
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from typing import Dict, Any    

from helpers import get_logger

mapping_cache = {}


def get_grid_one( cam_conf:Dict[str, Any] , img_h , img_w , n_hmap , xmax , xmin , ymax , ymin  , max_disp , camera_ext_x , camera_ext_y   ):
    remap_normed_inv = np.zeros((n_hmap , n_hmap , 2 ))
    # assert len(cam_conf) == 4 
    # f , cx , cy , tx = cam_conf
    # f = float( f )
    # cx = float( cx )
    # cy = float( cy )
    # tx = float( tx )
    
    logger = get_logger("get_grid_one")

    f = cam_conf['f']
    cx = cam_conf['cx']
    cy = cam_conf['cy']
    tx = cam_conf['tx']

    R = np.eye(3)
    if 'R' in cam_conf:
        R = cam_conf['R'][:3, :3]
    
    if torch.is_tensor(R):
            R = R.cpu().numpy()

    # logger.info(f"=================")
    # logger.info(f"R: {R}")
    # logger.info(f"=================\n")  

    key = str(f) + str(cx) + str(cy) + str(tx) + str(R.tobytes())

    if not key in mapping_cache:
        for X in range(n_hmap):
            for Y in range(n_hmap):
                # Get point in pointcloud coordinates
                z_pc = (xmax-xmin)*X/n_hmap + xmin - camera_ext_x  # Depth direction
                x_pc = (ymax-ymin)*Y/n_hmap + ymin - camera_ext_y  # Horizontal direction
                y_pc = 0  # Height (ground plane)
            
                # Apply rotation
                point_pc = np.array([x_pc, y_pc, z_pc])
                point_rotated = R @ point_pc
                
                # logger.info(f"=================")
                # logger.info(f"point_pc: {point_pc}")
                # logger.info(f"point_rotated: {point_rotated}")
                # logger.info(f"=================\n")

                # Keep original projection structure
                k = (f / (point_rotated[2]/tx)) / (max_disp/2) - 1
                j = ((f / (point_rotated[2]/tx)) * (point_rotated[0]/tx) + cx)/(img_w/2) - 1
            
                remap_normed_inv[Y, X, 0] = k
                remap_normed_inv[Y, X, 1] = j
            
        mapping_cache[key] = remap_normed_inv

    remap_normed_inv = mapping_cache[key]
    grid = torch.from_numpy(remap_normed_inv[None].astype('float32'))
    return grid

    


def pt_costvol_to_hmap( reduced_vol , cam_confs , sys_confs ):
    
    
    img_h = sys_confs['img_h']
    img_w = sys_confs['img_w']
    n_hmap = sys_confs['n_hmap']
    xmax = sys_confs['xmax']
    xmin = sys_confs['xmin']
    ymax = sys_confs['ymax']
    ymin = sys_confs['ymin']
    max_disp = sys_confs['max_disp']
    camera_ext_x = sys_confs['camera_ext_x']
    camera_ext_y     = sys_confs['camera_ext_y']
    
    logger = get_logger("pt_costvol_to_hmap")
    
    assert reduced_vol.shape[2] == img_w
    assert reduced_vol.shape[3] == max_disp
    
    bs = reduced_vol.shape[0]
    grids = []
    
    cam_conf: dict = {}
    cam_conf['f'] = cam_confs['f']
    cam_conf['cx'] = cam_confs['cx']
    cam_conf['cy'] = cam_confs['cy']
    cam_conf['tx'] = cam_confs['tx']
    
    # Handle rotation matrix initialization
    if 'R' in cam_confs:
        R_list = cam_confs['R']
    else:
        R_list = [np.eye(3)] * bs  # Default identity matrix
    
    for i in range(bs):
        cam_conf['R'] = R_list[i] if 'R' in cam_confs else np.eye(3)
        grids.append( get_grid_one( cam_conf , img_h=img_h , img_w=img_w , n_hmap=n_hmap , xmax=xmax , xmin=xmin , ymax=ymax , ymin=ymin   , max_disp=max_disp , camera_ext_x=camera_ext_x, camera_ext_y=camera_ext_y   ) )
    
    grid = torch.cat( grids  , 0).cuda()
        
    warped = torch.nn.functional.grid_sample( reduced_vol , grid,padding_mode='zeros') 
    return warped
        
    

    


def warp_p_scale( img , ipm_m , sys_confs  ):
    mm = ipm_m.cpu().numpy()
    m = mm[ : , :9 ].reshape( (-1,3,3) )
    for i in range(img.shape[0]):
        s = mm[i , 10] /  img[i].shape[2]
        m[  i , : , :2 ] *= s 
#         print("scale , " , s  ,  mm[i , 10] , img[i].shape[2] )
    m = Variable( torch.from_numpy(m)).cuda()
    
#     dbg[-1]  = mm
    
    ans =  torchgeometry.warp_perspective( img , m  , dsize=(sys_confs['n_hmap'] , sys_confs['n_hmap'] ))
    ans = torch.flip(ans , (3,))
    return ans.permute(0 , 1 , 3 , 2)




def build_cost_volume(refimg_fea , targetimg_fea , maxdisp  ):
    cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, maxdisp//4,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_(), volatile=False).cuda()

    for i in range(maxdisp//4):
        if i > 0 :
            cost[:, :refimg_fea.size()[1], i, :,i:]   = refimg_fea[:,:,:,i:]
            cost[:, refimg_fea.size()[1]:, i, :,i:] = targetimg_fea[:,:,:,:-i]
        else:
            cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
            cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea
    cost = cost.contiguous()
    
    return cost 


