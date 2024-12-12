
import torch
import torch.nn as nn
import torchgeometry
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from helpers import get_logger

logger = get_logger("bev_costvol_utils")

mapping_cache = {}


def get_grid_one( cam_conf , 
                 img_h , img_w , 
                 n_hmap , 
                 xmax , xmin , ymax , ymin  , 
                 max_disp , 
                 camera_ext_x , camera_ext_y   ):
    
    remap_normed_inv = np.zeros((n_hmap , n_hmap , 2 ))
    assert len(cam_conf) == 4 
    f , cx , cy , tx = cam_conf
    f = float( f )
    cx = float( cx )
    cy = float( cy )
    tx = float( tx )
    
    # logger.warning(f"=================")
    # logger.warning(f"(x_min, x_max) = {(xmin, xmax)}")
    # logger.warning(f"(y_min, y_max) = {(ymin, ymax)}")
    # logger.warning(f"=================")
    
    
    key = str(f) + str(cx) + str(cy) + str(tx)
    
    if not key in  mapping_cache:

        for X in range(n_hmap):
            for Y in range(n_hmap):
                # logger.warning(f"=================")
                # logger.warning(f"(X, Y) = {(X, Y)}")
                # logger.warning(f"=================\n")
                
                # # x: [depth]
                # k = ((( f  / (((xmax-xmin)*X/n_hmap + xmin - camera_ext_x)/tx ) ))) / ( max_disp/2) - 1 
                
                # Calculate t1 term
                t1 = 10 - ((xmax-xmin)*X/n_hmap + xmin)
                
                try:
                    # Term 1: Depth scaling
                    term1 = f / ((t1 - camera_ext_x)/tx)
                    # Term 2: Disparity normalization 
                    term2 = term1 / (max_disp/2)
                    # Term 3: Final offset
                    k = term2 - 1
                except ZeroDivisionError:
                    logger.info(f"=================")
                    logger.info("x calculations => ")
                    logger.info(f"t1 = {t1}")
                    logger.info(f"camera_ext_x = {camera_ext_x}")
                    logger.info(f"tx = {tx}")
                    logger.info(f"max_disp = {max_disp}")
                    logger.info(f"=================\n")
                    raise
                
                try:
                    # Term 1: Depth scaling
                    term1 = f / ((t1 - camera_ext_x)/tx)
                    # Term 2: Horizontal offset 
                    term2 = (((ymax-ymin)*Y/n_hmap + ymin - camera_ext_y)/tx)
                    # Term 3: Image coordinate normalization
                    j = ((term1 * term2 + cx)/(img_w/2)) - 1
                except ZeroDivisionError:
                    logger.warning(f"=================")
                    logger.warning("y calculations => ")
                    logger.warning(f"t1 = {t1}")
                    logger.warning(f"camera_ext_x = {camera_ext_x}") 
                    logger.warning(f"tx = {tx}")
                    logger.warning(f"img_w = {img_w}")
                    logger.warning(f"=================\n")
                    raise
               
                # break
            # break
        
        remap_normed_inv[ Y ,X, 0 ] = k # depth is along x lol
        remap_normed_inv[ Y , X  , 1 ] = j

        mapping_cache[key] = remap_normed_inv
    
    remap_normed_inv = mapping_cache[key]
    grid = torch.from_numpy( remap_normed_inv[None].astype('float32') )
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
    
    
    
    assert reduced_vol.shape[2] == img_w
    assert reduced_vol.shape[3] == max_disp
    
    bs = reduced_vol.shape[0]
    grids = []
    
    for i in range( bs ):
        grids.append( get_grid_one( cam_confs[i] , img_h=img_h , img_w=img_w , n_hmap=n_hmap , xmax=xmax , xmin=xmin , ymax=ymax , ymin=ymin   , max_disp=max_disp , camera_ext_x=camera_ext_x, camera_ext_y=camera_ext_y   ) )
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



