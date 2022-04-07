import os
import pdb
import sys
import numpy as np
import random
import pylab
import glob
import math
import re
from tqdm import tqdm, trange
import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from torch.autograd import grad
import torch.utils.data as data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from chamferdist import ChamferDistance
from scipy.spatial.transform import Rotation as R

from parser import *
from often_use import *
from train_pl import *





def make_point_cloud(est_depth, hit_rays_d, hit_rays_o):
    return est_depth[..., None] * hit_rays_d.reshape(-1, 3) + hit_rays_o






if __name__=='__main__':
    # Get args
    args = get_args()
    args.gpu_num = torch.cuda.device_count() # log used gpu num.
    args.same_instances = False

    # Get ckpts path.
    ckpt_dir = os.path.join('lightning_logs', f'{args.expname}_v{args.exp_version}', 'checkpoints/*')
    ckpt_path_list = sorted(glob.glob(ckpt_dir))
    latest_ckpt_path = ckpt_path_list[-1]

    # Make model and load ckpt.
    model = DDF(args)
    model = model.load_from_checkpoint(checkpoint_path=latest_ckpt_path, args=args)

    # model to gpu.
    device=torch.device('cuda')
    model.to(device)

    # pos is (batch, 3:xyz) and c2w is (batch, 3*3:SO3)
    path = '/home/yyoshitake/works/DeepSDF/project/far_test'
    # path = '/home/yyoshitake/works/DeepSDF/project/dataset/dugon/single_test_e488826128fe3854b300c4ca2f51c01b/e488826128fe3854b300c4ca2f51c01b/00010'
    pos, c2w = path2posc2w(path, model) # returen as torchtensor

    # Instance_id to latent code
    instance_id = torch.zeros(1, dtype=torch.long).to(model.device)

    # Get inverced depth map
    H = W = 512
    rays_d = get_ray_direction(H, args.fov, c2w)
    rays_o = pos[:, None, None, :].expand(-1, H, W, -1)
    input_lat_vec = model.lat_vecs(instance_id)
    est_depth, hit_obj_mask, _ = model.forward_from_far(rays_o, rays_d, input_lat_vec, inverced_depth=False)

    # Make point clouds.
    hit_rays_d = rays_d[hit_obj_mask]
    hit_rays_o = rays_o[hit_obj_mask]
    est_point = make_point_cloud(est_depth, hit_rays_d, hit_rays_o)

    # Make glound truth.
    H = W = 512
    rays_d = get_ray_direction(H, args.fov, c2w)
    rays_o = pos[:, None, None, :].expand(-1, H, W, -1)
    inverced_depth_map, blur_mask, depth_map, hit_obj_mask = path2depthinfo(path, model, with_depth = True, with_normal = False, H = H)
    gt_point = make_point_cloud(depth_map[hit_obj_mask], rays_d[hit_obj_mask], rays_o[hit_obj_mask])
    
    
    ##################################################
    ##################################################
    ##################################################

    
    # # Make fig
    # fig = pylab.figure(figsize=(5, 5))

    # ax1 = fig.add_subplot(1, 1, 1)
    # ax1.set_title('result')
    # ax1.imshow(depth_map[0].to('cpu').detach().numpy().copy())

    # ax1.xaxis.set_ticklabels([])
    # ax1.yaxis.set_ticklabels([])
    # path = os.path.join('tes.png')
    # fig.savefig(path, dpi=300)
    # pylab.close()
    
    
    ##################################################
    ##################################################
    ##################################################

    
    # Make fig
    fig = pylab.figure(figsize=(5, 5))

    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    point_1 = est_point.to('cpu').detach().numpy().copy().reshape(-1, 3)
    point_2 = gt_point.to('cpu').detach().numpy().copy().reshape(-1, 3)
    ax1.scatter(point_1[:, 0], point_1[:, 1], point_1[:, 2], marker="o", linestyle='None', c='m', s=0.01)
    ax1.scatter(point_2[:, 0], point_2[:, 1], point_2[:, 2], marker="o", linestyle='None', c='c', s=0.01)
    ax1.view_init(elev=90, azim=0)

    path = os.path.join('tes.png')
    fig.savefig(path, dpi=300)
    pylab.close()
    