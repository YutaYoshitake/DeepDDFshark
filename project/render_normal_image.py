import os
import pdb
import sys
from turtle import pd
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

torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732






def get_pixel_normal(position, mask):
    length = position.shape[0]
    diff_from_right = (position[1:length] - position[0:length-1])[:, 0:length-1]
    diff_from_under = (position[:, 1:length] - position[:, 0:length-1])[0:length-1]
    normal = F.normalize(torch.cross(diff_from_right, diff_from_under, dim=-1), dim=-1)
    color = (normal + 1) / 2
    normal_image = torch.zeros_like(color)
    normal_image[mask[0:length-1, 0:length-1]] = color[mask[0:length-1, 0:length-1]]
    return normal_image





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
    # path = '/home/yyoshitake/works/DeepSDF/project/far_test'
    path = '/home/yyoshitake/works/DeepSDF/project/dataset/dugon/validation_set_multi35/1a6f615e8b1b5ae4dbbc9440457e303e/00005'
    pos, c2w = path2posc2w(path, model) # returen as torchtensor
    inverced_depth_map, blur_mask, normal_map, normal_mask, depth_map, hit_obj_mask = path2depthinfo(path, model, with_depth = True, with_normal = True, H = 'unspecified')

    # Get instance id.
    instance_id = torch.zeros(1, dtype=torch.long).to(model.device)
    # instance_id = torch.full_like(instance_id, 5)


    # get inputs
    rays_d_cam = get_ray_direction(args.H, args.fov, False) # batch, H, W, 3:xyz
    rays_d_wrd = get_ray_direction(args.H, args.fov, c2w).to(c2w.dtype).to(c2w.device) # batch, H, W, 3:xyz
    rays_o = pos[:, None, None, :].expand(-1, args.H, args.W, -1)
    input_lat_vec = model.lat_vecs(instance_id)

    # # Check map.
    # check_map((normal_map[0] + 1.) * .5, 'tes_gt.png')


    # # Check pixel normal.
    # est_depth = model.render_depth_map(pos, c2w, instance_id, H=512, inverced_depth_map=False)
    # hit_obj_mask = est_depth < 3
    # est_point = est_depth[hit_obj_mask][..., None] * rays_d_wrd[0][hit_obj_mask].reshape(-1, 3) + pos
    # est_point_image = torch.zeros(args.H, args.W, 3).to(c2w.dtype).to(c2w.device)
    # est_point_image[hit_obj_mask] = est_point
    # normal_image = get_pixel_normal(est_point_image, hit_obj_mask)
    # check_map(normal_image, 'tes_gt.png')

    # Get inverced depth map
    est_inverced_depth = model(rays_o, rays_d_wrd, input_lat_vec, blur_mask=blur_mask)

    # Start
    hit_obj_mask = torch.full_like(blur_mask, False)
    hit_obj_mask[blur_mask] = est_inverced_depth > .5

    # get normal
    est_normal = model.get_normals(rays_o, rays_d_wrd, input_lat_vec, c2w, hit_obj_mask, est_inverced_depth[hit_obj_mask[blur_mask]])
    
    # get normal map
    est_normal_image = torch.zeros_like(rays_o)
    est_normal_image[hit_obj_mask] = (est_normal + 1) / 2
    check_map(est_normal_image[0])
    