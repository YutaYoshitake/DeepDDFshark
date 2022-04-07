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





class DDF_dataset(data.Dataset):
    def __init__(self, args, data_dir, N_views):
    
        self.data_list = []
        with open(args.instance_list_txt, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                self.data_list.append(os.path.join(data_dir, line.rstrip('\n')))

        self.H = args.H
        self.W = args.W
        self.fov = args.fov
        self.N_views = N_views
        self.use_normal = args.use_normal_data
        self.rays_d_cam = get_ray_direction(self.H, self.fov)[0]

        self.random_sample_rays = args.random_sample_rays
        self.coords = np.arange(0, self.H**2)
        self.sample_ratio = args.sample_ratio # The ratio of total pixels used in this batch
        self.inside_true_ratio = args.inside_true_ratio # The ratio used in this batch inside the mask
        self.outside_true_ratio = args.outside_true_ratio # The ratio used in this batch outside the mask

    def __getitem__(self, index):
        view_ind = random.randrange(0, self.N_views)
        path = os.path.join(self.data_list[index], str(view_ind).zfill(5))
        if not(os.path.exists(path+'_mask.pickle') and os.path.exists(path+'_pose.pickle')):
            print('Data doesnt exist')
            print(path)
            sys.exit()
        camera_info = pickle_load(path+'_pose.pickle')
        pos = camera_info['pos']
        c2w = camera_info['rot'].T

        depth_info = pickle_load(path+'_mask.pickle')
        inverced_depth = depth_info['inverced_depth']
        blur_mask = depth_info['blur_mask']
        if self.use_normal:
            if 'normal_map' not in depth_info.keys():
                print(path)
            normal_map = depth_info['normal_map']
            # normal_mask = depth_info['normal_mask']
            normal_mask = inverced_depth > 0
        else:
            normal_map = False
            normal_mask = False

        if self.random_sample_rays:

            blur_mask = blur_mask.reshape(-1)
            coord_inside_mask = self.coords[blur_mask]
            coord_outside_mask = self.coords[np.logical_not(blur_mask)]

            inside_true_num = int(self.sample_ratio * self.inside_true_ratio * len(coord_inside_mask))
            inside_false_num = len(coord_inside_mask) - inside_true_num
            outside_true_num = int(self.sample_ratio * self.outside_true_ratio * len(coord_outside_mask))

            np.random.shuffle(coord_inside_mask)
            np.random.shuffle(coord_outside_mask)

            blur_mask[coord_inside_mask[:inside_false_num]] = False
            blur_mask[coord_outside_mask[:outside_true_num]] = True

            # Get new mask.
            blur_mask = blur_mask.reshape(self.H, self.W)
            normal_mask = normal_mask * blur_mask

        return index, pos, c2w, self.rays_d_cam, inverced_depth, blur_mask, normal_map, normal_mask

    def __len__(self):
        return len(self.data_list)