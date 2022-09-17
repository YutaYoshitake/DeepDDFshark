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
import torchvision.transforms as T
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from chamferdist import ChamferDistance
from scipy.spatial.transform import Rotation as R

from parser import *
from often_use import *

torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
if device=='cuda':
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





class TaR_dataset(data.Dataset):
    def __init__(
        self, 
        args, 
        mode, 
        instance_list_txt, 
        data_dir, 
        N_views, 
        ):

        self.mode = mode
        self.N_views = N_views
        self.instance_path_list = []
        with open(instance_list_txt, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                if self.mode=='train':
                    for view_ind in range(self.N_views):
                        self.instance_path_list.append(
                            os.path.join(data_dir, line.rstrip('\n'), f'{str(view_ind+1).zfill(5)}.pickle')
                            )
                elif self.mode=='val':
                    for view_ind in range(self.N_views):
                        self.instance_path_list.append(
                            os.path.join(data_dir, line.rstrip('\n'), f'{str(view_ind+1).zfill(5)}.pickle')
                            )
        self.rand_idx_list = 'non'
        self.canonical_path = f'dataset/dugon/moving_camera/train/canonical/resolution{str(args.ddf_H_W_during_dfnet)}'
        # ###############################################################
        # self.rand_idx_list = 'hi'
        # pickle_path = '/home/yyoshitake/works/DeepSDF/project/txt/experiments/log/2022_08_28_16_22_22/log_error.pickle'
        # targets = pickle_load(pickle_path)
        # self.instance_path_list = targets['path']
        # self.instance_path_list = [
        #     '/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/train/kmean0_randn/'+instance_path
        #      for instance_path in self.instance_path_list]
        # self.rand_P_seed = targets['rand_P_seed']
        # self.rand_S_seed = targets['rand_S_seed']
        # self.randn_theta_seed = targets['randn_theta_seed']
        # self.randn_axis_idx = targets['randn_axis_idx']
        # ###############################################################
        # self.instance_path_list = self.instance_path_list[:16]

    def __getitem__(self, index):
        if self.mode=='train':
            path = self.instance_path_list[index]
        elif self.mode=='val':
            path = self.instance_path_list[index]
        instance_id = self.instance_path_list[index].split('/')[-2] # [-1]

        data_dict = pickle_load(path)
        frame_mask = data_dict['mask']
        frame_distance_map = data_dict['depth_map'].astype(np.float32)
        frame_camera_pos = data_dict['camera_pos'].astype(np.float32)
        frame_camera_rot = data_dict['camera_rot'].astype(np.float32)
        frame_obj_pos = data_dict['obj_pos'].astype(np.float32)
        frame_obj_rot = data_dict['obj_rot'].astype(np.float32)
        frame_obj_scale = np.squeeze(data_dict['obj_scale'].astype(np.float32))

        if self.mode=='val':
            log_path = '/'.join(path.split('/')[-2:])
            canonical_path = os.path.join(self.canonical_path, instance_id + '.pickle')
            canonical_data_dict = pickle_load(canonical_path)
            canonical_distance_map = canonical_data_dict['depth_map'].astype(np.float32)
            canonical_camera_pos = canonical_data_dict['camera_pos'].astype(np.float32)
            canonical_camera_rot = canonical_data_dict['camera_rot'].astype(np.float32)

            if self.rand_idx_list != 'non':
                rand_P_seed = self.rand_P_seed[index]
                rand_S_seed = self.rand_S_seed[index]
                randn_theta_seed = self.randn_theta_seed[index]
                randn_axis_idx = self.randn_axis_idx[index]
            else:
                rand_P_seed = 'non'
                rand_S_seed = 'non'
                randn_theta_seed = 'non'
                randn_axis_idx = 'non'
        
        if self.mode=='train':
            return frame_mask, frame_distance_map, frame_camera_pos, frame_camera_rot, frame_obj_pos, frame_obj_rot, frame_obj_scale, instance_id
        elif self.mode=='val':
            return frame_mask, frame_distance_map, frame_camera_pos, frame_camera_rot, frame_obj_pos, frame_obj_rot, frame_obj_scale, \
            canonical_distance_map, canonical_camera_pos, canonical_camera_rot, instance_id, log_path, rand_P_seed, rand_S_seed, randn_theta_seed, randn_axis_idx

    def __len__(self):
        return len(self.instance_path_list)
