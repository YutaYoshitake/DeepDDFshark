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
        self.re_wood = args.re_wood
        self.instance_path_list = []
        with open(instance_list_txt, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                if self.mode=='train':
                    self.instance_path_list.append(
                        os.path.join(data_dir, line.rstrip('\n'))
                        )
                elif self.mode=='val':
                    for view_ind in range(self.N_views//3):
                        self.instance_path_list.append(
                            os.path.join(data_dir, line.rstrip('\n'), f'{str(view_ind+1).zfill(5)}.pickle')
                            )
                    if self.re_wood:
                        self.instance_path_list = ['/home/yyoshitake/works/DeepSDF/project/pixellib_pkl/test_image/test.pickle']
                    # self.instance_path_list = ['/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/val/views16/e31c6c24a8d80ac35692a9640d6947fc/00005.pickle']
                    # self.instance_path_list = self.instance_path_list[:2]

    def __getitem__(self, index):
        # Load data
        if self.mode=='train':
            view_ind = random.randrange(1, self.N_views + 1)
            path = os.path.join(
                self.instance_path_list[index], 
                f'{str(view_ind).zfill(5)}.pickle')
        elif self.mode=='val':
            path = self.instance_path_list[index]
        data_dict = pickle_load(path)

        # frame_rgb_map = data_dict['rgb_map'].transpose(0, 3, 1, 2)
        frame_mask = data_dict['mask']
        frame_distance_map = data_dict['depth_map']
        frame_camera_pos = data_dict['camera_pos']
        frame_camera_rot = data_dict['camera_rot']
        frame_obj_rot = data_dict['obj_rot']
        frame_obj_pos = data_dict['obj_pos']
        frame_obj_scale = data_dict['obj_scale']

        # Preprocessing.
        frame_rgb_map = 0 # self.rgb_transform(frame_rgb_map)
        instance_id = self.instance_path_list[index].split('/')[-1]

        if self.mode=='val':
            splitted_path_list = path.split('/')
            log_path = splitted_path_list[-2]+'/'+splitted_path_list[-1]
            if not(self.re_wood):
                canonical_path = '/'.join(splitted_path_list[:-3]) + '/canoncal/' + splitted_path_list[-2] + '.pickle'
                canonical_data_dict = pickle_load(canonical_path)
                canonical_distance_map = canonical_data_dict['depth_map']
                canonical_camera_pos = canonical_data_dict['camera_pos']
                canonical_camera_rot = canonical_data_dict['camera_rot']
            else:
                canonical_path = '/'.join(splitted_path_list[:-3]) + '/canoncal/' + splitted_path_list[-2] + '.pickle'
                canonical_data_dict = pickle_load(canonical_path)
                canonical_path = 0
                canonical_data_dict = 0
                canonical_distance_map = 0
                canonical_camera_pos = 0
                canonical_camera_rot = 0
        
        if self.mode=='train':
            return frame_mask, frame_distance_map, frame_camera_pos, frame_camera_rot, frame_obj_pos, frame_obj_rot, frame_obj_scale, instance_id
        elif self.mode=='val':
            return frame_mask, frame_distance_map, frame_camera_pos, frame_camera_rot, frame_obj_pos, frame_obj_rot, frame_obj_scale, \
            canonical_distance_map, canonical_camera_pos, canonical_camera_rot, instance_id, log_path

    def __len__(self):
        return len(self.instance_path_list)