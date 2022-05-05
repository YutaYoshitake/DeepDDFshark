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
        instance_list_txt, 
        data_dir, 
        N_views
        ):
    
        self.instance_path_list = []
        with open(instance_list_txt, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                self.instance_path_list.append(os.path.join(data_dir, line.rstrip('\n')))
        self.H = args.H
        self.fov = args.fov
        self.N_views = N_views
        self.rgb_transform = T.Compose([
            T.Resize(self.H),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):

        # Load data
        view_ind = random.randrange(1, self.N_views + 1)
        path = os.path.join(
            self.instance_path_list[index], 
            f'{str(view_ind).zfill(5)}.pickle')
        data_dict = pickle_load(path)

        frame_rgb_map = data_dict['rgb_map'].transpose(0, 3, 1, 2)
        frame_mask = data_dict['mask']
        frame_depth_map = data_dict['depth_map']
        frame_camera_pos = data_dict['camera_pos']
        frame_camera_rot = data_dict['camera_rot']
        frame_obj_rot = data_dict['obj_rot']

        # Preprocessing.
        frame_rgb_map = torch.from_numpy(frame_rgb_map.astype(np.float32)).clone()
        frame_rgb_map = self.rgb_transform(frame_rgb_map)
        instance_id = self.instance_path_list[index].split('/')[-1]

        return frame_rgb_map, frame_mask, frame_depth_map, frame_camera_pos, frame_camera_rot, frame_obj_rot, instance_id

    def __len__(self):
        return len(self.instance_path_list)