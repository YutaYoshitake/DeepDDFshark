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
import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
import cv2
from PIL import Image

from ResNet import *
from parser import *
from dataset import *
from often_use import *
from train_initnet import *
from train_dfnet import *
from DDF.train_pl import DDF



RESOLUTION = 256
FOV = 49.134
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
if device=='cuda':
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_ray_direction_diff_HW(H, W, F):
    tan_fov_h = 0.5*H/F
    tan_fov_w = 0.5*W/F
    x_coord = np.tile(np.linspace(-tan_fov_w, tan_fov_w, W)[None], (H, 1))
    y_coord = np.tile(np.linspace(-tan_fov_h, tan_fov_h, H)[:, None], (1, W))
    ray_direction = np.stack([x_coord, y_coord, np.ones_like(x_coord)], axis=2)
    return ray_direction

ins = instanceSegmentation()
pkl_path = 'pixellib_pkl/pointrend_resnet50.pkl'
ins.load_model(pkl_path)

#########################
#####  each frames  #####
#########################
depth_path_list = ['pixellib_pkl/test_image/0000001-000000000000.png']
rgb_path_list = ['pixellib_pkl/test_image/0000001-000000000000.jpg']

rgb_map_list = []
mask_list = []
depth_map_list = []
camera_pos_list = []
camera_rot_list = []
obj_pos_list = []
obj_rot_list = []
obj_scale_list = []

for data_id, (depth_path, rgb_path) in enumerate(zip(depth_path_list, rgb_path_list)):
    segmentation_results, image = ins.segmentImage(rgb_path, show_bboxes=False)

    # 最も面積の大きいマスクを取得
    mask_area = [mask_i.sum() for mask_i in segmentation_results['masks'].transpose(2, 0, 1)]
    max_idx = mask_area.index(max(mask_area))

    if segmentation_results['class_names'][max_idx] == 'chair':

        rgb_map = np.array(Image.open(rgb_path))[:, :, :3]
        rgb_map_list.append(rgb_map)
        mask = segmentation_results['masks'][:, :, max_idx]
        mask_list.append(mask)
        depth_map = cv2.imread(depth_path, -1)
        depth_map = depth_map * np.linalg.norm(get_ray_direction_diff_HW(480, 640, 525), axis=-1)
        depth_map_list.append(depth_map.astype(np.float32))

        # # Check Depth.
        # rays_d = get_ray_direction_diff_HW(480, 640, 525)
        # rays_d_norm = np.linalg.norm(rays_d, axis=-1)
        # normalized_rays_d = rays_d / rays_d_norm[..., None]
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # point_1 = (depth_map[..., None] * normalized_rays_d).reshape(-1, 3) 
        # ax.scatter(point_1[::3, 0], point_1[::3, 1], point_1[::3, 2], marker="o", linestyle='None', c='m', s=0.05)
        # ax.view_init(elev=0, azim=-90)
        # fig.savefig("tes_1.png")
        # plt.close()
        # import pdb; pdb.set_trace()

        camera_pos_list.append(0)
        camera_rot_list.append(0)
        obj_pos_list.append(0)
        obj_rot_list.append(0)
        obj_scale_list.append(0)

# Save.
rgb_map_list = np.array(rgb_map_list)
mask_list = np.array(mask_list)
depth_map_list = np.array(depth_map_list)
camera_pos_list = np.array(camera_pos_list)
camera_rot_list = np.array(camera_rot_list)
obj_pos_list = np.array(obj_pos_list)
obj_rot_list = np.array(obj_rot_list)
obj_scale_list = np.array(obj_scale_list)
data_dict = {
    'rgb_map':rgb_map_list, 
    'mask':mask_list, 
    'depth_map':depth_map_list, 
    'camera_pos':camera_pos_list, 
    'camera_rot':camera_rot_list, 
    'obj_pos':obj_pos_list,
    'obj_rot':obj_rot_list,
    'obj_scale':obj_scale_list,
    }
pickle_dump(data_dict, 'test.pickle')
