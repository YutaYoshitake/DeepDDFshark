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
                # if self.mode=='train':
                #     self.instance_path_list.append(
                #         os.path.join(data_dir, line.rstrip('\n'))
                #         )
                if self.mode=='train':
                    for view_ind in range(self.N_views):
                        self.instance_path_list.append(
                            os.path.join(data_dir, line.rstrip('\n'), f'{str(view_ind+1).zfill(5)}.pickle')
                            ) # '/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/val/views16/{???}/00001.pickle' 
                elif self.mode=='val':
                    for view_ind in range(self.N_views):
                        self.instance_path_list.append(
                            os.path.join(data_dir, line.rstrip('\n'), f'{str(view_ind+1).zfill(5)}.pickle')
                            ) # '/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/val/views16/{???}/00001.pickle' 
        # self.instance_path_list = pickle_load('/home/yyoshitake/works/DeepSDF/project/adam_vs_deep_kmeans0_test.pickle')
        # self.instance_path_list = self.instance_path_list[:3]
        # self.instance_path_list = ['/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/train/views64/c967b1e07ef7fc0bebc740fe800c0367/00007.pickle',
        #                            '/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/train/views64/2249c62788a52c61613f0dbd986ed6f8/00006.pickle',
        #                            '/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/train/views64/c7786437606ac263b04cb542e2c50eb4/00008.pickle']


    def __getitem__(self, index):
        # Load data
        # if self.mode=='train':
        #     view_ind = random.randrange(1, self.N_views + 1)
        #     path = os.path.join(
        #         self.instance_path_list[index], 
        #         f'{str(view_ind).zfill(5)}.pickle')
        if self.mode=='train':
            path = self.instance_path_list[index]
        elif self.mode=='val':
            path = self.instance_path_list[index]

        data_dict = pickle_load(path)
        # frame_rgb_map = data_dict['rgb_map'].transpose(0, 3, 1, 2)
        frame_mask = data_dict['mask']
        frame_distance_map = data_dict['depth_map'].astype(np.float32)
        frame_camera_pos = data_dict['camera_pos'].astype(np.float32)
        frame_camera_rot = data_dict['camera_rot'].astype(np.float32)
        frame_obj_pos = data_dict['obj_pos'].astype(np.float32)
        frame_obj_rot = data_dict['obj_rot'].astype(np.float32)
        frame_obj_scale = data_dict['obj_scale'].astype(np.float32)

        # splitted_path_list = path.split('/')
        # log_path = splitted_path_list[-2]+'/'+splitted_path_list[-1]
        # canonical_path = '/'.join(splitted_path_list[:-3]) + '/canonical/' + splitted_path_list[-2] + '.pickle'
        # canonical_data_dict = pickle_load(canonical_path)
        # frame_mask = canonical_data_dict['mask']
        # frame_distance_map = canonical_data_dict['depth_map'].astype(np.float32)
        # frame_camera_pos = canonical_data_dict['camera_pos'].astype(np.float32)
        # frame_camera_rot = canonical_data_dict['camera_rot'].astype(np.float32)
        # frame_obj_pos = canonical_data_dict['obj_pos'].astype(np.float32)
        # frame_obj_rot = canonical_data_dict['obj_rot'].astype(np.float32)
        # frame_obj_scale = canonical_data_dict['obj_scale'].astype(np.float32)

        # Preprocessing.
        frame_rgb_map = 0 # self.rgb_transform(frame_rgb_map)
        instance_id = self.instance_path_list[index].split('/')[-2] # [-1]

        if self.mode=='val':
            splitted_path_list = path.split('/')
            log_path = splitted_path_list[-2]+'/'+splitted_path_list[-1]
            canonical_path = '/'.join(splitted_path_list[:-3]) + '/canonical/' + splitted_path_list[-2] + '.pickle'
            canonical_data_dict = pickle_load(canonical_path)
            canonical_distance_map = canonical_data_dict['depth_map'].astype(np.float32)
            canonical_camera_pos = canonical_data_dict['camera_pos'].astype(np.float32)
            canonical_camera_rot = canonical_data_dict['camera_rot'].astype(np.float32)
        
        if self.mode=='train':
            return frame_mask, frame_distance_map, frame_camera_pos, frame_camera_rot, frame_obj_pos, frame_obj_rot, frame_obj_scale, instance_id
        elif self.mode=='val':
            return frame_mask, frame_distance_map, frame_camera_pos, frame_camera_rot, frame_obj_pos, frame_obj_rot, frame_obj_scale, \
            canonical_distance_map, canonical_camera_pos, canonical_camera_rot, instance_id, log_path

    def __len__(self):
        return len(self.instance_path_list)





# if __name__=='__main__':

#     size = 256
#     fov = 49.134
#     rays_d_cam = get_ray_direction(size, fov).to('cpu').detach().numpy().copy()

#     data_path = '/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/train/views64/23e726da58f115f69b9f2eb77f5e247e/00001.pickle'
#     data_dict = pickle_load(data_path)
#     frame_mask = data_dict['mask']
#     frame_distance_map = data_dict['depth_map'].astype(np.float32)
#     frame_camera_pos = data_dict['camera_pos'].astype(np.float32)
#     frame_camera_rot = data_dict['camera_rot'].astype(np.float32)
#     frame_obj_pos = data_dict['obj_pos'].astype(np.float32)
#     frame_obj_rot = data_dict['obj_rot'].astype(np.float32)
#     frame_obj_scale = data_dict['obj_scale'].astype(np.float32)


    
#     depth_map_list = frame_distance_map
#     mask_list = frame_mask
#     camera_rot_list = frame_camera_rot
#     camera_pos_list = frame_camera_pos
#     obj_rot_list = frame_obj_rot
#     obj_pos_list = frame_obj_pos
#     normalized_rays_d = rays_d_cam[0]
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     for ind_1 in range(len(depth_map_list)):
#         point_1 = (depth_map_list[ind_1][..., None] * normalized_rays_d)[mask_list[ind_1]].reshape(-1, 3) 
#         point_1 = np.sum(point_1[:, None, :]*camera_rot_list[ind_1].T, -1) + camera_pos_list[ind_1]
#         point_1 = np.sum(point_1[:, None, :]*obj_rot_list[ind_1].T, -1) + obj_pos_list[ind_1]
#         ax.scatter(point_1[::3, 0], point_1[::3, 1], point_1[::3, 2], marker="o", linestyle='None', c='m', s=0.05)
#     # plt.show()
#     # ax.view_init(elev=0, azim=90)
#     fig.savefig("tes.png")
#     # print(rgb_path)
#     # import pdb; pdb.set_trace()
#     plt.close()




#     ind_1 = 0
#     point_1 = (depth_map_list[ind_1][..., None] * normalized_rays_d)
#     frame_surface_pos_cam_map = np.sum(point_1[..., None, :]*camera_rot_list[ind_1].T, -1) + camera_pos_list[ind_1]
#     frame_surface_pos_obj_map = np.sum(point_1[..., None, :]*obj_rot_list[ind_1].T, -1) + obj_pos_list[ind_1]
#     frame_surface_pos_obj_map -= frame_surface_pos_obj_map.min()
#     frame_surface_pos_obj_map /= frame_surface_pos_obj_map.max()
#     frame_surface_pos_obj_map[np.logical_not(mask_list[0])] = .0
#     import pdb; pdb.set_trace()
