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





# for kmeans_idx in ['0', '2']:
#     instance_list_txt = f'instance_lists/kmean/kmeans_list_{kmeans_idx}.txt'

#     instance_list = []
#     with open(instance_list_txt, 'r') as f:
#         lines = f.read().splitlines()
#         for line in lines:
#             instance_list.append(line)

#     random.shuffle(instance_list)






#     txt_path = {}
#     sta_num = {}
#     end_num = {}
#     txt_path['tes'] = f'instance_lists/kmean/kmeans{kmeans_idx}_test_list.txt'
#     txt_path['val'] = f'instance_lists/kmean/kmeans{kmeans_idx}_val_list.txt'
#     txt_path['tra'] = f'instance_lists/kmean/kmeans{kmeans_idx}_train_list.txt'
#     num_tes = 128
#     num_val = 128
#     sta_num['tes'] = 0
#     sta_num['val'] = num_tes
#     sta_num['tra'] = num_tes + num_val
#     end_num['tes'] = num_tes
#     end_num['val'] = num_tes + num_val





#     for key_i in ['tes', 'val', 'tra']:
#         if key_i in {'tes', 'val'}:
#             with open(txt_path[key_i], 'a') as f:
#                 for instance_id in instance_list[sta_num[key_i]:end_num[key_i]]:
#                     f.write(instance_id+'\n')
#         elif key_i == 'tra':
#             with open(txt_path[key_i], 'a') as f:
#                 for instance_id in instance_list[sta_num[key_i]:]:
#                     f.write(instance_id+'\n')


train_ins_list = glob.glob('/disks/local/yyoshitake/DeepSDF/train_data/*/')
train_instances = set([train_ins.split('/')[-2] for train_ins in train_ins_list])
ins_list = glob.glob('/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/val/canonical/*/')
random.shuffle(ins_list)
import pdb; pdb.set_trace()
for ins in ins_list:
    ins = ins.split('/')[-2]
    # if not ins in train_instances:
    #     with open('/home/yyoshitake/works/DeepSDF/project/instance_lists/kmean/total_test_novel.txt', 'a') as f:
    #         f.write(ins+'\n')