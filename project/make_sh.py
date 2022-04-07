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





with open('test_set_multi35_instance_list.txt', mode ='rt', encoding='utf-8') as f:
    read_data = f.readlines()
read_data = [read_data_i.split('\n')[0] for read_data_i in read_data]

cnt = 0
index = 0
for instance_name in read_data:
    # path = 'cp' + str(index).zfill(3) + '.sh'
    path = 'cp.sh'
    with open(path, 'a') as f:
        # f.write('mkdir /disks/local/yyoshitake/DeepSDF/train_data/' + instance_name +'/\n')
        # f.write('cp -v ~/works/make_depth_image/project/tmp/03001627/' + instance_name + '/000*_pose.pickle /disks/local/yyoshitake/DeepSDF/train_data/' + instance_name +'/\n')
        # f.write('cp -v ~/works/make_depth_image/project/tmp/03001627/' + instance_name + '/000*_mask.pickle /disks/local/yyoshitake/DeepSDF/train_data/' + instance_name +'/\n')
        # f.write('cp -v ~/works/make_depth_image/project/tmp/03001627/' + instance_name + '/001*_pose.pickle /disks/local/yyoshitake/DeepSDF/train_data/' + instance_name +'/\n')
        # f.write('cp -v ~/works/make_depth_image/project/tmp/03001627/' + instance_name + '/001*_mask.pickle /disks/local/yyoshitake/DeepSDF/train_data/' + instance_name +'/\n')
        # if len(glob.glob(f'/home/yyoshitake/works/DeepSDF/project/dataset/dugon/train_set_multi35/{instance_name}/*')) < 500:
        #     f.write(f'scp /media/yyoshitake/HDD/ShapeNetCore_RenderedImages/test_set_multi35/{instance_name}/000* yyoshitake@barracuda:/d/home/yyoshitake/works/DeepSDF/project/dataset/dugon/train_set_multi35/{instance_name}/' + '\n')
        #     f.write(f'scp /media/yyoshitake/HDD/ShapeNetCore_RenderedImages/test_set_multi35/{instance_name}/001* yyoshitake@barracuda:/d/home/yyoshitake/works/DeepSDF/project/dataset/dugon/train_set_multi35/{instance_name}/' + '/\n')
        #     f.write(f'scp /media/yyoshitake/HDD/ShapeNetCore_RenderedImages/test_set_multi35/{instance_name}/002* yyoshitake@barracuda:/d/home/yyoshitake/works/DeepSDF/project/dataset/dugon/train_set_multi35/{instance_name}/' + '/\n')
        print(len(glob.glob(f'/home/yyoshitake/works/DeepSDF/project/dataset/dugon/validation_set_multi35/{instance_name}/*')))
    #     cnt += 1
    # if cnt > 1101:
    #     cnt = 0
    #     index += 1

    # Change file number.
    # file_list = glob.glob(f'/home/yyoshitake/works/DeepSDF/project/dataset/dugon/validation_set_multi35/{instance_name}/015*_*')
    # for file_name in file_list:
    #     file_path_info = re.split('[/]', file_name)
    #     file_extension = re.split('[_]', file_path_info[-1])[-1]
    #     file_index = re.split('[_]', file_path_info[-1])[0]
    #     file_path_info[-1] = str(int(file_index) - 1500).zfill(5) + '_' + file_extension
    #     new_file_name = '/'.join(file_path_info)
    #     os.rename(file_name, new_file_name)





# # Check lacked file.
# bbb = sorted(glob.glob('/home/yyoshitake/works/DeepSDF/project/dataset/dugon/train_set_multi35/1ab8a3b55c14a7b27eaeab1f0c9120b7/*pose.ckpt'))
# cnt = 0
# for globed in bbb:
#     if int(re.sub(r"\D", "", globed.split('/')[-1])) != cnt:
#         print(f'{cnt}\n')
#         cnt = int(re.sub(r"\D", "", globed.split('/')[-1]))
#     cnt += 1