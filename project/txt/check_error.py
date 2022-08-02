import os
import pdb
import sys
from turtle import pd
import numpy as np
import random
import pylab
import glob
import math
import shutil
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



label_a = "trans_avg"
label_b = "origin"
pickle_name = 'Rad.pickle'
aaa = pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/log/2022_07_18_12_58_59/log_error.pickle')
bbb = pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/log/2022_07_18_12_59_04/log_error.pickle')

# for key in aaa.keys():
#     fig = pylab.figure()
#     ax = fig.add_subplot(1,1,1)
#     ax.hist([aaa[key], bbb[key]], bins=50, label=[label_a, label_b])
#     if key == 'path':
#         break
#     ax.legend()
#     fig.savefig(f"err_{key}.png")



target_dir = 'list1/ryouhou_ii'
parent_directory_path = f'sample_images/whether_it_works/{target_dir}/'
os.makedirs(parent_directory_path, exist_ok=True)
key = 'red'
criterion = bbb
target = aaa
criterion_min = 10 # criterionで誤差がこれ以上
target_max = 3 # criterionで誤差がこれ以上

min_mask = criterion[key] < 1
max_mask = target[key][min_mask] < 1 # > 10 
path_list = target['path'][min_mask][max_mask]

for path_i in path_list:
    split_dot = path_i.split('.')[0] # '738395f54b301d80b1f5d603f931c1aa/00001'
    split_slash = split_dot.split('/') # '738395f54b301d80b1f5d603f931c1aa', '00001'
    path_i = 'dataset/dugon/moving_camera/train/views64/' + split_dot + '.png'
    target_name = path_i.split('.')[0]
    shutil.copyfile(path_i, parent_directory_path + split_slash[0] + '_' + split_slash[1] + '.png')

# origin = aaa
# target = bbb
# result_path_list = []
# original_value_list = []
# target_value_list = []
# err_key = 'red'

# target_idx_list = np.flipud(np.argsort(origin[err_key])) # 誤差の大きい順で並べる
# origin_threshold = 70
# original_num = 128
# mask = origin[err_key][target_idx_list] > origin_threshold
# target_idx_list = target_idx_list[mask][:original_num] # しきい値以上のインデックスを取得
# # print(origin[err_key][target_idx_list]) # エラーを表示

# threshold = 5
# for idx in target_idx_list:
#     origin_value = origin[err_key][idx]
#     target_value = target[err_key][idx]
#     if target_value < threshold:
#         if target_value < origin_value:
#             result_path = '/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/train/views64/' + origin['path'][idx]
#             result_path_list.append(result_path)
#             print(f'origin : {origin[err_key][idx]:.1f}, target : {target[err_key][idx]:.1f}')
#             # original_value_list.append(origin[err_key][idx])
#             # target_value_list.append(target[err_key][idx])

# # print(original_value_list)
# # print(target_value_list)
# print('###     ' + str(len(result_path_list)) + '     ###')

# pickle_dump(result_path_list, pickle_name)
# import pdb; pdb.set_trace()



# mmm = []
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_13_03_02_05/log.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_13_03_02_11/log.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_13_03_02_14/log.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_13_03_02_18/log.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_13_03_02_21/log.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_13_03_02_24/log.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_13_03_02_27/log.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_13_03_02_31/log.pickle'))

# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_12_21_16_27/log_error.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_12_21_16_32/log_error.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_12_21_16_35/log_error.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_12_21_16_40/log_error.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_12_21_16_43/log_error.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_12_21_16_46/log_error.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_12_21_16_49/log_error.pickle'))
# mmm.append(pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/adams/2022_06_12_21_16_52/log_error.pickle'))

# ooo = {}
# for key in mmm[0].keys():
#     nnn = []
#     for mmm_i in mmm:
#         nnn.append(mmm_i[key])
#     nnn = np.concatenate(nnn)
#     ooo[key] = nnn
# pickle_dump(ooo, f'adam_ind_update.pickle')
# import pdb; pdb.set_trace()



# df = pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/log/2022_06_08_23_07_18/log_error.pickle')
# adam = pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/log/2022_06_07_12_16_00/log_error.pickle')

# for key in df.keys():
#     # x = df[key]
#     # y = adam[key]

#     # fig = pylab.figure()
#     # ax = fig.add_subplot(1,1,1)
#     # ax.set_title(f'{key}')
#     # ax.hist(x, bins=50)
#     # ax.hist(y, bins=50)
#     # fig.savefig(f"err_{key}.png")

#     if key != 'path':
#         for path in enumerate(adam['path'][np.argsort(adam[key])]):
#             with open(f'adam_{key}.txt', 'a') as f:
#                 f.writelines(f'{path[-1]}\n')
        
#         for path in enumerate(df['path'][np.argsort(df[key])]):
#             with open(f'df_{key}.txt', 'a') as f:
#                 f.writelines(f'{path[-1]}\n')
#     import pdb; pdb.set_trace()



# pickle_name = 'adam_vs_deep_kmeans0_test.pickle'
# mmm = pickle_load('/home/yyoshitake/works/DeepSDF/project/adam_vs_deep_kmeans0_test.pickle')
# for idx in range(len(mmm)):
#     mmm[idx] = '/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/train/views64/' + mmm[idx]
# pickle_dump(mmm, pickle_name)