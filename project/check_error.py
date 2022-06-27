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



label_a = "progressive"
label_b = "original"
pickle_name = 'list0.pickle'
aaa = pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/log/exp0626/2022_06_26_05_56_45/log_error.pickle')
bbb = pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/log/exp0626/2022_06_26_06_14_15/log_error.pickle')
pickle_name = 'list1.pickle'
aaa = pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/log/exp0626/2022_06_26_06_05_45/log_error.pickle')
bbb = pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/log/exp0626/2022_06_26_06_24_32/log_error.pickle')
pickle_name = 'list2.pickle'
aaa = pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/log/exp0626/2022_06_26_00_37_40/log_error.pickle')
bbb = pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/log/exp0626/2022_06_26_00_37_56/log_error.pickle')
# pickle_name = 'list3.pickle'
# aaa = pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/log/exp0626/2022_06_26_05_37_40/log_error.pickle')
# bbb = pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/log/exp0626/2022_06_26_05_53_36/log_error.pickle')
# pickle_name = 'list4.pickle'
# aaa = pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/log/exp0626/2022_06_26_05_47_02/log_error.pickle')
# bbb = pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/log/exp0626/2022_06_26_06_04_12/log_error.pickle')

# for key in aaa.keys():
#     fig = pylab.figure()
#     ax = fig.add_subplot(1,1,1)
#     ax.hist([aaa[key], bbb[key]], bins=50, label=[label_a, label_b])
#     if key == 'path':
#         break
#     ax.legend()
#     fig.savefig(f"err_{key}.png")

origin = aaa
target = bbb
result_path_list = []
original_value_list = []
target_value_list = []
err_key = 'red'

target_idx_list = np.flipud(np.argsort(origin[err_key])) # 誤差の大きい順で並べる
origin_threshold = 70
original_num = 128
mask = origin[err_key][target_idx_list] > origin_threshold
target_idx_list = target_idx_list[mask][:original_num] # しきい値以上のインデックスを取得
# print(origin[err_key][target_idx_list]) # エラーを表示

threshold = 30
for idx in target_idx_list:
    origin_value = origin[err_key][idx]
    target_value = target[err_key][idx]
    if target_value < threshold:
        if target_value < origin_value:
            result_path = '/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/train/views64/' + origin['path'][idx]
            result_path_list.append(result_path)
            print(f'origin : {origin[err_key][idx]:.1f}, target : {target[err_key][idx]:.1f}')
            # original_value_list.append(origin[err_key][idx])
            # target_value_list.append(target[err_key][idx])

# print(original_value_list)
# print(target_value_list)
print('###     ' + str(len(result_path_list)) + '     ###')

pickle_dump(result_path_list, pickle_name)
import pdb; pdb.set_trace()



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
