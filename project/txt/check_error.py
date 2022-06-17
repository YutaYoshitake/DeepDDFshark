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


aaa = pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/log/2022_06_06_00_01_53/log.pickle')
bbb = pickle_load('txt/experiments/log/2022_06_12_13_30_12/log_error.pickle')
ccc = pickle_load('/home/yyoshitake/works/DeepSDF/project/txt/experiments/total/until/log/2022_06_06_01_01_52/log.pickle')
ddd = pickle_load('txt/experiments/log/2022_06_10_20_26_44/log_error.pickle')
eee = pickle_load('txt/experiments/total/until/adams/adam_ind_update.pickle')
fff = pickle_load('txt/experiments/total/until/adams/adam_averaged_update.pickle')

for key in aaa.keys():
    
    fig = pylab.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(aaa[key], bins=50, label="wd_ind_U")
    ax.hist(bbb[key], bins=50, label="wd_ave_U")
    ax.hist(ccc[key], bins=50, label="wod_ind_U")
    ax.hist(ddd[key], bins=50, label="wod_ave_U")
    ax.hist(eee[key], bins=50, label="adam_ind_U")
    ax.hist(fff[key], bins=50, label="adam_ave_U")
    if key == 'path':
        break
    ax.legend()
    fig.savefig(f"err_{key}.png")

    # print('###############')
    # print(key)
    # print(aaa[key].mean())
    # print(bbb[key].mean())
    # print(ccc[key].mean())
    # print(ddd[key].mean())
    # print(eee[key].mean())
    # print(fff[key].mean())
    # print(sorted(aaa[key])[1040])
    # print(sorted(bbb[key])[1040])
    # print(sorted(ccc[key])[1040])
    # print(sorted(ddd[key])[1040])
    # print(sorted(eee[key])[1040])
    # print(sorted(fff[key])[1040])
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
