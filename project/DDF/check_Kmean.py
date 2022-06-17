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
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import shutil

from parser import *
from often_use import *
from train_pl import *

torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False





if __name__=='__main__':
    #########################
    #####               #####
    #########################
    # instance_list = []
    # with open('instance_list.txt', 'r') as f:
    #     lines = f.read().splitlines()
    #     for line in lines:
    #         instance_list.append(line.rstrip('\n'))
    # instance_list = np.array(instance_list)
    # pickle_dump(instance_list, 'instance_list.pickle')

    # error_dict = pickle_load('dep_err.pickle')
    # fig = pylab.figure()
    # ax = fig.add_subplot(1,1,1)
    # ax.hist(error_dict['error'], bins=50)
    # ax.legend()
    # fig.savefig(f"ddf_dep_err.png")

    # # outliers_idx = np.array(error_dict['ins_id'])[np.argsort(error_dict['error'])[2905:]]
    # # outliers_instance_list = instance_list[outliers_idx]
    # # with open('outliers_instance.txt', 'a') as f:
    # #     for instance_id in outliers_instance_list:
    # #         f.write(instance_id+'\n')

    # outliers_instance_list = []
    # with open('outliers_instance.txt', 'r') as f:
    #     lines = f.read().splitlines()
    #     for line in lines:
    #         outliers_instance_list.append(line.rstrip('\n'))
    # pickle_dump(outliers_instance_list, 'outliers_list.pickle')
    # import pdb; pdb.set_trace()

    # outliers_instance_list = pickle_load('outliers.pickle')
    # print(len(outliers_instance_list))

    # split_dir_path = f'/home/yyoshitake/works/DeepSDF/project/dataset/dugon/test_views/Kmeans/outliers/'
    # if not os.path.isdir(split_dir_path):
    #     os.mkdir(split_dir_path)
    # else:
    #     print('already exist!!')
    #     sys.exit()
    # for cnt, instance_idx in enumerate(outliers_instance_list):
    #     original = f'/home/yyoshitake/works/DeepSDF/project/dataset/dugon/test_views/ins3000_test_views/{instance_idx}/rgb_00000.png'
    #     target = split_dir_path + f'/{instance_idx}.png'
    #     shutil.copyfile(original, target)

    # instance_list = [ins for ins in instance_list if not ins in outliers_instance_list]
    # split_dir_path = f'/home/yyoshitake/works/DeepSDF/project/dataset/dugon/test_views/Kmeans/inliers/'
    # if not os.path.isdir(split_dir_path):
    #     os.mkdir(split_dir_path)
    # else:
    #     print('already exist!!')
    #     sys.exit()
    # for cnt, instance_idx in enumerate(instance_list):
    #     original = f'/home/yyoshitake/works/DeepSDF/project/dataset/dugon/test_views/ins3000_test_views/{instance_idx}/rgb_00000.png'
    #     target = split_dir_path + f'/{instance_idx}.png'
    #     shutil.copyfile(original, target)



    #########################
    #####               #####
    #########################
    # outliers_instance_list = []
    # with open('outliers_instance.txt', 'r') as f:
    #     lines = f.read().splitlines()
    #     for line in lines:
    #         outliers_instance_list.append(line.rstrip('\n'))
    # instance_path_list = [
    #                         'raw_kmeans_0.txt', 
    #                         'raw_kmeans_1.txt', 
    #                         'raw_kmeans_2.txt', 
    #                         'raw_kmeans_3.txt', 
    #                         'raw_kmeans_4.txt', 
    #                     ]
    # refined_list = [
    #                         'kmeans_list_0.pickle', 
    #                         'kmeans_list_1.pickle', 
    #                         'kmeans_list_2.pickle', 
    #                         'kmeans_list_3.pickle', 
    #                         'kmeans_list_4.pickle', 
    #                     ]
    
    # base_dir_path = '/home/yyoshitake/works/DeepSDF/project/dataset/dugon/test_views/Kmeans'
    # if not os.path.isdir(base_dir_path):
    #     os.mkdir(base_dir_path)

    # for instance_path, refined in zip(instance_path_list, refined_list):
    #     instance_list = []
    #     with open(instance_path, 'r') as f:
    #         lines = f.read().splitlines()
    #         for line in lines:
    #             instance_list.append(line.rstrip('\n'))
    #     instance_list = np.array(instance_list)

    #     refined_instance_list = [ins for ins in instance_list if not ins in outliers_instance_list]

    #     pickle_dump(refined_instance_list, refined)



    #########################
    #####               #####
    #########################
    split_dir_path = f'/home/yyoshitake/works/DeepSDF/project/dataset/dugon/test_views/Kmeans'
    if not os.path.isdir(split_dir_path):
        os.mkdir(split_dir_path)
    result_list = [
                            'top_256_kmeans_list_0.txt', 
                            'top_256_kmeans_list_1.txt', 
                            'top_256_kmeans_list_2.txt', 
                            'top_256_kmeans_list_3.txt', 
                            'top_256_kmeans_list_4.txt', 
                        ]
    for instance_path in result_list:
                
        instance_list = []
        with open(instance_path, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                instance_list.append(line.rstrip('\n'))
        
        split_name = instance_path.split('.')[0]
        split_dir_path = f'/home/yyoshitake/works/DeepSDF/project/dataset/dugon/test_views/Kmeans/{split_name}'
        if not os.path.isdir(split_dir_path):
            os.mkdir(split_dir_path)
        else:
            print('already exist!!')
            sys.exit()

        for cnt, instance_idx in enumerate(instance_list):
            original = f'/home/yyoshitake/works/DeepSDF/project/dataset/dugon/test_views/ins3000_test_views/{instance_idx}/rgb_00000.png'
            target = split_dir_path + f'/{instance_idx}.png'
            shutil.copyfile(original, target)
            # if cnt + 1 >= 100: break