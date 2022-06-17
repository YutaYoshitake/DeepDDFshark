# python Kmean.py --config=configs/chair/cat.txt

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

from parser import *
from often_use import *
from train_pl import *

torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False





if __name__=='__main__':
    TOP = 256

    # Get args
    args = get_args()
    args.gpu_num = torch.cuda.device_count() # log used gpu num.

    # Get ddf.
    ddf = DDF(args)
    ddf = ddf.load_from_checkpoint(
        checkpoint_path='/home/yyoshitake/works/DeepSDF/project/DDF/lightning_logs/chair/cat_depth_mae_normal_mae_seed0_normal001_lr00001/checkpoints/0000010000.ckpt', 
        args=args)
    ddf.eval()
    
    lat_vecs = []
    for lat_id in range(ddf.lat_vecs.num_embeddings):
        lat_vecs.append(ddf.lat_vecs(torch.tensor([lat_id], dtype=torch.long).to(ddf.device)))
    lat_vecs = torch.cat(lat_vecs, dim=0)
    lat_vecs = lat_vecs.to('cpu').detach().numpy().copy()

    instance_list = pickle_load('instance_list/instance_list.pickle').tolist()
    kmean_lists = [
                            '/home/yyoshitake/works/DeepSDF/project/DDF/instance_list/kmean/kmeans_list_0.txt', 
                            '/home/yyoshitake/works/DeepSDF/project/DDF/instance_list/kmean/kmeans_list_1.txt', 
                            '/home/yyoshitake/works/DeepSDF/project/DDF/instance_list/kmean/kmeans_list_2.txt', 
                            '/home/yyoshitake/works/DeepSDF/project/DDF/instance_list/kmean/kmeans_list_3.txt', 
                            '/home/yyoshitake/works/DeepSDF/project/DDF/instance_list/kmean/kmeans_list_4.txt', 
                        ]

    for label, kmean_list_path in enumerate(kmean_lists):

        kmean_list = []
        with open(kmean_list_path, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                kmean_list.append(line.rstrip('\n'))
        
        idx = [instance_list.index(kmean_ins) for kmean_ins in  kmean_list]
        kmean_lat_vecs = lat_vecs[np.array(idx)]
        mean_vec = kmean_lat_vecs.mean(0)
        norms = np.linalg.norm(kmean_lat_vecs-mean_vec[None, :], axis=-1)
        top_kmean = np.array(kmean_list)[np.argsort(norms)[:TOP]]
        
        path = f'top_256_kmeans_list_{label}.txt'
        with open(path, 'a') as f:
            for instance_id in top_kmean:
                f.write(instance_id+'\n')
        # import pdb; pdb.set_trace()
