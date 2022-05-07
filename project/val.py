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

from ResNet import *
from parser import *
from dataset import *
from often_use import *
from train_dfnet import *
from train_init_net import *
from DDF.train_pl import DDF

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





if __name__=='__main__':
    # Get args
    args = get_args()
    args.gpu_num = torch.cuda.device_count() # log used gpu num.

    # Set trainer.
    trainer = pl.Trainer(
        gpus=args.gpu_num, 
        strategy=DDPPlugin(find_unused_parameters=True), #=False), \
        enable_checkpointing = False,
        )    
    
    # Create dataloader
    val_dataset = TaR_dataset(
        args, 
        args.val_instance_list_txt, 
        args.val_data_dir, 
        args.val_N_views
        )
    val_dataloader = data_utils.DataLoader(
        val_dataset, 
        batch_size=args.N_batch, 
        num_workers=args.num_workers, 
        drop_last=False, 
        shuffle=False
        )

    # Get ddf.
    ddf = DDF(args)
    ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args)
    ddf.eval()

    # Get init net.
    init_net = TaR_init_only(args, ddf)
    init_net = init_net.load_from_checkpoint(
        checkpoint_path='./lightning_logs/DeepTaR/chair/test_initnet_0/checkpoints/0000003200.ckpt', 
        args=args, 
        ddf=ddf
        )
    
    # Get dfnet.
    df_net = TaR(args, ddf)
    df_net = df_net.load_from_checkpoint(
        checkpoint_path= './lightning_logs/DeepTaR/chair/test_dfnet_withx/checkpoints/0000003200.ckpt', 
        args=args, 
        ddf=ddf
        )

    # Val.
    model = init_net.eval()
    model = df_net.eval()
    ckpt_path = './lightning_logs/DeepTaR/chair/test_dfnet_withx/checkpoints/0000003200.ckpt'
    import datetime
    dt_now = datetime.datetime.now()
    time_log = dt_now.strftime('%Y_%m_%d_%H_%M_%S')

    file_name = time_log + '.txt'
    with open(file_name, 'a') as file:
        file.write('time_log : ' + time_log + '\n')
        file.write('ckpt_path : ' + ckpt_path + '\n')
    
    model.test_log_path = file_name
    trainer.test(model, val_dataloader)
    
    # Delite lightning log.
    import shutil
    shutil.rmtree('./lightning_logs/version_0')
