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
from train_init_net import *
from train_dfnet import *
from train_frame_dfnet import *
from DDF.train_pl import DDF

torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False

# seed = 0
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)
# if device=='cuda':
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False





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
    
    # Create dataloader.
    val_dataset = TaR_testset(
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

    # Create ddf.
    ddf = DDF(args)
    ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args)
    ddf.eval()

    # # # Create init net.
    # # init_net = TaR_init_only(args, ddf)
    # # checkpoint_path = './lightning_logs/DeepTaR/chair/test_initnet_0/checkpoints/0000003200.ckpt'
    # # init_net = init_net.load_from_checkpoint(
    # #     checkpoint_path=checkpoint_path, 
    # #     args=args, 
    # #     ddf=ddf
    # #     )
    # # model = init_net.eval()
    
    # # Create dfnet.
    # # if args.test_model=='nomal':
    # #     print('aaa')
    # args.use_gru = False
    # df_net = TaR(args, ddf)
    # checkpoint_path = './lightning_logs/DeepTaR/chair/test_dfnet_withx/checkpoints/0000003200.ckpt'
    # df_net = df_net.load_from_checkpoint(
    #     checkpoint_path=checkpoint_path, 
    #     args=args, 
    #     ddf=ddf
    #     )
    # df_net.eval()
    # model = df_net
    # model.test_mode = 'average'
    # checkpoint_path = checkpoint_path + '---' + model.test_mode + '---single'
    
    # # # Create dfnet.
    # # # elif args.test_model=='frame':
    # # #     print('bbb')
    # # args.use_gru = False
    # # frame_df_net = TaR_frame(args, ddf)
    # # checkpoint_path = './lightning_logs/DeepTaR/chair/test_dfnet_frame/checkpoints/0000003200.ckpt'
    # # frame_df_net = frame_df_net.load_from_checkpoint(
    # #     checkpoint_path=checkpoint_path, 
    # #     args=args, 
    # #     ddf=ddf
    # #     )
    # # frame_df_net.eval()
    # # model = frame_df_net
    # # # checkpoint_path = './lightning_logs/DeepTaR/chair/test_dfnet_framegru/checkpoints/0000003300.ckpt : opt=4 : [4, 2, 2]'
    # # model.test_mode = 'sequence'

    # # Val.
    # model.average_each_results = True
    # model.start_frame_idx = 0
    # model.frame_sequence_num = 3
    # model.half_lambda_max = 8
    # if model.test_mode == 'average':
    #     model.test_optim_num = [5, 5, 5]
    # if model.test_mode == 'sequence':
    #     model.test_optim_num = [5, 3, 2]

    # import datetime
    # dt_now = datetime.datetime.now()
    # time_log = dt_now.strftime('%Y_%m_%d_%H_%M_%S')

    # file_name = time_log + '.txt'
    # ckpt_path = checkpoint_path
    # with open(file_name, 'a') as file:
    #     file.write('time_log : ' + time_log + '\n')
    #     file.write('ckpt_path : ' + ckpt_path + '\n')
    
    # model.test_log_path = file_name
    # trainer.test(model, val_dataloader)
    
    # # # Delite lightning log.
    # # import shutil
    # # trash_log_list = glob.glob('./lightning_logs/version_*')
    # # for trash_log_path in trash_log_list:
    # #     shutil.rmtree(trash_log_path)

    ###########################################################################
    ###########################################################################
    ###########################################################################
    
    # # Create dfnet.
    # args.use_gru = False
    # df_net = TaR(args, ddf)
    # checkpoint_path = './lightning_logs/DeepTaR/chair/test_dfnet_withx/checkpoints/0000003200.ckpt'
    # df_net = df_net.load_from_checkpoint(
    #     checkpoint_path=checkpoint_path, 
    #     args=args, 
    #     ddf=ddf
    #     )
    # df_net.eval()
    # model = df_net
    # model.test_mode = 'average'
    # checkpoint_path = checkpoint_path + '---' + model.test_mode + '---single'

    # # Val.
    # model.average_each_results = True
    # model.start_frame_idx = 0
    # model.frame_sequence_num = 3
    # model.half_lambda_max = 8
    # if model.test_mode == 'average':
    #     model.test_optim_num = [5, 5, 5]
    # if model.test_mode == 'sequence':
    #     model.test_optim_num = [5, 3, 2]

    # import datetime
    # dt_now = datetime.datetime.now()
    # time_log = dt_now.strftime('%Y_%m_%d_%H_%M_%S')

    # file_name = time_log + '.txt'
    # ckpt_path = checkpoint_path
    # with open(file_name, 'a') as file:
    #     file.write('time_log : ' + time_log + '\n')
    #     file.write('ckpt_path : ' + ckpt_path + '\n')
    
    # model.test_log_path = file_name
    # trainer.test(model, val_dataloader)
    

    
    # # Create dfnet.
    # args.use_gru = False
    # frame_df_net = TaR_frame(args, ddf)
    # checkpoint_path = './lightning_logs/DeepTaR/chair/test_dfnet_withx/checkpoints/0000003200.ckpt'
    # frame_df_net = frame_df_net.load_from_checkpoint(
    #     checkpoint_path=checkpoint_path, 
    #     args=args, 
    #     ddf=ddf
    #     )
    # frame_df_net.eval()
    # model = frame_df_net
    # model.test_mode = 'sequence'

    # # Val.
    # model.average_each_results = True
    # model.start_frame_idx = 0
    # model.frame_sequence_num = 3
    # model.half_lambda_max = 8
    # if model.test_mode == 'average':
    #     model.test_optim_num = [5, 5, 5]
    # if model.test_mode == 'sequence':
    #     model.test_optim_num = [5, 3, 2]

    # import datetime
    # dt_now = datetime.datetime.now()
    # time_log = dt_now.strftime('%Y_%m_%d_%H_%M_%S')

    # file_name = time_log + '.txt'
    # ckpt_path = checkpoint_path
    # with open(file_name, 'a') as file:
    #     file.write('time_log : ' + time_log + '\n')
    #     file.write('ckpt_path : ' + ckpt_path + '\n')
    
    # model.test_log_path = file_name
    # trainer.test(model, val_dataloader)
    
    # Create dfnet.
    args.use_gru = True
    frame_df_net = TaR_frame(args, ddf)
    checkpoint_path = './lightning_logs/DeepTaR/chair/test_dfnet_framegru/checkpoints/0000003200.ckpt'
    frame_df_net = frame_df_net.load_from_checkpoint(
        checkpoint_path=checkpoint_path, 
        args=args, 
        ddf=ddf
        )
    frame_df_net.eval()
    model = frame_df_net
    model.test_mode = 'sequence'

    # Val.
    model.average_each_results = True
    model.start_frame_idx = 0
    model.frame_sequence_num = 3
    model.half_lambda_max = 8
    if model.test_mode == 'average':
        model.test_optim_num = [5, 5, 5]
    if model.test_mode == 'sequence':
        model.test_optim_num = [5, 3, 2]

    import datetime
    dt_now = datetime.datetime.now()
    time_log = dt_now.strftime('%Y_%m_%d_%H_%M_%S')

    file_name = time_log + '.txt'
    ckpt_path = checkpoint_path
    with open(file_name, 'a') as file:
        file.write('time_log : ' + time_log + '\n')
        file.write('ckpt_path : ' + ckpt_path + '\n')
    
    model.test_log_path = file_name
    trainer.test(model, val_dataloader)


    ###########################################################################
    ###########################################################################
    ###########################################################################
