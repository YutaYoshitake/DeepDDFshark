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
import pl_bolts
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from chamferdist import ChamferDistance
from scipy.spatial.transform import Rotation as R

from ResNet import *
from parser import *
from dataset import *
from often_use import *
from model import *
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



class original_optimizer(pl.LightningModule):

    def __init__(self, args, ddf):
        super().__init__()

        # Base configs
        self.model_mode = args.model_mode
        self.fov = args.fov
        self.input_H = args.input_H
        self.input_W = args.input_W
        self.x_coord = torch.arange(0, self.input_W)[None, :].expand(self.input_H, -1)
        self.y_coord = torch.arange(0, self.input_H)[:, None].expand(-1, self.input_W)
        self.image_coord = torch.stack([self.y_coord, self.x_coord], dim=-1) # [H, W, (Y and X)]
        fov = torch.deg2rad(torch.tensor(self.fov, dtype=torch.float))
        self.image_lengs = 2 * torch.tan(fov*.5)
        self.ddf_H = 256
        self.lr = args.lr
        self.rays_d_cam = get_ray_direction(self.ddf_H, self.fov)
        self.optim_num = args.optim_num
        self.itr_frame_num = args.itr_frame_num
        self.save_interval = args.save_interval
        self.ddf_instance_list = []
        with open(args.ddf_instance_list_txt, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                self.ddf_instance_list.append(line.rstrip('\n'))
        self.train_instance_list = []
        with open(args.train_instance_list_txt, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                self.train_instance_list.append(line.rstrip('\n'))
        self.rand_P_range = args.rand_P_range # [-self.rand_P_range, self.rand_P_range)で一様サンプル
        self.rand_S_range = args.rand_S_range # [1.-self.rand_S_range, 1.+self.rand_S_range)で一様サンプル
        self.rand_R_range = args.rand_R_range * torch.pi # .25 * torch.pi # [-self.rand_R_range, self.rand_R_range)で一様サンプル
        self.random_axis_num = 1024
        self.random_axis_list = torch.from_numpy(sample_fibonacci_views(self.random_axis_num).astype(np.float32)).clone()
        self.rand_Z_sigma = args.rand_Z_sigma
        self.train_instance_ids = [self.ddf_instance_list.index(instance_i) for instance_i in self.train_instance_list]
        self.rand_Z_center = ddf.lat_vecs(torch.tensor(self.train_instance_ids, device=ddf.device)).mean(0).clone().detach()
        self.optim_mode = args.optim_mode
        self.init_mode = args.init_mode

        # Make model
        self.input_type = args.input_type
        if self.input_type == 'depth':
            self.in_channel = 5
            self.output_diff_coordinate = args.output_diff_coordinate
        elif self.input_type == 'osmap':
            self.in_channel = 11
            self.output_diff_coordinate = args.output_diff_coordinate
        self.optimizer_type = args.optimizer_type
        if self.optimizer_type == 'optimize_former':
            self.output_diff_coordinate = 'obj'
            self.positional_encoding_mode = args.positional_encoding_mode
            self.df_net = optimize_former(
                            input_type=self.input_type, 
                            num_encoder_layers = args.num_encoder_layers, 
                            positional_encoding_mode=self.positional_encoding_mode, 
                            integration_mode = args.integration_mode, 
                            split_into_patch = args.split_into_patch, 
                            encoder_norm_type = args.encoder_norm_type, 
                            reset_transformer_params = args.reset_transformer_params, )

        # ??.
        self.lr_log = []



    def training_step(self, batch, batch_idx):

        inp = torch.ones(512+3, requires_grad=True).to(self.df_net.device)
        x = self.df_net.fc_pos(inp)
        loss = torch.norm(x)

        self.lr_log.append(self.trainer.optimizers[0].param_groups[0]['lr'])
        sch = self.lr_schedulers()

        sch.step()
        return loss



    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {"params": self.df_net.parameters()},
        ], lr=self.lr, betas=(0.9, 0.98), eps = 1.0e-9)
        scheduler = WarmupScheduler(optimizer, warmup_steps=4000)
        return [optimizer, ], [scheduler, ]
        # ], lr=self.lr, betas=(0.9, 0.98), eps = 1.0e-9)
        # return optimizer





if __name__=='__main__':
    # Get args
    args = get_args()
    args.gpu_num = torch.cuda.device_count() # log used gpu num.
    args.check_val_every_n_epoch = args.save_interval
    val_model_name = args.expname.split('/')[-1] + '_' + args.exp_version
    if args.model_ckpt_path=='non':
        # args.model_ckpt_path = f'lightning_logs/DeepTaR/trans/until0802/{val_model_name}/checkpoints/{str(args.val_model_epoch).zfill(10)}.ckpt'
        args.model_ckpt_path = f'lightning_logs/DeepTaR/chair/{val_model_name}/checkpoints/{str(args.val_model_epoch).zfill(10)}.ckpt'
    if args.initnet_ckpt_path=='non':
        args.initnet_ckpt_path = f'lightning_logs/DeepTaR/chair/{args.init_net_name}/checkpoints/{str(args.init_net_epoch).zfill(10)}.ckpt'


    # Set trainer.
    logger = pl.loggers.TensorBoardLogger(
        save_dir=os.getcwd(), 
        version=f'val_trash', 
        name='lightning_logs'
        )
    trainer = pl.Trainer(
        gpus=args.gpu_num, 
        strategy=DDPPlugin(find_unused_parameters=True), #=False), 
        logger=logger,
        max_epochs=3, 
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        )

    # Create dataloader
    train_dataset = TaR_dataset(
        args, 
        'train', 
        args.train_instance_list_txt, 
        args.train_data_dir, 
        args.train_N_views, 
        )
    train_dataloader = data_utils.DataLoader(
        train_dataset, 
        batch_size=args.N_batch, 
        num_workers=args.num_workers, 
        drop_last=False, 
        shuffle=True, 
        )
    
    # Set models and Start training.
    ddf = DDF(args)
    ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args)
    ddf.eval()

    # Get model.
    # ckpt_path = 'lightning_logs/DeepTaR/chair/dfnet_list0_randR05_origin_after_date0721/checkpoints/0000000700.ckpt'
    ckpt_path = None
    model = original_optimizer(args, ddf)
    if args.val_model_epoch > 0:
        ckpt_path = args.model_ckpt_path
    else:
        ckpt_path = None
    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader, 
        ckpt_path=ckpt_path, 
        val_dataloaders=None, 
        datamodule=None, 
        )

    

    lr_log = np.array(model.lr_log)
    steps = np.arange(1, lr_log.shape[0]+1)

    # def lr_lambda(step_num):
    #         d_model = 512
    #         warmup_steps = 4000

    #         step_num = step_num + 1
    #         return d_model**(-0.5) * min(step_num**(-0.5), step_num*(warmup_steps**(-1.5)))
    
    # steps = np.arange(1, 100001)
    
    # lr_log = []
    # for step in steps:
    #     lr_log.append(lr_lambda(step))
    # lr_log = np.array(lr_log)

    # d_model = 512
    # warmup_steps = 4000
    # lr_log = (d_model**(-0.5)) * np.stack([steps**(-0.5), steps*(warmup_steps**(-1.5))]).min(axis=0)

    x = steps
    y = lr_log
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y)
    ax.set_xlabel('steps')
    ax.set_ylabel('lr')
    fig.savefig('lr.png', dpi=300)
