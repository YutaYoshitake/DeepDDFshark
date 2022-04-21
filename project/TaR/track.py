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

from parser import *
from often_use import *
from dataset import *

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





class test_model(pl.LightningModule):

    def __init__(self, args):
        super().__init__()

        self.fc = nn.Sequential(
                nn.Linear(5, 5), nn.LeakyReLU(0.2)
                )
    
    def forward(self, inp):
        x = self.fc(inp)
        return x





class TaR(pl.LightningModule):

    def __init__(self, args):
        super().__init__()

        # Base configs
        self.H = args.H
        self.fov = args.fov
        self.lr = args.lr

        # Make model
        self.model = test_model(args)

        # log config
        self.save_interval = args.save_interval

        # loss func.
        self.mae = nn.L1Loss()

        # Model info
        self.model_params_dtype = False
        self.model_device = False



    def forward(self):
        return 0



    def training_step(self, batch, batch_idx):
        loss = 0
        import pdb; pdb.set_trace()
        return loss



    def training_epoch_end(self, outputs):

        # Log loss.
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        current_epoch = torch.tensor(self.current_epoch + 1., dtype=avg_loss.dtype)
        self.log_dict({'train/total_loss': avg_loss, "step": current_epoch})

        # Save ckpt.
        if (self.current_epoch + 1) % self.save_interval == 0:
            ckpt_name = str(self.current_epoch + 1).zfill(10) + '.ckpt'
            ckpt_path = os.path.join(self.trainer.log_dir, 'checkpoints', ckpt_name)
            trainer.save_checkpoint(ckpt_path)



    # def validation_step(self, batch, batch_idx):
    #     return {'depth_err': depth_err}



    # def validation_epoch_end(self, outputs):
    #     # Log loss.
    #     avg_depth_err = torch.stack([x['depth_err'] for x in outputs]).mean()
    #     current_epoch = torch.tensor(self.current_epoch + 1., dtype=avg_depth_err.dtype)
    #     self.log_dict({'validation/total_depth_err': avg_depth_err, "step": current_epoch})



    def check_model_info (self):
        self.model_params_dtype = list(self.mlp.parameters())[-1].dtype
        self.model_device = self.device



    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {"params": self.model.parameters()},
        ], lr=self.lr, betas=(0.9, 0.999),)
        return optimizer



if __name__=='__main__':
    # Get args
    args = get_args()
    args.gpu_num = torch.cuda.device_count() # log used gpu num.


    # Set trainer.
    logger = pl.loggers.TensorBoardLogger(
            save_dir=os.getcwd(),
            version=f'{args.expname}_{args.exp_version}',
            name='lightning_logs'
        )
    trainer = pl.Trainer(
        gpus=args.gpu_num, 
        strategy=DDPPlugin(find_unused_parameters=False), 
        logger=logger,
        max_epochs=args.N_epoch, 
        enable_checkpointing = False,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        )
    

    # Save config files.
    os.makedirs(os.path.join('lightning_logs', f'{args.expname}_{args.exp_version}'), exist_ok=True)
    f = os.path.join('lightning_logs', f'{args.expname}_{args.exp_version}', 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join('lightning_logs', f'{args.expname}_{args.exp_version}', 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())


    # Create dataloader
    train_dataset = TaR_dataset(
        args, 
        args.train_instance_list_txt, 
        args.train_data_dir, 
        args.train_N_views
        )
    train_dataloader = data_utils.DataLoader(
        train_dataset, 
        batch_size=args.N_batch, 
        num_workers=args.num_workers, 
        drop_last=False, 
        shuffle=True
        )
    # val_dataset = dataset(args, args.val_data_dir, args.N_val_views)
    # val_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.N_batch, num_workers=args.num_workers, drop_last=False, shuffle=True)

    # Get ckpts path.
    ckpt_dir = os.path.join('lightning_logs', f'{args.expname}_{args.exp_version}', 'checkpoints/*')
    ckpt_path_list = sorted(glob.glob(ckpt_dir))

    # Load ckpt and start training.
    if len(ckpt_path_list) == 0:
        model = TaR(args)
        trainer.fit(
            model=model, 
            train_dataloaders=train_dataloader, 
            # val_dataloaders=val_dataloader, 
            datamodule=None, 
            ckpt_path=None
            )

    elif len(ckpt_path_list) > 0:
        latest_ckpt_path = ckpt_path_list[-1]
        print('\n', f'+++ Reloading from {latest_ckpt_path} +++ \n')
        model = TaR(args)
        trainer.fit(
            model=model, 
            train_dataloaders=train_dataloader, 
            # val_dataloaders=val_dataloader, 
            datamodule=None, 
            ckpt_path=latest_ckpt_path
            )
