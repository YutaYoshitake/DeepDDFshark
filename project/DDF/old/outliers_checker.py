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
from train_pl import *
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





class DDF_checker_dataset(data.Dataset):
    def __init__(self, args, data_dir, N_views):
    
        self.data_list = []
        with open(args.instance_list_txt, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                self.data_list.append(os.path.join(data_dir, line.rstrip('\n')))

        self.H = args.H
        self.W = args.W
        self.fov = args.fov
        self.N_views = N_views
        self.use_normal = args.use_normal_data
        self.rays_d_cam = get_ray_direction(self.H, self.fov)[0].to(torch.float32)

    def __getitem__(self, index):
        index = 1005
        view_ind_list = np.random.randint(0, self.N_views, 8)
        ins_idx = []
        pos = []
        c2w = []
        rays_d_cam = []
        inverced_depth = []
        for view_ind in view_ind_list:
            path = os.path.join(self.data_list[index], str(view_ind).zfill(5))
            camera_info = pickle_load(path+'_pose.pickle')
            pos.append(camera_info['pos'].astype(np.float32))
            c2w.append(camera_info['rot'].astype(np.float32).T)
            depth_info = pickle_load(path+'_mask.pickle')
            inverced_depth.append(depth_info['inverced_depth'].astype(np.float32))
            ins_idx.append(index)
            rays_d_cam.append(self.rays_d_cam)
        
        ins_idx = np.array(ins_idx)
        pos = np.array(pos)
        c2w = np.array(c2w)
        rays_d_cam = torch.stack(rays_d_cam)
        inverced_depth = np.array(inverced_depth)
        return ins_idx, pos, c2w, rays_d_cam, inverced_depth

    def __len__(self):
        return len(self.data_list)





class DDF_checker(pl.LightningModule):

    def __init__(self, args, ddf):
        super().__init__()

        # Base configs
        self.checker_sample_interval = 2
        self.H = args.H
        self.W = args.W
        self.fov = args.fov
        self.same_instances = args.same_instances
        self.use_world_dir = args.use_world_dir
        if not self.use_world_dir:
            print('Support for world coordinate system only.')
            sys.exit()

        self.vec_lrate = args.vec_lrate
        self.model_lrate = args.model_lrate

        self.canonical_bbox_diagonal = 1.0
        self.use_3d_code = args.use_3d_code
        self.only_latent = args.only_latent
        self.latent_size = args.latent_size
        self.latent_3d_size = args.latent_3d_size

        # Latent vecs
        self.ddf = ddf

        # loss func.
        self.error_log = []
        self.instance_id_log = []



    def test_step(self, batch, batch_idx):

        # Get input
        instance_id, pos, c2w, rays_d_cam, inverced_depth_map = batch

        # Get ray direction
        current_batch_size = instance_id.shape[0]
        frame_num = instance_id.shape[-1]
        instance_id = instance_id.reshape(-1)
        pos = pos.reshape(-1, 3)
        c2w = c2w.reshape(-1, 3, 3)
        rays_d_cam = rays_d_cam.reshape(-1, self.H, self.W, 3)
        inverced_depth_map = inverced_depth_map[..., ::self.checker_sample_interval, ::self.checker_sample_interval]
        rays_o = pos[:, None, None, :].expand(-1, self.H, self.W, -1).detach()
        rays_d_wrd = torch.sum(rays_d_cam[:, :, :, None, :] * c2w[:, None, None, :, :], -1).detach()

        # Get latent code
        input_lat_vec = self.ddf.lat_vecs(instance_id)
        est_inverced_depth = self.ddf.forward(
            rays_o[:, ::self.checker_sample_interval, ::self.checker_sample_interval], 
            rays_d_wrd[:, ::self.checker_sample_interval, ::self.checker_sample_interval], 
            input_lat_vec
            )
        est_inverced_depth = est_inverced_depth.reshape(-1, frame_num, self.H//self.checker_sample_interval, self.W//self.checker_sample_interval)
        ##################################################
        total_map = []
        gt = inverced_depth_map
        est = est_inverced_depth
        for idx, (gt_i, est_i) in enumerate(zip(gt, est)):
            gt_map = torch.cat([gt_i_j for gt_i_j in gt_i], dim=-1)
            est_map = torch.cat([est_i_j for est_i_j in est_i], dim=-1)
            total_map.append(torch.cat([gt_map, est_map], dim=0))
        check_map(torch.cat(total_map, dim=0))
        import pdb; pdb.set_trace()
        ##################################################
        
        # Cal depth loss.
        instance_id = instance_id.reshape(-1, frame_num)
        error = torch.abs(inverced_depth_map-est_inverced_depth).mean(-1).mean(-1).mean(-1)
        for instance_id_i, error_i in zip(instance_id, error):
            self.error_log.append(instance_id_i[0].item())
            self.instance_id_log.append(error_i.item())
        return 0



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

    
    # Create dataloader
    tes_dataset = DDF_checker_dataset(args, args.train_data_dir, args.N_views)
    tes_dataloader = data_utils.DataLoader(tes_dataset, batch_size=args.N_batch, num_workers=args.num_workers, drop_last=False, shuffle=False)

    # Get ckpts path.
    ddf = DDF(args)
    ddf = ddf.load_from_checkpoint(
        checkpoint_path='/home/yyoshitake/works/DeepSDF/project/DDF/lightning_logs/chair/cat_depth_mae_normal_mae_seed0_normal001_lr00001/checkpoints/0000010000.ckpt', 
        args=args)
    ddf.eval()

    trainer = pl.Trainer(
        gpus=args.gpu_num, 
        strategy=DDPPlugin(find_unused_parameters=False), 
        enable_checkpointing = False,
        )
    model = DDF_checker(args, ddf)
    trainer.test(model, tes_dataloader)
    import pdb; pdb.set_trace()
