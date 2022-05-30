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

from pyrsistent import pdeque
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
from train_initnet import *
from DDF.train_pl import DDF

torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False

seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
if device=='cuda':
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





def get_depth_map_from_axis(
        H, 
        axis_green, 
        axis_red,
        cam_pos_wrd, 
        obj_pos_wrd, 
        rays_d_cam, 
        w2c, 
        input_lat_vec, 
        ddf, 
        with_invdepth_map = False, 
    ):
    # Get rotation matrix.
    axis_blue = torch.cross(axis_red, axis_green, dim=-1)
    axis_blue = F.normalize(axis_blue, dim=-1)
    orthogonal_axis_red = torch.cross(axis_green, axis_blue, dim=-1)
    est_o2c = torch.stack([orthogonal_axis_red, axis_green, axis_blue], dim=-1)

    # Get rays direction.
    rays_d_obj = torch.sum(rays_d_cam[..., None, :]*est_o2c[..., None, None, :, :].permute(0, 1, 2, 4, 3), -1)

    # Get rays origin.
    est_o2w = torch.bmm(w2c.to(est_o2c.dtype).permute(0, 2, 1), est_o2c)
    cam_pos_obj = torch.sum((cam_pos_wrd - obj_pos_wrd)[..., None, :] * est_o2w.permute(0, 2, 1), dim=-1) # = -obj_T_c2o
    rays_o_obj = cam_pos_obj[:, None, None, :].expand(-1, H, H, -1)

    # Get rays inputs.
    rays_d = rays_d_obj
    rays_o = rays_o_obj

    # Estimating.
    est_invdepth_map = ddf.forward(rays_o, rays_d, input_lat_vec)
    est_mask = est_invdepth_map > .5
    est_depth_map = torch.zeros_like(est_invdepth_map)
    est_depth_map[est_mask] = 1. / est_invdepth_map[est_mask]

    if with_invdepth_map:
        return est_invdepth_map, est_mask, est_depth_map
    else:
        return est_mask, est_depth_map





class TaR(pl.LightningModule):

    def __init__(self, args, ddf, init_net=False):
        super().__init__()

        # Base configs
        self.dynamic = args.dynamic
        self.H = args.H
        self.fov = args.fov
        self.lr = args.lr
        self.rays_d_cam = get_ray_direction(self.H, self.fov)
        self.ddf_instance_list = []
        with open(args.ddf_instance_list_txt, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                self.ddf_instance_list.append(line.rstrip('\n'))
        self.save_interval = args.save_interval
        self.train_optim_num = args.train_optim_num
        self.frame_num = args.frame_num
        self.model_params_dtype = False
        self.model_device = False

        # Make model
        self.ddf = ddf
        self.init_net = resnet_encoder_prot(args, in_channel=2) #init_net
        self.use_gru = args.use_gru
        if self.use_gru:
            self.df_net = df_resnet_encoder_with_gru(args, in_channel=5)
        else:
            self.df_net = df_resnet_encoder(args, in_channel=5)

        # loss func.
        self.use_depth_error = args.use_depth_error
        self.l1 = torch.nn.L1Loss()
        self.cossim = nn.CosineSimilarity(dim=-1)



    def training_step(self, batch, batch_idx):

        frame_rgb_map, frame_mask, frame_depth_map, frame_camera_pos, frame_camera_rot, frame_obj_rot, instance_id = batch
        batch_size = len(instance_id)
        
        loss_axis_green = []
        loss_axis_red = []
        loss_shape_code = []
        if self.use_depth_error:
            depth_simulation_error = []

        # for frame_idx in range(self.frame_num):
        frame_idx = random.randint(0, frame_depth_map.shape[1]-1)
        for optim_idx in range(self.train_optim_num):

            with torch.no_grad():
                # Get ground truth.
                batch_size = len(instance_id)
                instance_idx = [self.ddf_instance_list.index(instance_id_i) for instance_id_i in instance_id]
                gt_shape_code = self.ddf.lat_vecs(torch.tensor(instance_idx, device=self.ddf.device)).detach()

                # # Get gt
                o2w = batch_pi2rot_y(frame_obj_rot[:, frame_idx]).permute(0, 2, 1)
                w2c = frame_camera_rot[:, frame_idx]
                o2c = torch.bmm(w2c, o2w) # とりあえずこれを推論する
                o2w = torch.bmm(w2c.permute(0, 2, 1), o2c)
                o2o = torch.bmm(o2w.permute(0, 2, 1), o2w)
                gt_axis_green = o2c[:, :, 1] # Y
                gt_axis_red = o2c[:, :, 0] # X

                # Get input.
                mask = frame_mask[:, frame_idx]
                depth_map = frame_depth_map[:, frame_idx]

            # Estimating.
            if optim_idx == 0:
                inp = torch.stack([depth_map, mask], 1)
                if self.use_gru:
                    est_axis_green, est_axis_red, est_shape_code, feature_vec = self.init_net(inp, self.use_gru)
                    pre_hidden_state = feature_vec
                else:
                    est_axis_green, est_axis_red, est_shape_code = self.init_net(inp, self.use_gru)
            elif optim_idx > 0:
                inp = torch.stack([depth_map, mask, pre_depth_map, pre_mask, depth_map - pre_depth_map], 1)
                if self.use_gru:
                    diff_axis_green, diff_axis_red, diff_shape_code, feature_vec, pre_hidden_state = self.df_net(inp, pre_axis_green, pre_axis_red, pre_shape_code, pre_hidden_state)
                else:
                    diff_axis_green, diff_axis_red, diff_shape_code = self.df_net(inp, pre_axis_green, pre_axis_red, pre_shape_code)
                est_axis_green = F.normalize(pre_axis_green + diff_axis_green, dim=-1)
                est_axis_red = F.normalize(pre_axis_red + diff_axis_red, dim=-1)
                est_shape_code = pre_shape_code + diff_shape_code

            # Cal loss.
            loss_axis_green.append(torch.mean(-self.cossim(est_axis_green, gt_axis_green) + 1.))
            loss_axis_red.append(torch.mean(-self.cossim(est_axis_red, gt_axis_red) + 1.))
            loss_shape_code.append(F.mse_loss(est_shape_code, gt_shape_code))

            # Estimating depth map.
            if optim_idx < self.train_optim_num:
                if not self.use_depth_error:
                    with torch.no_grad():
                        # Estimating.
                        rays_d_cam = self.rays_d_cam.expand(batch_size, -1, -1, -1).to(frame_camera_rot.device)
                        obj_pos_wrd = torch.zeros(batch_size, 3, device=frame_camera_rot.device)
                        est_invdepth_map, est_mask, est_depth_map = get_depth_map_from_axis(
                                                                        H = self.H, 
                                                                        axis_green = est_axis_green, 
                                                                        axis_red = est_axis_red,
                                                                        cam_pos_wrd = frame_camera_pos[:, frame_idx].detach(), 
                                                                        obj_pos_wrd = obj_pos_wrd.detach(), 
                                                                        rays_d_cam = rays_d_cam.detach(), 
                                                                        w2c = w2c.detach(), 
                                                                        input_lat_vec = est_shape_code, 
                                                                        ddf = self.ddf, 
                                                                        with_invdepth_map = True, 
                                                                        )

                        # get next inputs
                        pre_axis_green = est_axis_green.detach()
                        pre_axis_red = est_axis_red.detach()
                        pre_shape_code = est_shape_code.detach()
                        pre_mask = est_mask.detach()
                        pre_depth_map = est_depth_map.detach()

                elif self.use_depth_error:
                    # Estimating.
                    rays_d_cam = self.rays_d_cam.expand(batch_size, -1, -1, -1).to(frame_camera_rot.device)
                    obj_pos_wrd = torch.zeros(batch_size, 3, device=frame_camera_rot.device)
                    est_invdepth_map, est_mask, est_depth_map = get_depth_map_from_axis(
                                                                    H = self.H, 
                                                                    axis_green = est_axis_green, 
                                                                    axis_red = est_axis_red,
                                                                    cam_pos_wrd = frame_camera_pos[:, frame_idx].detach(), 
                                                                    obj_pos_wrd = obj_pos_wrd.detach(), 
                                                                    rays_d_cam = rays_d_cam.detach(), 
                                                                    w2c = w2c.detach(), 
                                                                    input_lat_vec = est_shape_code, 
                                                                    ddf = self.ddf, 
                                                                    with_invdepth_map = True, 
                                                                    )

                    # Cal depth_simulation_error.
                    invdepth_map = torch.zeros_like(depth_map)
                    invdepth_map[mask] = 1. / depth_map[mask]
                    depth_simulation_error.append(self.l1(est_invdepth_map, invdepth_map.detach()))

                    # get next inputs
                    pre_axis_green = est_axis_green.detach()
                    pre_axis_red = est_axis_red.detach()
                    pre_shape_code = est_shape_code.detach()
                    pre_mask = est_mask.detach()
                    pre_depth_map = est_depth_map.detach()

        # Cal total loss.
        num_stacked_loss = len(loss_axis_green)
        loss_axis_green = sum(loss_axis_green) / num_stacked_loss
        loss_axis_red = sum(loss_axis_red) / num_stacked_loss
        loss_shape_code = sum(loss_shape_code) / num_stacked_loss
        if not self.use_depth_error:
            loss_axis = loss_axis_green + .5 * loss_axis_red
            loss = loss_axis + 1e1 * loss_shape_code
        elif self.use_depth_error:
            depth_simulation_error = sum(depth_simulation_error) / num_stacked_loss
            loss_axis = loss_axis_green + .5 * loss_axis_red
            loss = loss_axis + 1e1 * loss_shape_code + 1e1 * depth_simulation_error
        
        # if (self.current_epoch + 1) % 10 == 0 and batch_idx==0:
        #     for batch_idx in range(batch_size):
        #         check_map_i = torch.cat([depth_map[batch_idx], pre_depth_map[batch_idx]], dim=0)
        #         check_map_1.append(check_map_i)
        #     check_map_1 = torch.cat(check_map_1, dim=1)
        #     check_map_torch(check_map_1, f'test_dfnet_png/check_map_{self.current_epoch + 1}.png', figsize=[10,2])
        #     check_map_1 = []
        #     # import pdb; pdb.set_trace()

        if not self.use_depth_error:
            return {'loss': loss, 'loss_axis': loss_axis.detach(), 'loss_shape_code': loss_shape_code.detach()}
        elif self.use_depth_error:
            return {'loss': loss, 'loss_axis': loss_axis.detach(), 'loss_shape_code': loss_shape_code.detach(), 'depth_simulation_error': depth_simulation_error.detach()}



    def training_epoch_end(self, outputs):
        # Get epoch.
        current_epoch = torch.tensor(self.current_epoch + 1., dtype=outputs[0]['loss_axis'].dtype)
        
        # Log loss.
        avg_loss_axis = torch.stack([x['loss_axis'] for x in outputs]).mean()
        self.log_dict({'train/loss_axis': avg_loss_axis, "step": current_epoch})

        avg_loss_shape_code = torch.stack([x['loss_shape_code'] for x in outputs]).mean()
        self.log_dict({'train/loss_shape_code': avg_loss_shape_code, "step": current_epoch})

        if self.use_depth_error:
            avg_depth_simulation_error = torch.stack([x['depth_simulation_error'] for x in outputs]).mean()
            self.log_dict({'train/depth_simulation_error': avg_depth_simulation_error, "step": current_epoch})

        # Save ckpt.
        if (self.current_epoch + 1) % self.save_interval == 0:
            ckpt_name = str(self.current_epoch + 1).zfill(10) + '.ckpt'
            ckpt_path = os.path.join(self.trainer.log_dir, 'checkpoints', ckpt_name)
            trainer.save_checkpoint(ckpt_path)



    def validation_step(self, batch, batch_idx):

        frame_rgb_map, frame_mask, frame_depth_map, frame_camera_pos, frame_camera_rot, frame_obj_rot, instance_id = batch
        batch_size = len(instance_id)
        
        # for frame_idx in range(self.frame_num):
        frame_idx = random.randint(0, frame_depth_map.shape[1]-1)
        for optim_idx in range(self.train_optim_num):

            with torch.no_grad():
                # Get ground truth.
                batch_size = len(instance_id)

                # Get gt
                o2w = batch_pi2rot_y(frame_obj_rot[:, frame_idx]).permute(0, 2, 1)
                w2c = frame_camera_rot[:, frame_idx]
                o2c = torch.bmm(w2c, o2w) # とりあえずこれを推論する
                o2w = torch.bmm(w2c.permute(0, 2, 1), o2c)
                o2o = torch.bmm(o2w.permute(0, 2, 1), o2w)
                gt_axis_green = o2c[:, :, 1] # Y
                gt_axis_red = o2c[:, :, 0] # X

                # Get input.
                mask = frame_mask[:, frame_idx]
                depth_map = frame_depth_map[:, frame_idx]

            # Estimating.
            if optim_idx == 0:
                inp = torch.stack([depth_map, mask], 1)
                if self.use_gru:
                    est_axis_green, est_axis_red, est_shape_code, feature_vec = self.init_net(inp, self.use_gru)
                    pre_hidden_state = feature_vec
                else:
                    est_axis_green, est_axis_red, est_shape_code = self.init_net(inp, self.use_gru)
            elif optim_idx > 0:
                inp = torch.stack([depth_map, mask, pre_depth_map, pre_mask, depth_map - pre_depth_map], 1)
                if self.use_gru:
                    diff_axis_green, diff_axis_red, diff_shape_code, feature_vec, pre_hidden_state = self.df_net(inp, pre_axis_green, pre_axis_red, pre_shape_code, pre_hidden_state)
                else:
                    diff_axis_green, diff_axis_red, diff_shape_code = self.df_net(inp, pre_axis_green, pre_axis_red, pre_shape_code)
                est_axis_green = F.normalize(pre_axis_green + diff_axis_green, dim=-1)
                est_axis_red = F.normalize(pre_axis_red + diff_axis_red, dim=-1)
                est_shape_code = pre_shape_code + diff_shape_code

            # Estimating depth map.
            if optim_idx < self.train_optim_num:
                with torch.no_grad():
                    # Estimating.
                    rays_d_cam = self.rays_d_cam.expand(batch_size, -1, -1, -1).to(frame_camera_rot.device)
                    obj_pos_wrd = torch.zeros(batch_size, 3, device=frame_camera_rot.device)
                    est_invdepth_map, est_mask, est_depth_map = get_depth_map_from_axis(
                                                                    H = self.H, 
                                                                    axis_green = est_axis_green, 
                                                                    axis_red = est_axis_red,
                                                                    cam_pos_wrd = frame_camera_pos[:, frame_idx].detach(), 
                                                                    obj_pos_wrd = obj_pos_wrd.detach(), 
                                                                    rays_d_cam = rays_d_cam.detach(), 
                                                                    w2c = w2c.detach(), 
                                                                    input_lat_vec = est_shape_code, 
                                                                    ddf = self.ddf, 
                                                                    with_invdepth_map = True, 
                                                                    )

                    # get next inputs
                    pre_axis_green = est_axis_green.detach()
                    pre_axis_red = est_axis_red.detach()
                    pre_shape_code = est_shape_code.detach()
                    pre_mask = est_mask.detach()
                    pre_depth_map = est_depth_map.detach()
                    
        # Cal last error.
        err_axis_green = torch.mean(-self.cossim(est_axis_green, gt_axis_green) + 1.)
        err_axis_red = torch.mean(-self.cossim(est_axis_red, gt_axis_red) + 1.)

        return {'err_axis_green': err_axis_green.detach(), 'err_axis_red': err_axis_red.detach()}



    def validation_epoch_end(self, outputs):
        # Log loss.
        avg_err_axis_green = torch.stack([x['err_axis_green'] for x in outputs]).mean()
        current_epoch = torch.tensor(self.current_epoch + 1., dtype=avg_err_axis_green.dtype)
        self.log_dict({'validation/err_axis_green': avg_err_axis_green, "step": current_epoch})

        avg_err_axis_red = torch.stack([x['err_axis_red'] for x in outputs]).mean()
        self.log_dict({'validation/err_axis_red': avg_err_axis_red, "step": current_epoch})
        


    def check_model_info (self):
        self.model_params_dtype = list(self.mlp.parameters())[-1].dtype
        self.model_device = self.device



    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {"params": self.init_net.parameters()},
            {"params": self.df_net.parameters()},
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
        strategy=DDPPlugin(find_unused_parameters=True), #=False), 
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

    # # Get init net.
    # init_net = TaR_init_only(args, ddf)
    # init_net = init_net.load_from_checkpoint(
    #     checkpoint_path='./lightning_logs/DeepTaR/chair/test_initnet_0/checkpoints/0000005000.ckpt', 
    #     args=args, 
    #     ddf=ddf
    #     ).model
    # init_net.eval()

    # Get df net.
    args.use_gru = False
    model = TaR(args, ddf)
    
    # Get ckpts path.
    ckpt_dir = os.path.join('lightning_logs', f'{args.expname}_{args.exp_version}', 'checkpoints/*')
    ckpt_path_list = sorted(glob.glob(ckpt_dir))

    # # # Load ckpt and start training.
    # # # if len(ckpt_path_list) == 0:
    # trainer.fit(
    #     model=model, 
    #     train_dataloaders=train_dataloader, 
    #     val_dataloaders=val_dataloader, 
    #     datamodule=None, 
    #     ckpt_path=None
    #     )

    # # elif len(ckpt_path_list) > 0:
    # #     latest_ckpt_path = ckpt_path_list[-1]
    # #     print('\n', f'+++ Reloading from {latest_ckpt_path} +++ \n')
    ckpt_path = './lightning_logs/DeepTaR/chair/test_dfnet_withx/checkpoints/0000003200.ckpt'
    model = TaR(args, ddf)
    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=val_dataloader, 
        datamodule=None, 
        ckpt_path=ckpt_path # latest_ckpt_path
        )
