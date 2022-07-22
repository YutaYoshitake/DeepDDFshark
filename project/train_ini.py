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
# from model import *
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



class initializer(pl.LightningModule):

    def __init__(self, args, in_channel=2):
        super().__init__()

        self.backbone_encoder = nn.Sequential(
                ResNet50(args, in_channel=in_channel), 
                )
        self.backbone_fc = nn.Sequential(
                nn.Linear(2048 + 7, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, 512), nn.LeakyReLU(0.2),
                )
        self.fc_axis_green = nn.Sequential(
                nn.Linear(512, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 3), 
                )
        self.fc_axis_red = nn.Sequential(
                nn.Linear(512, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 3), 
                )
        self.fc_shape_code = nn.Sequential(
                nn.Linear(512, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, args.latent_size), 
                )
        self.fc_pos = nn.Sequential(
                nn.Linear(512, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 3), 
                )
        self.fc_scale = nn.Sequential(
                nn.Linear(512, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 1), nn.Softplus(beta=.7), 
                )
        # self.fc_weight = nn.Sequential(
        #         nn.Linear(512, 256), nn.LeakyReLU(0.2),
        #         nn.Linear(256, 256), nn.LeakyReLU(0.2),
        #         nn.Linear(256, 1), nn.Sigmoid(), 
        #         )

    
    def forward(self, inp, bbox_info):
        # Backbone.
        x = self.backbone_encoder(inp)
        x = x.reshape(inp.shape[0], -1)
        x = self.backbone_fc(torch.cat([x, bbox_info], dim=-1))

        # Get pose.
        x_pos = self.fc_pos(x)

        # Get axis.
        x_green = self.fc_axis_green(x)
        axis_green = F.normalize(x_green, dim=-1)
        x_red = self.fc_axis_red(x)
        axis_red = F.normalize(x_red, dim=-1)

        # Get scale.
        scale_diff = self.fc_scale(x) + 1e-5 # Prevent scale=0.

        # Get shape code.
        shape_code = self.fc_shape_code(x)

        return x_pos, axis_green, axis_red, scale_diff, shape_code, 0





class deep_optimizer(pl.LightningModule):

    def __init__(self, args, in_channel=5):
        super().__init__()

        self.backbone_encoder = nn.Sequential(
                ResNet50(args, in_channel=in_channel), 
                )
        self.backbone_fc = nn.Sequential(
                nn.Linear(2048 + 7, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, 512), nn.LeakyReLU(0.2),
                )
        self.fc_axis_green = nn.Sequential(
                nn.Linear(512 + 3, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 3), 
                )
        self.fc_axis_red = nn.Sequential(
                nn.Linear(512 + 3, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 3), 
                )
        self.fc_shape_code = nn.Sequential(
                nn.Linear(512 + args.latent_size, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, args.latent_size), 
                )
        self.fc_pos = nn.Sequential(
                nn.Linear(512 + 3, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 3), 
                )
        self.fc_scale = nn.Sequential(
                nn.Linear(512 + 1, 256), nn.LeakyReLU(0.2), 
                nn.Linear(256, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 1), nn.Softplus(beta=.7), 
                )


    def forward(self, inp, bbox_info, pre_pos, pre_axis_green, pre_axis_red, pre_scale, pre_shape_code, with_x=False):
        # Backbone.
        x = self.backbone_encoder(inp)
        x = x.reshape(inp.shape[0], -1)
        x = self.backbone_fc(torch.cat([x, bbox_info], dim=-1))

        # Get pose diff.
        diff_pos = self.fc_pos(torch.cat([x, pre_pos], dim=-1))
        # diff_pos = torch.zeros_like(pre_pos)

        # Get axis diff.
        diff_axis_green = self.fc_axis_green(torch.cat([x, pre_axis_green], dim=-1))
        diff_axis_red = self.fc_axis_red(torch.cat([x, pre_axis_red], dim=-1))

        # Get scale diff.
        diff_scale_cim = self.fc_scale(torch.cat([x, pre_scale], dim=-1)) + 1e-5 # Prevent scale=0.

        # Get shape code diff.
        diff_shape_code = self.fc_shape_code(torch.cat([x, pre_shape_code], dim=-1))

        if not with_x:
            return diff_pos, diff_axis_green, diff_axis_red, diff_scale_cim, diff_shape_code

        elif with_x:
            return diff_pos, diff_axis_green, diff_axis_red, diff_scale_cim, diff_shape_code, x





class only_init_net(pl.LightningModule):

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
        self.ddf_H = 256
        self.lr = args.lr
        self.rays_d_cam = get_ray_direction(self.ddf_H, self.fov)
        self.ddf_instance_list = []
        with open(args.ddf_instance_list_txt, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                self.ddf_instance_list.append(line.rstrip('\n'))
        self.save_interval = args.save_interval
        self.itr_frame_num = 3

        # Make model
        self.ddf = ddf
        self.init_net = initializer(args, in_channel=2) #init_net
        self.df_net = deep_optimizer(args, in_channel=5)

        # loss func.
        self.l1 = torch.nn.L1Loss()
        self.cossim = nn.CosineSimilarity(dim=-1)
        self.cosssim_min = -1+1e-8
        self.cosssim_max = 1-1e-8
        self.use_depth_error = args.use_depth_error
        self.depth_sampling_type = args.depth_sampling_type # 'clopping'
        self.sampling_interval = 8
        self.clopping_size = 100
        self.L_p = args.L_p
        self.L_s = args.L_s
        self.L_a = args.L_a
        self.L_c = args.L_c
        self.L_d = args.L_d
        self.automatic_optimization = False
    


    def preprocess(self, batch, mode):
        # Get batch data.
        if mode=='train':
            frame_mask, frame_distance_map, frame_camera_pos, frame_camera_rot, frame_obj_pos, frame_obj_rot, frame_obj_scale, instance_id = batch
        if mode=='val':
            frame_mask, frame_distance_map, frame_camera_pos, frame_camera_rot, frame_obj_pos, frame_obj_rot, frame_obj_scale, \
            canonical_distance_map, canonical_camera_pos, canonical_camera_rot, instance_id, path = batch
        batch_size = len(instance_id)

        # Get ground truth.
        if mode=='train':
            instance_idx = [self.ddf_instance_list.index(instance_id_i) for instance_id_i in instance_id]
            gt_shape_code = self.ddf.lat_vecs(torch.tensor(instance_idx, device=self.ddf.device)).detach()

        # Clop distance map.
        raw_invdistance_map = torch.zeros_like(frame_distance_map)
        raw_invdistance_map[frame_mask] = 1. / frame_distance_map[frame_mask]
        clopped_mask, clopped_distance_map, bbox_list = clopping_distance_map(
                                                            frame_mask.reshape(-1, self.input_H, self.input_W), 
                                                            frame_distance_map.reshape(-1, self.input_H, self.input_W), 
                                                            self.image_coord, 
                                                            self.input_H, 
                                                            self.input_W, 
                                                            self.ddf_H
                                                            )

        # Get normalized depth map.
        rays_d_cam = get_clopped_rays_d_cam(self.ddf_H, self.fov, bbox_list).to(frame_camera_rot)
        clopped_depth_map, normalized_depth_map, avg_depth_map = get_normalized_depth_map(
                                                                    clopped_mask, 
                                                                    clopped_distance_map, 
                                                                    rays_d_cam
                                                                    )
        bbox_info = torch.cat([bbox_list.reshape(-1, 4), 
                                bbox_list.mean(1), 
                                avg_depth_map.to('cpu')[:, None]], dim=-1).to(frame_camera_rot)

        # Reshaping maps.
        frame_raw_invdistance_map = raw_invdistance_map
        frame_clopped_mask = clopped_mask.reshape(batch_size, -1, self.ddf_H, self.ddf_H)
        frame_clopped_distance_map = clopped_distance_map.reshape(batch_size, -1, self.ddf_H, self.ddf_H)
        frame_bbox_list = bbox_list.reshape(batch_size, -1, 2, 2)
        frame_rays_d_cam = rays_d_cam.reshape(batch_size, -1, self.ddf_H, self.ddf_H, 3)
        frame_clopped_depth_map = clopped_depth_map.reshape(batch_size, -1, self.ddf_H, self.ddf_H)
        frame_normalized_depth_map = normalized_depth_map.reshape(batch_size, -1, self.ddf_H, self.ddf_H)
        frame_avg_depth_map = avg_depth_map.reshape(batch_size, -1)
        frame_bbox_info = bbox_info.reshape(batch_size, -1, 7)

        # Get ground truth.
        o2w = frame_obj_rot.reshape(-1, 3, 3)
        w2c = frame_camera_rot.reshape(-1, 3, 3)
        o2c = torch.bmm(w2c, o2w) # とりあえずこれを推論する
        gt_obj_axis_green_cam = o2c[:, :, 1] # Y
        gt_obj_axis_red_cam = o2c[:, :, 0] # X
        gt_obj_axis_green_wrd = torch.sum(gt_obj_axis_green_cam[..., None, :]*w2c.permute(0, 2, 1), -1) # Y_w
        gt_obj_axis_red_wrd = torch.sum(gt_obj_axis_red_cam[..., None, :]*w2c.permute(0, 2, 1), -1) # X_w

        # Reshaping ground truth.
        frame_w2c = w2c.reshape(batch_size, -1, 3, 3)
        frame_gt_obj_axis_green_wrd = gt_obj_axis_green_wrd.reshape(batch_size, -1, 3)
        frame_gt_obj_axis_red_wrd = gt_obj_axis_red_wrd.reshape(batch_size, -1, 3)

        if mode=='train':
            return batch_size, frame_raw_invdistance_map, frame_clopped_mask, \
                    frame_clopped_distance_map, frame_bbox_list, frame_rays_d_cam, \
                    frame_clopped_depth_map, frame_normalized_depth_map, frame_avg_depth_map, frame_bbox_info, gt_shape_code, \
                    frame_w2c, frame_gt_obj_axis_green_wrd, frame_gt_obj_axis_red_wrd, \
                    frame_camera_pos, frame_obj_pos, frame_obj_scale
        elif mode=='val':
            return batch_size, frame_raw_invdistance_map, frame_clopped_mask, \
                    frame_clopped_distance_map, frame_bbox_list, frame_rays_d_cam, \
                    frame_clopped_depth_map, frame_normalized_depth_map, frame_avg_depth_map, frame_bbox_info, \
                    frame_w2c, frame_gt_obj_axis_green_wrd, frame_gt_obj_axis_red_wrd, \
                    frame_camera_pos, frame_obj_pos, frame_obj_scale, \
                    canonical_distance_map, canonical_camera_pos, canonical_camera_rot, instance_id, path



    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            batch_size, frame_raw_invdistance_map, frame_clopped_mask, \
            frame_clopped_distance_map, frame_bbox_list, frame_rays_d_cam, \
            frame_clopped_depth_map, frame_normalized_depth_map, frame_avg_depth_map, frame_bbox_info, gt_shape_code, \
            frame_w2c, frame_gt_obj_axis_green_wrd, frame_gt_obj_axis_red_wrd, \
            frame_camera_pos, frame_obj_pos, frame_obj_scale = self.preprocess(batch, mode='train')


        ###################################
        #####     Start training      #####
        ###################################
        # Preprocessing
        with torch.no_grad():
            # Set frames
            frame_idx_list = random.sample(list(range(self.frame_sequence_num)), 1) # Frameは一つ選択
        
            # Get current maps.
            raw_invdistance_map = frame_raw_invdistance_map[:, frame_idx_list].reshape(-1, self.input_H, self.input_W).detach()
            clopped_mask = frame_clopped_mask[:, frame_idx_list].reshape(-1, self.ddf_H, self.ddf_H).detach()
            clopped_distance_map = frame_clopped_distance_map[:, frame_idx_list].reshape(-1, self.ddf_H, self.ddf_H).detach()
            bbox_list = frame_bbox_list[:, frame_idx_list].reshape(-1, 2, 2).detach()
            rays_d_cam = frame_rays_d_cam[:, frame_idx_list].reshape(-1, self.ddf_H, self.ddf_H, 3).detach()
            clopped_depth_map = frame_clopped_depth_map[:, frame_idx_list].reshape(-1, self.ddf_H, self.ddf_H).detach()
            normalized_depth_map = frame_normalized_depth_map[:, frame_idx_list].reshape(-1, self.ddf_H, self.ddf_H).detach()
            avg_depth_map = frame_avg_depth_map[:, frame_idx_list].reshape(-1).detach()
            bbox_info = frame_bbox_info[:, frame_idx_list].reshape(-1, 7).detach()

            # Get current GT.
            w2c = frame_w2c[:, frame_idx_list].reshape(-1, 3, 3).detach()
            gt_obj_axis_green_wrd = frame_gt_obj_axis_green_wrd[:, frame_idx_list].detach()
            gt_obj_axis_red_wrd = frame_gt_obj_axis_red_wrd[:, frame_idx_list].detach()
            cam_pos_wrd = frame_camera_pos[:, frame_idx_list].reshape(-1, 3).detach()
            gt_obj_pos_wrd = frame_obj_pos[:, frame_idx_list].detach()
            gt_obj_scale = frame_obj_scale[:, frame_idx_list][..., None].detach()

        # Train only initnet.
        first_iterartion = True

        # Set optimizers
        opt = self.optimizers()
        opt.zero_grad()

        # Get loss lists.
        loss_pos = []
        loss_scale = []
        loss_axis_green = []
        loss_axis_red = []
        loss_shape_code = []
        loss_depth_sim = []

        ###################################
        #####     Perform initnet.    #####
        ###################################
        # print('ini')
        # if first_iterartion:
        # Est.
        inp = torch.stack([normalized_depth_map, clopped_mask], 1).detach()
        est_obj_pos_cim, est_obj_axis_green_cam, est_obj_axis_red_cam, est_scale_cim, est_shape_code, _ = self.init_net(inp, bbox_info)
        est_obj_pos_cam, est_obj_scale, cim2im_scale, im2cam_scale, bbox_center = diff2estimation(
                                                                                    est_obj_pos_cim, 
                                                                                    est_scale_cim, 
                                                                                    bbox_list, 
                                                                                    avg_depth_map, 
                                                                                    self.fov, 
                                                                                    with_cim2cam_info=True)
        est_obj_pos_wrd = torch.sum(est_obj_pos_cam[..., None, :]*w2c.permute(0, 2, 1), dim=-1) + cam_pos_wrd
        est_obj_axis_green_wrd = torch.sum(est_obj_axis_green_cam[..., None, :]*w2c.permute(0, 2, 1), -1)
        est_obj_axis_red_wrd = torch.sum(est_obj_axis_red_cam[..., None, :]*w2c.permute(0, 2, 1), -1)

        # Cal loss.
        loss_pos_ini = F.mse_loss(est_obj_pos_wrd, gt_obj_pos_wrd[:, -1].detach())
        loss_scale_ini = F.mse_loss(est_obj_scale, gt_obj_scale[:, -1].detach())
        loss_green_ini = torch.mean(-self.cossim(est_obj_axis_green_wrd, gt_obj_axis_green_wrd[:, -1].detach()) + 1.)
        loss_red_ini = torch.mean(-self.cossim(est_obj_axis_red_wrd, gt_obj_axis_red_wrd[:, -1].detach()) + 1.)
        loss_shape_ini = F.mse_loss(est_shape_code, gt_shape_code.detach())
        depth_err_ini = torch.zeros_like(loss_pos_ini).detach() # Dummy

        # Append loss.
        loss_pos.append(loss_pos_ini)
        loss_scale.append(loss_scale_ini)
        loss_axis_green.append(loss_green_ini)
        loss_axis_red.append(loss_red_ini)
        loss_shape_code.append(loss_shape_ini)
        loss_depth_sim.append(depth_err_ini)

        # Integrate each optim step losses.
        num_stacked_loss = len(loss_pos)
        loss_pos = sum(loss_pos) / num_stacked_loss
        loss_scale = sum(loss_scale) / num_stacked_loss
        loss_axis_green = sum(loss_axis_green) / num_stacked_loss
        loss_axis_red = sum(loss_axis_red) / num_stacked_loss
        loss_shape_code = sum(loss_shape_code) / num_stacked_loss
        loss_depth_sim = sum(loss_depth_sim) / num_stacked_loss
        loss_axis = loss_axis_green + loss_axis_red
        loss = self.L_p * loss_pos + self.L_s * loss_scale + self.L_a * loss_axis + self.L_c * loss_shape_code + self.L_d * loss_depth_sim

        # Optimizer step.
        self.manual_backward(loss)
        opt.step()

        # Save pre estimation.
        with torch.no_grad():
            pre_obj_pos_wrd = est_obj_pos_wrd.clone().detach()
            pre_obj_scale = est_obj_scale.clone().detach()
            pre_obj_axis_green_wrd = est_obj_axis_green_wrd.clone().detach()
            pre_obj_axis_red_wrd = est_obj_axis_red_wrd.clone().detach()
            pre_shape_code = est_shape_code.clone().detach()

        return {'loss': loss.detach(), 
                'loss_pos':loss_pos.detach(), 
                'loss_scale': loss_scale.detach(), 
                'loss_axis_red': loss_axis_red.detach(), 
                'loss_shape_code': loss_shape_code.detach(), 
                'loss_depth_sim': loss_depth_sim.detach()}


    def training_epoch_end(self, outputs):

        # Log loss.
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        current_epoch = torch.tensor(self.current_epoch + 1., dtype=avg_loss.dtype)
        self.log_dict({'train/total_loss': avg_loss, "step": current_epoch})

        avg_loss_pos = torch.stack([x['loss_pos'] for x in outputs]).mean()
        self.log_dict({'train/loss_pos': avg_loss_pos, "step": current_epoch})

        avg_loss_scale = torch.stack([x['loss_scale'] for x in outputs]).mean()
        self.log_dict({'train/loss_scale': avg_loss_scale, "step": current_epoch})

        avg_loss_axis_red = torch.stack([x['loss_axis_red'] for x in outputs]).mean()
        self.log_dict({'train/loss_axis_red': avg_loss_axis_red, "step": current_epoch})

        avg_loss_shape_code = torch.stack([x['loss_shape_code'] for x in outputs]).mean()
        self.log_dict({'train/loss_shape_code': avg_loss_shape_code, "step": current_epoch})

        avg_loss_depth_sim = torch.stack([x['loss_depth_sim'] for x in outputs]).mean()
        self.log_dict({'train/loss_depth_sim': avg_loss_depth_sim, "step": current_epoch})

        # Save ckpt.
        if (self.current_epoch + 1) % self.save_interval == 0:
            ckpt_name = str(self.current_epoch + 1).zfill(10) + '.ckpt'
            ckpt_path = os.path.join(self.trainer.log_dir, 'checkpoints', ckpt_name)
            trainer.save_checkpoint(ckpt_path)



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
    # ckpt_path = '/home/yyoshitake/works/DeepSDF/project/old_files/until/DeepTaR_until0703/lightning_logs/original_list0_0621/checkpoints/0000001000.ckpt'
    ckpt_path = None
    model = only_init_net(args, ddf)
    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader, 
        ckpt_path=ckpt_path
        )
        