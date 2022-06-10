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
from train_initnet import *
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





class df_resnet_encoder(pl.LightningModule):

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
                nn.Linear(256, 1), 
                )
        self.use_gru = args.use_gru
        if self.use_gru:
            self.backbone_gru = nn.GRU(input_size=512, hidden_size=512)


    def forward(self, inp, bbox_info, pre_pos, pre_axis_green, pre_axis_red, pre_scale, pre_shape_code, h_0=False):
        # Backbone.
        x = self.backbone_encoder(inp)
        x = x.reshape(inp.shape[0], -1)
        x = self.backbone_fc(torch.cat([x, bbox_info], dim=-1))
        if self.use_gru:
            x, post_h = self.gru(x.unsqueeze(0), h_0.unsqueeze(0))
            x = x.reshape(batch_size, -1)
            h_1 = post_h.reshape(batch_size, -1)
        elif not self.use_gru:
            h_1 = 0

        # Get pose diff.
        diff_pos = self.fc_pos(torch.cat([x, pre_pos], dim=-1))

        # Get axis diff.
        diff_axis_green = self.fc_axis_green(torch.cat([x, pre_axis_green], dim=-1))
        diff_axis_red = self.fc_axis_red(torch.cat([x, pre_axis_red], dim=-1))

        # Get scale diff.
        x_scale = self.fc_scale(torch.cat([x, pre_scale], dim=-1))
        diff_scale_cim = x_scale + torch.ones_like(x_scale)

        # Get shape code diff.
        diff_shape_code = self.fc_shape_code(torch.cat([x, pre_shape_code], dim=-1))

        return diff_pos, diff_axis_green, diff_axis_red, diff_scale_cim, diff_shape_code, h_1





class TaR(pl.LightningModule):

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
        self.model_params_dtype = False
        self.model_device = False
        self.use_gru = args.use_gru
        self.frame_num = args.frame_num
        self.use_depth_error = args.use_depth_error
        self.frame_sequence_num = args.frame_sequence_num
        self.train_optim_num = [args.train_optim_num] * self.frame_sequence_num
        if self.model_mode == 'sequence':
            self.train_optim_num = [optim_num - int_clamp(i, 0, 1) for i, optim_num in enumerate(self.train_optim_num)]

        # Make model
        self.ddf = ddf
        self.init_net = resnet_encoder(args, in_channel=2) #init_net
        self.df_net = df_resnet_encoder(args, in_channel=5)

        # loss func.
        self.l1 = torch.nn.L1Loss()
        self.cossim = nn.CosineSimilarity(dim=-1)
        self.depth_sampling_type = 'clopping'
        self.sampling_interval = 8
        self.clopping_size = 100



    def training_step(self, batch, batch_idx):
        # Get batch data.
        frame_mask, frame_distance_map, frame_camera_pos, frame_camera_rot, frame_obj_pos, frame_obj_rot, frame_obj_scale, instance_id = batch
        batch_size = len(instance_id)

        # Get ground truth.
        instance_idx = [self.ddf_instance_list.index(instance_id_i) for instance_id_i in instance_id]
        gt_shape_code = self.ddf.lat_vecs(torch.tensor(instance_idx, device=self.ddf.device)).detach()

        # Set frame.
        if self.model_mode == 'single' or self.model_mode == 'only_init':
            frame_idx_list = [random.randint(0, frame_mask.shape[1]-1)]
        elif self.model_mode == 'sequence':
            start = random.randint(0, frame_mask.shape[1]-self.frame_sequence_num)
            frame_idx_list = list(range(start, start+self.frame_sequence_num))
        
        # Start optimization.
        loss_pos = []
        loss_scale = []
        loss_axis_green = []
        loss_axis_red = []
        loss_shape_code = []
        if self.use_depth_error:
            depth_simulation_error = []
        
        # ######################################################################
        # self.init_net.eval(), self.df_net.eval()
        # ######################################################################

        ###################################
        #####     Start Inference     #####
        ###################################
        for frame_sequence_idx, frame_idx in enumerate(frame_idx_list):
            optim_num = self.train_optim_num[frame_sequence_idx]

            # Preprocess.
            with torch.no_grad():
                # Clop distance map.
                raw_mask = frame_mask[:, frame_idx]
                raw_distance_map = frame_distance_map[:, frame_idx]
                clopped_mask, clopped_distance_map, bbox_list = clopping_distance_map(
                                                                    raw_mask, raw_distance_map, self.image_coord, self.input_H, self.input_W, self.ddf_H
                                                                    )

                # Get normalized depth map.
                rays_d_cam = get_clopped_rays_d_cam(self.ddf_H, self.fov, bbox_list).to(frame_camera_rot.device)
                clopped_depth_map, normalized_depth_map, avg_depth_map = get_normalized_depth_map(
                                                                            clopped_mask, clopped_distance_map, rays_d_cam
                                                                            )
                bbox_info = torch.cat([bbox_list.reshape(-1, 4), bbox_list.mean(1), avg_depth_map.to('cpu')[:, None]], dim=-1).to(raw_distance_map)
                gt_invdistance_map = torch.zeros_like(clopped_distance_map)
                gt_invdistance_map[clopped_mask] = 1. / clopped_distance_map[clopped_mask]

                # Get ground truth.
                o2w = frame_obj_rot[:, frame_idx].to(torch.float)
                w2c = frame_camera_rot[:, frame_idx].to(torch.float)
                o2c = torch.bmm(w2c, o2w).to(torch.float) # とりあえずこれを推論する
                gt_obj_axis_green_cam = o2c[:, :, 1] # Y
                gt_obj_axis_red_cam = o2c[:, :, 0] # X
                gt_obj_axis_green_wrd = torch.sum(gt_obj_axis_green_cam[..., None, :]*w2c.permute(0, 2, 1), -1) # Y
                gt_obj_axis_red_wrd = torch.sum(gt_obj_axis_red_cam[..., None, :]*w2c.permute(0, 2, 1), -1) # X
                cam_pos_wrd = frame_camera_pos[:, frame_idx].to(torch.float)
                gt_obj_pos_wrd = frame_obj_pos[:, frame_idx].to(torch.float)
                gt_obj_scale = frame_obj_scale[:, frame_idx][:, None].to(torch.float)

            ###################################
            #####    Start Optim Step     #####
            ###################################
            for optim_idx in range(optim_num):
                if self.model_mode == 'single':
                    perform_init_est = optim_idx == 0
                elif self.model_mode == 'sequence':
                    perform_init_est = optim_idx == 0 and frame_sequence_idx==0
                elif self.model_mode == 'only_init':
                    perform_init_est = True

                # Estimating.
                # est_x_cim, est_y_cim : クロップされた画像座標（[-1, 1]で定義）における物体中心の予測, 
                # est_z_cim : デプス画像の正則に用いた平均から、物体中心がどれだけズレているか？, 
                # est_obj_axis_green_cam : カメラ座標系での物体の上方向, 
                # est_obj_axis_red_cam : カメラ座標系での物体の右方向, 
                # est_scale_cim : Clopping-BBoxの対角と物体のカノニカルBBoxの対角がどれくらいずれているか, 
                if perform_init_est:
                    inp = torch.stack([normalized_depth_map, clopped_mask], 1).detach()
                    est_obj_pos_cim, est_obj_axis_green_cam, est_obj_axis_red_cam, est_scale_cim, est_shape_code, pre_hidden_state = self.init_net(inp, bbox_info)
                    est_obj_pos_cam, est_obj_scale = diff2estimation(est_obj_pos_cim, est_scale_cim, bbox_list, avg_depth_map, self.fov)
                    est_obj_pos_wrd = torch.sum(est_obj_pos_cam[..., None, :]*w2c.permute(0, 2, 1), dim=-1) + cam_pos_wrd
                    est_obj_axis_green_wrd = torch.sum(est_obj_axis_green_cam[..., None, :]*w2c.permute(0, 2, 1), -1)
                    est_obj_axis_red_wrd = torch.sum(est_obj_axis_red_cam[..., None, :]*w2c.permute(0, 2, 1), -1)
                elif not perform_init_est:
                    # Get inputs.
                    print('nex')
                    with torch.no_grad():
                        pre_obj_axis_green_cam = torch.sum(pre_obj_axis_green_wrd[..., None, :]*w2c, -1)
                        pre_obj_axis_red_cam = torch.sum(pre_obj_axis_red_wrd[..., None, :]*w2c, -1)
                        est_clopped_invdistance_map, est_clopped_mask, est_clopped_distance_map = render_distance_map_from_axis(
                                                                                                        H = self.ddf_H, 
                                                                                                        obj_pos_wrd = pre_obj_pos_wrd, # gt_obj_pos_wrd, 
                                                                                                        axis_green = pre_obj_axis_green_cam, # gt_obj_axis_green_cam, 
                                                                                                        axis_red = pre_obj_axis_red_cam, # gt_obj_axis_red_cam, 
                                                                                                        obj_scale = pre_obj_scale[:, 0], # gt_obj_scale[:, 0].to(pre_obj_scale), 
                                                                                                        input_lat_vec = pre_shape_code, # gt_shape_code, 
                                                                                                        cam_pos_wrd = cam_pos_wrd, 
                                                                                                        rays_d_cam = rays_d_cam, 
                                                                                                        w2c = w2c.detach(), 
                                                                                                        ddf = self.ddf, 
                                                                                                        with_invdistance_map = True, 
                                                                                                        )
                        pre_mask = est_clopped_mask.detach()
                        _, pre_depth_map, _ = get_normalized_depth_map(est_clopped_mask, est_clopped_distance_map, rays_d_cam, avg_depth_map)
                    # Estimating update values.
                    inp = torch.stack([normalized_depth_map, clopped_mask, pre_depth_map, pre_mask, normalized_depth_map - pre_depth_map], 1).detach()
                    diff_pos_cim, diff_obj_axis_green_cam, diff_obj_axis_red_cam, diff_scale_cim, diff_shape_code, pre_hidden_state = self.df_net(
                                                                                                                                inp = inp, 
                                                                                                                                bbox_info = bbox_info, 
                                                                                                                                pre_pos = pre_obj_pos_cim, 
                                                                                                                                pre_axis_green = pre_obj_axis_green_cam, 
                                                                                                                                pre_axis_red = pre_obj_axis_red_cam, 
                                                                                                                                pre_scale = pre_obj_scale_cim, 
                                                                                                                                pre_shape_code = pre_shape_code, 
                                                                                                                                h_0 = pre_hidden_state)
                    # Convert deff2est.
                    est_obj_pos_cim = pre_obj_pos_cim + diff_pos_cim
                    est_scale_cim = pre_obj_scale_cim * diff_scale_cim
                    est_obj_pos_cam, est_obj_scale = diff2estimation(est_obj_pos_cim, est_scale_cim, bbox_list, avg_depth_map, self.fov)
                    est_obj_pos_wrd = torch.sum(est_obj_pos_cam[..., None, :]*w2c.permute(0, 2, 1), dim=-1) + cam_pos_wrd
                    est_obj_axis_green_cam = F.normalize(pre_obj_axis_green_cam + diff_obj_axis_green_cam, dim=-1)
                    est_obj_axis_green_wrd = torch.sum(est_obj_axis_green_cam[..., None, :]*w2c.permute(0, 2, 1), -1)
                    est_obj_axis_red_cam = F.normalize(pre_obj_axis_red_cam + diff_obj_axis_red_cam, dim=-1)
                    est_obj_axis_red_wrd = torch.sum(est_obj_axis_red_cam[..., None, :]*w2c.permute(0, 2, 1), -1)
                    est_shape_code = pre_shape_code + diff_shape_code

                # # Check inp.
                # check_map = []
                # for inp_i in inp:
                #     check_map.append(torch.cat([inp_i_i for inp_i_i in inp_i], dim=-1))
                #     if len(check_map)>5:
                #         break
                # check_map = torch.cat(check_map, dim=0)
                # check_map_torch(check_map, f'sample_images/frame{frame_idx}_optim{optim_idx}.png')
                # import pdb; pdb.set_trace()

                # Cal loss.
                loss_pos.append(F.mse_loss(est_obj_pos_wrd, gt_obj_pos_wrd.detach()))
                loss_scale.append(F.mse_loss(est_obj_scale, gt_obj_scale.detach()))
                loss_axis_green.append(torch.mean(-self.cossim(est_obj_axis_green_wrd, gt_obj_axis_green_wrd.detach()) + 1.))
                loss_axis_red.append(torch.mean(-self.cossim(est_obj_axis_red_wrd, gt_obj_axis_red_wrd.detach()) + 1.))
                loss_shape_code.append(F.mse_loss(est_shape_code, gt_shape_code.detach()))

                # Get next inputs to estimate updates.
                pre_obj_pos_cim = est_obj_pos_cim.detach()
                pre_obj_pos_wrd = est_obj_pos_wrd.detach()
                pre_obj_scale_cim = est_scale_cim.detach()
                pre_obj_scale = est_obj_scale.detach()
                pre_obj_axis_green_wrd = est_obj_axis_green_wrd.detach()
                pre_obj_axis_red_wrd = est_obj_axis_red_wrd.detach()
                pre_shape_code = est_shape_code.detach()

                if self.model_mode == 'only_init':
                    break

                # Cal depth error.
                if 0 < optim_idx and self.use_depth_error:
                    if self.depth_sampling_type == 'clopping':
                        clopping_start = np.random.randint(0, self.ddf_H-self.clopping_size, 2)
                        clopping_end = clopping_start + self.clopping_size
                        clopped_H = self.clopping_size
                        sampled_rays_d_cam = rays_d_cam[:, clopping_start[0]:clopping_end[0], clopping_start[1]:clopping_end[1]]
                        gt_invdistance_map_for_deptherr = gt_invdistance_map[:, clopping_start[0]:clopping_end[0], clopping_start[1]:clopping_end[1]]
                    elif self.depth_sampling_type == 'sparse':
                        clopped_H = self.ddf_H//self.sampling_interval, 
                        sampling_start = np.random.randint(0, self.sampling_interval, 2)
                        sampled_rays_d_cam = rays_d_cam[:, sampling_start[0]::self.sampling_interval, sampling_start[1]::self.sampling_interval]
                        gt_invdistance_map_for_deptherr = gt_invdistance_map[:, sampling_start[0]::self.sampling_interval, sampling_start[1]::self.sampling_interval]
                    est_invdistance_map_for_deptherr, _, _ = render_distance_map_from_axis(
                                                                H = clopped_H, 
                                                                obj_pos_wrd = est_obj_pos_wrd, # gt_obj_pos_wrd, 
                                                                axis_green = est_obj_axis_green_cam, # gt_obj_axis_green_cam, 
                                                                axis_red = est_obj_axis_red_cam, # gt_obj_axis_red_cam, 
                                                                obj_scale = est_obj_scale[:, 0], # gt_obj_scale[:, 0].to(est_obj_scale), 
                                                                input_lat_vec = est_shape_code, # gt_shape_code, 
                                                                cam_pos_wrd = cam_pos_wrd, 
                                                                rays_d_cam = sampled_rays_d_cam, 
                                                                w2c = w2c.detach(), 
                                                                ddf = self.ddf, 
                                                                with_invdistance_map = True, 
                                                                )
                    depth_simulation_error.append(self.l1(est_invdistance_map_for_deptherr, gt_invdistance_map_for_deptherr.detach()))
                    # # Check map.
                    # check_map = []
                    # for gt, est in zip(gt_invdistance_map_for_deptherr, est_invdistance_map_for_deptherr):
                    #     check_map.append(torch.cat([gt, est, torch.abs(gt-est)], dim=-1))
                    #     if len(check_map)>5:
                    #         break
                    # check_map = torch.cat(check_map, dim=0)
                    # check_map_torch(check_map, f'sample_images/tes.png')
                    # import pdb; pdb.set_trace()

        # Cal total loss.
        num_stacked_loss = len(loss_pos)
        loss_pos = sum(loss_pos) / num_stacked_loss
        loss_scale = sum(loss_scale) / num_stacked_loss
        loss_axis_green = sum(loss_axis_green) / num_stacked_loss
        loss_axis_red = sum(loss_axis_red) / num_stacked_loss
        loss_shape_code = sum(loss_shape_code) / num_stacked_loss

        if not self.use_depth_error:
            loss = 1e1 * loss_pos + 1e1 * loss_scale + loss_axis_green + loss_axis_red + 1e1 * loss_shape_code
            return {'loss': loss, 
                    'loss_pos':loss_pos.detach(), 
                    'loss_scale': loss_scale.detach(), 
                    'loss_axis_red': loss_axis_red.detach(), 
                    'loss_shape_code': loss_shape_code.detach()}

        elif self.use_depth_error:
            num_stacked_loss = len(depth_simulation_error)
            depth_simulation_error = sum(depth_simulation_error) / num_stacked_loss
            loss_axis = loss_axis_green + loss_axis_red
            loss = 1e1 * loss_pos + 1e1 * loss_scale + 1e0 * loss_axis + 1e1 * loss_shape_code +  + 1e0 * depth_simulation_error
            return {'loss': loss, 
                    'loss_pos':loss_pos.detach(), 
                    'loss_scale': loss_scale.detach(), 
                    'loss_axis_red': loss_axis_red.detach(), 
                    'loss_shape_code': loss_shape_code.detach(), 
                    'depth_simulation_error': depth_simulation_error.detach()}


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

        if self.use_depth_error:
            avg_depth_simulation_error = torch.stack([x['depth_simulation_error'] for x in outputs]).mean()
            self.log_dict({'train/depth_simulation_error': avg_depth_simulation_error, "step": current_epoch})

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
    val_dataset = TaR_dataset(
        args, 
        'val', 
        args.val_instance_list_txt, 
        args.val_data_dir, 
        args.val_N_views, 
        )
    val_dataloader = data_utils.DataLoader(
        val_dataset, 
        batch_size=args.N_batch, 
        num_workers=args.num_workers, 
        drop_last=False, 
        shuffle=False, 
        )

    # Get ddf.
    ddf = DDF(args)
    ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args)
    ddf.eval()
    
    # Get ckpts path.
    ckpt_dir = os.path.join('lightning_logs', f'{args.expname}_{args.exp_version}', 'checkpoints/*')
    ckpt_path_list = sorted(glob.glob(ckpt_dir))

    # Load ckpt and start training.
    model = TaR(args, ddf)
    # trainer.fit(
    #     model=model, 
    #     train_dataloaders=train_dataloader, 
    #     ckpt_path=None
    #     )
    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=val_dataloader, 
        datamodule=None, 
        ckpt_path='./lightning_logs/DeepTaR/chair/initnet_first/checkpoints/0000001010.ckpt'
        )
