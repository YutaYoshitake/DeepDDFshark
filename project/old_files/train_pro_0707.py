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
        self.fc_weight = nn.Sequential(
                nn.Linear(512, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 1), nn.Sigmoid(), 
                )

    
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

        # Get weight.
        frame_weight = self.fc_weight(x)

        return x_pos, axis_green, axis_red, scale_diff, shape_code, frame_weight





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


    def forward(self, inp, bbox_info, pre_pos, pre_axis_green, pre_axis_red, pre_scale, pre_shape_code):
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

        return diff_pos, diff_axis_green, diff_axis_red, diff_scale_cim, diff_shape_code









class progressive_optimizer(pl.LightningModule):

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
        self.frame_sequence_num = args.frame_sequence_num
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
        for new_frame_idx in range(self.frame_sequence_num):
            if new_frame_idx == 0: first_iterartion = True
            else: first_iterartion = False

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

            # Preprocessing
            with torch.no_grad():
                # Set frames
                if new_frame_idx < self.itr_frame_num:
                    frame_idx_list = list(range(new_frame_idx+1)) # フレームの長さが短いときは、全てのフレームを使用
                else:
                    frame_idx_list = random.sample(list(range(new_frame_idx)), self.itr_frame_num-1) # それ以外ではランダムに抽出
                    frame_idx_list += [new_frame_idx] # 新しいフレームは必ず使用

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


            ###################################
            #####     Perform initnet.    #####
            ###################################
            inp = torch.stack([normalized_depth_map, clopped_mask], 1).detach()
            est_obj_pos_cim, est_obj_axis_green_cam, est_obj_axis_red_cam, est_scale_cim, est_shape_code, est_frame_weight = self.init_net(inp, bbox_info)
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

            # Cal loss for init net.
            est_obj_pos_wrd = est_obj_pos_wrd.reshape(batch_size, -1, 3)[:, -1]
            est_obj_scale = est_obj_scale.reshape(batch_size, -1, 1)[:, -1]
            est_obj_axis_green_wrd = est_obj_axis_green_wrd.reshape(batch_size, -1, 3)[:, -1]
            est_obj_axis_red_wrd = est_obj_axis_red_wrd.reshape(batch_size, -1, 3)[:, -1]
            est_shape_code = est_shape_code.reshape(batch_size, -1, self.ddf.latent_size)[:, -1]

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


            ###################################
            #####      Perform dfnet.     #####
            ###################################
            if not first_iterartion:
                with torch.no_grad():
                    # Reshape to (batch, frame, ?)
                    opt_frame_num = len(frame_idx_list)
                    inp_pre_obj_pos_wrd = pre_obj_pos_wrd[:, None, :].expand(-1, opt_frame_num, -1).reshape(-1, 3)
                    inp_pre_obj_scale = pre_obj_scale[:, None, :].expand(-1, opt_frame_num, -1).reshape(-1, 1)
                    inp_pre_obj_axis_green_wrd = pre_obj_axis_green_wrd[:, None, :].expand(-1, opt_frame_num, -1).reshape(-1, 3)
                    inp_pre_obj_axis_red_wrd = pre_obj_axis_red_wrd[:, None, :].expand(-1, opt_frame_num, -1).reshape(-1, 3)
                    inp_pre_shape_code = pre_shape_code[:, None, :].expand(-1, opt_frame_num, -1).reshape(-1, self.ddf.latent_size)
                    inp_pre_obj_pos_cam = torch.sum((inp_pre_obj_pos_wrd - cam_pos_wrd)[..., None, :]*w2c, dim=-1)
                    inp_pre_obj_pos_cim = torch.cat([
                                            (inp_pre_obj_pos_cam[:, :-1] / im2cam_scale[:, None] - bbox_center) / cim2im_scale[:, None], 
                                            (inp_pre_obj_pos_cam[:, -1] - avg_depth_map)[:, None]], dim=-1)
                    inp_pre_obj_scale_cim = inp_pre_obj_scale / (im2cam_scale[:, None] * cim2im_scale[:, None] * 2 * math.sqrt(2))

                    # Simulate DDF.
                    inp_pre_obj_axis_green_cam = torch.sum(inp_pre_obj_axis_green_wrd[..., None, :]*w2c, -1)
                    inp_pre_obj_axis_red_cam = torch.sum(inp_pre_obj_axis_red_wrd[..., None, :]*w2c, -1)
                    est_clopped_mask, est_clopped_distance_map = render_distance_map_from_axis(
                                                                    H = self.ddf_H, 
                                                                    obj_pos_wrd = inp_pre_obj_pos_wrd, 
                                                                    axis_green = inp_pre_obj_axis_green_cam, 
                                                                    axis_red = inp_pre_obj_axis_red_cam, 
                                                                    obj_scale = inp_pre_obj_scale[:, 0], 
                                                                    input_lat_vec = inp_pre_shape_code, 
                                                                    cam_pos_wrd = cam_pos_wrd, 
                                                                    rays_d_cam = rays_d_cam, 
                                                                    w2c = w2c.detach(), 
                                                                    ddf = self.ddf, 
                                                                    with_invdistance_map = False)
                    inp_pre_mask = est_clopped_mask.detach()
                    _, inp_pre_depth_map, _ = get_normalized_depth_map(est_clopped_mask, est_clopped_distance_map, rays_d_cam, avg_depth_map)
                
                # Get update.
                inp = torch.stack([normalized_depth_map, 
                                   clopped_mask, 
                                   inp_pre_depth_map, 
                                   inp_pre_mask, 
                                   normalized_depth_map - inp_pre_depth_map], 1).detach()
                diff_pos_cim, diff_obj_axis_green_cam, diff_obj_axis_red_cam, diff_scale, diff_shape_code = self.df_net(
                                                                                                                    inp = inp, 
                                                                                                                    bbox_info = bbox_info, 
                                                                                                                    pre_pos = inp_pre_obj_pos_cim, 
                                                                                                                    pre_axis_green = inp_pre_obj_axis_green_cam, 
                                                                                                                    pre_axis_red = inp_pre_obj_axis_red_cam, 
                                                                                                                    pre_scale = inp_pre_obj_scale_cim, 
                                                                                                                    pre_shape_code = inp_pre_shape_code)

                # Convert cordinates.
                diff_pos_cam = diffcim2diffcam(diff_pos_cim, cim2im_scale, im2cam_scale)
                diff_pos_wrd = torch.sum(diff_pos_cam[..., None, :]*w2c.permute(0, 2, 1), -1)
                diff_obj_axis_green_wrd = torch.sum(diff_obj_axis_green_cam[..., None, :]*w2c.permute(0, 2, 1), -1)
                diff_obj_axis_red_wrd = torch.sum(diff_obj_axis_red_cam[..., None, :]*w2c.permute(0, 2, 1), -1)
                
                # Get weight.
                est_frame_weight = est_frame_weight.reshape(batch_size, opt_frame_num)
                est_frame_weight = est_frame_weight / torch.sum(est_frame_weight, dim=-1)[:, None]
                
                # Get integrated_update.
                diff_pos_wrd = torch.sum(est_frame_weight[..., None]*diff_pos_wrd.reshape(batch_size, -1, 3), dim=1)
                diff_scale = torch.sum(est_frame_weight[..., None]*diff_scale.reshape(batch_size, -1, 1), dim=1)
                diff_obj_axis_green_wrd = torch.sum(est_frame_weight[..., None]*diff_obj_axis_green_wrd.reshape(batch_size, -1, 3), dim=1)
                diff_obj_axis_red_wrd = torch.sum(est_frame_weight[..., None]*diff_obj_axis_red_wrd.reshape(batch_size, -1, 3), dim=1)
                diff_shape_code = torch.sum(est_frame_weight[..., None]*diff_shape_code.reshape(batch_size, -1, self.ddf.latent_size), dim=1)

                # Update estimations.
                est_obj_pos_wrd = pre_obj_pos_wrd + diff_pos_wrd
                est_obj_scale = pre_obj_scale * diff_scale
                est_obj_axis_green_wrd = F.normalize(pre_obj_axis_green_wrd + diff_obj_axis_green_wrd, dim=-1)
                est_obj_axis_red_wrd = F.normalize(pre_obj_axis_red_wrd + diff_obj_axis_red_wrd, dim=-1)
                est_shape_code = pre_shape_code + diff_shape_code

                # Cal loss.
                loss_pos_df = F.mse_loss(est_obj_pos_wrd, gt_obj_pos_wrd[:, -1].detach())
                loss_scale_df = F.mse_loss(est_obj_scale, gt_obj_scale[:, -1].detach())
                loss_green_df = torch.mean(-self.cossim(est_obj_axis_green_wrd, gt_obj_axis_green_wrd[:, -1].detach()) + 1.)
                loss_red_df = torch.mean(-self.cossim(est_obj_axis_red_wrd, gt_obj_axis_red_wrd[:, -1].detach()) + 1.)
                loss_shape_df = F.mse_loss(est_shape_code, gt_shape_code.detach())

                # Get depth loss.
                # self.use_depth_error = True
                # with torch.no_grad():
                if self.use_depth_error:
                    clopped_H = self.ddf_H
                    sampled_rays_d_cam = self.rays_d_cam.expand(batch_size*opt_frame_num, -1, -1, -1).to(w2c)
                    est_obj_axis_green_cam = torch.sum(est_obj_axis_green_wrd[:, None, :].expand(-1, opt_frame_num, -1).reshape(-1, 3)[..., None, :]*w2c, -1)
                    est_obj_axis_red_cam = torch.sum(est_obj_axis_red_wrd[:, None, :].expand(-1, opt_frame_num, -1).reshape(-1, 3)[..., None, :]*w2c, -1)
                    gt_invdistance_map = raw_invdistance_map
                    est_invdistance_map, _, _ = render_distance_map_from_axis(
                                                H = clopped_H, 
                                                ###########################################
                                                obj_pos_wrd = est_obj_pos_wrd[:, None, :].expand(-1, opt_frame_num, -1).reshape(-1, 3), 
                                                axis_green = est_obj_axis_green_cam, 
                                                axis_red = est_obj_axis_red_cam, 
                                                obj_scale = est_obj_scale[:, None, :].expand(-1, opt_frame_num, -1).reshape(-1), 
                                                input_lat_vec = est_shape_code[:, None, :].expand(-1, opt_frame_num, -1).reshape(-1, self.ddf.latent_size), 
                                                ###########################################
                                                # obj_pos_wrd = gt_obj_pos_wrd.reshape(-1, 3), 
                                                # axis_green = gt_obj_axis_green_cam.reshape(batch_size, -1, 3)[:, frame_idx_list].reshape(-1, 3), 
                                                # axis_red = gt_obj_axis_red_cam.reshape(batch_size, -1, 3)[:, frame_idx_list].reshape(-1, 3), 
                                                # obj_scale = gt_obj_scale.reshape(-1), 
                                                # input_lat_vec = gt_shape_code[:, None, :].expand(-1, opt_frame_num, -1).reshape(-1, self.ddf.latent_size), 
                                                # ###########################################
                                                cam_pos_wrd = cam_pos_wrd, 
                                                rays_d_cam = sampled_rays_d_cam, 
                                                w2c = w2c.detach(), 
                                                ddf = self.ddf, 
                                                with_invdistance_map = True)
                    depth_err_df = self.l1(est_invdistance_map, gt_invdistance_map.detach())
                    # ##################################################
                    # # Check map.
                    # check_map = []
                    # for gt, est in zip(gt_invdistance_map.detach(), est_invdistance_map):
                    #     check_map.append(torch.cat([gt, est, torch.abs(gt-est)], dim=-1))
                    # check_map = torch.cat(check_map, dim=0)
                    # check_map_torch(check_map, f'{batch_idx}_{new_frame_idx}.png')
                    # import pdb; pdb.set_trace()
                    # ##################################################
                else:
                    depth_err_df = torch.zeros_like(loss_pos_df).detach() # Dummy

                # Append loss.
                loss_pos.append(loss_pos_df)
                loss_scale.append(loss_scale_df)
                loss_axis_green.append(loss_green_df)
                loss_axis_red.append(loss_red_df)
                loss_shape_code.append(loss_shape_df)
                loss_depth_sim.append(depth_err_df)

            # ##################################################
            # # Check inp.
            # check_map = []
            # for inp_i in inp:
            #     check_map.append(torch.cat([inp_i_i for inp_i_i in inp_i], dim=-1))
            #     # if len(check_map)>5:
            #     #     break
            # check_map = torch.cat(check_map, dim=0)
            # check_map_torch(check_map, f'tes.png')
            # import pdb; pdb.set_trace()
            # ##################################################

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

    
    def test_step(self, batch, batch_idx):
        # with torch.no_grad():
        batch_size, frame_raw_invdistance_map, frame_clopped_mask, \
        frame_clopped_distance_map, frame_bbox_list, frame_rays_d_cam, \
        frame_clopped_depth_map, frame_normalized_depth_map, frame_avg_depth_map, frame_bbox_info, \
        frame_w2c, frame_gt_obj_axis_green_wrd, frame_gt_obj_axis_red_wrd, \
        frame_camera_pos, frame_obj_pos, frame_obj_scale, \
        canonical_distance_map, canonical_camera_pos, canonical_camera_rot, instance_id, path = self.preprocess(batch, mode='val')


        ###################################
        #####     Start training      #####
        ###################################
        for new_frame_idx in range(self.frame_sequence_num):
            if new_frame_idx == 0: first_iterartion = True
            else: first_iterartion = False

            # Set frames
            if new_frame_idx < self.itr_frame_num:
                frame_idx_list = list(range(new_frame_idx+1)) # フレームの長さが短いときは、全てのフレームを使用
            else:
                frame_idx_list = random.sample(list(range(new_frame_idx)), self.itr_frame_num-1) # それ以外ではランダムに抽出
                frame_idx_list += [new_frame_idx] # 新しいフレームは必ず使用
            opt_frame_num = len(frame_idx_list)

            # Get current maps.
            raw_invdistance_map = frame_raw_invdistance_map[:, frame_idx_list].reshape(-1, self.input_H, self.input_W).detach()
            clopped_mask = frame_clopped_mask[:, frame_idx_list].reshape(-1, self.input_H, self.input_W).detach()
            clopped_distance_map = frame_clopped_distance_map[:, frame_idx_list].reshape(-1, self.input_H, self.input_W).detach()
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


            ###################################
            #####     Perform initnet.    #####
            ###################################
            print('ini')
            # Get new frame.
            inp = torch.stack([normalized_depth_map.reshape(batch_size, opt_frame_num, self.ddf_H, self.ddf_H)[:, -1], 
                               clopped_mask.reshape(batch_size, opt_frame_num, self.ddf_H, self.ddf_H)[:, -1]]
                               , 1).detach()
            latest_bbox_info = bbox_info.reshape(batch_size, opt_frame_num, 7)[:, -1]
            latest_bbox_list = bbox_list.reshape(batch_size, opt_frame_num, 2, 2)[:, -1]
            latest_avg_depth = avg_depth_map.reshape(batch_size, opt_frame_num)[:, -1]
            est_obj_pos_cim, est_obj_axis_green_cam, est_obj_axis_red_cam, est_scale_cim, est_shape_code, frame_weight_i = self.init_net(inp, latest_bbox_info)
            est_obj_pos_cam, est_obj_scale, cim2im_scale_i, im2cam_scale_i, bbox_center_i = diff2estimation(
                                                                                                est_obj_pos_cim, 
                                                                                                est_scale_cim, 
                                                                                                latest_bbox_list, 
                                                                                                latest_avg_depth, 
                                                                                                self.fov, 
                                                                                                with_cim2cam_info=True)

            # Save results.
            latest_w2c = w2c.reshape(batch_size, opt_frame_num, 3, 3)[:, -1]
            latest_cam_pos_wrd = cam_pos_wrd.reshape(batch_size, opt_frame_num, 3)[:, -1]
            est_obj_pos_wrd = torch.sum(est_obj_pos_cam[..., None, :]*latest_w2c.permute(0, 2, 1), dim=-1) + latest_cam_pos_wrd
            est_obj_axis_green_wrd = torch.sum(est_obj_axis_green_cam[..., None, :]*latest_w2c.permute(0, 2, 1), -1)
            est_obj_axis_red_wrd = torch.sum(est_obj_axis_red_cam[..., None, :]*latest_w2c.permute(0, 2, 1), -1)

            est_obj_pos_wrd = est_obj_pos_wrd.reshape(batch_size, -1, 3)[:, -1]
            est_obj_scale = est_obj_scale.reshape(batch_size, -1, 1)[:, -1]
            est_obj_axis_green_wrd = est_obj_axis_green_wrd.reshape(batch_size, -1, 3)[:, -1]
            est_obj_axis_red_wrd = est_obj_axis_red_wrd.reshape(batch_size, -1, 3)[:, -1]
            est_shape_code = est_shape_code.reshape(batch_size, -1, self.ddf.latent_size)[:, -1]

            if first_iterartion:
                frame_weight_list = frame_weight_i
                cim2im_scale_list = cim2im_scale_i.unsqueeze(1)
                im2cam_scale_list = im2cam_scale_i.unsqueeze(1)
                bbox_center_list = bbox_center_i.unsqueeze(1)
            
            else:
                frame_weight_list = torch.cat([frame_weight_list, frame_weight_i], dim=1)
                cim2im_scale_list = torch.cat([cim2im_scale_list, cim2im_scale_i.unsqueeze(1)], dim=1)
                im2cam_scale_list = torch.cat([im2cam_scale_list, im2cam_scale_i.unsqueeze(1)], dim=1)
                bbox_center_list = torch.cat([bbox_center_list, bbox_center_i.unsqueeze(1)], dim=1)

            frame_weight = frame_weight_list[:, frame_idx_list]
            cim2im_scale = cim2im_scale_list[:, frame_idx_list].reshape(-1)
            im2cam_scale = im2cam_scale_list[:, frame_idx_list].reshape(-1)
            bbox_center = bbox_center_list[:, frame_idx_list].reshape(-1, 2)

            ### Save init-estimation.
            current_ini_est_obj_pos_wrd = est_obj_pos_wrd.clone().detach()
            current_ini_est_obj_scale = est_obj_scale.clone().detach()
            current_ini_est_obj_axis_red_wrd = est_obj_axis_red_wrd.clone().detach()
            current_ini_est_obj_axis_green_wrd = est_obj_axis_green_wrd.clone().detach()
            current_ini_est_shape_code = est_shape_code.clone().detach()


            ###################################
            #####      Perform dfnet.     #####
            ###################################
            if not first_iterartion:
                print('df')
                ### Append pre init estimation. [最適化, 初期推定]のペア
                pre_obj_pos_wrd_w_ini = torch.stack([pre_obj_pos_wrd, current_ini_est_obj_pos_wrd], dim=0) # [2*batch, xyz]
                pre_obj_scale_w_ini = torch.stack([pre_obj_scale, current_ini_est_obj_scale], dim=0) # [2*batch, 1]
                pre_obj_axis_red_wrd_w_ini = torch.stack([pre_obj_axis_red_wrd, current_ini_est_obj_axis_red_wrd], dim=0) # [2*batch, d_xyz]
                pre_obj_axis_green_wrd_w_ini = torch.stack([pre_obj_axis_green_wrd, current_ini_est_obj_axis_green_wrd], dim=0) # [2*batch, d_xyz]
                pre_shape_code_w_ini = torch.stack([pre_shape_code, current_ini_est_shape_code], dim=0) # [2*batch, 256]

                inp_pre_obj_pos_wrd = pre_obj_pos_wrd_w_ini[..., None, :].expand(-1, -1, opt_frame_num, -1).reshape(-1, 3)# [2*batch*opt_frame_num, xyz]
                inp_pre_obj_scale = pre_obj_scale_w_ini[..., None, :].expand(-1, -1, opt_frame_num, -1).reshape(-1, 1)# [2*batch*opt_frame_num, 1]
                inp_pre_obj_axis_red_wrd = pre_obj_axis_red_wrd_w_ini[..., None, :].expand(-1, -1, opt_frame_num, -1).reshape(-1, 3)# [2*batch*opt_frame_num, d_xyz]
                inp_pre_obj_axis_green_wrd = pre_obj_axis_green_wrd_w_ini[..., None, :].expand(-1, -1, opt_frame_num, -1).reshape(-1, 3)# [2*batch*opt_frame_num, d_xyz]
                inp_pre_shape_code = pre_shape_code_w_ini[..., None, :].expand(-1, -1, opt_frame_num, -1).reshape(-1, self.ddf.latent_size)# [2*batch*opt_frame_num, 256]

                avg_depth_map_w_ini = frame_avg_depth_map[:, frame_idx_list][None, ...].expand(2, -1, -1).reshape(-1)
                w2c_w_ini = frame_w2c[:, frame_idx_list][None, ...].expand(2, -1, -1, -1, -1).reshape(-1, 3, 3)
                cam_pos_wrd_w_ini = frame_camera_pos[:, frame_idx_list][None, ...].expand(2, -1, -1, -1).reshape(-1, 3)
                rays_d_cam_w_ini = frame_rays_d_cam[:, frame_idx_list][None, ...].expand(2, -1, -1, -1, -1, -1).reshape(-1, self.ddf_H, self.ddf_H, 3)

                # Simulate DDF.
                inp_pre_obj_axis_green_cam = torch.sum(inp_pre_obj_axis_green_wrd[..., None, :]*w2c_w_ini, -1)
                inp_pre_obj_axis_red_cam = torch.sum(inp_pre_obj_axis_red_wrd[..., None, :]*w2c_w_ini, -1)
                est_clopped_invdistance_map_w_ini, est_clopped_mask_w_ini, est_clopped_distance_map_w_ini = render_distance_map_from_axis(
                                                                                                                H = self.ddf_H, 
                                                                                                                obj_pos_wrd = inp_pre_obj_pos_wrd, 
                                                                                                                axis_green = inp_pre_obj_axis_green_cam, 
                                                                                                                axis_red = inp_pre_obj_axis_red_cam, 
                                                                                                                obj_scale = inp_pre_obj_scale[:, 0], 
                                                                                                                input_lat_vec = inp_pre_shape_code, 
                                                                                                                cam_pos_wrd = cam_pos_wrd_w_ini, 
                                                                                                                rays_d_cam = rays_d_cam_w_ini, 
                                                                                                                w2c = w2c_w_ini.detach(), 
                                                                                                                ddf = self.ddf, 
                                                                                                                with_invdistance_map = True)

                ### Which one ?
                opt_est_clopped_invdistance_map, ini_est_clopped_invdistance_map = est_clopped_invdistance_map_w_ini.reshape(2, -1, self.ddf_H, self.ddf_H)
                opt_est_clopped_mask, ini_est_clopped_mask = est_clopped_mask_w_ini.reshape(2, -1, self.ddf_H, self.ddf_H)
                opt_est_clopped_distance_map, ini_est_clopped_distance_map = est_clopped_distance_map_w_ini.reshape(2, -1, self.ddf_H, self.ddf_H)
                clopped_invdistance_map = torch.zeros_like(clopped_distance_map)
                clopped_invdistance_map[clopped_mask] = 1. / clopped_distance_map[clopped_mask]
                opt_err = torch.abs(clopped_invdistance_map-opt_est_clopped_invdistance_map).reshape(batch_size, -1).mean(-1)
                ini_err = torch.abs(clopped_invdistance_map-ini_est_clopped_invdistance_map).reshape(batch_size, -1).mean(-1)
                ini_is_better = opt_err > ini_err

                # Replace the the previous estimation with the initial estimation.
                pre_obj_pos_wrd[ini_is_better] = current_ini_est_obj_pos_wrd[ini_is_better].clone().detach()
                pre_obj_scale[ini_is_better] = current_ini_est_obj_scale[ini_is_better].clone().detach()
                pre_obj_axis_red_wrd[ini_is_better] = current_ini_est_obj_axis_red_wrd[ini_is_better].clone().detach()
                pre_obj_axis_green_wrd[ini_is_better] = current_ini_est_obj_axis_green_wrd[ini_is_better].clone().detach()
                pre_shape_code[ini_is_better] = current_ini_est_shape_code[ini_is_better].clone().detach()

                # Get depth map.
                ini_is_better_opt_frame = ini_is_better[..., None].expand(-1, opt_frame_num,).reshape(-1)
                inp_pre_mask = opt_est_clopped_mask
                inp_pre_mask[ini_is_better_opt_frame] = ini_est_clopped_mask[ini_is_better_opt_frame]
                est_clopped_distance_map = opt_est_clopped_distance_map #[batch, opt_frame_num, H, W]
                est_clopped_distance_map[ini_is_better_opt_frame] = ini_est_clopped_distance_map[ini_is_better_opt_frame]
                _, inp_pre_depth_map, _ = get_normalized_depth_map(inp_pre_mask, est_clopped_distance_map, rays_d_cam, avg_depth_map)
                # ##################################################
                # # Check which one?
                # gt_map = torch.cat([gt_i for gt_i in clopped_invdistance_map], dim=-1)
                # opt_map = torch.cat([opt_i for opt_i in opt_est_clopped_invdistance_map], dim=-1)
                # ini_map = torch.cat([ini_i for ini_i in ini_est_clopped_invdistance_map], dim=-1)
                # result_map = torch.cat([res_i for res_i in est_clopped_distance_map], dim=-1)
                # check_map_torch(torch.cat([gt_map, opt_map, ini_map, result_map], dim=0), f'which_{batch_idx}_{new_frame_idx}.png')
                # print(ini_is_better)
                # ##################################################

                inp_pre_obj_pos_wrd = pre_obj_pos_wrd[:, None, :].expand(-1, opt_frame_num, -1).reshape(-1, 3)
                inp_pre_obj_scale = pre_obj_scale[:, None, :].expand(-1, opt_frame_num, -1).reshape(-1, 1)
                inp_pre_obj_axis_green_wrd = pre_obj_axis_green_wrd[:, None, :].expand(-1, opt_frame_num, -1).reshape(-1, 3)
                inp_pre_obj_axis_red_wrd = pre_obj_axis_red_wrd[:, None, :].expand(-1, opt_frame_num, -1).reshape(-1, 3)
                inp_pre_shape_code = pre_shape_code[:, None, :].expand(-1, opt_frame_num, -1).reshape(-1, self.ddf.latent_size)
                
                inp_pre_obj_pos_cam = torch.sum((inp_pre_obj_pos_wrd - cam_pos_wrd)[..., None, :]*w2c, dim=-1)
                inp_pre_obj_pos_cim = torch.cat([
                                        (inp_pre_obj_pos_cam[:, :-1] / im2cam_scale[:, None] - bbox_center) / cim2im_scale[:, None], 
                                        (inp_pre_obj_pos_cam[:, -1] - avg_depth_map)[:, None]], dim=-1)
                inp_pre_obj_scale_cim = inp_pre_obj_scale / (im2cam_scale[:, None] * cim2im_scale[:, None] * 2 * math.sqrt(2))
                inp_pre_obj_axis_green_cam = torch.sum(inp_pre_obj_axis_green_wrd[..., None, :]*w2c, -1)
                inp_pre_obj_axis_red_cam = torch.sum(inp_pre_obj_axis_red_wrd[..., None, :]*w2c, -1)
                
                pre_error = torch.abs(inp_pre_depth_map - normalized_depth_map)
                pre_error = pre_error.reshape(batch_size, opt_frame_num, self.ddf_H, self.ddf_H).mean(dim=-1).mean(dim=-1)
                pre_error = pre_error.mean(dim=-1)

                # Get update.
                inp = torch.stack([normalized_depth_map, 
                                   clopped_mask, 
                                   inp_pre_depth_map, 
                                   inp_pre_mask, 
                                   normalized_depth_map - inp_pre_depth_map], 1).detach()
                # ##################################################
                # # Check inp.
                # check_map = []
                # for inp_i in inp:
                #     check_map.append(torch.cat([inp_i_i for inp_i_i in inp_i], dim=-1))
                #     # if len(check_map)>5:
                #     #     break
                # check_map = torch.cat(check_map, dim=0)
                # check_map_torch(check_map, f'{batch_idx}_pro_{new_frame_idx}.png')
                # # import pdb; pdb.set_trace()
                # ##################################################
                diff_pos_cim, diff_obj_axis_green_cam, diff_obj_axis_red_cam, diff_scale, diff_shape_code = self.df_net(
                                                                                                                    inp = inp, 
                                                                                                                    bbox_info = bbox_info, 
                                                                                                                    pre_pos = inp_pre_obj_pos_cim, 
                                                                                                                    pre_axis_green = inp_pre_obj_axis_green_cam, 
                                                                                                                    pre_axis_red = inp_pre_obj_axis_red_cam, 
                                                                                                                    pre_scale = inp_pre_obj_scale_cim, 
                                                                                                                    pre_shape_code = inp_pre_shape_code)

                # Convert cordinates.
                diff_pos_cam = diffcim2diffcam(diff_pos_cim, cim2im_scale, im2cam_scale)
                diff_pos_wrd = torch.sum(diff_pos_cam[..., None, :]*w2c.permute(0, 2, 1), -1)
                diff_obj_axis_green_wrd = torch.sum(diff_obj_axis_green_cam[..., None, :]*w2c.permute(0, 2, 1), -1)
                diff_obj_axis_red_wrd = torch.sum(diff_obj_axis_red_cam[..., None, :]*w2c.permute(0, 2, 1), -1)

                # Get weight.
                frame_weight = frame_weight / torch.sum(frame_weight, dim=-1)[:, None]

                # Get integrated_update.
                diff_pos_wrd = torch.sum(frame_weight[..., None]*diff_pos_wrd.reshape(batch_size, -1, 3), dim=1)
                diff_scale = torch.sum(frame_weight[..., None]*diff_scale.reshape(batch_size, -1, 1), dim=1)
                diff_obj_axis_green_wrd = torch.sum(frame_weight[..., None]*diff_obj_axis_green_wrd.reshape(batch_size, -1, 3), dim=1)
                diff_obj_axis_red_wrd = torch.sum(frame_weight[..., None]*diff_obj_axis_red_wrd.reshape(batch_size, -1, 3), dim=1)
                diff_shape_code = torch.sum(frame_weight[..., None]*diff_shape_code.reshape(batch_size, -1, self.ddf.latent_size), dim=1)

                # Update estimations.
                est_obj_pos_wrd = pre_obj_pos_wrd + diff_pos_wrd
                est_obj_scale = pre_obj_scale * diff_scale
                est_obj_axis_green_wrd = F.normalize(pre_obj_axis_green_wrd + diff_obj_axis_green_wrd, dim=-1)
                est_obj_axis_red_wrd = F.normalize(pre_obj_axis_red_wrd + diff_obj_axis_red_wrd, dim=-1)
                est_shape_code = pre_shape_code + diff_shape_code


                ###################################
                #####    Start Lamda Step     #####
                ###################################
                for half_lambda_idx in range(self.half_lambda_max):
                    print(f'lamda{half_lambda_idx}')
                    
                    # Reshape to (batch, frame, ?)
                    inp_est_obj_pos_wrd = est_obj_pos_wrd[:, None, :].expand(-1, opt_frame_num, -1).reshape(-1, 3)
                    inp_est_obj_scale = est_obj_scale[:, None, :].expand(-1, opt_frame_num, -1).reshape(-1, 1)
                    inp_est_obj_axis_green_wrd = est_obj_axis_green_wrd[:, None, :].expand(-1, opt_frame_num, -1).reshape(-1, 3)
                    inp_est_obj_axis_red_wrd = est_obj_axis_red_wrd[:, None, :].expand(-1, opt_frame_num, -1).reshape(-1, 3)
                    inp_est_shape_code = est_shape_code[:, None, :].expand(-1, opt_frame_num, -1).reshape(-1, self.ddf.latent_size)
                    inp_est_obj_pos_cam = torch.sum((inp_est_obj_pos_wrd - cam_pos_wrd)[..., None, :]*w2c, dim=-1)
                    inp_est_obj_pos_cim = torch.cat([
                                            (inp_est_obj_pos_cam[:, :-1] / im2cam_scale[:, None] - bbox_center) / cim2im_scale[:, None], 
                                            (inp_est_obj_pos_cam[:, -1] - avg_depth_map)[:, None]], dim=-1)
                    inp_est_obj_scale_cim = inp_est_obj_scale / (im2cam_scale[:, None] * cim2im_scale[:, None] * 2 * math.sqrt(2))

                    # Simulate DDF.
                    inp_est_obj_axis_green_cam = torch.sum(inp_est_obj_axis_green_wrd[..., None, :]*w2c, -1)
                    inp_est_obj_axis_red_cam = torch.sum(inp_est_obj_axis_red_wrd[..., None, :]*w2c, -1)
                    est_mask, est_distance_map = render_distance_map_from_axis(
                                                    H = self.ddf_H, 
                                                    obj_pos_wrd = inp_est_obj_pos_wrd, 
                                                    axis_green = inp_est_obj_axis_green_cam, 
                                                    axis_red = inp_est_obj_axis_red_cam, 
                                                    obj_scale = inp_est_obj_scale[:, 0], 
                                                    input_lat_vec = inp_est_shape_code, 
                                                    cam_pos_wrd = cam_pos_wrd, 
                                                    rays_d_cam = rays_d_cam, 
                                                    w2c = w2c.detach(), 
                                                    ddf = self.ddf, 
                                                    with_invdistance_map = False)
                    _, est_normalized_depth_map, _ = get_normalized_depth_map(
                                                        est_mask, est_distance_map, rays_d_cam, avg_depth_map, 
                                                        )
                    error = torch.abs(est_normalized_depth_map - normalized_depth_map)
                    error = error.reshape(batch_size, opt_frame_num, self.ddf_H, self.ddf_H).mean(dim=-1).mean(dim=-1)
                    error = error.mean(dim=-1)

                    # Make update mask.
                    un_update_mask = (pre_error - error) < 0.
                    decade_all_error = not un_update_mask.any()
                    over_lamda_step = half_lambda_idx + 1 == self.half_lambda_max

                    # 更新により、エラーが全てのバッチで小さくなった or ラムダステップの最大まで行った
                    # -> 次の最適化ステップかフレームへ
                    if decade_all_error or over_lamda_step:
                        # Update values.
                        est_obj_pos_wrd[un_update_mask] = pre_obj_pos_wrd[un_update_mask]
                        est_obj_scale[un_update_mask] = pre_obj_scale[un_update_mask]
                        est_obj_axis_green_wrd[un_update_mask] = pre_obj_axis_green_wrd[un_update_mask]
                        est_obj_axis_red_wrd[un_update_mask] = pre_obj_axis_red_wrd[un_update_mask]
                        est_shape_code[un_update_mask] = pre_shape_code[un_update_mask]
                        break # ラムダステップ終了。

                    # 更新により、エラーが全てのバッチで小さくななかった
                    # -> ならなかったUpdateを半減させて再計算
                    # -> 具体的にはun_update_maskを変更
                    else:
                        lamda_i = 1 / 2**(half_lambda_idx+1)
                        est_obj_pos_wrd[un_update_mask] = pre_obj_pos_wrd[un_update_mask] + lamda_i * diff_pos_wrd[un_update_mask]
                        est_obj_scale[un_update_mask] = pre_obj_scale[un_update_mask] * (1. + lamda_i * (diff_scale[un_update_mask] - 1.))
                        est_obj_axis_green_wrd[un_update_mask] = F.normalize(pre_obj_axis_green_wrd[un_update_mask] + lamda_i * diff_obj_axis_green_wrd[un_update_mask], dim=-1)
                        est_obj_axis_red_wrd[un_update_mask] = F.normalize(pre_obj_axis_red_wrd[un_update_mask] + lamda_i * diff_obj_axis_red_wrd[un_update_mask], dim=-1)
                        est_shape_code[un_update_mask] = pre_shape_code[un_update_mask] + lamda_i * diff_shape_code[un_update_mask]


            # Save pre estimation.
            with torch.no_grad():
                pre_obj_pos_wrd = est_obj_pos_wrd.clone().detach()
                pre_obj_scale = est_obj_scale.clone().detach()
                pre_obj_axis_red_wrd = est_obj_axis_red_wrd.clone().detach()
                pre_obj_axis_green_wrd = est_obj_axis_green_wrd.clone().detach()
                pre_shape_code = est_shape_code.clone().detach()

        # # Check attention.
        # total_check_map = []
        # for b_i in range(batch_size):
        #     check_map_list = frame_clopped_distance_map[b_i][torch.argsort(frame_weight_list[b_i])]
        #     check_map = torch.cat([check_map_i for check_map_i in check_map_list], dim=-1)
        #     total_check_map.append(check_map)
        # check_map_torch(torch.cat(total_check_map, dim=0), 'tes.png')
        # import pdb; pdb.set_trace()

        # # Check final outputs.
        # total_check_map = []
        # for b_i in range(batch_size):
        #     check_map = torch.cat([normalized_depth_map[::3][b_i], 
        #                            est_normalized_depth_map[::3][b_i], 
        #                            torch.abs(normalized_depth_map[::3][b_i]-est_normalized_depth_map[::3][b_i])], dim=-1)
        #     total_check_map.append(check_map)
        # check_map_torch(torch.cat(total_check_map, dim=0), f'last_{batch_idx}_pro.png')
        # import pdb; pdb.set_trace()


        ###################################
        #####       Check shape       #####
        ###################################
        depth_error = []
        for shape_i, (gt_distance_map, cam_pos_wrd, w2c) in enumerate(zip(canonical_distance_map.permute(1, 0, 2, 3), 
                                                                            canonical_camera_pos.permute(1, 0, 2), 
                                                                            canonical_camera_rot.permute(1, 0, 2, 3))):
            # Get simulation results.
            rays_d_cam = get_ray_direction(self.ddf_H, self.fov).expand(batch_size, -1, -1, -1).to(w2c)
            _, est_distance_map = get_canonical_map(
                                        H = self.ddf_H, 
                                        cam_pos_wrd = cam_pos_wrd, 
                                        rays_d_cam = rays_d_cam, 
                                        w2c = w2c, 
                                        input_lat_vec = pre_shape_code, 
                                        ddf = self.ddf, 
                                        )
            depth_error.append(torch.abs(gt_distance_map-est_distance_map).mean(dim=-1).mean(dim=-1))


        # Cal err.
        err_pos = torch.abs(pre_obj_pos_wrd - frame_obj_pos[:, 0]).mean(dim=-1)
        err_scale = torch.abs(1 - pre_obj_scale[:, 0] / frame_obj_scale[:, 0])
        err_axis_red_cos_sim = self.cossim(pre_obj_axis_red_wrd, frame_gt_obj_axis_red_wrd[:, 0]).clamp(min=self.cosssim_min, max=self.cosssim_max)
        err_axis_red = torch.acos(err_axis_red_cos_sim) * 180 / torch.pi
        err_axis_green_cos_sim = self.cossim(pre_obj_axis_green_wrd, frame_gt_obj_axis_green_wrd[:, 0]).clamp(min=self.cosssim_min, max=self.cosssim_max)
        err_axis_green = torch.acos(err_axis_green_cos_sim) * 180 / torch.pi
        depth_error = torch.stack(depth_error, dim=-1).mean(dim=-1)

        return {'err_pos':err_pos.detach(), 
                'err_scale': err_scale.detach(), 
                'err_axis_red': err_axis_red.detach(), 
                'err_axis_green':err_axis_green.detach(), 
                'depth_error':depth_error.detach(), 
                'path':np.array(path), 
                'pos':pre_obj_pos_wrd.detach(), 
                'scale':pre_obj_scale.detach(), 
                'axis_red':pre_obj_axis_red_wrd.detach(), 
                'axis_green':pre_obj_axis_green_wrd.detach(), 
                'shape_code':pre_shape_code.detach()}



    def test_epoch_end(self, outputs):
        # Log loss.
        err_pos_list = torch.cat([x['err_pos'] for x in outputs], dim=0)
        half_list_length = len(err_pos_list) // 2
        mode_err_pos = sorted(err_pos_list)[half_list_length]
        avg_err_pos = err_pos_list.mean()
        err_scale_list = torch.cat([x['err_scale'] for x in outputs], dim=0)
        mode_err_scale = sorted(err_scale_list)[half_list_length]
        avg_err_scale = err_scale_list.mean()
        err_red_list = torch.cat([x['err_axis_red'] for x in outputs], dim=0)
        mode_err_red = sorted(err_red_list)[half_list_length]
        avg_err_red = err_red_list.mean()
        err_green_list = torch.cat([x['err_axis_green'] for x in outputs], dim=0)
        mode_err_green = sorted(err_green_list)[half_list_length]
        avg_err_green = err_green_list.mean()
        err_depth_list = torch.cat([x['depth_error'] for x in outputs], dim=0)
        mode_err_depth = sorted(err_depth_list)[half_list_length]
        avg_err_depth = err_depth_list.mean()
        path_list = np.concatenate([x['path'] for x in outputs])
        with open(self.test_log_path, 'a') as file:
            file.write('avg_err_pos  ' + ' : '  + str(avg_err_pos.item())   + ' : ' + str(mode_err_pos.item()) + '\n')
            file.write('avg_err_scale' + ' : '  + str(avg_err_scale.item()) + ' : ' + str(mode_err_scale.item()) + '\n')
            file.write('avg_err_red  ' + ' : '  + str(avg_err_red.item())   + ' : ' + str(mode_err_red.item()) + '\n')
            file.write('avg_err_green' + ' : '  + str(avg_err_green.item()) + ' : ' + str(mode_err_green.item()) + '\n')
            file.write('avg_err_depth' + ' : '  + str(avg_err_depth.item()) + ' : ' + str(mode_err_depth.item()) + '\n')

        err_pos_list = err_pos_list.to('cpu').detach().numpy().copy()
        err_scale_list = err_scale_list.to('cpu').detach().numpy().copy()
        err_red_list = err_red_list.to('cpu').detach().numpy().copy()
        err_green_list = err_green_list.to('cpu').detach().numpy().copy()
        err_depth_list = err_depth_list.to('cpu').detach().numpy().copy()
        log_dict = {'pos':err_pos_list, 
                    'scale':err_scale_list, 
                    'red':err_red_list, 
                    'green':err_green_list, 
                    'depth':err_depth_list, 
                    'path': path_list}
        pickle_dump(log_dict, self.test_log_path.split('.txt')[0] + '_error.pickle')

        pos_list = torch.cat([x['pos'] for x in outputs], dim=0)
        scale_list = torch.cat([x['scale'] for x in outputs], dim=0)
        axis_red_list = torch.cat([x['axis_red'] for x in outputs], dim=0)
        axis_green_list = torch.cat([x['axis_green'] for x in outputs], dim=0)
        shape_code_list = torch.cat([x['shape_code'] for x in outputs], dim=0)
        log_dict = {'pos':pos_list, 
                    'scale':scale_list, 
                    'red':axis_red_list, 
                    'green':axis_green_list, 
                    'shape':shape_code_list, 
                    'path': path_list}
        pickle_dump(log_dict, self.test_log_path.split('.txt')[0] + '_estval.pickle')



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
    # # Get args
    # args = get_args()
    # args.gpu_num = torch.cuda.device_count() # log used gpu num.

    # # Set trainer.
    # logger = pl.loggers.TensorBoardLogger(
    #         save_dir=os.getcwd(),
    #         version=f'{args.expname}_{args.exp_version}',
    #         name='lightning_logs'
    #     )
    # trainer = pl.Trainer(
    #     gpus=args.gpu_num, 
    #     strategy=DDPPlugin(find_unused_parameters=True), #=False), 
    #     logger=logger,
    #     max_epochs=args.N_epoch, 
    #     enable_checkpointing = False,
    #     check_val_every_n_epoch=args.check_val_every_n_epoch,
    #     )

    # # Save config files.
    # os.makedirs(os.path.join('lightning_logs', f'{args.expname}_{args.exp_version}'), exist_ok=True)
    # f = os.path.join('lightning_logs', f'{args.expname}_{args.exp_version}', 'args.txt')
    # with open(f, 'w') as file:
    #     for arg in sorted(vars(args)):
    #         attr = getattr(args, arg)
    #         file.write('{} = {}\n'.format(arg, attr))
    # if args.config is not None:
    #     f = os.path.join('lightning_logs', f'{args.expname}_{args.exp_version}', 'config.txt')
    #     with open(f, 'w') as file:
    #         file.write(open(args.config, 'r').read())

    # # Create dataloader
    # train_dataset = TaR_dataset(
    #     args, 
    #     'train', 
    #     args.train_instance_list_txt, 
    #     args.train_data_dir, 
    #     args.train_N_views, 
    #     )
    # train_dataloader = data_utils.DataLoader(
    #     train_dataset, 
    #     batch_size=args.N_batch, 
    #     num_workers=args.num_workers, 
    #     drop_last=False, 
    #     shuffle=True, 
    #     )

    # # Set models and Start training.
    # ddf = DDF(args)
    # ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args)
    # ddf.eval()
    # # ckpt_path = '/home/yyoshitake/works/DeepSDF/project/lightning_logs/DeepTaR/chair/dfnetwfd_list0_date0616/checkpoints/0000001500.ckpt'
    # ckpt_path = None
    # model = progressive_optimizer(args, ddf)
    # trainer.fit(
    #     model=model, 
    #     train_dataloaders=train_dataloader, 
    #     ckpt_path=ckpt_path
    #     )



    # Get args
    args = get_args()
    args.gpu_num = torch.cuda.device_count() # log used gpu num.
    args.val_data_dir='/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/train/views64'
    args.val_N_views = 32
    args.use_gru = False

    # Create dataloader.
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

    # Set models and Start training.
    ddf = DDF(args)
    ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args)
    ddf.eval()
    model = progressive_optimizer(args, ddf)
    model = model.load_from_checkpoint(checkpoint_path=args.model_ckpt_path, args=args, ddf=ddf)
    ###########################################################################
    model_ = progressive_optimizer(args, ddf)
    init_net_skpt = 'lightning_logs/DeepTaR/chair/progressive_list0_0621/checkpoints/0000000500.ckpt'
    # init_net_skpt = 'lightning_logs/DeepTaR/chair/progressivewfd_list0_0621/checkpoints/0000000500.ckpt'
    model_ = model_.load_from_checkpoint(checkpoint_path=init_net_skpt, args=args, ddf=ddf)
    model.init_net = model_.init_net
    ###########################################################################

    # Setting model.
    model.start_frame_idx = 0
    model.half_lambda_max = 3
    if model.model_mode == 'only_init':
        model.only_init_net = True
        model.test_optim_num = 1
    else:
        model.only_init_net = False
    model.use_deep_optimizer = True
    model.use_adam_optimizer = not(model.use_deep_optimizer)

    # Save logs.
    import datetime
    dt_now = datetime.datetime.now()
    time_log = dt_now.strftime('%Y_%m_%d_%H_%M_%S')

    os.mkdir('./txt/experiments/log/' + time_log)
    file_name = './txt/experiments/log/' + time_log + '/log.txt'
    model.test_log_path = file_name
    ckpt_path = args.model_ckpt_path
    with open(file_name, 'a') as file:
        file.write('script_name : ' + 'val adam multi' + '\n')
        file.write('time_log : ' + time_log + '\n')
        file.write('ckpt_path : ' + ckpt_path + '\n')
        file.write('val_N_views : ' + str(args.val_N_views) + '\n')
        file.write('val_instance_list_txt : ' + str(args.val_instance_list_txt) + '\n')
        file.write('\n')
        file.write('only_init_net : ' + str(model.only_init_net) + '\n')
        file.write('start_frame_idx : ' + str(model.start_frame_idx) + '\n')
        file.write('frame_sequence_num : ' + str(model.frame_sequence_num) + '\n')
        file.write('half_lambda_max : ' + str(model.half_lambda_max) + '\n')
        file.write('itr_frame_num : ' + str(model.itr_frame_num) + '\n')
        file.write('\n')

    # Test model.
    trainer = pl.Trainer(
        gpus=args.gpu_num, 
        strategy=DDPPlugin(find_unused_parameters=True), #=False), 
        enable_checkpointing = False,
        check_val_every_n_epoch = args.check_val_every_n_epoch,
        logger = pl.loggers.TensorBoardLogger(save_dir=os.getcwd(), version=f'val_trash', name='lightning_logs')
        )
    trainer.test(model, val_dataloader)
