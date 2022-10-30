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
        self.transformer_model = args.transformer_model
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
                            transformer_model=args.transformer_model, 
                            input_type=self.input_type, 
                            num_encoder_layers = args.num_encoder_layers, 
                            positional_encoding_mode=self.positional_encoding_mode, 
                            integration_mode = args.integration_mode, 
                            split_into_patch = args.split_into_patch, 
                            encoder_norm_type = args.encoder_norm_type, 
                            reset_transformer_params = args.reset_transformer_params, 
                            hidden_dim = args.hidden_dim, 
                            dim_feedforward = args.dim_feedforward, 
                            num_head = args.num_head, 
                            dropout = args.dropout, 
                            loss_timing = args.loss_timing, 
                            )
        # elif self.optimizer_type == 'origin':
        #     self.df_net = deep_optimizer(input_type=self.input_type, output_diff_coordinate=self.output_diff_coordinate)
        self.ddf = ddf
        self.loss_timing = args.loss_timing

        # loss func.
        self.automatic_optimization = False
        self.min_loss = float('inf')
        self.min_epoch = 0
        self.L_p = args.L_p
        self.L_s = args.L_s
        self.L_a = args.L_a
        self.L_c = args.L_c
        self.L_d = args.L_d
        self.depth_error_mode = args.depth_error_mode
        self.cossim = nn.CosineSimilarity(dim=-1)
        self.cosssim_min = - 1 + 1e-8
        self.cosssim_max = 1 - 1e-8
        self.train_rand_idx = 0
        self.val_rand_idx = 0
        self.tes_rand_idx = 0
    


    def preprocess(self, batch, mode):
        # Get batch data.
        if mode=='train':
            frame_mask, frame_distance_map, frame_camera_pos, frame_camera_rot, frame_obj_pos, frame_obj_rot, frame_obj_scale, instance_id = batch
        if mode in {'val', 'tes'}:
            frame_mask, frame_distance_map, frame_camera_pos, frame_camera_rot, frame_obj_pos, frame_obj_rot, frame_obj_scale, \
            canonical_distance_map, canonical_camera_pos, canonical_camera_rot, instance_id, path = batch
        batch_size = len(instance_id)

        # Get ground truth shape code.
        if mode in {'train', 'val', 'tes'}:
            instance_idx = [self.ddf_instance_list.index(instance_id_i) for instance_id_i in instance_id]
            gt_shape_code = self.ddf.lat_vecs(torch.tensor(instance_idx, device=self.ddf.device)).detach()
        else:
            gt_shape_code = False

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
        clopped_invdistance_map = torch.zeros_like(clopped_distance_map)
        clopped_invdistance_map[clopped_mask] = 1. / clopped_distance_map[clopped_mask]

        # Get normalized depth map.
        rays_d_cam = get_clopped_rays_d_cam(self.ddf_H, bbox_list, self.rays_d_cam).to(frame_camera_rot)
        clopped_depth_map, normalized_depth_map, avg_depth_map = get_normalized_depth_map(
                                                                    clopped_mask, 
                                                                    clopped_distance_map, 
                                                                    rays_d_cam
                                                                    )
        bbox_info = torch.cat([bbox_list.reshape(-1, 4), 
                                bbox_list.mean(1), 
                                avg_depth_map.to('cpu')[:, None]], dim=-1)

        # Reshaping maps.
        frame_raw_invdistance_map = raw_invdistance_map
        frame_clopped_mask = clopped_mask.reshape(batch_size, -1, self.ddf_H, self.ddf_H)
        frame_clopped_distance_map = clopped_distance_map.reshape(batch_size, -1, self.ddf_H, self.ddf_H)
        frame_clopped_invdistance_map = clopped_invdistance_map.reshape(batch_size, -1, self.ddf_H, self.ddf_H)
        frame_bbox_list = bbox_list.reshape(batch_size, -1, 2, 2).to(frame_camera_rot)
        frame_rays_d_cam = rays_d_cam.reshape(batch_size, -1, self.ddf_H, self.ddf_H, 3)
        frame_clopped_depth_map = clopped_depth_map.reshape(batch_size, -1, self.ddf_H, self.ddf_H)
        frame_normalized_depth_map = normalized_depth_map.reshape(batch_size, -1, self.ddf_H, self.ddf_H)
        frame_avg_depth_map = avg_depth_map.reshape(batch_size, -1)
        frame_bbox_info = bbox_info.reshape(batch_size, -1, 7).to(frame_camera_rot)

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
        frame_gt_obj_axis_green_cam = gt_obj_axis_green_cam.reshape(batch_size, -1, 3)
        frame_gt_obj_axis_red_cam = gt_obj_axis_red_cam.reshape(batch_size, -1, 3)
        frame_gt_obj_axis_green_wrd = gt_obj_axis_green_wrd.reshape(batch_size, -1, 3)
        frame_gt_obj_axis_red_wrd = gt_obj_axis_red_wrd.reshape(batch_size, -1, 3)

        ###################################
        #####     Initialization.     #####
        ###################################
        if mode=='train':
            randn_path = f'randn/list0randn_batch10_train/{str(int(self.train_rand_idx)).zfill(10)}.pickle'
            self.train_rand_idx = self.train_rand_idx + 1
            if self.train_rand_idx >= 1e6: self.train_rand_idx = 0
        elif mode=='val':
            randn_path = f'randn/list0randn_batch10_val/{str(int(self.val_rand_idx)).zfill(10)}.pickle'
            self.val_rand_idx = self.val_rand_idx + 1
            if self.val_rand_idx >= 1e5: self.val_rand_idx = 0
        elif mode=='tes':
            randn_path = f'randn/list0randn_batch10_tes/{str(int(self.tes_rand_idx)).zfill(10)}.pickle'
            self.tes_rand_idx = self.tes_rand_idx + 1
            if self.tes_rand_idx >= 1e5: self.tes_rand_idx = 0
        randn = pickle_load(randn_path)
        rand_P_seed = randn[0][:batch_size] # torch.rand(batch_size, 1)
        rand_S_seed = randn[1][:batch_size] # torch.rand(batch_size, 1)
        randn_theta_seed = randn[2][:batch_size] # torch.rand(batch_size)
        randn_axis_idx = randn[3][:batch_size] # np.random.choice(self.random_axis_num, batch_size)

        # Get initial position.
        rand_P = 2 * self.rand_P_range * (rand_P_seed - .5) #  * (torch.rand(batch_size, 1) - .5)
        ini_obj_pos = frame_obj_pos[:, 0, :] + rand_P.to(frame_obj_pos)

        # Get initial scale.
        rand_S = 2 * self.rand_S_range * (rand_S_seed - .5) + 1. #  * (torch.rand(batch_size, 1) - .5) + 1.
        ini_obj_scale = frame_obj_scale[:, 0].unsqueeze(-1) * rand_S.to(frame_obj_scale)

        # Get initial red.
        randn_theta = 2 * self.rand_R_range * (randn_theta_seed - .5) # (torch.rand(batch_size) - .5) # torch.tensor([.5*torch.pi]*batch_size)
        randn_axis_idx = randn_axis_idx # np.random.choice(self.random_axis_num, batch_size)
        randn_axis = self.random_axis_list[randn_axis_idx] # F.normalize(torch.tensor([[0.5, 0., 0.5]]*batch_size), dim=-1)
        cos_t = torch.cos(randn_theta)
        sin_t = torch.sin(randn_theta)
        n_x = randn_axis[:, 0]
        n_y = randn_axis[:, 1]
        n_z = randn_axis[:, 2]
        rand_R = torch.stack([torch.stack([cos_t+n_x*n_x*(1-cos_t), n_x*n_y*(1-cos_t)-n_z*sin_t, n_z*n_x*(1-cos_t)+n_y*sin_t], dim=-1), 
                                torch.stack([n_x*n_y*(1-cos_t)+n_z*sin_t, cos_t+n_y*n_y*(1-cos_t), n_y*n_z*(1-cos_t)-n_x*sin_t], dim=-1), 
                                torch.stack([n_z*n_x*(1-cos_t)-n_y*sin_t, n_y*n_z*(1-cos_t)+n_x*sin_t, cos_t+n_z*n_z*(1-cos_t)], dim=-1)], dim=1)
        ini_o2w = frame_obj_rot[:, 0, :, :]
        ini_o2w = torch.bmm(rand_R.to(ini_o2w), ini_o2w)
        ini_obj_axis_red_wrd = ini_o2w[:, :, 0] # X_w

        # Get initial green.
        ini_obj_axis_green_wrd = ini_o2w[:, :, 1] # Y_w

        # Get initial shape.
        # randn_Z = self.rand_Z_sigma * torch.randn_like(gt_shape_code)
        ini_shape_code = self.rand_Z_center.unsqueeze(0).expand(batch_size, -1).to(gt_shape_code) # + randn_Z

        if mode=='train':
            return batch_size, frame_raw_invdistance_map, frame_clopped_mask, \
                    frame_clopped_distance_map, frame_clopped_invdistance_map, frame_bbox_list, \
                    frame_rays_d_cam, frame_clopped_depth_map, frame_normalized_depth_map, frame_avg_depth_map, frame_bbox_info, gt_shape_code, frame_w2c, \
                    frame_gt_obj_axis_green_cam, frame_gt_obj_axis_red_cam, frame_gt_obj_axis_green_wrd, \
                    frame_gt_obj_axis_red_wrd, frame_camera_pos, frame_obj_pos, frame_obj_scale, frame_obj_rot, \
                    ini_obj_pos, ini_obj_scale, ini_obj_axis_red_wrd, ini_obj_axis_green_wrd,ini_shape_code

        elif mode in {'val', 'tes'}:
            return batch_size, frame_raw_invdistance_map, frame_clopped_mask, \
                    frame_clopped_distance_map, frame_bbox_list, frame_rays_d_cam, \
                    frame_clopped_depth_map, frame_normalized_depth_map, frame_avg_depth_map, frame_bbox_info, gt_shape_code, \
                    frame_w2c, frame_gt_obj_axis_green_wrd, frame_gt_obj_axis_red_wrd, \
                    frame_camera_pos, frame_obj_pos, frame_obj_scale, frame_obj_rot, \
                    canonical_distance_map, canonical_camera_pos, canonical_camera_rot, instance_id, path, \
                    ini_obj_pos, ini_obj_scale, ini_obj_axis_red_wrd, ini_obj_axis_green_wrd,ini_shape_code,
    


    # def forward(self, batch_size, opt_frame_num, normalized_depth_map, clopped_mask, clopped_distance_map, pre_etimations, 
    #     w2c, cam_pos_wrd, rays_d_cam, bbox_info, frame_obj_rot=False, gt_obj_pos_wrd=False, 
    #     gt_obj_axis_green_wrd=False, gt_obj_axis_red_wrd=False, gt_obj_scale=False, gt_shape_code=False, frame_idx_list=False, 
    #     cim2im_scale=False, im2cam_scale=False, bbox_center=False, avg_depth_map=False, 
    #     model_mode='train', batch_idx=0, optim_idx=0):
    def forward(self, batch_size, opt_frame_num, normalized_depth_map, clopped_mask, clopped_distance_map, pre_etimations, 
        w2c, cam_pos_wrd, rays_d_cam, bbox_info, cim2im_scale=False, im2cam_scale=False, bbox_center=False, avg_depth_map=False, 
        model_mode='train', batch_idx=0, optim_idx=0):
        pre_obj_pos_wrd = pre_etimations['pos']
        pre_obj_axis_green_wrd = pre_etimations['green']
        pre_obj_axis_red_wrd = pre_etimations['red']
        pre_obj_scale_wrd = pre_etimations['scale']
        pre_shape_code = pre_etimations['shape']

        with torch.no_grad():
            # Reshape pre estimation to [batch, frame, ?]
            inp_pre_obj_pos_wrd = pre_obj_pos_wrd[:, None, :].expand(-1, opt_frame_num, -1).reshape(-1, 3)
            inp_pre_obj_axis_green_wrd = pre_obj_axis_green_wrd[:, None, :].expand(-1, opt_frame_num, -1).reshape(-1, 3)
            inp_pre_obj_axis_red_wrd = pre_obj_axis_red_wrd[:, None, :].expand(-1, opt_frame_num, -1).reshape(-1, 3)
            inp_pre_obj_scale_wrd = pre_obj_scale_wrd[:, None, :].expand(-1, opt_frame_num, -1).reshape(-1, 1)
            inp_pre_shape_code = pre_shape_code[:, None, :].expand(-1, opt_frame_num, -1).reshape(-1, self.ddf.latent_size)
            pre_obj_axis_blue_wrd = torch.cross(pre_obj_axis_red_wrd, pre_obj_axis_green_wrd, dim=-1)
            pre_obj_axis_blue_wrd = F.normalize(pre_obj_axis_blue_wrd, dim=-1)
            orthogonal_red = torch.cross(pre_obj_axis_green_wrd, pre_obj_axis_blue_wrd, dim=-1)
            pre_o2w = torch.stack([orthogonal_red, pre_obj_axis_green_wrd, pre_obj_axis_blue_wrd], dim=-1)
            inp_pre_o2w = pre_o2w[:, None, :, :].expand(-1, opt_frame_num, -1, -1).reshape(-1, 3, 3)
            # ###############################################################
            # inp_pre_o2w = frame_obj_rot[:, frame_idx_list].reshape(-1, 3, 3)
            # inp_pre_obj_pos_wrd = gt_obj_pos_wrd.reshape(-1, 3)
            # inp_pre_obj_axis_green_wrd = gt_obj_axis_green_wrd.expand(-1, opt_frame_num, -1).reshape(-1, 3)
            # inp_pre_obj_axis_red_wrd = gt_obj_axis_red_wrd.expand(-1, opt_frame_num, -1).reshape(-1, 3)
            # inp_pre_obj_scale_wrd = gt_obj_scale.reshape(-1, 1)
            # inp_pre_shape_code = gt_shape_code[:, None, :].expand(-1, opt_frame_num, -1).reshape(-1, self.ddf.latent_size)
            # ###############################################################

            # Simulate DDF.
            inp_pre_obj_axis_green_cam = torch.sum(inp_pre_obj_axis_green_wrd[..., None, :]*w2c, dim=-1)
            inp_pre_obj_axis_red_cam = torch.sum(inp_pre_obj_axis_red_wrd[..., None, :]*w2c, dim=-1)
            est_clopped_mask, est_clopped_distance_map = render_distance_map_from_axis(
                                                            H = self.ddf_H, 
                                                            obj_pos_wrd = inp_pre_obj_pos_wrd, 
                                                            axis_green = inp_pre_obj_axis_green_cam, 
                                                            axis_red = inp_pre_obj_axis_red_cam, 
                                                            obj_scale = inp_pre_obj_scale_wrd[:, 0], 
                                                            input_lat_vec = inp_pre_shape_code, 
                                                            cam_pos_wrd = cam_pos_wrd, 
                                                            rays_d_cam = rays_d_cam, 
                                                            w2c = w2c.detach(), 
                                                            ddf = self.ddf, 
                                                            with_invdistance_map = False)
            inp_pre_mask = est_clopped_mask.detach()
            _, inp_pre_depth_map, _ = get_normalized_depth_map(est_clopped_mask, est_clopped_distance_map, rays_d_cam, avg_depth_map)

            # Get inputs.
            if self.input_type == 'depth':
                inp = torch.stack([
                            normalized_depth_map, 
                            clopped_mask, 
                            inp_pre_depth_map, 
                            inp_pre_mask, 
                            normalized_depth_map - inp_pre_depth_map]
                            , dim=1).reshape(batch_size, opt_frame_num, -1, self.ddf_H, self.ddf_H).detach()
            if self.input_type == 'osmap':
                obs_OSMap_obj = get_OSMap(clopped_distance_map, rays_d_cam, w2c, cam_pos_wrd, inp_pre_o2w, inp_pre_obj_pos_wrd, inp_pre_obj_scale_wrd, clopped_mask)
                est_OSMap_obj = get_OSMap(est_clopped_distance_map, rays_d_cam, w2c, cam_pos_wrd, inp_pre_o2w, inp_pre_obj_pos_wrd, inp_pre_obj_scale_wrd, est_clopped_mask)
                diff_distance_map = clopped_distance_map - est_clopped_distance_map
                diff_mask = est_clopped_mask + clopped_mask
                diff_OSMap_obj = get_diff_OSMap(diff_distance_map, rays_d_cam, w2c, inp_pre_o2w, inp_pre_obj_scale_wrd, diff_mask)
                inp = torch.cat([
                            obs_OSMap_obj.permute(0, 3, 1, 2),    # [batch*seq, 3, H, W]
                            clopped_mask[:, None, :, :],          # [batch*seq, 1, H, W]
                            est_OSMap_obj.permute(0, 3, 1, 2),    # [batch*seq, 3, H, W]
                            est_clopped_mask[:, None, :, :],      # [batch*seq, 1, H, W]
                            diff_OSMap_obj.permute(0, 3, 1, 2)]   # [batch*seq, 3, H, W]
                            , dim=1).reshape(batch_size, opt_frame_num, -1, self.ddf_H, self.ddf_H).detach()
                ##################################################
                fig = plt.figure()
                ax = Axes3D(fig)
                for ind_1 in range(opt_frame_num):
                    ind_1 += 1 * opt_frame_num
                    point_1 = est_OSMap_obj[ind_1][est_clopped_mask[ind_1]]
                    point_1 = point_1.to('cpu').detach().numpy().copy()
                    ax.scatter(point_1[::3, 0], point_1[::3, 1], point_1[::3, 2], marker="o", linestyle='None', c='c', s=0.05)
                    point_2 = obs_OSMap_obj[ind_1][clopped_mask[ind_1]]
                    point_2 = point_2.to('cpu').detach().numpy().copy()
                    ax.scatter(point_2[::3, 0], point_2[::3, 1], point_2[::3, 2], marker="o", linestyle='None', c='m', s=0.05)
                ax.view_init(elev=0, azim=90)
                fig.savefig("tes_00_90.png")
                ax.view_init(elev=0, azim=0)
                fig.savefig("tes_00_00.png")
                ax.view_init(elev=45, azim=45)
                fig.savefig("tes_45_45.png")
                plt.close()
                import pdb; pdb.set_trace()
                ##################################################

        # Get update value.
        if self.optimizer_type == 'optimize_former':
            if self.positional_encoding_mode in {'non'}:
                positional_encoding_target = False
            elif self.positional_encoding_mode in {'add', 'cat'}:
                positional_encoding_target = get_positional_encoding(
                                                cam_pos_wrd, rays_d_cam, batch_size, opt_frame_num, w2c, 
                                                inp_pre_obj_pos_wrd, inp_pre_o2w, inp_pre_obj_scale_wrd, self.image_lengs, bbox_info)

            diff_pos_wrd, diff_obj_axis_green_wrd, diff_obj_axis_red_wrd, diff_scale, diff_shape_code \
                = self.df_net(inp, rays_d_cam, pre_obj_scale_wrd, pre_shape_code, pre_o2w, 
                inp_pre_obj_scale_wrd, inp_pre_shape_code, inp_pre_o2w, positional_encoding_target, model_mode=model_mode)

        elif self.optimizer_type == 'origin':
            diff_pos_wrd, diff_obj_axis_green_wrd, diff_obj_axis_red_wrd, diff_scale, diff_shape_code \
                = self.df_net(inp, inp_pre_obj_pos_wrd, inp_pre_obj_axis_green_wrd, inp_pre_obj_axis_red_wrd, inp_pre_obj_scale_wrd, inp_pre_shape_code, \
                cim2im_scale, im2cam_scale, bbox_center, avg_depth_map, w2c, cam_pos_wrd, rays_d_cam, inp_pre_o2w, model_mode = model_mode)
        
        # ##################################################
        # # if model_mode=='train':
        # gt = clopped_distance_map
        # est = est_clopped_distance_map
        # str_batch_idx = str(batch_idx).zfill(5)
        # str_optim_idx = str(optim_idx).zfill(2)
        # check_map = torch.cat([gt, est, torch.abs(gt-est)], dim=-1)
        # check_map = torch.cat([map_i for map_i in check_map], dim=0)
        # check_map_torch(check_map, f'tes_{str_batch_idx}_{str_optim_idx}.png')
        # import pdb; pdb.set_trace()
        # #################################################
        # # end_step = model_mode=='train' and optim_idx==(self.optim_num-1)
        # # end_step = end_step or (model_mode=='val' and optim_idx==self.optim_num)
        # # if end_step: import pdb; pdb.set_trace()
        # # ##################################################
        # for idx, layers in enumerate(self.df_net.encoder.layers):
        #     # attn_weights = layers.attn_output_weights_sa.detach().to('cpu').numpy()
        #     import pdb; pdb.set_trace()
        #     attn_weights = layers.attn_output_weights_sa.detach().to('cpu').numpy()
        # # ##################################################

        return diff_pos_wrd, diff_obj_axis_green_wrd, diff_obj_axis_red_wrd, diff_scale, diff_shape_code



    def perform_init_estimation(self, batch_size, init_frame_list, 
        normalized_depth_map, clopped_mask, w2c, cam_pos_wrd, bbox_list, bbox_info, avg_depth_map):

        # Est.
        inp = torch.stack([normalized_depth_map, clopped_mask], 1).detach()
        est_obj_pos_cim, est_obj_axis_green_cam, est_obj_axis_red_cam, est_scale_cim, est_shape_code, _ = self.init_net(inp, bbox_info)

        # Convert coordinate.
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

        # Get average of init estimations.
        est_obj_pos_wrd = est_obj_pos_wrd.reshape(batch_size, -1, 3)[:, init_frame_list].mean(1)
        est_obj_scale = est_obj_scale.reshape(batch_size, -1, 1)[:, init_frame_list].mean(1)
        est_obj_axis_green_wrd = est_obj_axis_green_wrd.reshape(batch_size, -1, 3)[:, init_frame_list].mean(1)
        est_obj_axis_red_wrd = est_obj_axis_red_wrd.reshape(batch_size, -1, 3)[:, init_frame_list].mean(1)
        est_shape_code = est_shape_code.reshape(batch_size, -1, self.ddf.latent_size)[:, init_frame_list].mean(1)

        return est_obj_pos_wrd, est_obj_scale, est_obj_axis_green_wrd, est_obj_axis_red_wrd, est_shape_code



    def training_step(self, batch, batch_idx):

        # Get batch info.
        with torch.no_grad():
            batch_size, frame_raw_invdistance_map, frame_clopped_mask, \
            frame_clopped_distance_map, frame_clopped_invdistance_map, frame_bbox_list, \
            frame_rays_d_cam, frame_clopped_depth_map, frame_normalized_depth_map, \
            frame_avg_depth_map, frame_bbox_info, gt_shape_code, frame_w2c, \
            frame_gt_obj_axis_green_cam, frame_gt_obj_axis_red_cam, frame_gt_obj_axis_green_wrd, \
            frame_gt_obj_axis_red_wrd, frame_camera_pos, frame_obj_pos, frame_obj_scale, frame_obj_rot, \
            ini_obj_pos, ini_obj_scale, ini_obj_axis_red_wrd, ini_obj_axis_green_wrd,ini_shape_code = self.preprocess(batch, mode='train')

        # Get current frames.
        with torch.no_grad():
            # Set frames
            frame_idx_list = list(range(self.itr_frame_num))
            opt_frame_num = len(frame_idx_list)
        
            # Get current maps.
            raw_invdistance_map = frame_raw_invdistance_map[:, frame_idx_list].reshape(-1, self.input_H, self.input_W).detach()
            clopped_mask = frame_clopped_mask[:, frame_idx_list].reshape(-1, self.ddf_H, self.ddf_H).detach()
            clopped_distance_map = frame_clopped_distance_map[:, frame_idx_list].reshape(-1, self.ddf_H, self.ddf_H).detach()
            clopped_invdistance_map = frame_clopped_invdistance_map[:, frame_idx_list].reshape(-1, self.ddf_H, self.ddf_H).detach()
            bbox_list = frame_bbox_list[:, frame_idx_list].reshape(-1, 2, 2).detach()
            rays_d_cam = frame_rays_d_cam[:, frame_idx_list].reshape(-1, self.ddf_H, self.ddf_H, 3).detach()
            clopped_depth_map = frame_clopped_depth_map[:, frame_idx_list].reshape(-1, self.ddf_H, self.ddf_H).detach()
            normalized_depth_map = frame_normalized_depth_map[:, frame_idx_list].reshape(-1, self.ddf_H, self.ddf_H).detach()
            avg_depth_map = frame_avg_depth_map[:, frame_idx_list].reshape(-1).detach()
            bbox_info = frame_bbox_info[:, frame_idx_list].reshape(-1, 7).detach()
            cim2im_scale, im2cam_scale, bbox_center = get_clopping_infos(bbox_list, avg_depth_map, self.fov)

            # Get current GT.
            w2c = frame_w2c[:, frame_idx_list].reshape(-1, 3, 3).detach()
            gt_obj_axis_green_cam = frame_gt_obj_axis_green_cam[:, frame_idx_list].detach()
            gt_obj_axis_red_cam = frame_gt_obj_axis_red_cam[:, frame_idx_list].detach()
            gt_obj_axis_green_wrd = frame_gt_obj_axis_green_wrd[:, frame_idx_list].detach()
            gt_obj_axis_red_wrd = frame_gt_obj_axis_red_wrd[:, frame_idx_list].detach()
            cam_pos_wrd = frame_camera_pos[:, frame_idx_list].reshape(-1, 3).detach()
            gt_obj_pos_wrd = frame_obj_pos[:, frame_idx_list].detach()
            gt_obj_scale = frame_obj_scale[:, frame_idx_list][..., None].detach()

        ###################################
        #####    Start Optimizing     #####
        ###################################
        # Initialization.
        pre_obj_pos_wrd = ini_obj_pos.detach()
        pre_obj_scale_wrd = ini_obj_scale.detach()
        pre_obj_axis_green_wrd = ini_obj_axis_green_wrd.detach()
        pre_obj_axis_red_wrd = ini_obj_axis_red_wrd.detach()
        pre_shape_code = ini_shape_code.detach()

        if batch_idx == 0:
            self.ddf.eval()

        if self.lr == -1: # warm_up
            sch = self.lr_schedulers()

        for optim_idx in range(self.optim_num):

            # Set optimizers
            opt = self.optimizers()
            opt.zero_grad()

            ###################################
            #####      Perform dfnet.     #####
            ###################################
            # Get update values.
            pre_etimations = {
                            'pos' : pre_obj_pos_wrd, 
                            'green' : pre_obj_axis_green_wrd, 
                            'red' : pre_obj_axis_red_wrd, 
                            'scale' : pre_obj_scale_wrd, 
                            'shape' : pre_shape_code, 
                            }
            diff_pos_wrd, diff_obj_axis_green_wrd, diff_obj_axis_red_wrd, diff_scale, diff_shape_code \
                = self.forward(batch_size, opt_frame_num, normalized_depth_map, clopped_mask, clopped_distance_map, pre_etimations, 
                w2c, cam_pos_wrd, rays_d_cam, bbox_info, cim2im_scale, im2cam_scale, bbox_center, avg_depth_map, 'train', batch_idx, optim_idx)

            # Update estimations.
            est_obj_pos_wrd = pre_obj_pos_wrd + diff_pos_wrd
            est_obj_scale = pre_obj_scale_wrd * diff_scale
            est_obj_axis_green_wrd = F.normalize(pre_obj_axis_green_wrd + diff_obj_axis_green_wrd, dim=-1)
            est_obj_axis_red_wrd = F.normalize(pre_obj_axis_red_wrd + diff_obj_axis_red_wrd, dim=-1)
            est_shape_code = pre_shape_code + diff_shape_code

            # Cal loss against integrated estimations.
            if self.loss_timing=='after_mean':
                loss_pos = F.mse_loss(est_obj_pos_wrd, gt_obj_pos_wrd[:, -1].detach())
                loss_scale = F.mse_loss(est_obj_scale, gt_obj_scale[:, -1].detach())
                loss_axis_red = torch.mean(-self.cossim(est_obj_axis_red_wrd, gt_obj_axis_red_wrd[:, -1].detach()) + 1.)
                loss_axis_green = torch.mean(-self.cossim(est_obj_axis_green_wrd, gt_obj_axis_green_wrd[:, -1].detach()) + 1.)
                loss_shape_code = F.mse_loss(est_shape_code, gt_shape_code.detach())
            elif self.loss_timing=='before_mean':
                loss_pos = F.mse_loss(est_obj_pos_wrd, gt_obj_pos_wrd[None, :, -1].expand(self.itr_frame_num, -1, -1).detach())
                loss_scale = F.mse_loss(est_obj_scale, gt_obj_scale[None, :, -1].expand(self.itr_frame_num, -1, -1).detach())
                loss_axis_red = torch.mean(-self.cossim(est_obj_axis_red_wrd, gt_obj_axis_red_wrd[None, :, -1].expand(self.itr_frame_num, -1, -1).detach()) + 1.)
                loss_axis_green = torch.mean(-self.cossim(est_obj_axis_green_wrd, gt_obj_axis_green_wrd[None, :, -1].expand(self.itr_frame_num, -1, -1).detach()) + 1.)
                loss_shape_code = F.mse_loss(est_shape_code, gt_shape_code[None, :, :].expand(self.itr_frame_num, -1, -1).detach())
                # Integrate estimations.
                est_obj_pos_wrd = est_obj_pos_wrd.mean(0)
                est_obj_scale = est_obj_scale.mean(0)
                est_obj_axis_green_wrd = est_obj_axis_green_wrd.mean(0)
                est_obj_axis_red_wrd = est_obj_axis_red_wrd.mean(0)
                est_shape_code = est_shape_code.mean(0)

            # Get depth loss.
            loss_depth = torch.zeros_like(loss_pos).detach() # Dummy

            # Integrate each optim step losses.
            loss_axis = loss_axis_green + loss_axis_red
            loss = self.L_p * loss_pos + self.L_s * loss_scale + self.L_a * loss_axis + self.L_c * loss_shape_code + self.L_d * loss_depth

            # Optimizer step.
            self.manual_backward(loss)
            opt.step()

            # Save pre estimation.
            with torch.no_grad():
                pre_obj_pos_wrd = est_obj_pos_wrd.clone().detach()
                pre_obj_scale_wrd = est_obj_scale.clone().detach()
                pre_obj_axis_green_wrd = est_obj_axis_green_wrd.clone().detach()
                pre_obj_axis_red_wrd = est_obj_axis_red_wrd.clone().detach()
                pre_shape_code = est_shape_code.clone().detach()
        
        if self.lr == -1: # warm_up
            sch.step()

        return {'loss': loss.detach(), 
                'loss_pos':loss_pos.detach(), 
                'loss_scale': loss_scale.detach(), 
                'loss_axis_red': loss_axis_red.detach(), 
                'loss_shape_code': loss_shape_code.detach(), 
                'loss_depth': loss_depth.detach()}



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
        avg_loss_depth = torch.stack([x['loss_depth'] for x in outputs]).mean()
        self.log_dict({'train/loss_depth': avg_loss_depth, "step": current_epoch})
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log_dict({'train/lr': current_lr, "step": current_epoch})

        # Save ckpt.
        if (self.current_epoch + 1) % self.save_interval == 0:
            ckpt_name = str(self.current_epoch + 1).zfill(10) + '.ckpt'
            ckpt_path = os.path.join(self.trainer.log_dir, 'checkpoints', ckpt_name)
            trainer.save_checkpoint(ckpt_path)



    def test_step(self, batch, batch_idx, step_mode='tes'):
        batch_size, frame_raw_invdistance_map, frame_clopped_mask, \
        frame_clopped_distance_map, frame_bbox_list, frame_rays_d_cam, \
        frame_clopped_depth_map, frame_normalized_depth_map, frame_avg_depth_map, frame_bbox_info, gt_shape_code, \
        frame_w2c, frame_gt_obj_axis_green_wrd, frame_gt_obj_axis_red_wrd, \
        frame_camera_pos, frame_obj_pos, frame_obj_scale, frame_obj_rot, \
        canonical_distance_map, canonical_camera_pos, canonical_camera_rot, instance_id, path, \
        ini_obj_pos, ini_obj_scale, ini_obj_axis_red_wrd, ini_obj_axis_green_wrd,ini_shape_code = self.preprocess(batch, mode=step_mode)


        ###################################
        #####     Start inference.    #####
        ###################################
        # import time # Time.
        # time_sta = time.time()
        ###################################
        itr_err_list = {'pos':[], 'scale':[], 'red':[], 'green':[], 'shape':[]}
        for optim_idx in range(self.optim_num+1):

            if self.optim_mode=='optimall': # 初期化も、その後の最適化も全てのフレームを使って行う。
                frame_idx_list = init_frame_list = list(range(self.itr_frame_num))
                opt_frame_num = len(frame_idx_list)

            elif self.optim_mode=='progressive':
                if optim_idx == 0:
                    frame_idx_list = list(range(self.itr_frame_num))
                    if self.init_mode=='all':
                        init_frame_list = frame_idx_list
                    elif self.init_mode=='single':
                        init_frame_list = [0]
                else:
                    end_frame = min(self.itr_frame_num, optim_idx)
                    frame_idx_list = list(range(end_frame))
                    opt_frame_num = len(frame_idx_list)

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
            ###   Perform initialization.   ###
            ###################################
            if optim_idx == 0:
                if self.use_init_net:
                    # Get init values.
                    est_obj_pos_wrd, est_obj_scale, est_obj_axis_green_wrd, est_obj_axis_red_wrd, est_shape_code \
                    = self.perform_init_estimation(batch_size, init_frame_list, normalized_depth_map, clopped_mask,
                    w2c, cam_pos_wrd, bbox_list, bbox_info, avg_depth_map)
                else:
                    # Get init values.
                    est_obj_pos_wrd = ini_obj_pos.detach()
                    est_obj_scale = ini_obj_scale.detach()
                    est_obj_axis_green_wrd = ini_obj_axis_green_wrd.detach()
                    est_obj_axis_red_wrd = ini_obj_axis_red_wrd.detach()
                    est_shape_code = ini_shape_code.detach()
                # Save bbox preprocess.
                cim2im_scale, im2cam_scale, bbox_center = get_clopping_infos(bbox_list, avg_depth_map, self.fov)
                cim2im_scale_list = cim2im_scale.reshape(batch_size, self.itr_frame_num).detach()
                im2cam_scale_list = im2cam_scale.reshape(batch_size, self.itr_frame_num).detach()
                bbox_center_list = bbox_center.reshape(batch_size, self.itr_frame_num, 2).detach()

            ###################################
            #####      Perform dfnet.     #####
            ###################################
            else:
                # Get update values.
                pre_etimations = {
                                'pos' : pre_obj_pos_wrd, 
                                'green' : pre_obj_axis_green_wrd, 
                                'red' : pre_obj_axis_red_wrd, 
                                'scale' : pre_obj_scale_wrd, 
                                'shape' : pre_shape_code, 
                                }
                diff_pos_wrd, diff_obj_axis_green_wrd, diff_obj_axis_red_wrd, diff_scale, diff_shape_code \
                    = self.forward(batch_size, opt_frame_num, normalized_depth_map, clopped_mask, clopped_distance_map, pre_etimations, 
                    w2c, cam_pos_wrd, rays_d_cam, bbox_info, cim2im_scale, im2cam_scale, bbox_center, avg_depth_map, 'val', batch_idx, optim_idx)
                    # "bbox_info", frame_obj_rot, gt_obj_pos_wrd, gt_obj_axis_green_wrd, gt_obj_axis_red_wrd, gt_obj_scale, gt_shape_code, frame_idx_list, "cim2im_scale"

                # Update estimations.
                est_obj_pos_wrd = pre_obj_pos_wrd + diff_pos_wrd
                est_obj_scale = pre_obj_scale_wrd * diff_scale
                est_obj_axis_green_wrd = F.normalize(pre_obj_axis_green_wrd + diff_obj_axis_green_wrd, dim=-1)
                est_obj_axis_red_wrd = F.normalize(pre_obj_axis_red_wrd + diff_obj_axis_red_wrd, dim=-1)
                est_shape_code = pre_shape_code + diff_shape_code


            # Save pre estimation.
            pre_obj_pos_wrd = est_obj_pos_wrd.clone().detach()
            pre_obj_scale_wrd = est_obj_scale.clone().detach()
            pre_obj_axis_red_wrd = est_obj_axis_red_wrd.clone().detach()
            pre_obj_axis_green_wrd = est_obj_axis_green_wrd.clone().detach()
            pre_shape_code = est_shape_code.clone().detach()

            # ##################################################
            # err_pos_i = torch.abs(pre_obj_pos_wrd - frame_obj_pos[:, 0]).mean(dim=-1)
            # err_scale_i = torch.abs(1 - pre_obj_scale_wrd[:, 0] / frame_obj_scale[:, 0])
            # err_axis_red_cos_sim_i = self.cossim(pre_obj_axis_red_wrd, frame_gt_obj_axis_red_wrd[:, 0]).clamp(min=self.cosssim_min, max=self.cosssim_max)
            # err_axis_red_i = torch.acos(err_axis_red_cos_sim_i) * 180 / torch.pi
            # err_axis_green_cos_sim_i = self.cossim(pre_obj_axis_green_wrd, frame_gt_obj_axis_green_wrd[:, 0]).clamp(min=self.cosssim_min, max=self.cosssim_max)
            # err_axis_green_i = torch.acos(err_axis_green_cos_sim_i) * 180 / torch.pi
            # depth_error = []
            # for shape_i, (gt_distance_map, cam_pos_wrd, w2c) in enumerate(zip(canonical_distance_map.permute(1, 0, 2, 3), 
            #                                                                     canonical_camera_pos.permute(1, 0, 2), 
            #                                                                     canonical_camera_rot.permute(1, 0, 2, 3))):
            #     # Get simulation results.
            #     rays_d_cam = get_ray_direction(self.ddf_H, self.fov).expand(batch_size, -1, -1, -1).to(w2c)
            #     _, est_distance_map = get_canonical_map(
            #                                 H = self.ddf_H, 
            #                                 cam_pos_wrd = cam_pos_wrd, 
            #                                 rays_d_cam = rays_d_cam, 
            #                                 w2c = w2c, 
            #                                 input_lat_vec = pre_shape_code, 
            #                                 ddf = self.ddf, 
            #                                 )
            #     depth_error.append(torch.abs(gt_distance_map-est_distance_map).mean(dim=-1).mean(dim=-1))
            # depth_error_i = torch.stack(depth_error, dim=-1).mean(dim=-1)
            # itr_err_list['pos'].append(err_pos_i)
            # itr_err_list['scale'].append(err_scale_i)
            # itr_err_list['red'].append(err_axis_red_i)
            # itr_err_list['green'].append(err_axis_green_i)
            # itr_err_list['shape'].append(depth_error_i)
            # ##################################################

            if self.only_init_net:
                break

        ###################################
        # time_end = time.time()
        # with open(f'time.txt', 'a') as file:
        #     file.write(str(time_end- time_sta) + '\n')
        # ###################################

        if step_mode=='tes':
            # Val shape.
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
            err_pos = torch.abs(pre_obj_pos_wrd - gt_obj_pos_wrd[:, -1]).mean(dim=-1)
            err_scale = 100 * torch.abs((pre_obj_scale_wrd - gt_obj_scale[:, -1]) / gt_obj_scale[:, -1])
            err_axis_red_cos_sim = self.cossim(pre_obj_axis_red_wrd, gt_obj_axis_red_wrd[:, -1]).clamp(min=self.cosssim_min, max=self.cosssim_max)
            err_axis_red = torch.acos(err_axis_red_cos_sim) * 180 / torch.pi
            err_axis_green_cos_sim = self.cossim(pre_obj_axis_green_wrd, gt_obj_axis_green_wrd[:, -1]).clamp(min=self.cosssim_min, max=self.cosssim_max)
            err_axis_green = torch.acos(err_axis_green_cos_sim) * 180 / torch.pi
            depth_error = torch.stack(depth_error, dim=-1).mean(dim=-1)

            return {'err_pos':err_pos.detach(), 
                    'err_scale': err_scale.detach(), 
                    'err_axis_red': err_axis_red.detach(), 
                    'err_axis_green':err_axis_green.detach(), 
                    'depth_error':depth_error.detach(), 
                    'path':np.array(path), 
                    'itr_err_list':itr_err_list, }
        
        elif step_mode=='val':
            # Cal loss to monitor performance.
            loss_pos = F.mse_loss(est_obj_pos_wrd, gt_obj_pos_wrd[:, -1].detach())
            loss_scale = F.mse_loss(est_obj_scale, gt_obj_scale[:, -1].detach())
            loss_axis_red = torch.mean(-self.cossim(est_obj_axis_red_wrd, gt_obj_axis_red_wrd[:, -1].detach()) + 1.)
            loss_axis_green = torch.mean(-self.cossim(est_obj_axis_green_wrd, gt_obj_axis_green_wrd[:, -1].detach()) + 1.)
            loss_shape_code = F.mse_loss(est_shape_code, gt_shape_code.detach())
            loss_axis = loss_axis_green + loss_axis_red
            loss = self.L_p * loss_pos + self.L_s * loss_scale + self.L_a * loss_axis + self.L_c * loss_shape_code

            # Cal err.
            err_axis_red_cos_sim = self.cossim(pre_obj_axis_red_wrd, frame_gt_obj_axis_red_wrd[:, 0]).clamp(min=self.cosssim_min, max=self.cosssim_max)
            err_axis_red = torch.acos(err_axis_red_cos_sim) * 180 / torch.pi
            err_axis_green_cos_sim = self.cossim(pre_obj_axis_green_wrd, frame_gt_obj_axis_green_wrd[:, 0]).clamp(min=self.cosssim_min, max=self.cosssim_max)
            err_axis_green = torch.acos(err_axis_green_cos_sim) * 180 / torch.pi
            
            return {'loss':loss.detach(), 
                    'loss_pos':loss_pos.detach(), 
                    'loss_scale':loss_scale.detach(), 
                    'loss_axis_red':loss_axis_red.detach(), 
                    'loss_axis_green':loss_axis_green.detach(), 
                    'loss_shape_code':loss_shape_code.detach(), 
                    'err_axis_red': err_axis_red.detach(), 
                    'err_axis_green':err_axis_green.detach(), }



    def test_epoch_end(self, outputs):
        # Log loss.
        err_pos_list = torch.cat([x['err_pos'] for x in outputs], dim=0).to('cpu').detach().numpy().copy()
        med_err_pos = np.median(err_pos_list)
        avg_err_pos = np.mean(err_pos_list)
        err_scale_list = torch.cat([x['err_scale'] for x in outputs], dim=0).to('cpu').detach().numpy().copy()
        med_err_scale = np.median(err_scale_list)
        avg_err_scale = np.mean(err_scale_list)
        err_red_list = torch.cat([x['err_axis_red'] for x in outputs], dim=0).to('cpu').detach().numpy().copy()
        med_err_red = np.median(err_red_list)
        avg_err_red = np.mean(err_red_list)
        err_green_list = torch.cat([x['err_axis_green'] for x in outputs], dim=0).to('cpu').detach().numpy().copy()
        med_err_green = np.median(err_green_list)
        avg_err_green = np.mean(err_green_list)
        err_depth_list = torch.cat([x['depth_error'] for x in outputs], dim=0).to('cpu').detach().numpy().copy()
        med_err_depth = np.median(err_depth_list)
        avg_err_depth = np.mean(err_depth_list)
        path_list = np.concatenate([x['path'] for x in outputs])

        with open(self.test_log_path, 'a') as file:
            file.write('avg_err_pos  ' + ' : '  + str(avg_err_pos.item())   + ' : ' + str(med_err_pos.item()) + '\n')
            file.write('avg_err_scale' + ' : '  + str(avg_err_scale.item()) + ' : ' + str(med_err_scale.item()) + '\n')
            file.write('avg_err_red  ' + ' : '  + str(avg_err_red.item())   + ' : ' + str(med_err_red.item()) + '\n')
            file.write('avg_err_green' + ' : '  + str(avg_err_green.item()) + ' : ' + str(med_err_green.item()) + '\n')
            file.write('avg_err_depth' + ' : '  + str(avg_err_depth.item()) + ' : ' + str(med_err_depth.item()) + '\n')
        log_dict = {'pos':err_pos_list, 
                    'scale':err_scale_list, 
                    'red':err_red_list, 
                    'green':err_green_list, 
                    'depth':err_depth_list, 
                    'path': path_list}
        pickle_dump(log_dict, self.test_log_path.split('.txt')[0] + '_error.pickle')

        # # Log error at each step.
        # for key_i in ['pos', 'scale', 'red', 'green', 'shape']:
        #     itr_err_list = torch.cat([
        #         torch.stack(x['itr_err_list'][key_i], dim=-1) for x in outputs]
        #         , dim=0).to('cpu').detach().numpy().copy()
        #     itr_err_mediam = np.median(itr_err_list, axis=0)
        #     itr_err_mean = np.mean(itr_err_list, axis=0)
        #     itr_log_median = 'median'
        #     itr_log_mean = 'mean'
        #     for mediam, mean in zip(itr_err_mediam, itr_err_mean):
        #         itr_log_median += (', ' + str(mediam))
        #         itr_log_mean += (', ' + str(mean))
        #     with open(self.test_log_path, 'a') as file:
        #         file.write('\n')
        #         file.write(key_i + '\n')
        #         file.write(itr_log_median + '\n')
        #         file.write(itr_log_mean + '\n')
        return 0



    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx, step_mode='val')


        
    def validation_epoch_end(self, outputs):
        # Cal loss.
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_loss_pos = torch.stack([x['loss_pos'] for x in outputs]).mean()
        avg_loss_scale = torch.stack([x['loss_scale'] for x in outputs]).mean()
        avg_loss_axis_green = torch.stack([x['loss_axis_green'] for x in outputs]).mean()
        avg_loss_axis_red = torch.stack([x['loss_axis_red'] for x in outputs]).mean()
        avg_loss_shape_code = torch.stack([x['loss_shape_code'] for x in outputs]).mean()

        # Cal error.
        err_red_list = torch.cat([x['err_axis_red'] for x in outputs], dim=0)
        med_err_red = torch.median(err_red_list)
        avg_err_red = torch.mean(err_red_list)
        err_green_list = torch.cat([x['err_axis_green'] for x in outputs], dim=0)
        med_err_green = torch.median(err_green_list)
        avg_err_green = torch.mean(err_green_list)

        # Log.
        current_epoch = torch.tensor(self.current_epoch + 1., dtype=avg_loss.dtype)
        self.log_dict({'val/total_loss': avg_loss, "step": current_epoch})
        self.log_dict({'val/loss_pos': avg_loss_pos, "step": current_epoch})
        self.log_dict({'val/loss_scale': avg_loss_scale, "step": current_epoch})
        self.log_dict({'val/loss_axis_green': avg_loss_axis_green, "step": current_epoch})
        self.log_dict({'val/loss_axis_red': avg_loss_axis_red, "step": current_epoch})
        self.log_dict({'val/loss_shape_code': avg_loss_shape_code, "step": current_epoch})
        self.log_dict({'val/err_red_med': med_err_red, "step": current_epoch})
        self.log_dict({'val/err_red_avg': avg_err_red, "step": current_epoch})
        self.log_dict({'val/err_green_med': med_err_green, "step": current_epoch})
        self.log_dict({'val/err_green_avg': avg_err_green, "step": current_epoch})
        return 0



    def check_model_info (self):
        self.model_params_dtype = list(self.mlp.parameters())[-1].dtype
        self.model_device = self.device



    def configure_optimizers(self):
        if self.lr == -1:
            optimizer = torch.optim.Adam([
                {"params": self.df_net.parameters()},
            ], betas=(0.9, 0.98), eps = 1.0e-9)
            scheduler = WarmupScheduler(optimizer, warmup_steps=4000)
            return [optimizer, ], [scheduler, ]
        elif self.lr > 0.:
            optimizer = torch.optim.Adam([
                {"params": self.df_net.parameters()},
            ], lr=self.lr, betas=(0.9, 0.999),)
            return optimizer





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


    if args.code_mode == 'TRAIN':
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
        
        if args.val_instance_list_txt == 'non':
            # Set models and Start training.
            ddf = DDF(args)
            ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args)
            ddf.eval()

            # Get model.
            # ckpt_path = 'lightning_logs/DeepTaR/chair/dfnet_list0_randR05_origin_after_date0721/checkpoints/0000000700.ckpt'
            ckpt_path = None
            model = original_optimizer(args, ddf)
            trainer.fit(
                model=model, 
                train_dataloaders=train_dataloader, 
                ckpt_path=ckpt_path, 
                val_dataloaders=None, 
                datamodule=None, 
                )
        
        else:
            # Create val dataloader.
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

            # Set model.
            if args.val_model_epoch > 0:
                ckpt_path = args.model_ckpt_path
            else:
                ckpt_path = None
            model = original_optimizer(args, ddf)
            model.only_init_net = False

            # Set validation configs.
            model.optim_mode = 'optimall'
            model.use_init_net = False
            if model.use_init_net:
                from train_ini import only_init_net
                model_ = only_init_net(args, ddf)
                model_ = model_.load_from_checkpoint(checkpoint_path=args.initnet_ckpt_path, args=args, ddf=ddf)
                model.init_net = model_.init_net
                del model_

            # Start training.
            trainer.fit(
                model=model, 
                train_dataloaders=train_dataloader, 
                val_dataloaders=val_dataloader, 
                datamodule=None, 
                ckpt_path=ckpt_path, 
                )


    elif args.code_mode == 'VAL':
        # Create dataloader.
        test_dataset = TaR_dataset(
            args, 
            'val', 
            args.test_instance_list_txt, 
            args.test_data_dir, 
            args.test_N_views, 
            )
        test_dataloader = data_utils.DataLoader(
            test_dataset, 
            batch_size=args.N_batch, 
            num_workers=args.num_workers, 
            drop_last=False, 
            shuffle=False, 
            )

        # Set models and Start training.
        ddf = DDF(args)
        ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args)
        ddf.eval()

        # Set model
        model = original_optimizer(args, ddf)
        model = model.load_from_checkpoint(checkpoint_path=args.model_ckpt_path, args=args, ddf=ddf)

        # Set init net
        model.use_init_net = False
        # if model.use_init_net:
        #     from train_ini import only_init_net
        #     model_ = only_init_net(args, ddf)
        #     model_ = model_.load_from_checkpoint(checkpoint_path=args.initnet_ckpt_path, args=args, ddf=ddf)
        #     model.init_net = model_.init_net
        #     del model_

        # Setting model.
        model.start_frame_idx = 0
        model.half_lambda_max = 0
        # model.model_mode = args.model_mode
        # if model.model_mode == 'only_init':
        #     model.only_init_net = True
        # else:
        model.only_init_net = False
        model.use_deep_optimizer = True
        model.use_adam_optimizer = not(model.use_deep_optimizer)
        model.optim_num = args.optim_num

        # Save logs.
        import datetime
        dt_now = datetime.datetime.now()
        time_log = dt_now.strftime('%Y_%m_%d_%H_%M_%S')

        os.mkdir('./txt/experiments/log/' + time_log)
        file_name = './txt/experiments/log/' + time_log + '/log.txt'
        model.test_log_path = file_name
        with open(file_name, 'a') as file:
            file.write('time_log : ' + time_log + '\n')
            if model.use_init_net:
                file.write('ini_ckpt_path : ' + args.initnet_ckpt_path + '\n')
            else:
                file.write('ini_ckpt_path : ' + 'non' + '\n')
                file.write('rand_P_range : ' + str(model.rand_P_range) + '\n')
                file.write('rand_S_range : ' + str(model.rand_S_range) + '\n')
                file.write('rand_R_range : ' + str(model.rand_R_range) + '\n')
                file.write('random_axis_num : ' + str(model.random_axis_num) + '\n')
                file.write('rand_Z_sigma : ' + str(model.rand_Z_sigma) + '\n')
            file.write('ckpt_path : ' + args.model_ckpt_path + '\n')
            file.write('test_N_views : ' + str(args.test_N_views) + '\n')
            file.write('test_instance_list_txt : ' + str(args.test_instance_list_txt) + '\n')
            file.write('\n')
            file.write('only_init_net : ' + str(model.only_init_net) + '\n')
            file.write('start_frame_idx : ' + str(model.start_frame_idx) + '\n')
            file.write('itr_frame_num : ' + str(model.itr_frame_num) + '\n')
            file.write('half_lambda_max : ' + str(model.half_lambda_max) + '\n')
            file.write('optim_num : ' + str(model.optim_num) + '\n')
            file.write('init_mode : ' + str(model.init_mode) + '\n')
            file.write('optim_mode : ' + str(model.optim_mode) + '\n')
            file.write('\n')

        # Test model.
        trainer = pl.Trainer(
            gpus=args.gpu_num, 
            strategy=DDPPlugin(find_unused_parameters=True), #=False), 
            enable_checkpointing = False,
            check_val_every_n_epoch = args.check_val_every_n_epoch,
            logger = pl.loggers.TensorBoardLogger(save_dir=os.getcwd(), version=f'val_trash', name='lightning_logs')
            )
        trainer.test(model, test_dataloader)
