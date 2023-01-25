print('start')
import os
import sys
import numpy as np
import random
import glob
import math
from copy import deepcopy
from tqdm import tqdm, trange
import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
import torch.utils.data as data
import time
from BackBone.train_backbone import *
from resnet import *
from parser_get_arg import *
from dataset import *
from often_use import *
from DDF.train_pl import DDF
from model import optimize_former
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732





class original_optimizer(pl.LightningModule):

    def __init__(self, args, pretrain_models):
        super().__init__()

        # Base configs
        self.fov = args.fov
        self.inp_H = args.input_H
        self.inp_W = args.input_W
        self.x_coord = torch.arange(0, self.inp_W)[None, :].expand(self.inp_H, -1)
        self.y_coord = torch.arange(0, self.inp_H)[:, None].expand(-1, self.inp_W)
        self.image_coord = torch.stack([self.y_coord, self.x_coord], dim=-1) # [H, W, (Y and X)]
        fov = torch.deg2rad(torch.tensor(self.fov, dtype=torch.float))
        self.image_lengs = 2 * torch.tan(fov*.5)
        self.ddf_H = args.ddf_H_W_during_dfnet
        self.rays_d_cam = get_ray_direction(self.ddf_H, self.fov)
        self.canonical_rays_d_cam = get_ray_direction(self.ddf_H, args.canonical_fov)
        self.total_obs_num = args.total_obs_num
        self.save_interval = args.save_interval
        self.ddf_instance_list = txt2list(args.ddf_instance_list_txt)
        train_instance_list = txt2list(args.train_instance_list_txt)
        self.train_instance_ids = [self.ddf_instance_list.index(instance_i) for instance_i in train_instance_list]
        self.rand_Z_center = ddf.lat_vecs(torch.tensor(self.train_instance_ids, device=ddf.device)).mean(0).clone().detach()
        self.view_position = args.view_position
        self.view_selection = args.view_selection
        self.total_itr = args.total_itr
        if self.view_selection == 'simultaneous':
            self.itr_per_frame = self.total_itr
        if self.view_selection == 'sequential':
            self.itr_per_frame = args.itr_per_frame # self.total_itr // self.total_obs_num
        self.lr = args.lr # / args.total_itr
        self.code_mode = args.code_mode

        # Make model
        self.ddf = pretrain_models['ddf']
        self.input_type = args.input_type
        self.backbone_encoder = pretrain_models['backbone_encoder'] # torch.nn.ModuleList()
        self.backbone_embedding_dim = self.backbone_encoder.embedding_dim
        self.main_layers_name = args.main_layers_name
        self.positional_encoding_mode = args.positional_encoding_mode
        self.df_net = optimize_former(view_selection = args.view_selection, 
                                      main_layers_name = args.main_layers_name, 
                                      positional_encoding_mode = args.positional_encoding_mode, 
                                      num_encoder_layers = args.num_encoder_layers, 
                                      num_decoder_layers = args.num_decoder_layers, 
                                      add_conf=args.add_conf, 
                                      layer_wise_attention = args.layer_wise_attention, 
                                      use_cls=args.use_cls, 
                                      latent_size = args.latent_size, 
                                      use_attn_mask=args.use_attn_mask, 
                                      total_obs_num=self.total_obs_num, 
                                      inp_itr_num=args.inp_itr_num, 
                                      itr_per_frame=self.itr_per_frame, 
                                      total_itr=args.total_itr, 
                                      dec_inp_type=args.dec_inp_type, 
                                      scale_dim=args.scale_dim)
        self.rs = {'train': np.random.RandomState(3407)} # 3407 151
        self.rand_P_range = args.rand_P_range # [-self.rand_P_range, self.rand_P_range)で一様サンプル
        self.rand_S_range = args.rand_S_range # [1.-self.rand_S_range, 1.+self.rand_S_range)で一様サンプル
        self.rand_R_range = args.rand_R_range * np.pi # [-self.rand_R_range, self.rand_R_range)で一様サンプル
        self.random_axis_num = 1024
        self.random_axis_list = torch.from_numpy(sample_fibonacci_views(self.random_axis_num).astype(np.float32)).clone()
        self.gt_scale_range = 0.2
        if self.main_layers_name == 'autoreg':
            self.total_inp_itr = args.inp_itr_num + 1
        elif self.main_layers_name in {'only_mlp', 'encoder'}:
            self.total_inp_itr = args.inp_itr_num
        self.until_convergence = args.until_convergence
        self.convergence_thr = args.convergence_thr
        self.convergence_thr_shape = args.convergence_thr_shape
        self.scale_dim = args.scale_dim
        self.cnt_prg_step = args.cnt_prg_step
        self.add_conf = args.add_conf
        self.data_type = args.data_type

        # loss func.
        self.min_loss = float('inf')
        self.min_epoch = 0
        self.L_p = args.L_p
        self.L_s = args.L_s
        self.L_a = args.L_a
        self.L_r = args.L_r
        self.L_c = args.L_c
        self.L_d = args.L_d
        self.depth_error_mode = args.depth_error_mode
        self.mseloss = nn.MSELoss() # nn.SmoothL1Loss()
        self.cossim = nn.CosineSimilarity(dim=-1)
        self.cosssim_min = - 1 + 1e-8
        self.cosssim_max = 1 - 1e-8
        self.automatic_optimization = False
        self.dep_loss_use_num_max = 100000
        self.prg = args.prg
        if self.prg == 'yes': self.max_itr = self.total_itr
        self.use_sym_label = args.sym_label_path != 'no'
        self.sym_axis = {'__SYM_NONE'       : torch.tensor([[ 1., 0., 0.], [ 1., 0., 0.], [ 1., 0., 0.], [ 1., 0., 0.], ]), 
                         'SYM_ROTATE_UP_2'  : torch.tensor([[ 1., 0., 0.], [ 1., 0., 0.], [-1., 0., 0.], [-1., 0., 0.], ]), 
                         'SYM_ROTATE_UP_4'  : torch.tensor([[ 1., 0., 0.], [-1., 0., 0.], [0., 0.,  1.], [0., 0., -1.], ]), 
                         'SYM_ROTATE_UP_INF': torch.tensor([[ 1., 0., 0.], [-1., 0., 0.], [0., 0.,  1.], [0., 0., -1.], ])}



    def preprocess(self, batch, mode):
        # Get batch data. 
        if self.data_type == 'shapenet': # 書き換えた gt_model_3D_bbox_obj
            frame_mask, frame_distance_map, instance_id, frame_camera_pos_wrd, frame_w2c, frame_bbox_diagonal, bbox_list, frame_gt_obj_pos_wrd, \
            frame_gt_o2w, frame_gt_obj_green_wrd, frame_gt_obj_red_wrd, frame_gt_o2c, frame_gt_obj_green_cam, frame_gt_obj_red_cam, \
            frame_gt_obj_scale_wrd, canonical_distance_map, canonical_camera_pos, canonical_camera_rot, path, rand_seed, scene_id, gt_pc_obj, sym_label = batch
            frame_clopped_mask, frame_clopped_distance_map, frame_clopped_rays_d_cam = \
                clopping_distance_map_from_bbox(self.ddf_H, bbox_list, self.rays_d_cam, frame_mask, frame_distance_map)
            gt_model_3D_bbox_obj = False
        elif self.data_type == 'scan2cad': # 書き換えた gt_model_3D_bbox_obj
            frame_clopped_mask, frame_clopped_distance_map, instance_id, frame_camera_pos_wrd, frame_w2c, frame_bbox_diagonal, bbox_list, frame_clopped_rays_d_cam, \
            frame_gt_obj_pos_wrd, frame_gt_o2w, frame_gt_obj_green_wrd, frame_gt_obj_red_wrd, frame_gt_o2c, frame_gt_obj_green_cam, frame_gt_obj_red_cam, \
            frame_gt_obj_scale_wrd, gt_model_3D_bbox_obj, canonical_distance_map, canonical_camera_pos, canonical_camera_rot, gt_pc_obj, rand_seed, sym_label, path, scene_id = batch
        
        # Get baych size
        batch_size = len(instance_id)

        # Get Ground truth.
        if self.data_type == 'scan2cad' or mode in {'train', 'val'}:
            instance_idx = [self.ddf_instance_list.index(instance_id_i) for instance_id_i in instance_id]
            gt_shape_code = self.ddf.lat_vecs(torch.tensor(instance_idx, device=self.ddf.device)).detach()
        else:
            gt_shape_code = instance_idx = False

        # Log random seeds.
        if rand_seed['rand_P_seed'][0] == 'not_given':
            gt_S_seed = self.rs[mode].rand(batch_size, 1).astype(np.float32)
            rand_P_seed = self.rs[mode].rand(batch_size, 3).astype(np.float32)
            if self.data_type == 'shapenet':
                rand_S_seed = self.rs[mode].rand(batch_size, 1).astype(np.float32)
            elif self.data_type == 'scan2cad':
                rand_S_seed = self.rs[mode].rand(batch_size, self.scale_dim).astype(np.float32)
            randn_theta_seed = self.rs[mode].rand(batch_size).astype(np.float32)
            randn_axis_idx = self.rs[mode].choice(range(self.random_axis_num), batch_size)
        else:
            gt_S_seed = rand_seed['gt_S_seed'].to('cpu').detach().numpy().copy()
            rand_P_seed = rand_seed['rand_P_seed'].to('cpu').detach().numpy().copy()
            rand_S_seed = rand_seed['rand_S_seed'].to('cpu').detach().numpy().copy()
            randn_theta_seed = rand_seed['randn_theta_seed'].to('cpu').detach().numpy().copy()
            randn_axis_idx = rand_seed['randn_axis_idx'].to('cpu').detach().numpy().copy()
        if mode in {'val', 'tes'}:
            rand_seed = {}
            rand_seed['gt_S_seed'] = deepcopy(gt_S_seed)
            rand_seed['rand_P_seed'] = deepcopy(rand_P_seed)
            rand_seed['rand_S_seed'] = deepcopy(rand_S_seed)
            rand_seed['randn_theta_seed'] = deepcopy(randn_theta_seed)
            rand_seed['randn_axis_idx'] = deepcopy(randn_axis_idx)

        # Change gt scale.
        if self.data_type == 'shapenet':
            gt_S_seed = torch.from_numpy(gt_S_seed).clone().to(frame_w2c.device)
            frame_gt_obj_scale_wrd = self.gt_scale_range * 2 * (gt_S_seed[:, None, :].expand(-1, self.total_obs_num, -1) - 0.5) + 1.
            frame_camera_pos_wrd *= frame_gt_obj_scale_wrd
            frame_gt_obj_pos_wrd *= frame_gt_obj_scale_wrd
            frame_distance_map *= frame_gt_obj_scale_wrd[:, :, :, None]
            frame_clopped_distance_map *= frame_gt_obj_scale_wrd[:, :, :, None]
        elif self.data_type == 'scan2cad': # 'scan2cad'でも訓練時に変えればオーグメンテーションに成る？
            ...

        # Get inits.
        rand_P_seed = torch.from_numpy(rand_P_seed).clone().to(frame_w2c.device)
        rand_P = 2 * self.rand_P_range * (rand_P_seed - .5)
        ini_obj_pos = frame_gt_obj_pos_wrd[:, 0, :] + rand_P
        rand_S_seed = torch.from_numpy(rand_S_seed).clone().to(frame_w2c.device)
        rand_S = 2 * self.rand_S_range * (rand_S_seed - .5) + 1.
        ini_obj_scale = frame_gt_obj_scale_wrd[:, 0, :] * rand_S

        randn_axis = self.random_axis_list[randn_axis_idx].to(frame_w2c.device)
        randn_axis = torch.sum(randn_axis[:, None, :]*frame_gt_o2w[:, 0, :, :], dim=-1)
        randn_theta_seed = torch.from_numpy(randn_theta_seed).clone().to(frame_w2c.device)
        randn_theta = self.rand_R_range * 2 * (randn_theta_seed - .5)
        cos_t = torch.cos(randn_theta)
        sin_t = torch.sin(randn_theta)
        n_x, n_y, n_z = randn_axis.permute(1, 0)
        rand_R = torch.stack([torch.stack([cos_t+n_x*n_x*(1-cos_t), n_x*n_y*(1-cos_t)-n_z*sin_t, n_z*n_x*(1-cos_t)+n_y*sin_t], dim=-1), 
                              torch.stack([n_x*n_y*(1-cos_t)+n_z*sin_t, cos_t+n_y*n_y*(1-cos_t), n_y*n_z*(1-cos_t)-n_x*sin_t], dim=-1), 
                              torch.stack([n_z*n_x*(1-cos_t)-n_y*sin_t, n_y*n_z*(1-cos_t)+n_x*sin_t, cos_t+n_z*n_z*(1-cos_t)], dim=-1)], dim=1)
        ini_obj_axis_green_wrd = torch.sum(frame_gt_obj_green_wrd[:, :1, :]*rand_R, dim=-1)
        ini_obj_axis_red_wrd = torch.sum(frame_gt_obj_red_wrd[:, :1, :]*rand_R, dim=-1)

        ini_shape_code = self.rand_Z_center.unsqueeze(0).expand(batch_size, -1).to(ini_obj_pos)

        # Retern.
        if mode=='train':
            return batch_size, frame_clopped_mask, frame_clopped_distance_map, frame_clopped_rays_d_cam, gt_shape_code, frame_bbox_diagonal, \
                   frame_w2c, frame_gt_obj_green_wrd, frame_gt_obj_red_wrd, frame_camera_pos_wrd, frame_gt_obj_pos_wrd, frame_gt_obj_scale_wrd, \
                   ini_obj_pos, ini_obj_scale, ini_obj_axis_red_wrd, ini_obj_axis_green_wrd,ini_shape_code, frame_gt_o2w, instance_id, scene_id, sym_label
        elif mode in {'val', 'tes'}: # 書き換えた gt_model_3D_bbox_obj
            return batch_size, frame_clopped_mask, frame_clopped_distance_map, frame_clopped_rays_d_cam, gt_shape_code, \
                   frame_bbox_diagonal, frame_w2c, frame_gt_obj_green_wrd, frame_gt_obj_red_wrd, frame_camera_pos_wrd, \
                   frame_gt_obj_pos_wrd, frame_gt_obj_scale_wrd, gt_model_3D_bbox_obj, canonical_distance_map, canonical_camera_pos, \
                   canonical_camera_rot, instance_id, path, ini_obj_pos, ini_obj_scale, ini_obj_axis_red_wrd, \
                   ini_obj_axis_green_wrd,ini_shape_code, rand_seed, frame_gt_o2w, instance_id, scene_id, gt_pc_obj, sym_label



    def forward(self, batch_size, current_frame_num, clopped_mask, clopped_distance_map, past_itr_log, pre_obj_pos_wrd, 
                pre_obj_axis_green_wrd, pre_obj_axis_red_wrd, pre_obj_scale_wrd, pre_shape_code, w2c, cam_pos_wrd, 
                rays_d_cam, bbox_diagonal, batch_idx, optim_idx, print_debug, model_mode):

        with torch.no_grad():
            # Reshape pre estimation to [batch, frame, ?]
            inp_pre_obj_pos_wrd = pre_obj_pos_wrd[:, None, :].expand(-1, current_frame_num, -1)
            inp_pre_obj_axis_green_wrd = pre_obj_axis_green_wrd[:, None, :].expand(-1, current_frame_num, -1)
            inp_pre_obj_axis_red_wrd = pre_obj_axis_red_wrd[:, None, :].expand(-1, current_frame_num, -1)
            inp_pre_obj_scale_wrd = pre_obj_scale_wrd[:, None, :].expand(-1, current_frame_num, -1)
            inp_pre_shape_code = pre_shape_code[:, None, :].expand(-1, current_frame_num, -1)
            pre_o2w = axis2rotation(pre_obj_axis_green_wrd, pre_obj_axis_red_wrd)
            inp_pre_o2w = pre_o2w[:, None, :, :].expand(-1, current_frame_num, -1, -1)

            # Simulate DDF.
            inp_pre_obj_axis_green_cam = torch.sum(inp_pre_obj_axis_green_wrd.reshape(-1, 3)[..., None, :]*w2c, dim=-1)
            inp_pre_obj_axis_red_cam = torch.sum(inp_pre_obj_axis_red_wrd.reshape(-1, 3)[..., None, :]*w2c, dim=-1)
            if self.data_type == 'shapenet':
                est_clopped_mask, est_clopped_distance_map = render_distance_map_from_axis(
                                                                H = self.ddf_H, 
                                                                obj_pos_wrd = inp_pre_obj_pos_wrd.reshape(-1, 3), 
                                                                axis_green = inp_pre_obj_axis_green_cam, 
                                                                axis_red = inp_pre_obj_axis_red_cam, 
                                                                obj_scale = inp_pre_obj_scale_wrd.reshape(-1), 
                                                                input_lat_vec = inp_pre_shape_code.reshape(-1, self.ddf.latent_size), 
                                                                cam_pos_wrd = cam_pos_wrd, 
                                                                rays_d_cam = rays_d_cam, 
                                                                w2c = w2c.detach(), 
                                                                ddf = self.ddf, 
                                                                with_invdistance_map = False)
            elif self.data_type == 'scan2cad':
                est_clopped_mask, est_clopped_distance_map = render_distance_map_from_axis_for_scannet(
                                                                H = self.ddf_H, 
                                                                obj_pos_wrd = inp_pre_obj_pos_wrd.reshape(-1, 3), 
                                                                axis_green = inp_pre_obj_axis_green_cam, 
                                                                axis_red = inp_pre_obj_axis_red_cam, 
                                                                obj_scale = inp_pre_obj_scale_wrd.reshape(-1, self.scale_dim), 
                                                                input_lat_vec = inp_pre_shape_code.reshape(-1, self.ddf.latent_size), 
                                                                cam_pos_wrd = cam_pos_wrd, 
                                                                rays_d_cam = rays_d_cam, 
                                                                w2c = w2c.detach(), 
                                                                ddf = self.ddf, 
                                                                with_invdistance_map = False)

            # Log current maps.
            past_itr_log['itr_length'].append(current_frame_num)
            past_itr_log['obs_mask'].append(clopped_mask.reshape(batch_size, current_frame_num, self.ddf_H, self.ddf_H).clone())
            past_itr_log['est_mask'].append(est_clopped_mask.reshape(batch_size, current_frame_num, self.ddf_H, self.ddf_H).clone())
            past_itr_log['w2c'].append(w2c.reshape(batch_size, current_frame_num, 3, 3).clone())
            past_itr_log['cam_pos_wrd'].append(cam_pos_wrd.reshape(batch_size, current_frame_num, 3).clone())
            past_itr_log['rays_d_cam'].append(rays_d_cam.reshape(batch_size, current_frame_num, self.ddf_H, self.ddf_H, 3).clone())
            past_itr_log['bbox_diagonal'].append(bbox_diagonal.reshape(batch_size, current_frame_num, 1).clone())
            # if self.input_type == 'depthmap':
            #     for map_idx, (map_i, mask_i) in enumerate(zip(clopped_distance_map, clopped_mask)):
            #         avg_i = map_i[mask_i].mean()
            #         clopped_distance_map[map_idx][clopped_mask[map_idx]] = clopped_distance_map[map_idx][clopped_mask[map_idx]] - avg_i
            #         est_clopped_distance_map[map_idx][est_clopped_mask[map_idx]] = est_clopped_distance_map[map_idx][est_clopped_mask[map_idx]] - avg_i
            past_itr_log['obs_d'].append(clopped_distance_map.reshape(batch_size, current_frame_num, self.ddf_H, self.ddf_H).clone())
            past_itr_log['est_d'].append(est_clopped_distance_map.reshape(batch_size, current_frame_num, self.ddf_H, self.ddf_H).clone())

            # Get past maps.
            total_view_num = sum(past_itr_log['itr_length'])
            acc_obs_distmap = torch.cat(past_itr_log['obs_d'], dim=1).reshape(-1, self.ddf_H, self.ddf_H)
            acc_est_distmap = torch.cat(past_itr_log['est_d'], dim=1).reshape(-1, self.ddf_H, self.ddf_H)
            acc_obs_mask = torch.cat(past_itr_log['obs_mask'], dim=1).reshape(-1, self.ddf_H, self.ddf_H)
            acc_est_mask = torch.cat(past_itr_log['est_mask'], dim=1).reshape(-1, self.ddf_H, self.ddf_H)
            acc_w2c = torch.cat(past_itr_log['w2c'], dim=1).reshape(-1, 3, 3)
            acc_cam_pos_wrd = torch.cat(past_itr_log['cam_pos_wrd'], dim=1).reshape(-1, 3)
            acc_rays_d_cam = torch.cat(past_itr_log['rays_d_cam'], dim=1).reshape(-1, self.ddf_H, self.ddf_H, 3)
            acc_bbox_diagonal = torch.cat(past_itr_log['bbox_diagonal'], dim=1)
            acc_latest_o2w = pre_o2w[:, None, :, :].expand(-1, total_view_num, -1, -1).reshape(-1, 3, 3)
            acc_latest_obj_pos_wrd = pre_obj_pos_wrd[:, None, :].expand(-1, total_view_num, -1).reshape(-1, 3)
            if self.data_type == 'shapenet':
                acc_latest_obj_scale_wrd = pre_obj_scale_wrd[:, None, :].expand(-1, total_view_num, -1).reshape(-1, 1)
            elif self.data_type == 'scan2cad':
                acc_latest_obj_scale_wrd = pre_obj_scale_wrd[:, None, :].expand(-1, total_view_num, -1).reshape(-1, self.scale_dim)
            
            # Get map.
            past_view_num = 0
            if len(past_itr_log['itr_length']) > 1:
                # Expanding current est to past est shapes.
                past_view_num = total_view_num - past_itr_log['itr_length'][-1]
                repped_latest_distmap = torch.cat([past_itr_log['est_d'][view_idx+1][:, :view_num] for view_idx, view_num in enumerate(past_itr_log['itr_length'][:-1])], dim=1).reshape(-1, self.ddf_H, self.ddf_H)
                repped_latest_mask    = torch.cat([past_itr_log['est_mask'][view_idx+1][:, :view_num] for view_idx, view_num in enumerate(past_itr_log['itr_length'][:-1])], dim=1).reshape(-1, self.ddf_H, self.ddf_H)
                repped_latest_o2w = pre_o2w[:, None, :, :].expand(-1, past_view_num, -1, -1).reshape(-1, 3, 3)
                if self.data_type == 'shapenet':
                    repped_latest_obj_scale_wrd = pre_obj_scale_wrd[:, None, :].expand(-1, past_view_num, -1).reshape(-1, 1)
                elif self.data_type == 'scan2cad':
                    repped_latest_obj_scale_wrd = pre_obj_scale_wrd[:, None, :].expand(-1, past_view_num, -1).reshape(-1, self.scale_dim)
                # Expanding current est to past est shapes.
                past_est_distmap = acc_est_distmap.reshape(batch_size, total_view_num, self.ddf_H, self.ddf_H)[:, :past_view_num].reshape(-1, self.ddf_H, self.ddf_H)
                past_est_mask = acc_est_mask.reshape(batch_size, total_view_num, self.ddf_H, self.ddf_H)[:, :past_view_num].reshape(-1, self.ddf_H, self.ddf_H)
                past_rays_d_cam = acc_rays_d_cam.reshape(batch_size, total_view_num, self.ddf_H, self.ddf_H, 3)[:, :past_view_num].reshape(-1, self.ddf_H, self.ddf_H, 3)
                past_w2c = acc_w2c.reshape(batch_size, total_view_num, 3, 3)[:, :past_view_num].reshape(-1, 3, 3)
            if self.input_type == 'osmap':
                obs_OSMap_obj = get_OSMap_obj(acc_obs_distmap, acc_obs_mask, acc_rays_d_cam, acc_w2c, acc_cam_pos_wrd, acc_latest_o2w, acc_latest_obj_pos_wrd, acc_latest_obj_scale_wrd, self.data_type, self.add_conf)
                est_OSMap_obj = get_OSMap_obj(acc_est_distmap, acc_est_mask, acc_rays_d_cam, acc_w2c, acc_cam_pos_wrd, acc_latest_o2w, acc_latest_obj_pos_wrd, acc_latest_obj_scale_wrd, self.data_type, self.add_conf)
                dif_OSMap_obj = get_diffOSMap_obj(acc_obs_distmap, acc_est_distmap, acc_obs_mask, acc_est_mask, acc_rays_d_cam, acc_w2c, acc_latest_o2w, acc_latest_obj_scale_wrd, self.data_type, self.add_conf)
                if len(past_itr_log['itr_length']) > 1:
                    tra_OSMap_obj = get_diffOSMap_obj(past_est_distmap, repped_latest_distmap, past_est_mask, repped_latest_mask, past_rays_d_cam, past_w2c, repped_latest_o2w, repped_latest_obj_scale_wrd)
                else:
                    tra_OSMap_obj = torch.tensor([])[:, None, None, None].expand(-1, self.ddf_H, self.ddf_H, 4).to(obs_OSMap_obj)
                inp_maps = torch.cat([obs_OSMap_obj, est_OSMap_obj, dif_OSMap_obj, tra_OSMap_obj], dim=0).permute(0, 3, 1, 2).contiguous().detach()
            elif self.input_type == 'depthmap':
                obs_distmap = torch.stack([acc_obs_distmap, acc_obs_mask], dim=-1)
                est_distmap = torch.stack([acc_est_distmap, acc_est_mask], dim=-1)
                dif_distmap = torch.stack([acc_obs_distmap-acc_est_distmap, torch.logical_xor(acc_obs_mask, acc_est_mask)], dim=-1)
                if len(past_itr_log['itr_length']) > 1:
                    tra_distmap = torch.stack([past_est_distmap - repped_latest_distmap, torch.logical_xor(past_est_mask, repped_latest_mask)], dim=-1)
                else:
                    tra_distmap = torch.tensor([])[:, None, None, None].expand(-1, self.ddf_H, self.ddf_H, 2).to(acc_obs_distmap)
                inp_maps = torch.cat([obs_distmap, est_distmap, dif_distmap, tra_distmap], dim=0).permute(0, 3, 1, 2).contiguous().detach()

        # Get embeddings.
        inp_embed = self.backbone_encoder(inp_maps)
        obs_embed = inp_embed[:1*batch_size*total_view_num].reshape(batch_size, total_view_num, self.backbone_embedding_dim)
        est_embed = inp_embed[1*batch_size*total_view_num:2*batch_size*total_view_num].reshape(batch_size, total_view_num, self.backbone_embedding_dim)
        dif_embed = inp_embed[2*batch_size*total_view_num:3*batch_size*total_view_num].reshape(batch_size, total_view_num, self.backbone_embedding_dim)
        tra_embed = inp_embed[3*batch_size*total_view_num:].reshape(batch_size, past_view_num, self.backbone_embedding_dim)

        # Get pe target.
        pe_target = None
        if self.positional_encoding_mode == 'yes':
            if self.data_type == 'shapenet':
                pe_target = get_pe_target(acc_cam_pos_wrd.reshape(batch_size, -1, 3), acc_w2c.reshape(batch_size, -1, 3, 3), acc_rays_d_cam.reshape(batch_size, -1, self.ddf_H, self.ddf_H, 3), 
                                        acc_latest_obj_pos_wrd.reshape(batch_size, -1, 3), acc_latest_o2w.reshape(batch_size, -1, 3, 3), acc_latest_obj_scale_wrd.reshape(batch_size, -1, 1), 
                                        acc_bbox_diagonal, self.ddf_H, self.add_conf)
            elif self.data_type == 'scan2cad':
                pe_target = get_pe_target(acc_cam_pos_wrd.reshape(batch_size, -1, 3), acc_w2c.reshape(batch_size, -1, 3, 3), acc_rays_d_cam.reshape(batch_size, -1, self.ddf_H, self.ddf_H, 3), 
                                        acc_latest_obj_pos_wrd.reshape(batch_size, -1, 3), acc_latest_o2w.reshape(batch_size, -1, 3, 3), acc_latest_obj_scale_wrd.reshape(batch_size, -1, self.scale_dim), 
                                        acc_bbox_diagonal, self.ddf_H, self.add_conf)

        # Get update value.
        est_obj_pos_wrd, est_obj_axis_green_wrd, est_obj_axis_red_wrd, est_obj_scale, est_shape_code = \
        self.df_net(obs_embed, est_embed, dif_embed, tra_embed, past_itr_log['itr_length'], optim_idx, inp_pre_obj_pos_wrd, inp_pre_obj_axis_green_wrd, inp_pre_obj_axis_red_wrd, inp_pre_o2w, \
                    inp_pre_obj_scale_wrd, inp_pre_shape_code, pre_obj_pos_wrd, pre_obj_axis_green_wrd, pre_obj_axis_red_wrd, pre_o2w, pre_obj_scale_wrd, pre_shape_code, pe_target, print_debug) # , self.backbone_decoder)

        ##################################################
        if optim_idx in {0, 5}: # if optim_idx > 1:
            for sample_id in range(batch_size):
                raw_1 = obs_OSMap_obj.reshape(batch_size, total_view_num, self.ddf_H, self.ddf_H, 4)[:, -current_frame_num:]
                raw_2 = est_OSMap_obj.reshape(batch_size, total_view_num, self.ddf_H, self.ddf_H, 4)[:, -current_frame_num:]
                # raw_1 = obs_OSMap_obj.reshape(batch_size, total_view_num, self.ddf_H, self.ddf_H, 4)[:, -sum(past_itr_log['itr_length'][-2:]):-current_frame_num]
                # raw_2 = est_OSMap_obj.reshape(batch_size, total_view_num, self.ddf_H, self.ddf_H, 4)[:, -current_frame_num:]
                # raw_3 = est_OSMap_obj.reshape(batch_size, total_view_num, self.ddf_H, self.ddf_H, 4)[:, -sum(past_itr_log['itr_length'][-2:]):-current_frame_num]
                map_1, mask_1 = raw_1[..., :-1], raw_1[..., -1]
                map_2, mask_2 = raw_2[..., :-1], raw_2[..., -1]
                # map_3, mask_3 = raw_3[..., :-1], raw_3[..., -1]
                fig = plt.figure()
                ax = Axes3D(fig)
                c_list_1 = c_list_2 = ['b', 'g', 'r', 'm', 'y']
                for ind_1 in range(current_frame_num):
                    c_1, c_2 = c_list_1[ind_1], c_list_2[ind_1]
                    # if past_itr_log['itr_length'][-2] > ind_1:
                    point_1 = map_1[sample_id, ind_1][mask_1[sample_id, ind_1] > 0]
                    point_1 = point_1.to('cpu').detach().numpy().copy()
                    ax.scatter(point_1[:, 0], point_1[:, 1], point_1[:, 2], marker="o", linestyle='None', c='g', s=0.1) #, c=c_1, s=0.1)
                    point_2 = map_2[sample_id, ind_1][mask_2[sample_id, ind_1] > 0]
                    point_2 = point_2.to('cpu').detach().numpy().copy()
                    ax.scatter(point_2[::2, 0], point_2[::2, 1], point_2[::2, 2], marker="o", linestyle='None', c='m', s=0.1) # , c=c_2, s=0.05)
                    # if past_itr_log['itr_length'][-2] > ind_1:
                    #     point_3 = map_3[sample_id, ind_1][mask_3[sample_id, ind_1] > 0]
                    #     point_3 = point_3.to('cpu').detach().numpy().copy()
                    #     ax.scatter(point_3[::2, 0], point_3[::2, 1], point_3[::2, 2], marker="o", linestyle='None', c='g', s=0.05)
                str_optim_idx = str(optim_idx).zfill(2)
                ax.view_init(elev=0, azim=90)
                fig.savefig(f"{sample_id}_{str_optim_idx}_90.png")
                ax.view_init(elev=0, azim=0)
                fig.savefig(f"{sample_id}_{str_optim_idx}_00.png")
                ax.view_init(elev=45, azim=45)
                fig.savefig(f"{sample_id}_{str_optim_idx}_45.png")
                plt.close()
                # import pdb; pdb.set_trace()
        ##################################################
        gt = clopped_distance_map
        est = est_clopped_distance_map
        gt_mask = clopped_mask
        est_mask = est_clopped_mask
        str_batch_idx = str(batch_idx).zfill(5)
        str_optim_idx = str(optim_idx).zfill(2)
        check_map = torch.cat([gt, gt_mask, est, est_mask, torch.abs(gt-est)], dim=-1)
        check_map = torch.cat([map_i for map_i in check_map], dim=0)
        check_map_torch(check_map, f'tes_{str_batch_idx}_{str_optim_idx}.png')
        # if optim_idx in {6}:
        #     import pdb; pdb.set_trace()
        # ##################################################
        # batch_i = 0
        # gt = gt.reshape(batch_size, current_frame_num, self.ddf_H, self.ddf_H)[batch_i]
        # est = est.reshape(batch_size, current_frame_num, self.ddf_H, self.ddf_H)[batch_i]
        # os.makedirs(f'paper_fig/depth_map/raw_gt_est/{self.test_log_name}', exist_ok=True)
        # for F_i in range(current_frame_num):
        #     pickle_dump(torch.stack([gt[F_i], est[F_i]], dim=0).to('cpu').detach().numpy().copy(), 
        #                 f'paper_fig/depth_map/raw_gt_est/{self.test_log_name}/{str(batch_idx).zfill(5)}_{F_i}_{str(optim_idx).zfill(2)}.pickle')
        # print(sum([layer_i.attn_output_weights.mean(1) for layer_i in self.df_net.main_layers.encoder.layers])/3)
        # print(sum([layer_i.self_attn_weights.mean(1) for layer_i in self.df_net.main_layers.decoder.layers])/3)
        # print(sum([layer_i.cross_attn_weights.mean(1) for layer_i in self.df_net.main_layers.decoder.layers])/3)
        # import pdb; pdb.set_trace()
        # ##################################################
        return est_obj_pos_wrd, est_obj_axis_green_wrd, est_obj_axis_red_wrd, est_obj_scale, est_shape_code



    def on_train_epoch_start(self):
        self.ddf.eval()
        self.rays_d_cam = self.rays_d_cam.to(self.df_net.device)
        np.random.seed(0 + self.current_epoch)
        train_dataset.dataset_current_epoch = self.current_epoch + 1
        # self.backbone_encoder.eval(), self.backbone_decoder.eval()
        if self.view_selection == 'simultaneous' and self.prg == 'yes':            # if self.view_selection == 'sequential': # 
            self.total_itr = int(min(self.max_itr, self.current_epoch // 80 + 3))  # if self.current_epoch < 200: self.total_itr, self.itr_per_frame = 5, 1 # 
            self.itr_per_frame = self.total_itr                                    # else: self.total_itr, self.itr_per_frame = 25, 5 #
        if self.view_selection == 'sequential' and self.prg == 'yes':
            if self.current_epoch < self.cnt_prg_step:
                self.total_itr = 5
                self.itr_per_frame = 1
            else:
                self.total_itr = 10
                self.itr_per_frame = 2



    def training_step(self, batch, batch_idx):
        # Get batch info.
        with torch.no_grad():
            batch_size, frame_clopped_mask, frame_clopped_distance_map, frame_clopped_rays_d_cam, gt_shape_code, frame_bbox_diagonal, frame_w2c, \
            frame_gt_obj_green_wrd, frame_gt_obj_red_wrd, frame_camera_pos_wrd, frame_gt_obj_pos_wrd, frame_gt_obj_scale_wrd, ini_obj_pos, \
            ini_obj_scale, ini_obj_axis_red_wrd, ini_obj_axis_green_wrd, ini_shape_code, frame_gt_o2w, instance_id, scene_id, sym_label = self.preprocess(batch, mode='train')

        # Initialization.
        pre_obj_pos_wrd = ini_obj_pos.detach()
        pre_obj_scale_wrd = ini_obj_scale.detach()
        pre_obj_axis_green_wrd = ini_obj_axis_green_wrd.detach()
        pre_obj_axis_red_wrd = ini_obj_axis_red_wrd.detach()
        pre_shape_code = ini_shape_code.detach()

        ###################################
        #####     Start training.     #####
        ###################################
        frame_optim_idx = 0
        if self.view_selection == 'simultaneous':
            current_frame_num = self.total_obs_num
        elif self.view_selection == 'sequential':
            current_frame_num = 1

        past_itr_log = {'itr_length': [], 'obs_d': [], 'est_d': [], 'obs_mask': [], 'est_mask': [], 'w2c': [], 'cam_pos_wrd': [], 'rays_d_cam': [], 'bbox_diagonal': []}
        frame_clopped_invdist = torch.zeros_like(frame_clopped_distance_map)
        frame_clopped_invdist[frame_clopped_mask] = 1 / frame_clopped_distance_map[frame_clopped_mask]
        # frame_gb_mask = self.gb(frame_clopped_mask)
        for optim_idx in range(self.total_itr):
            # Set optimizers
            opt = self.optimizers()
            opt.zero_grad()
            # for optim_idx in range(self.total_itr):
            # Set frames
            frame_idx_list = list(range(current_frame_num))

            # Get current maps.
            clopped_mask = frame_clopped_mask[:, frame_idx_list].reshape(-1, self.ddf_H, self.ddf_H).detach()
            clopped_distance_map = frame_clopped_distance_map[:, frame_idx_list].reshape(-1, self.ddf_H, self.ddf_H).detach()
            clopped_invdist = frame_clopped_invdist[:, frame_idx_list].reshape(-1, self.ddf_H, self.ddf_H).detach()
            # gb_mask = frame_gb_mask[:, frame_idx_list].reshape(-1, self.ddf_H, self.ddf_H).detach()

            # Get camera infos.
            w2c = frame_w2c[:, frame_idx_list].reshape(-1, 3, 3).detach()
            cam_pos_wrd = frame_camera_pos_wrd[:, frame_idx_list].reshape(-1, 3).detach()
            rays_d_cam = frame_clopped_rays_d_cam[:, frame_idx_list].reshape(-1, self.ddf_H, self.ddf_H, 3).detach()
            bbox_diagonal = frame_bbox_diagonal[:, frame_idx_list].reshape(-1, 1).detach()
            print_debug = self.current_epoch == 0 and batch_idx == 0 and optim_idx in {0, 2, 23}
            if print_debug:
                if optim_idx == 0:
                    print('########## TRAIN ##########')
                    print(f'---     ins     --- : {instance_id[:2]}')
                    print(f'---    scene    --- : {scene_id[0][:2]}')
                    print(f'---   gtscale   --- : {frame_gt_obj_scale_wrd[:5, 0, 0]}')
                    print(f'---   ini_red   --- : {ini_obj_axis_red_wrd[:5, 0]}')
                    print(f'---   nearpos   --- : {(torch.norm(frame_camera_pos_wrd[:, :-1]-frame_camera_pos_wrd[:, 1:], dim=-1)/frame_gt_obj_scale_wrd[:, :-1, 0]).max()<1.04}')
                print(f'#-- current_itr --# : {optim_idx+1} / {self.total_itr}')
                print(f'---  itr_frame  --- : {frame_idx_list}')
            # pre_obj_pos_wrd = frame_gt_obj_pos_wrd[:, 0].clone().detach()
            # pre_obj_pos_wrd = frame_gt_obj_pos_wrd[:, 0].clone().detach() + \
            #     (1. - ((optim_idx+1) / self.total_itr)) * \
            #     torch.sum((torch.tensor([0.3, 0, 0]).to(ini_obj_pos))[None, None, :].expand(batch_size, -1, -1)*frame_gt_o2w[:, 0], dim=-1)
            # pre_obj_scale_wrd = frame_gt_obj_scale_wrd[:, 0].clone().detach()
            # pre_obj_axis_red_wrd = frame_gt_obj_red_wrd[:, 0].clone().detach()
            # pre_obj_axis_green_wrd = frame_gt_obj_green_wrd[:, 0].clone().detach()
            # pre_shape_code = gt_shape_code.clone().detach()

            ###################################
            #####      Perform dfnet.     #####
            ###################################
            est_obj_pos_wrd, est_obj_axis_green_wrd, est_obj_axis_red_wrd, est_obj_scale, est_shape_code = \
                self.forward(batch_size, current_frame_num, clopped_mask.detach(), clopped_distance_map.detach(), past_itr_log, pre_obj_pos_wrd.detach(), pre_obj_axis_green_wrd.detach(), 
                pre_obj_axis_red_wrd.detach(), pre_obj_scale_wrd.detach(), pre_shape_code.detach(), w2c.detach(), cam_pos_wrd.detach(), rays_d_cam.detach(), bbox_diagonal.detach(), batch_idx, optim_idx, print_debug, 'train')

            # Cal loss against integrated estimations.
            loss_pos = self.mseloss(est_obj_pos_wrd, frame_gt_obj_pos_wrd[:, -1].detach())
            loss_axis_green = torch.mean(-self.cossim(est_obj_axis_green_wrd, frame_gt_obj_green_wrd[:, -1].detach()) + 1.)
            #########################
            if not self.use_sym_label:
                loss_axis_red = torch.mean(-self.cossim(est_obj_axis_red_wrd, frame_gt_obj_red_wrd[:, -1].detach()) + 1.)
            else:
                # 軸の対称性に注意
                # sym_label = ['SYM_ROTATE_UP_2', 'SYM_ROTATE_UP_INF']
                sym_axis = []
                for sym_label_i in sym_label:
                    sym_axis.append(self.sym_axis[sym_label_i])
                sym_axis = torch.stack(sym_axis).to(frame_gt_o2w)
                sym_axis_wrd = torch.sum(sym_axis[:, :, None, :]*frame_gt_o2w[:, 0][:, None, :, :], dim=-1)
                loss_sym_red = - self.cossim(est_obj_axis_red_wrd[:, None, :], sym_axis_wrd) + 1.
                sym_axis_id = torch.argmin(loss_sym_red, dim=-1)
                gt_red_wrd = sym_axis_wrd[range(batch_size), sym_axis_id] # frame_gt_obj_red_wrd[:, 0]
                sym_inf_mask = torch.tensor([sym_label_i != 'SYM_ROTATE_UP_INF' for sym_label_i in sym_label]).to(gt_red_wrd)
                err_axis_red_cos_sim_i = self.cossim(est_obj_axis_red_wrd, gt_red_wrd.detach()) * sym_inf_mask.detach() + (1.0 - sym_inf_mask.detach())
                loss_axis_red = torch.mean(- err_axis_red_cos_sim_i + 1.)
                # # 点群デバッグ
                # pt_cam = frame_clopped_distance_map.unsqueeze(-1) * frame_clopped_rays_d_cam
                # pt_wrd = torch.sum(pt_cam[..., None, :] * frame_w2c.permute(0, 1, 3, 2)[:, :, None, None, :], dim=-1) + frame_camera_pos_wrd[:, :, None, None, :]
                # for btc_id in range(batch_size):
                #     fig = plt.figure()
                #     ax = Axes3D(fig)
                #     for obs_id in range(pt_wrd.shape[1]):
                #         point = pt_wrd[btc_id, obs_id][frame_clopped_mask[btc_id, obs_id]].to('cpu').detach().numpy().copy()
                #         ax.scatter(point[:, 0], point[:, 1], point[:, 2], marker="o", linestyle='None', c='c', s=0.05)
                #     point_3 = (sym_axis_wrd[btc_id] + frame_gt_obj_pos_wrd[:, 0][btc_id]).to('cpu').detach().numpy().copy() # GTの候補
                #     ax.scatter(point_3[:, 0], point_3[:, 1], point_3[:, 2], marker="o", linestyle='None', c='g', s=1.0)
                #     point_4 = (gt_red_wrd[btc_id] + frame_gt_obj_pos_wrd[:, 0][btc_id]).to('cpu').detach().numpy().copy() # GT
                #     ax.scatter(point_4[0], point_4[1], point_4[2], marker="o", linestyle='None', c='r', s=5.0)
                #     point_5 = (pre_obj_axis_red_wrd[btc_id] + frame_gt_obj_pos_wrd[:, 0][btc_id]).to('cpu').detach().numpy().copy() # 推定値
                #     ax.scatter(point_5[0], point_5[1], point_5[2], marker="o", linestyle='None', c='b', s=5.0) # GT
                #     ax.view_init(elev=0, azim=90)
                #     fig.savefig(f"{btc_id}_00_90.png")
                #     ax.view_init(elev=0, azim=0)
                #     fig.savefig(f"{btc_id}_00_00.png")
                #     ax.view_init(elev=45, azim=45)
                #     fig.savefig(f"{btc_id}_45_45.png")
                #     plt.close()
                # import pdb; pdb.set_trace()
            #########################
            if self.scale_dim == 1:
                loss_scale = self.mseloss(est_obj_scale, frame_gt_obj_scale_wrd[:, -1].detach())
            elif self.scale_dim == 3:
                gt_scale = frame_gt_obj_scale_wrd[:, -1].detach().expand(-1, self.scale_dim)
                loss_scale = self.mseloss(est_obj_scale, gt_scale)
                est_obj_scale = est_obj_scale.mean(-1).unsqueeze(-1)
            loss_shape_code = self.mseloss(est_shape_code, gt_shape_code.detach())
            loss_depth = torch.zeros_like(loss_pos).detach() # Dummy
            loss_axis = loss_axis_green + self.L_r * loss_axis_red
            loss = self.L_p * loss_pos + self.L_s * loss_scale + self.L_a * loss_axis + self.L_c * loss_shape_code + self.L_d * loss_depth

            # Optimizer step.
            self.manual_backward(loss)
            opt.step()

            # Update frame infos.
            if optim_idx >= 0:
                frame_optim_idx += 1
            if frame_optim_idx >= self.itr_per_frame:
                if current_frame_num < self.total_obs_num:
                    current_frame_num += 1
                    frame_optim_idx = 0

            # Save pre estimation.
            with torch.no_grad():
                pre_obj_pos_wrd = est_obj_pos_wrd.clone().detach()
                pre_obj_scale_wrd = est_obj_scale.clone().detach()
                pre_obj_axis_green_wrd = est_obj_axis_green_wrd.clone().detach()
                pre_obj_axis_red_wrd = est_obj_axis_red_wrd.clone().detach()
                pre_shape_code = est_shape_code.clone().detach()
        
        # opt.step() # 64, 417, 457, 475, 492, 
        
        # Return loss.
        return {'pos':   loss_pos.detach(), 
                'green': loss_axis_green.detach(), 
                'red':   loss_axis_red.detach(), 
                'scale': loss_scale.detach(), 
                'shape': loss_shape_code.detach(), 
                'depth': loss_depth.detach(), 
                'total': loss.detach(), }



    def training_epoch_end(self, outputs):
        # Log loss.
        for key_i in ['pos', 'green', 'red', 'scale', 'shape', 'total', 'depth']:
            avg_loss = torch.stack([x[key_i] for x in outputs]).mean()
            current_epoch = torch.tensor(self.current_epoch + 1., dtype=avg_loss.dtype)
            self.log_dict({f'train/loss_{key_i}': avg_loss, "step": current_epoch})
        
        # Save ckpt.
        if (self.current_epoch + 1) % self.save_interval == 0:
            ckpt_name = str(self.current_epoch + 1).zfill(10) + '.ckpt'
            ckpt_path = os.path.join(self.trainer.log_dir, 'checkpoints', ckpt_name)
            trainer.save_checkpoint(ckpt_path)



    def on_test_epoch_start(self):
        self.rs['tes'] = np.random.RandomState(73)
        torch.backends.cudnn.deterministic = True
        self.rays_d_cam = self.rays_d_cam.to(self.df_net.device)
        self.canonical_rays_d_cam = self.canonical_rays_d_cam.to(self.df_net.device)



    def test_step(self, batch, batch_idx, step_mode='tes'):
        batch_size, frame_clopped_mask, frame_clopped_distance_map, frame_clopped_rays_d_cam, gt_shape_code, \
        frame_bbox_diagonal, frame_w2c, frame_gt_obj_green_wrd, frame_gt_obj_red_wrd, frame_camera_pos_wrd, \
        frame_gt_obj_pos_wrd, frame_gt_obj_scale_wrd, gt_model_3D_bbox_obj, canonical_distance_map, canonical_camera_pos, \
        canonical_camera_rot, instance_id, path, ini_obj_pos, ini_obj_scale, ini_obj_axis_red_wrd, \
        ini_obj_axis_green_wrd,ini_shape_code, rand_seed, frame_gt_o2w, instance_id, scene_id, gt_pc_obj, sym_label = self.preprocess(batch, step_mode)

        ###################################
        #####     Start inference.    #####
        ###################################
        frame_optim_idx = 0
        if self.view_selection == 'simultaneous':
            current_frame_num = self.total_obs_num
        elif self.view_selection == 'sequential':
            current_frame_num = 1 # 2
        
        past_itr_log = {'itr_length': [], 'obs_d': [], 'est_d': [], 'obs_mask': [], 'est_mask': [], 'w2c': [], 'cam_pos_wrd': [], 'rays_d_cam': [], 'bbox_diagonal': []}
        itr_err_list = {'pos':[], 'rot':[], 'scale':[], 'red':[], 'green':[], 'chm_obj':[], 'chm_s_obj':[], 'chm_ss_obj':[], 'chm_wrd':[], 'chm_wrd_s':[], 'shape':[], 'sec':[], 'cnv':[], 'est_p':[], 'est_g':[], 'est_r':[], 'est_s':[], 'est_z':[]}
        loss_list = {'total':[], 'pos':[], 'scale':[], 'red':[], 'green':[], 'shape':[], 'depth':[]}

        frame_clopped_invdist = torch.zeros_like(frame_clopped_distance_map)
        frame_clopped_invdist[frame_clopped_mask] = 1 / frame_clopped_distance_map[frame_clopped_mask]
        # frame_gb_mask = self.gb(frame_clopped_mask)
        for optim_idx in range(-1, self.total_itr):
            if self.add_conf == 'only_once':
                if optim_idx == 0:
                    optim_idx = self.total_itr - 1

            # Get frame index info.
            if optim_idx < 0 and self.view_selection == 'sequential':
                frame_idx_list = [0]
            else:
                frame_idx_list = list(range(current_frame_num))

            # Get current maps and camera infos.
            clopped_mask = frame_clopped_mask[:, frame_idx_list].reshape(-1, self.ddf_H, self.ddf_H).detach()
            clopped_distance_map = frame_clopped_distance_map[:, frame_idx_list].reshape(-1, self.ddf_H, self.ddf_H).detach()
            clopped_invdist = frame_clopped_invdist[:, frame_idx_list].reshape(-1, self.ddf_H, self.ddf_H).detach()
            # gb_mask = frame_gb_mask[:, frame_idx_list].reshape(-1, self.ddf_H, self.ddf_H).detach()
            rays_d_cam = frame_clopped_rays_d_cam[:, frame_idx_list].reshape(-1, self.ddf_H, self.ddf_H, 3).detach()
            w2c = frame_w2c[:, frame_idx_list].reshape(-1, 3, 3).detach()
            cam_pos_wrd = frame_camera_pos_wrd[:, frame_idx_list].reshape(-1, 3).detach()
            bbox_diagonal = frame_bbox_diagonal[:, frame_idx_list].reshape(-1, 1)
            print_debug = (self.current_epoch==0 or step_mode=='tes') and batch_idx == 0 and optim_idx in {0, 2}
            if print_debug:
                if optim_idx == 0 or self.add_conf == 'only_once':
                    print('########## VAL ##########')
                    print(f'---     ins     --- : {instance_id[:2]}')
                    print(f'---    scene    --- : {scene_id[0][:2]}')
                    print(f'---   gtscale   --- : {frame_gt_obj_scale_wrd[:5, 0, 0]}')
                    print(f'---   ini_red   --- : {ini_obj_axis_red_wrd[:5, 0]}')
                    print(f'---   nearpos   --- : {(torch.norm(frame_camera_pos_wrd[:, :-1]-frame_camera_pos_wrd[:, 1:], dim=-1)/frame_gt_obj_scale_wrd[:, :-1, 0]).max()<1.04}')
                print(f'#-- current_itr --# : {optim_idx+1} / {self.total_itr}')
                print(f'---  itr_frame  --- : {frame_idx_list}')

            # Set timer.
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            # Perform initialization.
            if optim_idx < 0:
                est_obj_pos_wrd = ini_obj_pos.detach()
                est_obj_scale = ini_obj_scale.detach()
                est_obj_axis_green_wrd = ini_obj_axis_green_wrd.detach()
                est_obj_axis_red_wrd = ini_obj_axis_red_wrd.detach()
                est_shape_code = ini_shape_code.detach()

            # Perform dfnet.
            else:
                est_obj_pos_wrd, est_obj_axis_green_wrd, est_obj_axis_red_wrd, est_obj_scale, est_shape_code = \
                    self.forward(batch_size, current_frame_num, clopped_mask, clopped_distance_map, past_itr_log, pre_obj_pos_wrd, pre_obj_axis_green_wrd, 
                    pre_obj_axis_red_wrd, pre_obj_scale_wrd, pre_shape_code, w2c, cam_pos_wrd, rays_d_cam, bbox_diagonal, batch_idx, optim_idx, print_debug, 'val')
                # スケールの次元を揃える
                if self.data_type == 'shapenet':
                    if self.scale_dim == 3:
                        est_obj_scale = est_obj_scale.mean(-1).unsqueeze(-1)

            # Stop timer.
            end.record()
            torch.cuda.synchronize()
            itr_err_list['sec'].append(start.elapsed_time(end) / 1000)

            # Log est.
            itr_err_list['est_p'].append(est_obj_pos_wrd.detach().clone())
            itr_err_list['est_g'].append(est_obj_axis_green_wrd.detach().clone())
            itr_err_list['est_r'].append(est_obj_axis_red_wrd.detach().clone())
            itr_err_list['est_s'].append(est_obj_scale.detach().clone())
            itr_err_list['est_z'].append(est_shape_code.detach().clone())
            
            # Convergence. 
            update_mask = None
            # print(clopped_mask.shape)
            if self.until_convergence=='yes' and (
                (self.view_selection=='simultaneous' and optim_idx==0) or 
                (self.view_selection=='sequential' and 0 <= optim_idx and optim_idx < self.total_obs_num)):
                # Save pre dif.
                pre_dif_norm_obj_pos = torch.norm(est_obj_pos_wrd - pre_obj_pos_wrd, dim=-1) / math.sqrt(3)
                pre_dif_norm_obj_green = torch.norm(est_obj_axis_green_wrd - pre_obj_axis_green_wrd, dim=-1) / math.sqrt(3)
                pre_dif_norm_obj_red = torch.norm(est_obj_axis_red_wrd - pre_obj_axis_red_wrd, dim=-1) / math.sqrt(3)
                pre_dif_norm_obj_scale = torch.norm(est_obj_scale - pre_obj_scale_wrd, dim=-1) / math.sqrt(self.scale_dim)
                pre_dif_norm_obj_shape = torch.norm(est_shape_code - pre_shape_code, dim=-1) / math.sqrt(self.ddf.latent_size)
                # Save pre estimation.
                pre_obj_pos_wrd = est_obj_pos_wrd.clone().detach()
                pre_obj_axis_green_wrd = est_obj_axis_green_wrd.clone().detach()
                pre_obj_axis_red_wrd = est_obj_axis_red_wrd.clone().detach()
                pre_obj_scale_wrd = est_obj_scale.clone().detach()
                pre_shape_code = est_shape_code.clone().detach()

            elif self.until_convergence=='yes' and optim_idx > 0:
                dif_norm_obj_pos = torch.norm(est_obj_pos_wrd - pre_obj_pos_wrd, dim=-1) / math.sqrt(3)
                dif_norm_obj_axis_green = torch.norm(est_obj_axis_green_wrd - pre_obj_axis_green_wrd, dim=-1) / math.sqrt(3)
                dif_norm_obj_axis_red = torch.norm(est_obj_axis_red_wrd - pre_obj_axis_red_wrd, dim=-1) / math.sqrt(3)
                dif_norm_obj_scale = torch.norm(est_obj_scale - pre_obj_scale_wrd, dim=-1) / math.sqrt(self.scale_dim)
                dif_norm_obj_shape = torch.norm(est_shape_code - pre_shape_code, dim=-1) / math.sqrt(self.ddf.latent_size)

                change_flg_pos = 100 * torch.abs(dif_norm_obj_pos / pre_dif_norm_obj_pos) > self.convergence_thr
                change_flg_green = 100 * torch.abs(dif_norm_obj_axis_green / pre_dif_norm_obj_green) > self.convergence_thr
                change_flg_red = 100 * torch.abs(dif_norm_obj_axis_red / pre_dif_norm_obj_red) > self.convergence_thr
                change_flg_scale = 100 * torch.abs(dif_norm_obj_scale / pre_dif_norm_obj_scale) > self.convergence_thr
                change_flg_shape = 100 * torch.abs(dif_norm_obj_shape / pre_dif_norm_obj_shape) > self.convergence_thr_shape

                # print(100 * torch.abs(dif_norm_obj_pos / pre_dif_norm_obj_pos))
                # print(100 * torch.abs(dif_norm_obj_axis_green / pre_dif_norm_obj_green))
                # print(100 * torch.abs(dif_norm_obj_axis_red / pre_dif_norm_obj_red))
                # print(100 * torch.abs(dif_norm_obj_scale / pre_dif_norm_obj_scale))
                # print(100 * torch.abs(dif_norm_obj_shape / pre_dif_norm_obj_shape))
                update_mask = torch.stack([change_flg_pos, change_flg_green, change_flg_red, change_flg_scale, change_flg_shape], dim=-1).any(dim=-1)
                # print(update_mask)
                # import pdb; pdb.set_trace()

                pre_dif_norm_obj_pos[torch.logical_not(update_mask)] = torch.full_like(pre_dif_norm_obj_pos[torch.logical_not(update_mask)], 1e5)
                pre_dif_norm_obj_green[torch.logical_not(update_mask)] = torch.full_like(pre_dif_norm_obj_green[torch.logical_not(update_mask)], 1e5)
                pre_dif_norm_obj_red[torch.logical_not(update_mask)] = torch.full_like(pre_dif_norm_obj_red[torch.logical_not(update_mask)], 1e5)
                pre_dif_norm_obj_scale[torch.logical_not(update_mask)] = torch.full_like(pre_dif_norm_obj_scale[torch.logical_not(update_mask)], 1e5)
                pre_dif_norm_obj_shape[torch.logical_not(update_mask)] = torch.full_like(pre_dif_norm_obj_shape[torch.logical_not(update_mask)], 1e5)

                pre_obj_pos_wrd[update_mask] = est_obj_pos_wrd[update_mask].clone().detach()
                pre_obj_axis_green_wrd[update_mask] = est_obj_axis_green_wrd[update_mask].clone().detach()
                pre_obj_axis_red_wrd[update_mask] = est_obj_axis_red_wrd[update_mask].clone().detach()
                pre_obj_scale_wrd[update_mask] = est_obj_scale[update_mask].clone().detach()
                pre_shape_code[update_mask] = est_shape_code[update_mask].clone().detach()

            else:
                # Save pre estimation.
                pre_obj_pos_wrd = est_obj_pos_wrd.clone().detach()
                pre_obj_axis_green_wrd = est_obj_axis_green_wrd.clone().detach()
                pre_obj_axis_red_wrd = est_obj_axis_red_wrd.clone().detach()
                pre_obj_scale_wrd = est_obj_scale.clone().detach()
                pre_shape_code = est_shape_code.clone().detach()
            # pre_obj_pos_wrd = frame_gt_obj_pos_wrd[:, 0].clone().detach() + \
            #     (1. - ((optim_idx+1) / self.total_itr)) * torch.sum((torch.tensor([0.2, 0, 0]).to(ini_obj_pos))[None, None, :].expand(batch_size, -1, -1)*frame_gt_o2w[:, 0], dim=-1)
            # pre_obj_pos_wrd = frame_gt_obj_pos_wrd[:, 0].clone().detach() # + torch.sum(torch.tensor([[0, 0.2, 0]]).to(ini_obj_pos)[:, None, :]*frame_gt_o2w[:, 0, :, :], dim=-1)
            # pre_obj_axis_green_wrd = frame_gt_obj_green_wrd[:, 0].clone().detach()
            # pre_obj_axis_red_wrd = frame_gt_obj_red_wrd[:, 0].clone().detach()
            # pre_obj_scale_wrd = frame_gt_obj_scale_wrd[:, 0].clone().detach() # * 0.5
            # pre_shape_code = gt_shape_code.clone().detach()

            # Cal err at each iteration.
            #########################
            # 軸の対称性に注意
            if self.use_sym_label:
                sym_axis = []
                for sym_label_i in sym_label:
                    sym_axis.append(self.sym_axis[sym_label_i])
                sym_axis = torch.stack(sym_axis).to(frame_gt_o2w)
                sym_axis_wrd = torch.sum(sym_axis[:, :, None, :]*frame_gt_o2w[:, 0][:, None, :, :], dim=-1)
                loss_sym_red = - self.cossim(est_obj_axis_red_wrd[:, None, :], sym_axis_wrd) + 1.
                sym_axis_id = torch.argmin(loss_sym_red, dim=-1)
                gt_red_wrd = sym_axis_wrd[range(batch_size), sym_axis_id] # frame_gt_obj_red_wrd[:, 0]
                sym_inf_mask = torch.tensor([sym_label_i != 'SYM_ROTATE_UP_INF' for sym_label_i in sym_label]).to(gt_red_wrd)
            #########################
            if self.data_type == 'shapenet':
                if step_mode=='tes' or optim_idx==(self.total_itr-1): # optim_idx==(self.total_itr-1): 
                    itr_err_list['pos'].append(torch.abs(pre_obj_pos_wrd - frame_gt_obj_pos_wrd[:, 0]).mean(dim=-1))
                    err_axis_green_cos_sim_i = self.cossim(pre_obj_axis_green_wrd, frame_gt_obj_green_wrd[:, 0]).clamp(min=self.cosssim_min, max=self.cosssim_max)
                    itr_err_list['green'].append(torch.acos(err_axis_green_cos_sim_i) * 180 / torch.pi)
                    if not self.use_sym_label:
                        err_axis_red_cos_sim_i = self.cossim(pre_obj_axis_red_wrd, frame_gt_obj_red_wrd[:, 0]).clamp(min=self.cosssim_min, max=self.cosssim_max)
                    else:
                        err_axis_red_cos_sim_i = self.cossim(pre_obj_axis_red_wrd, gt_red_wrd).clamp(min=self.cosssim_min, max=self.cosssim_max) * sym_inf_mask + (1.0 - sym_inf_mask)
                    itr_err_list['red'].append(torch.acos(err_axis_red_cos_sim_i) * 180 / torch.pi)
                    itr_err_list['scale'].append(100 * torch.abs(pre_obj_scale_wrd[:, 0] - frame_gt_obj_scale_wrd[:, 0, 0]) / frame_gt_obj_scale_wrd[:, 0, 0])
                    pre_o2w = axis2rotation(pre_obj_axis_green_wrd, pre_obj_axis_red_wrd)
                    itr_err_list['rot'].append(get_riemannian_distance(pre_o2w, frame_gt_o2w[:, 0]).to(pre_obj_pos_wrd))
                    depth_error = []
                    if step_mode=='tes':
                        est_pc_wrd_list = []
                        est_pc_obj_list = []
                        gt_pc_obj_list = []
                        for i_ in range(batch_size):
                            est_pc_wrd_list.append([])
                            est_pc_obj_list.append([])
                            gt_pc_obj_list.append([])
                    for gt_distance_map, cam_pos_wrd, w2c in zip(canonical_distance_map.permute(1, 0, 2, 3), 
                                                                canonical_camera_pos.permute(1, 0, 2), 
                                                                canonical_camera_rot.permute(1, 0, 2, 3)):
                        # Rendering
                        rays_d_cam = self.canonical_rays_d_cam.expand(batch_size, -1, -1, -1)
                        est_mask, est_distance_map = get_canonical_map(
                                                    H = self.ddf_H, 
                                                    cam_pos_wrd = cam_pos_wrd, 
                                                    rays_d_cam = rays_d_cam, 
                                                    w2c = w2c, 
                                                    input_lat_vec = pre_shape_code, 
                                                    ddf = self.ddf, )
                        depth_error.append(torch.abs(gt_distance_map-est_distance_map).mean(dim=-1).mean(dim=-1)) 
                        # check_map_torch(torch.abs(gt_distance_map-est_distance_map).reshape(-1, 128), 'tes.png')
                        # import pdb; pdb.set_trace()
                        # Get est pt
                        if step_mode=='tes':
                            est_mask = torch.logical_and(est_mask, est_distance_map < 1.25) # ?
                            b_est_mask = est_mask.clone()
                            b_mask_thr = 1
                            b_est_mask[:, b_mask_thr:, :] = torch.logical_and(b_est_mask[:, b_mask_thr:, :], est_mask[:, :-b_mask_thr, :])
                            b_est_mask[:, :-b_mask_thr, :] = torch.logical_and(b_est_mask[:, :-b_mask_thr, :], est_mask[:, b_mask_thr:, :])
                            b_est_mask[:, :, b_mask_thr:] = torch.logical_and(b_est_mask[:, :, b_mask_thr:], est_mask[:, :, :-b_mask_thr])
                            b_est_mask[:, :, :-b_mask_thr] = torch.logical_and(b_est_mask[:, :, :-b_mask_thr], est_mask[:, :, b_mask_thr:])
                            est_pc_cam = rays_d_cam * est_distance_map[..., None]
                            est_pc_obj_cnc_i = torch.sum(est_pc_cam[..., None, :]*w2c.permute(0, 2, 1)[:, None, None, :, :], dim=-1) + cam_pos_wrd[:, None, None, :]
                            est_pc_wrd_cnc_i = torch.sum(est_pc_obj_cnc_i.clone()[..., None, :]*pre_o2w[..., None, None, :, :], dim=-1) * pre_obj_scale_wrd[..., None, None, :] + pre_obj_pos_wrd[..., None, None, :]
                            for c_btc_idx, (pt_obj_i, pt_wrd_i, mask_i) in enumerate(zip(est_pc_obj_cnc_i, est_pc_wrd_cnc_i, b_est_mask)):
                                est_pc_obj_list[c_btc_idx].append(pt_obj_i[mask_i].to('cpu').detach().numpy().copy())
                                est_pc_wrd_list[c_btc_idx].append(pt_wrd_i[mask_i].to('cpu').detach().numpy().copy())
                            # 正順マップデバッグ
                            gt_mask = gt_distance_map != 0.0
                            gt_mappc_cam = rays_d_cam * gt_distance_map[..., None]
                            gt_mappc_obj_cnc_i = torch.sum(gt_mappc_cam[..., None, :]*w2c.permute(0, 2, 1)[:, None, None, :, :], dim=-1) + cam_pos_wrd[:, None, None, :]
                            for c_btc_idx, (pt_obj_i, mask_i) in enumerate(zip(gt_mappc_obj_cnc_i, gt_mask)):
                                gt_pc_obj_list[c_btc_idx].append(pt_obj_i[mask_i].to('cpu').detach().numpy().copy())
                            
                    itr_err_list['shape'].append(torch.stack(depth_error, dim=-1).mean(dim=-1))
                if step_mode=='tes': # and optim_idx==(self.total_itr-1):
                    # Get np pt.
                    est_pc_obj = [np.concatenate(pt_list) for pt_list in est_pc_obj_list]
                    est_pc_wrd = [np.concatenate(pt_list) for pt_list in est_pc_wrd_list]
                    gt_mappc_wrd = [np.concatenate(pt_list) for pt_list in gt_pc_obj_list]
                    gt_pc_wrd = torch.sum(gt_pc_obj[..., None, :]*frame_gt_o2w[:, 0][:, None, :, :], dim=-1) * frame_gt_obj_scale_wrd[:, 0][:, None, :] + frame_gt_obj_pos_wrd[:, 0][:, None, :]
                    gt_pc_obj_np = gt_pc_obj.to('cpu').detach().numpy().copy()
                    gt_pc_wrd_np = gt_pc_wrd.to('cpu').detach().numpy().copy()
                    # Cal Chamfer.
                    chm_dist_obj = []
                    chm_sdist_obj = []
                    # for btc_id, (est_pc_i, gt_pc_i) in enumerate(zip(est_pc_obj, gt_pc_obj_np)):
                    # for btc_id, (est_pc_i, gt_pc_i, sym_axis_wrd_i) in enumerate(zip(est_pc_wrd, gt_pc_wrd_np, sym_axis_wrd)):
                    for btc_id, (est_pc_i, gt_pc_i, gt_mappc_i) in enumerate(zip(est_pc_obj, gt_pc_obj_np, gt_mappc_wrd)):
                        chm_dist_obj.append(compute_trimesh_chamfer(est_pc_i, gt_pc_i))
                        est_pc_i = pre_obj_scale_wrd[btc_id, 0].to('cpu').detach().numpy().copy() * est_pc_i
                        gt_pc_i = frame_gt_obj_scale_wrd[btc_id, 0, 0].to('cpu').detach().numpy().copy() * gt_pc_i
                        gt_mappc_i = frame_gt_obj_scale_wrd[btc_id, 0, 0].to('cpu').detach().numpy().copy() * gt_mappc_i
                        chm_sdist_obj.append(compute_trimesh_chamfer(est_pc_i, gt_pc_i))
                        # # 点群デバッグ
                        # fig = plt.figure()
                        # ax = Axes3D(fig)
                        # point_1 = est_pc_i
                        # ax.scatter(point_1[:, 0], point_1[:, 1], point_1[:, 2], marker="o", linestyle='None', c='c', s=0.05)
                        # point_2 = gt_pc_i
                        # ax.scatter(point_2[::2, 0], point_2[::2, 1], point_2[::2, 2], marker="o", linestyle='None', c='m', s=0.05)
                        # point_3 = gt_mappc_i
                        # ax.scatter(point_3[::2, 0], point_3[::2, 1], point_3[::2, 2], marker="o", linestyle='None', c='g', s=0.05)
                        # ax.view_init(elev=0, azim=90)
                        # fig.savefig(f"{btc_id}_00_90.png")
                        # ax.view_init(elev=0, azim=0)
                        # fig.savefig(f"{btc_id}_00_00.png")
                        # ax.view_init(elev=45, azim=45)
                        # fig.savefig(f"{btc_id}_45_45.png")
                        # plt.close()
                        # fig = plt.figure()
                        # ax = Axes3D(fig)
                        # point_1 = est_pc_i
                        # ax.scatter(point_1[:, 0], point_1[:, 1], point_1[:, 2], marker="o", linestyle='None', c='c', s=0.05)
                        # point_2 = gt_pc_i
                        # ax.scatter(point_2[::2, 0], point_2[::2, 1], point_2[::2, 2], marker="o", linestyle='None', c='m', s=0.05)
                        # point_3 = (sym_axis_wrd_i + frame_gt_obj_pos_wrd[:, 0][btc_id]).to('cpu').detach().numpy().copy()
                        # ax.scatter(point_3[:, 0], point_3[:, 1], point_3[:, 2], marker="o", linestyle='None', c='g', s=1.0)
                        # point_4 = (gt_red_wrd[btc_id] + frame_gt_obj_pos_wrd[:, 0][btc_id]).to('cpu').detach().numpy().copy()
                        # ax.scatter(point_4[0], point_4[1], point_4[2], marker="o", linestyle='None', c='r', s=5.0)
                        # point_5 = (pre_obj_axis_red_wrd[btc_id] + frame_gt_obj_pos_wrd[:, 0][btc_id]).to('cpu').detach().numpy().copy()
                        # ax.scatter(point_5[0], point_5[1], point_5[2], marker="o", linestyle='None', c='b', s=5.0)
                        # ax.view_init(elev=0, azim=90)
                        # fig.savefig(f"{btc_id+100}_00_90.png")
                        # ax.view_init(elev=0, azim=0)
                        # fig.savefig(f"{btc_id+100}_00_00.png")
                        # ax.view_init(elev=45, azim=45)
                        # fig.savefig(f"{btc_id+100}_45_45.png")
                        # plt.close()
                        # import pdb; pdb.set_trace()
                    itr_err_list['chm_obj'].append(torch.tensor(chm_dist_obj).to(pre_obj_pos_wrd))
                    itr_err_list['chm_s_obj'].append(torch.tensor(chm_sdist_obj).to(pre_obj_pos_wrd))
                    itr_err_list['chm_ss_obj'].append(torch.tensor(chm_sdist_obj).to(pre_obj_pos_wrd) / frame_gt_obj_scale_wrd[:, 0, 0])
                    chm_dist_wrd = []
                    for btc_id, (est_pc_i, gt_pc_i) in enumerate(zip(est_pc_wrd, gt_pc_wrd_np)):
                        chm_dist_wrd.append(compute_trimesh_chamfer(est_pc_i, gt_pc_i))
                    itr_err_list['chm_wrd'].append(torch.tensor(chm_dist_wrd).to(pre_obj_pos_wrd))
                    itr_err_list['chm_wrd_s'].append(torch.tensor(chm_dist_wrd).to(pre_obj_pos_wrd) / frame_gt_obj_scale_wrd[:, 0, 0])
            # else: itr_err_list['shape'].append(torch.zeros_like(itr_err_list['scale'][-1]))

            # Save steps.
            if update_mask is not None:
                itr_err_list['cnv'].append(update_mask.to(pre_shape_code))
            else:
                itr_err_list['cnv'].append(torch.ones(batch_size).to(pre_shape_code))

            # Update frame infos.
            if optim_idx >= 0:
                frame_optim_idx += 1
            if frame_optim_idx >= self.itr_per_frame:
                if current_frame_num < self.total_obs_num:
                    current_frame_num += 1
                    frame_optim_idx = 0

        # Cal loss to monitor performance.
        if step_mode == 'val':
            loss_list['pos'] = self.mseloss(est_obj_pos_wrd, frame_gt_obj_pos_wrd[:, 0]).detach()
            loss_list['green'] = torch.mean(-self.cossim(est_obj_axis_green_wrd, frame_gt_obj_green_wrd[:, 0]) + 1.).detach()
            if not self.use_sym_label:
                err_axis_red_cos_sim_i = self.cossim(est_obj_axis_red_wrd, frame_gt_obj_red_wrd[:, 0])
            else:
                err_axis_red_cos_sim_i = self.cossim(est_obj_axis_red_wrd, gt_red_wrd) * sym_inf_mask + (1.0 - sym_inf_mask)
            loss_list['red'] = torch.mean(- err_axis_red_cos_sim_i + 1.).detach()
            loss_list['scale'] = self.mseloss(est_obj_scale, frame_gt_obj_scale_wrd[:, 0]).detach()
            loss_list['shape'] = self.mseloss(est_shape_code, gt_shape_code).detach()
            loss_list['depth'] = torch.zeros_like(loss_list['shape']).detach()
            loss_axis = loss_list['green'] + self.L_r * loss_list['red']
            loss_list['total'] = (self.L_p * loss_list['pos'] + self.L_s * loss_list['scale'] + self.L_a * loss_axis + self.L_c * loss_list['shape'] + self.L_d * loss_list['depth']).detach()

        # ##################################################
        # batch_i = 0
        # os.makedirs(f'paper_fig/normal_map/src/{self.test_log_name}', exist_ok=True)
        # pickle_dump({'gt_pos'   : frame_gt_obj_pos_wrd[batch_i, 0, :].to('cpu').detach().numpy().copy(), 
        #              'gt_o2w'   : frame_gt_o2w[batch_i, 0, :, :].to('cpu').detach().numpy().copy(), 
        #              'gt_scale' : frame_gt_obj_scale_wrd[batch_i, 0, :].to('cpu').detach().numpy().copy(), 
        #              'est_pos'  : pre_obj_pos_wrd[batch_i, :].to('cpu').detach().numpy().copy(), 
        #              'est_o2w'  : axis2rotation(pre_obj_axis_green_wrd, pre_obj_axis_red_wrd)[batch_i, :, :].to('cpu').detach().numpy().copy(), 
        #              'est_scale': pre_obj_scale_wrd[batch_i, :].to('cpu').detach().numpy().copy(), 
        #              'est_shape': pre_shape_code[batch_i, :].to('cpu').detach().numpy().copy()}, 
        #             f'paper_fig/normal_map/src/{self.test_log_name}/{str(batch_idx).zfill(5)}_{instance_id[batch_i]}.pickle') # 'tes.pickle')
        # import pdb; pdb.set_trace()
        # ##################################################
        import pdb; pdb.set_trace()

        # Return.
        itr_err_list['cnv'] = sum(itr_err_list['cnv']) - 1.
        return {'loss_list': loss_list, 
                'itr_err_list': itr_err_list, 
                'path': np.array(path).T, 
                'gt_S_seed': rand_seed['gt_S_seed'], 
                'rand_P_seed': rand_seed['rand_P_seed'], 
                'rand_S_seed': rand_seed['rand_S_seed'], 
                'randn_theta_seed': rand_seed['randn_theta_seed'], 
                'randn_axis_idx': rand_seed['randn_axis_idx'], }



    def test_epoch_end(self, outputs):
        log_dict = {}
        sec_start, avg_start, med_start = 0, 2, 15
        log_txt = ['???_sec_log = np.array([', # ])
                   '???_avg_log = {', 
                   '    \'pos\'       : np.array([', # ]), 
                   '    \'rot\'       : np.array([', # ]), 
                   '    \'chm_ss_obj\': np.array([', # ]), 
                   '    \'chm_s_obj\' : np.array([', # ]), 
                   '    \'green\'     : np.array([', # ]), 
                   '    \'red\'       : np.array([', # ]), 
                   '    \'scale\'     : np.array([', # ]), 
                   '    \'shape\'     : np.array([', # ]), 
                   '    \'chm_obj\'   : np.array([', # ]), 
                   '    \'chm_wrd\'   : np.array([', # ]), 
                   '    \'chm_wrd_s\' : np.array([', # ]), 
                   '    }', 
                   '???_med_log = {', 
                   '    \'pos\'       : np.array([', # ]), 
                   '    \'rot\'       : np.array([', # ]), 
                   '    \'chm_ss_obj\': np.array([', # ]), 
                   '    \'chm_s_obj\' : np.array([', # ]), 
                   '    \'green\'     : np.array([', # ]), 
                   '    \'red\'       : np.array([', # ]), 
                   '    \'scale\'     : np.array([', # ]), 
                   '    \'shape\'     : np.array([', # ]), 
                   '    \'chm_obj\'   : np.array([', # ]), 
                   '    \'chm_wrd\'   : np.array([', # ]), 
                   '    \'chm_wrd_s\' : np.array([', # ]), 
                   '    }', 
                   ]

        # log err
        for idx, key_i in enumerate(['pos', 'rot', 'chm_ss_obj', 'chm_s_obj', 'green', 'red', 'scale', 'shape', 'chm_obj', 'chm_wrd', 'chm_wrd_s']):
            itr_err = torch.cat([torch.stack(x['itr_err_list'][key_i], dim=-1) for x in outputs], dim=0)
            itr_err = itr_err.to('cpu').detach().numpy().copy()
            itr_err_med = np.median(itr_err, axis=0).tolist()
            itr_err_avg = np.mean(itr_err, axis=0).tolist()
            log_dict[key_i] = itr_err # itr_err[:, -1]
            log_txt[avg_start+idx] += (', ').join([str(n) for n in itr_err_avg]) + ']), '
            log_txt[med_start+idx] += (', ').join([str(n) for n in itr_err_med]) + ']), '
            with open(self.test_log_path, 'a') as file:
                file.write(f'{key_i}, {str(itr_err_avg[-1])}, {str(itr_err_med[-1])}\n')
        # log sec
        itr_sec = np.stack([x['itr_err_list']['sec'] for x in outputs], axis=0)
        itr_sec_txt = (', ').join([str(n) for n in itr_sec[:-1].mean(0).tolist()])
        log_txt[sec_start] += itr_sec_txt + '])'
        # log cnv
        itr_cnv = torch.cat([x['itr_err_list']['cnv'] for x in outputs], dim=0)
        itr_cnv = itr_cnv.to('cpu').detach().numpy().copy()
        itr_cnv_med = np.median(itr_cnv)
        itr_cnv_avg = np.average(itr_cnv)
        log_dict['cnv'] = itr_cnv # itr_err[:, -1]
        with open(self.test_log_path, 'a') as file:
            file.write(f'cnv, {str(itr_cnv_avg)}, {str(itr_cnv_med)}\n')

        for idx, key_i in enumerate(['path', 'gt_S_seed', 'rand_P_seed', 'rand_S_seed', 'randn_theta_seed', 'randn_axis_idx']):
            log_dict[key_i] = np.concatenate([x[key_i] for x in outputs])
        for idx, key_i in enumerate(['est_p', 'est_g', 'est_r', 'est_s', 'est_z']):
            est_key_i = torch.cat([torch.stack(x['itr_err_list'][key_i]) for x in outputs], dim=1)
            log_dict[key_i] = est_key_i.to('cpu').detach().numpy().copy()
        pickle_dump(log_dict, self.test_log_path.split('.txt')[0] + '_error.pickle')
        with open(self.test_log_path, 'a') as file:
            for txt in log_txt:
                file.write(txt + '\n')



    def on_validation_epoch_start(self):
        self.rs['val'] = np.random.RandomState(72)
        self.rays_d_cam = self.rays_d_cam.to(self.df_net.device)
        self.canonical_rays_d_cam = self.canonical_rays_d_cam.to(self.df_net.device)
        if self.view_selection == 'simultaneous' and self.prg == 'yes':            # if self.view_selection == 'sequential': # 
            self.total_itr = int(min(self.max_itr, self.current_epoch // 80 + 3))  # if self.current_epoch < 200: self.total_itr, self.itr_per_frame = 5, 1 # 
            self.itr_per_frame = self.total_itr                                    # else: self.total_itr, self.itr_per_frame = 25, 5 #
        if self.view_selection == 'sequential' and self.prg == 'yes':
            if self.current_epoch < self.cnt_prg_step:
                self.total_itr = 5
                self.itr_per_frame = 1
            else:
                self.total_itr = 10
                self.itr_per_frame = 2



    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx, step_mode='val')



    def validation_epoch_end(self, outputs):
        if self.code_mode == 'REVAL':
            avg_loss = torch.stack([x['loss_list'][ 'total'] for x in outputs]).mean()
            print(f'VAL_LOSS:::{self.current_epoch}:::{avg_loss}')
            with open(self.reval_output_file, 'a') as f:
                print(f'VAL_LOSS:::{self.current_epoch}:::{avg_loss}', file=f)
        
        else:
            for key_i in {'pos', 'green', 'red', 'scale', 'shape', 'total', 'depth'}:
                avg_loss = torch.stack([x['loss_list'][key_i] for x in outputs]).mean()
                current_epoch = torch.tensor(self.current_epoch + 1., dtype=avg_loss.dtype)
                if key_i in {'pos', 'green', 'red', 'scale', 'shape', 'total'}:
                    self.log_dict({f'val/loss_{key_i}': avg_loss, "step": current_epoch})

                if key_i in {'pos', 'green', 'red', 'scale', 'shape'}:
                    itr_err = torch.cat([torch.stack(x['itr_err_list'][key_i], dim=-1) for x in outputs], dim=0)
                    last_err_med = torch.median(itr_err[:, -1])
                    self.log_dict({f'val/err_{key_i}_med': last_err_med, "step": current_epoch})
                    last_err_avg = torch.mean(itr_err[:, -1])
                    self.log_dict({f'val/err_{key_i}_avg': last_err_avg, "step": current_epoch})



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.df_net.parameters()) + list(self.backbone_encoder.parameters()), 
        lr=self.lr, betas=(0.9, 0.999),)
        return optimizer



if __name__=='__main__':
    # Get args
    args = get_args()
    args.check_val_every_n_epoch = args.save_interval
    args.gpu_num = torch.cuda.device_count()
    pre_trained_ckpt_base_dir = f'lightning_logs/{args.expname}'
    pre_trained_val_model_name = args.expname.split('/')[-1] + args.exp_version
    # 合成データで事前学習したモデルのパス
    pre_trained_path = f'{pre_trained_ckpt_base_dir}/{pre_trained_val_model_name}/checkpoints/{str(args.pre_train_epoch).zfill(10)}.ckpt'
    # ScanNet用のパス
    args.expname = os.path.join('ScanNet', args.expname.split('/')[1])
    
    if args.code_mode == 'TRAIN':
        ##########################################################################################################################################################################################################################################################
        ##########################################################################################################################################################################################################################################################
        ##########################################################################################################################################################################################################################################################
        # CUDA_VISIBLE_DEVICES=7 python train_check_scannet.py --code_mode=TRAIN --config=configs/paper_exp/chair_re/view5/rdn.txt --exp_version=rdn_autoreg --N_batch=8 --pre_train_epoch=360 --data_type=scan2cad
        # 書き換える
        args.cat_id = '03001627'
        args.scannet_data_list = txt2list('/home/yyoshitake/works/DeepSDF/project/scannet/results_chair_train/data_list.txt')
        args.pretrain_model_path = '/home/yyoshitake/works/DeepSDF/project/lightning_logs/DeepTaR/chair_re/rdn_autoreg/checkpoints/0000000360.ckpt'

        # Make dummy data loader
        from scannet_dataset import scannet_dataset, render_distance_map_from_axis_for_scannet
        tes_dataset = scannet_dataset(args, 'train', ['scene0000_00'])
        tes_dataloader = data_utils.DataLoader(tes_dataset, batch_size=1, num_workers=0)
        # 書き換える
        tes_dataset.data_list = args.scannet_data_list
        ##########################################################################################################################################################################################################################################################
        ##########################################################################################################################################################################################################################################################
        ##########################################################################################################################################################################################################################################################

        # Set trainer.
        logger = pl.loggers.TensorBoardLogger(save_dir=os.getcwd(), version=os.path.join(args.expname, args.exp_version), name='lightning_logs')
        trainer = pl.Trainer(
            gpus = args.gpu_num, 
            strategy = DDPPlugin(find_unused_parameters=True), #=False), 
            logger = logger,
            max_epochs = args.N_epoch, 
            check_val_every_n_epoch = 1,
            )

        # Save config files.
        os.makedirs(os.path.join('lightning_logs', os.path.join(args.expname, args.exp_version)), exist_ok=True)
        f = os.path.join('lightning_logs', os.path.join(args.expname, args.exp_version), 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        if args.config is not None:
            f = os.path.join('lightning_logs', os.path.join(args.expname, args.exp_version), 'config.txt')
            with open(f, 'w') as file:
                file.write(open(args.config, 'r').read())

        # Set ddf.
        ddf = DDF(args)
        ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args)
        ddf.eval()

        # Set back bone.
        backbone = backbone_encoder_decoder(args)

        # Set model.
        model = original_optimizer(args, {'ddf':ddf, 'backbone_encoder':backbone.encoder_2dcnn})

        # Create dataloader
        def seed_worker(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id)
        train_generator = torch.Generator().manual_seed(7)
        train_dataset = txt2dataset(args, 'train', args.train_instance_list_txt, 100, args.train_txtfile)
        train_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.N_batch, num_workers=args.num_workers, shuffle=True, worker_init_fn=seed_worker, generator=train_generator)
        
        # val_generator = torch.Generator().manual_seed(8)
        # val_dataset = txt2dataset(args, 'val', args.val_instance_list_txt, 0, args.val_data_list)
        # val_dataloader = data_utils.DataLoader(val_dataset, batch_size=args.N_batch, num_workers=args.num_workers, shuffle=False, generator=val_generator)

        # 事前学習済みモデルの読み込み
        pretrained_model = original_optimizer(args, {'ddf':ddf, 'backbone_encoder':backbone.encoder_2dcnn})
        pretrained_model = pretrained_model.load_from_checkpoint(checkpoint_path=pre_trained_path, args=args, pretrain_models={'ddf':ddf, 'backbone_encoder':backbone.encoder_2dcnn})
        model.df_net = pretrained_model.df_net
        model.ddf = pretrained_model.ddf
        model.backbone_encoder = pretrained_model.backbone_encoder
        import pdb; pdb.set_trace()
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=False, datamodule=None, ckpt_path=None)

    elif args.code_mode == 'TES':
        # Reload args.
        args = reload_args(args, sys.argv)

        # Create dataloader.
        if args.data_type == 'shapenet':
            tes_generator = torch.Generator().manual_seed(1)
            tes_dataset = txt2dataset(args, 'tes', args.test_instance_list_txt, args.test_data_dir, args.tes_data_list)
            tes_dataloader = data_utils.DataLoader(tes_dataset, batch_size=args.N_batch, num_workers=8, shuffle=False, generator=tes_generator)
        
        ##########################################################################################################################################################################################################################################################
        ##########################################################################################################################################################################################################################################################
        ##########################################################################################################################################################################################################################################################
        # CUDA_VISIBLE_DEVICES=0 python train_check_shapenet.py --code_mode=TES --config=configs/paper_exp/chair_a/view5/tes.txt --exp_version=rndsim_encoder_tau01 --N_batch=1 --val_model_epoch=400 --data_type=scan2cad
        elif args.data_type == 'scan2cad':
            # 書き換える
            args.cat_id = '03001627'
            args.tes_scene_list = ['scene0314_00']
            args.scannet_data_dir = 'scannet/results_tes'
            tgt_obj_id = 'c585ee093bfd52af6512b7b24f3d84_001' # '3289bcc9bf8f5dab48d8ff57878739ca_012'
            check_tgt_sampled_image_names = ['00005_00000_00023', '00010_00000_00028', '00015_00000_00033', '00020_00000_00038', '00025_00000_00043']
            # check_tgt_sampled_image_names = ['00003_00002_00488', '00010_00002_00495', '00021_00002_00506', '00027_00002_00512', '00030_00002_00515']
            check_tgt_obj_data_list = [os.path.join(args.scannet_data_dir, args.cat_id, args.tes_scene_list[0], tgt_obj_id)]

            # Make dummy data loader
            from scannet_dataset import scannet_dataset, render_distance_map_from_axis_for_scannet
            tes_dataset = scannet_dataset(args, 'tes', args.tes_scene_list)
            tes_dataloader = data_utils.DataLoader(tes_dataset, batch_size=1, num_workers=0)

            # 書き換える
            tes_dataset.check_tgt_sampled_image_names = check_tgt_sampled_image_names
            tes_dataset.data_list = check_tgt_obj_data_list
        ##########################################################################################################################################################################################################################################################
        ##########################################################################################################################################################################################################################################################
        ##########################################################################################################################################################################################################################################################

        # Set models and Start training.
        ddf = DDF(args)
        ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args)
        ddf.eval()
        
        # Set back bone.
        backbone = backbone_encoder_decoder(args)

        # Set model.
        model = original_optimizer(args, {'ddf':ddf, 'backbone_encoder':backbone.encoder_2dcnn})
        model = model.load_from_checkpoint(checkpoint_path=args.model_ckpt_path, args=args, pretrain_models={'ddf':ddf, 'backbone_encoder':backbone.encoder_2dcnn})
        # スケール１で訓練してしまったモデルがある。
        if args.data_type == 'scan2cad':
            model.scale_dim = 3

        # Save logs.
        if args.data_type == 'shapenet':
            import datetime
            dt_now = datetime.datetime.now()
            time_log = dt_now.strftime('%Y_%m_%d_%H_%M_%S')
            time_log += '_' + args.model_ckpt_path.split('/')[-3] + '_epo_' + args.model_ckpt_path.split('/')[-1].split('.')[0]
            os.mkdir('./txt/experiments/log/' + time_log)
            file_name = './txt/experiments/log/' + time_log + '/log.txt'
            model.test_log_path = file_name
            with open(file_name, 'a') as file:
                # Base setting log.
                file.write('ckpt_path : ' + args.model_ckpt_path + '\n')
                file.write('test_instance_list_txt : ' + str(args.test_instance_list_txt) + '\n')
                file.write('total_obs_num : ' + str(model.total_obs_num) + '\n')
                file.write('total_itr : ' + str(model.total_itr) + '\n')
                file.write('tes_data_list : ' + str(args.tes_data_list) + '\n')
                file.write('view_position : ' + str(args.view_position) + '\n')
                file.write('view_selection : ' + str(args.view_selection) + '\n')
                file.write('gt_scale_range : ' + str(model.gt_scale_range) + '\n')
                file.write('rand_P_range : ' + str(model.rand_P_range) + '\n')
                file.write('rand_S_range : ' + str(model.rand_S_range) + '\n')
                file.write('rand_R_range : ' + str(model.rand_R_range) + '\n')
                file.write('random_axis_num : ' + str(model.random_axis_num) + '\n')
                file.write('\n' + '\n')
            model.test_log_name = args.model_ckpt_path.split('/')[-3] + '_epo_' + args.model_ckpt_path.split('/')[-1].split('.')[0]

        # Test model.
        trainer = pl.Trainer(
            gpus=args.gpu_num, 
            strategy=DDPPlugin(find_unused_parameters=True), #=False), 
            enable_checkpointing = False,
            check_val_every_n_epoch = args.check_val_every_n_epoch,
            logger = pl.loggers.TensorBoardLogger(save_dir=os.getcwd(), version=f'val_trash', name='lightning_logs')
            )
        trainer.test(model, tes_dataloader)

    elif args.code_mode == 'REVAL':
        # Reload args.
        args = reload_args(args, sys.argv)

        # Set trainer.
        trainer = pl.Trainer(
            gpus=args.gpu_num, 
            strategy=DDPPlugin(find_unused_parameters=True), #=False), 
            max_epochs=args.N_epoch, 
            )

        # Set ddf.
        ddf = DDF(args)
        ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args)
        ddf.eval()

        # Set back bone.
        backbone = backbone_encoder_decoder(args)

        # Set model.
        model_ckpt_list = sorted(glob.glob(f'{ckpt_base_dir}/{val_model_name}/checkpoints/*'))[48:]
        model = original_optimizer(args, {'ddf':ddf, 'backbone_encoder':backbone.encoder_2dcnn})
        expname = args.expname.split('/')[-2]
        model.reval_output_file = f'{expname}_{args.exp_version}'
        print('\n', f'### {model.reval_output_file} ###', '\n')

        # Create dataloader
        def seed_worker(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        val_generator = torch.Generator().manual_seed(8)
        val_dataset = txt2dataset(args, 'val', args.val_instance_list_txt, 0, args.val_data_list)
        val_dataloader = data_utils.DataLoader(val_dataset, batch_size=args.N_batch, num_workers=args.num_workers, shuffle=False, generator=val_generator)

        # Start training.
        for ckpt_path in model_ckpt_list:
            trainer.validate(model=model, val_dataloaders=val_dataloader, ckpt_path=ckpt_path)
        