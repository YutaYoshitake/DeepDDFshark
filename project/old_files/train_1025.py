import os
import pdb
import sys
from turtle import pd
import numpy as np
import random
import pylab
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
from torch.autograd import grad
import torch.utils.data as data
import time
# import torchvision
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from chamferdist import ChamferDistance
# from scipy.spatial.transform import Rotation as R

from BackBone.train_backbone import *
from resnet import *
from parser_get_arg import *
from dataset import *
from often_use import *
from DDF.train_pl import DDF
from model import optimize_former

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
        self.lr = args.lr
        self.rays_d_cam = get_ray_direction(self.ddf_H, self.fov)
        self.canonical_rays_d_cam = get_ray_direction(self.ddf_H, args.canonical_fov)
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
        self.train_instance_ids = [self.ddf_instance_list.index(instance_i) for instance_i in self.train_instance_list]
        self.rand_Z_center = ddf.lat_vecs(torch.tensor(self.train_instance_ids, device=ddf.device)).mean(0).clone().detach()
        self.optim_mode = args.optim_mode

        # Make model
        self.main_layers_name = args.main_layers_name
        self.positional_encoding_mode = args.positional_encoding_mode
        self.df_net = optimize_former( # _old
                            main_layers_name = args.main_layers_name, 
                            input_type = args.input_type,
                            positional_encoding_mode = args.positional_encoding_mode, 
                            num_encoder_layers = args.num_encoder_layers, 
                            num_decoder_layers = args.num_decoder_layers, 
                            add_conf=args.add_conf, 
                            itr_frame_num=self.itr_frame_num, )
        self.ddf = pretrain_models['ddf']
        #########################
        self.backbone_encoder = pretrain_models['backbone_encoder'] # torch.nn.ModuleList()
        self.backbone_embedding_dim = self.backbone_encoder.embedding_dim
        #########################
        self.backbone_training_strategy = args.backbone_training_strategy
        self.lr_backbone = args.lr_backbone
        self.rs = {'train': np.random.RandomState(71), 
                   'val'  : np.random.RandomState(72), 
                   'tes'  : np.random.RandomState(73), }
        self.rand_P_range = args.rand_P_range # [-self.rand_P_range, self.rand_P_range)で一様サンプル
        self.rand_S_range = args.rand_S_range # [1.-self.rand_S_range, 1.+self.rand_S_range)で一様サンプル
        self.rand_R_range = args.rand_R_range * np.pi # [-self.rand_R_range, self.rand_R_range)で一様サンプル
        self.random_axis_num = 1024
        self.random_axis_list = torch.from_numpy(sample_fibonacci_views(self.random_axis_num).astype(np.float32)).clone()
        self.gt_scale_range = 0.2

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
        self.mseloss = nn.MSELoss() # nn.SmoothL1Loss()
        self.cossim = nn.CosineSimilarity(dim=-1)
        self.cosssim_min = - 1 + 1e-8
        self.cosssim_max = 1 - 1e-8
        self.randn_from_pickle = True
        self.rand_idx = {'train':0, 'val':0, 'tes':0}
        self.rand_idx_max = {'train':1e6, 'val':1e5, 'tes':1e5}
        self.loss_timing = args.loss_timing



    def preprocess(self, batch, mode):
        # Get batch data.
        frame_mask, frame_distance_map, bbox_list, instance_id, frame_camera_pos_wrd, frame_w2c, frame_gt_obj_pos_wrd, frame_gt_o2w, \
        frame_gt_obj_green_wrd, frame_gt_obj_red_wrd, frame_gt_o2c, frame_gt_obj_green_cam, frame_gt_obj_red_cam, frame_gt_obj_scale_wrd, \
        canonical_distance_map, canonical_camera_pos, canonical_camera_rot, path, rand_seed, scene_id = batch
        
        # Get baych size
        batch_size = len(instance_id)

        # Get Ground truth.
        frame_clopped_mask, frame_clopped_distance_map, frame_clopped_rays_d_cam = \
            clopping_distance_map_from_bbox(self.ddf_H, bbox_list, self.rays_d_cam, frame_mask, frame_distance_map)
        instance_idx = [self.ddf_instance_list.index(instance_id_i) for instance_id_i in instance_id]
        gt_shape_code = self.ddf.lat_vecs(torch.tensor(instance_idx, device=self.ddf.device)).detach()

        # Log random seeds.
        if rand_seed[0] == 'not_given':
            gt_S_seed = self.rs[mode].rand(batch_size, 1).astype(np.float32)
            rand_P_seed = self.rs[mode].rand(batch_size, 3).astype(np.float32)
            rand_S_seed = self.rs[mode].rand(batch_size, 1).astype(np.float32)
            randn_theta_seed = self.rs[mode].rand(batch_size).astype(np.float32)
            randn_axis_idx = self.rs[mode].choice(range(self.random_axis_num), batch_size)
        else:
            gt_S_seed = 0
            rand_P_seed = 0
            rand_S_seed = 0
            randn_theta_seed = 0
            randn_axis_idx = 0
        rand_seed = {}
        rand_seed['gt_S_seed'] = deepcopy(gt_S_seed)
        rand_seed['rand_P_seed'] = deepcopy(rand_P_seed)
        rand_seed['rand_S_seed'] = deepcopy(rand_S_seed)
        rand_seed['randn_theta_seed'] = deepcopy(randn_theta_seed)
        rand_seed['randn_axis_idx'] = deepcopy(randn_axis_idx)
        gt_S_seed = torch.from_numpy(gt_S_seed).clone().to(frame_w2c.device)
        rand_P_seed = torch.from_numpy(rand_P_seed).clone().to(frame_w2c.device)
        rand_S_seed = torch.from_numpy(rand_S_seed).clone().to(frame_w2c.device)
        randn_theta_seed = torch.from_numpy(randn_theta_seed).clone().to(frame_w2c.device)
        
        # Change gt scale.
        frame_gt_obj_scale_wrd = self.gt_scale_range * 2 * (gt_S_seed[:, None, :].expand(-1, self.itr_frame_num, -1) - 0.5) + 1.
        frame_camera_pos_wrd *= frame_gt_obj_scale_wrd
        frame_gt_obj_pos_wrd *= frame_gt_obj_scale_wrd
        frame_distance_map *= frame_gt_obj_scale_wrd[:, :, :, None]
        frame_clopped_distance_map *= frame_gt_obj_scale_wrd[:, :, :, None]

        # Get inits.
        rand_P = 2 * self.rand_P_range * (rand_P_seed - .5)
        ini_obj_pos = frame_gt_obj_pos_wrd[:, 0, :] + rand_P
        rand_S = 2 * self.rand_S_range * (rand_S_seed - .5) + 1.
        ini_obj_scale = frame_gt_obj_scale_wrd[:, 0, :] * rand_S

        randn_axis = self.random_axis_list[randn_axis_idx].to(frame_w2c.device)
        randn_axis = torch.sum(randn_axis[:, None, :]*frame_gt_o2w[:, 0, :, :], dim=-1)
        randn_theta = self.rand_R_range * 2 * (randn_theta_seed - .5)
        cos_t = torch.cos(randn_theta)
        sin_t = torch.sin(randn_theta)
        n_x, n_y, n_z = randn_axis.permute(1, 0)
        rand_R = torch.stack([torch.stack([cos_t+n_x*n_x*(1-cos_t), n_x*n_y*(1-cos_t)-n_z*sin_t, n_z*n_x*(1-cos_t)+n_y*sin_t], dim=-1), 
                              torch.stack([n_x*n_y*(1-cos_t)+n_z*sin_t, cos_t+n_y*n_y*(1-cos_t), n_y*n_z*(1-cos_t)-n_x*sin_t], dim=-1), 
                              torch.stack([n_z*n_x*(1-cos_t)-n_y*sin_t, n_y*n_z*(1-cos_t)+n_x*sin_t, cos_t+n_z*n_z*(1-cos_t)], dim=-1)], dim=1)
        ini_obj_axis_green_wrd = torch.sum(frame_gt_obj_green_wrd[:, :1, :]*rand_R, dim=-1)
        ini_obj_axis_red_wrd = torch.sum(frame_gt_obj_red_wrd[:, :1, :]*rand_R, dim=-1)

        ini_shape_code = self.rand_Z_center.unsqueeze(0).expand(batch_size, -1).to(gt_shape_code)

        # Retern.
        if mode=='train':
            return batch_size, frame_clopped_mask, frame_clopped_distance_map, frame_clopped_rays_d_cam, gt_shape_code, frame_w2c, \
                   frame_gt_obj_green_wrd, frame_gt_obj_red_wrd, frame_camera_pos_wrd, frame_gt_obj_pos_wrd, frame_gt_obj_scale_wrd, \
                   ini_obj_pos, ini_obj_scale, ini_obj_axis_red_wrd, ini_obj_axis_green_wrd,ini_shape_code
        elif mode in {'val', 'tes'}:
            return batch_size, frame_clopped_mask, frame_clopped_distance_map, frame_clopped_rays_d_cam, \
                   gt_shape_code, frame_w2c, frame_gt_obj_green_wrd, frame_gt_obj_red_wrd, frame_camera_pos_wrd, \
                   frame_gt_obj_pos_wrd, frame_gt_obj_scale_wrd, canonical_distance_map, canonical_camera_pos, \
                   canonical_camera_rot, instance_id, path, ini_obj_pos, ini_obj_scale, ini_obj_axis_red_wrd, \
                   ini_obj_axis_green_wrd,ini_shape_code, rand_seed, frame_gt_o2w



    def forward(self, batch_size, current_frame_num, clopped_mask, clopped_distance_map, past_itr_log, pre_obj_pos_wrd, pre_obj_axis_green_wrd, pre_obj_axis_red_wrd, pre_obj_scale_wrd, pre_shape_code, w2c, cam_pos_wrd, rays_d_cam, batch_idx, optim_idx, model_mode):

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

            # Log current maps.
            past_itr_log['itr_length'].append(current_frame_num)
            past_itr_log['obs_d'].append(clopped_distance_map.reshape(batch_size, current_frame_num, self.ddf_H, self.ddf_H).clone())
            past_itr_log['obs_mask'].append(clopped_mask.reshape(batch_size, current_frame_num, self.ddf_H, self.ddf_H).clone())
            past_itr_log['est_d'].append(est_clopped_distance_map.reshape(batch_size, current_frame_num, self.ddf_H, self.ddf_H).clone())
            past_itr_log['est_mask'].append(est_clopped_mask.reshape(batch_size, current_frame_num, self.ddf_H, self.ddf_H).clone())
            past_itr_log['w2c'].append(w2c.reshape(batch_size, current_frame_num, 3, 3).clone())
            past_itr_log['cam_pos_wrd'].append(cam_pos_wrd.reshape(batch_size, current_frame_num, 3).clone())
            past_itr_log['rays_d_cam'].append(rays_d_cam.reshape(batch_size, current_frame_num, self.ddf_H, self.ddf_H, 3).clone())

            # Get past maps.
            total_view_num = sum(past_itr_log['itr_length'])
            acc_obs_distmap = torch.cat(past_itr_log['obs_d'], dim=1).reshape(-1, self.ddf_H, self.ddf_H)
            acc_est_distmap = torch.cat(past_itr_log['est_d'], dim=1).reshape(-1, self.ddf_H, self.ddf_H)
            acc_obs_mask = torch.cat(past_itr_log['obs_mask'], dim=1).reshape(-1, self.ddf_H, self.ddf_H)
            acc_est_mask = torch.cat(past_itr_log['est_mask'], dim=1).reshape(-1, self.ddf_H, self.ddf_H)
            acc_w2c = torch.cat(past_itr_log['w2c'], dim=1).reshape(-1, 3, 3)
            acc_cam_pos_wrd = torch.cat(past_itr_log['cam_pos_wrd'], dim=1).reshape(-1, 3)
            acc_rays_d_cam = torch.cat(past_itr_log['rays_d_cam'], dim=1).reshape(-1, self.ddf_H, self.ddf_H, 3)
            acc_latest_o2w = pre_o2w[:, None, :, :].expand(-1, total_view_num, -1, -1).reshape(-1, 3, 3)
            acc_latest_obj_pos_wrd = pre_obj_pos_wrd[:, None, :].expand(-1, total_view_num, -1).reshape(-1, 3)
            acc_latest_obj_scale_wrd = pre_obj_scale_wrd[:, None, :].expand(-1, total_view_num, -1).reshape(-1, 1)
            
            # Get map.
            obs_OSMap_obj = get_OSMap_obj(acc_obs_distmap, acc_obs_mask, acc_rays_d_cam, acc_w2c, acc_cam_pos_wrd, acc_latest_o2w, acc_latest_obj_pos_wrd, acc_latest_obj_scale_wrd)
            est_OSMap_obj = get_OSMap_obj(acc_est_distmap, acc_est_mask, acc_rays_d_cam, acc_w2c, acc_cam_pos_wrd, acc_latest_o2w, acc_latest_obj_pos_wrd, acc_latest_obj_scale_wrd)
            dif_OSMap_obj = get_diffOSMap_obj(acc_obs_distmap, acc_est_distmap, acc_obs_mask, acc_est_mask, acc_rays_d_cam, acc_w2c, acc_latest_o2w, acc_latest_obj_scale_wrd)
            if len(past_itr_log['itr_length']) > 1:
                # Expanding current est to past est shapes.
                past_view_num = total_view_num - past_itr_log['itr_length'][-1]
                repped_latest_distmap = torch.cat([past_itr_log['est_d'][-1][:, :view_num] for view_num in past_itr_log['itr_length'][:-1]], dim=1).reshape(-1, self.ddf_H, self.ddf_H)
                repped_latest_mask = torch.cat([past_itr_log['est_mask'][-1][:, :view_num] for view_num in past_itr_log['itr_length'][:-1]], dim=1).reshape(-1, self.ddf_H, self.ddf_H)
                repped_latest_o2w = pre_o2w[:, None, :, :].expand(-1, past_view_num, -1, -1).reshape(-1, 3, 3)
                repped_latest_obj_scale_wrd = pre_obj_scale_wrd[:, None, :].expand(-1, past_view_num, -1).reshape(-1, 1)
                # Expanding current est to past est shapes.
                past_est_distmap = acc_est_distmap.reshape(batch_size, total_view_num, self.ddf_H, self.ddf_H)[:, :past_view_num].reshape(-1, self.ddf_H, self.ddf_H)
                past_est_mask = acc_est_mask.reshape(batch_size, total_view_num, self.ddf_H, self.ddf_H)[:, :past_view_num].reshape(-1, self.ddf_H, self.ddf_H)
                past_rays_d_cam = acc_rays_d_cam.reshape(batch_size, total_view_num, self.ddf_H, self.ddf_H, 3)[:, :past_view_num].reshape(-1, self.ddf_H, self.ddf_H, 3)
                past_w2c = acc_w2c.reshape(batch_size, total_view_num, 3, 3)[:, :past_view_num].reshape(-1, 3, 3)
                # Make transition map.
                tra_OSMap_obj = get_diffOSMap_obj(past_est_distmap, repped_latest_distmap, past_est_mask, repped_latest_mask, past_rays_d_cam, past_w2c, repped_latest_o2w, repped_latest_obj_scale_wrd)
            else:
                # Make dummy map.
                past_view_num = 0
                tra_OSMap_obj = torch.tensor([])[:, None, None, None].expand(-1, self.ddf_H, self.ddf_H, 4).to(obs_OSMap_obj)
            inp_maps = torch.cat([obs_OSMap_obj, est_OSMap_obj, dif_OSMap_obj, tra_OSMap_obj], dim=0).permute(0, 3, 1, 2).contiguous().detach()

        # Get embeddings.
        inp_embed = self.backbone_encoder(inp_maps)
        obs_embed = inp_embed[:1*batch_size*total_view_num].reshape(batch_size, total_view_num, self.backbone_embedding_dim)
        est_embed = inp_embed[1*batch_size*total_view_num:2*batch_size*total_view_num].reshape(batch_size, total_view_num, self.backbone_embedding_dim)
        dif_embed = inp_embed[2*batch_size*total_view_num:3*batch_size*total_view_num].reshape(batch_size, total_view_num, self.backbone_embedding_dim)
        tra_embed = inp_embed[3*batch_size*total_view_num:].reshape(batch_size, past_view_num, self.backbone_embedding_dim)

        # Get update value.
        est_obj_pos_wrd, est_obj_axis_green_wrd, est_obj_axis_red_wrd, est_obj_scale, est_shape_code = \
        self.df_net(obs_embed, est_embed, dif_embed, tra_embed, past_itr_log['itr_length'], optim_idx, inp_pre_obj_pos_wrd, inp_pre_obj_axis_green_wrd, inp_pre_obj_axis_red_wrd, inp_pre_o2w, \
                    inp_pre_obj_scale_wrd, inp_pre_shape_code, pre_obj_pos_wrd, pre_obj_axis_green_wrd, pre_obj_axis_red_wrd, pre_o2w, pre_obj_scale_wrd, pre_shape_code) #, self.backbone_decoder)

        ##################################################
        # if optim_idx > 1:
        #     for sample_id in range(batch_size):
        #         # raw_1 = obs_OSMap_obj.reshape(batch_size, total_view_num, self.ddf_H, self.ddf_H, 4)[:, -current_frame_num:]
        #         # raw_2 = est_OSMap_obj.reshape(batch_size, total_view_num, self.ddf_H, self.ddf_H, 4)[:, -current_frame_num:]
        #         raw_1 = obs_OSMap_obj.reshape(batch_size, total_view_num, self.ddf_H, self.ddf_H, 4)[:, -sum(past_itr_log['itr_length'][-2:]):-current_frame_num]
        #         raw_2 = est_OSMap_obj.reshape(batch_size, total_view_num, self.ddf_H, self.ddf_H, 4)[:, -current_frame_num:]
        #         raw_3 = est_OSMap_obj.reshape(batch_size, total_view_num, self.ddf_H, self.ddf_H, 4)[:, -sum(past_itr_log['itr_length'][-2:]):-current_frame_num]
        #         map_1 = raw_1[..., :-1]
        #         mask_1 = raw_1[..., -1]
        #         map_2 = raw_2[..., :-1]
        #         mask_2 = raw_2[..., -1]
        #         map_3 = raw_3[..., :-1]
        #         mask_3 = raw_3[..., -1]
        #         fig = plt.figure()
        #         ax = Axes3D(fig)
        #         c_list_1 = ['b', 'g', 'r', 'm', 'y']
        #         c_list_2 = ['b', 'g', 'r', 'm', 'y']
        #         for ind_1 in range(current_frame_num):
        #             c_1 = c_list_1[ind_1]
        #             c_2 = c_list_2[ind_1]
        #             if past_itr_log['itr_length'][-2] > ind_1:
        #                 point_1 = map_1[sample_id, ind_1][mask_1[sample_id, ind_1] > 0]
        #                 point_1 = point_1.to('cpu').detach().numpy().copy()
        #                 ax.scatter(point_1[:, 0], point_1[:, 1], point_1[:, 2], marker="o", linestyle='None', c='c', s=0.05)
        #             point_2 = map_2[sample_id, ind_1][mask_2[sample_id, ind_1] > 0]
        #             point_2 = point_2.to('cpu').detach().numpy().copy()
        #             ax.scatter(point_2[::2, 0], point_2[::2, 1], point_2[::2, 2], marker="o", linestyle='None', c='m', s=0.05)
        #             if past_itr_log['itr_length'][-2] > ind_1:
        #                 point_3 = map_3[sample_id, ind_1][mask_3[sample_id, ind_1] > 0]
        #                 point_3 = point_3.to('cpu').detach().numpy().copy()
        #                 ax.scatter(point_3[::2, 0], point_3[::2, 1], point_3[::2, 2], marker="o", linestyle='None', c='g', s=0.05)
        #         ax.view_init(elev=0, azim=90)
        #         fig.savefig(f"{sample_id}_00_90.png")
        #         ax.view_init(elev=0, azim=0)
        #         fig.savefig(f"{sample_id}_00_00.png")
        #         ax.view_init(elev=45, azim=45)
        #         fig.savefig(f"{sample_id}_45_45.png")
        #         plt.close()
        # ##################################################
        # if model_mode=='train' and optim_idx==0:
        #     gt = clopped_distance_map
        #     est = est_clopped_distance_map
        #     str_batch_idx = str(batch_idx).zfill(5)
        #     str_optim_idx = str(optim_idx).zfill(2)
        #     check_map = torch.cat([gt, est, torch.abs(gt-est)], dim=-1)
        #     check_map = torch.cat([map_i for map_i in check_map], dim=0)
        #     check_map_torch(check_map, f'tes_{str_batch_idx}_{str_optim_idx}.png')
        #     # import pdb; pdb.set_trace()
        # check_map_torch(obs_OSMap_obj.reshape(batch_size, -1, self.ddf_H, self.ddf_H, 4)[0, :, :, :, -1].reshape(-1, self.ddf_H), 'tes_obs.png')
        # check_map_torch(est_OSMap_obj.reshape(batch_size, -1, self.ddf_H, self.ddf_H, 4)[0, :, :, :, -1].reshape(-1, self.ddf_H), 'tes_est.png')
        # check_map_torch(dif_OSMap_obj.reshape(batch_size, -1, self.ddf_H, self.ddf_H, 4)[0, :, :, :, -1].reshape(-1, self.ddf_H), 'tes_dif.png')
        # check_map_torch(tra_OSMap_obj.reshape(batch_size, -1, self.ddf_H, self.ddf_H, 4)[0, :, :, :, -1].reshape(-1, self.ddf_H), 'tes_tra.png')
        # self.df_net.main_layers.encoder.layers[0].attn_output_weights
        # ##################################################
        # batch_i = 0
        # str_batch_i = str(batch_i).zfill(5)
        # gt = gt.reshape(batch_size, current_frame_num, self.ddf_H, self.ddf_H)[batch_i]
        # est = est.reshape(batch_size, current_frame_num, self.ddf_H, self.ddf_H)[batch_i]
        # for F_i in range(current_frame_num):
        #     pickle_dump(torch.stack([gt[F_i], est[F_i]], dim=0).to('cpu').detach().numpy().copy(), 
        #                 f'paper_fig/depth_map/raw_gt_est/{str_batch_i}_{F_i}_{str_optim_idx}.pickle')
        # pickle_dump(est_shape_code.to('cpu').detach().numpy().copy(), f'paper_fig/normal_map/latvec/{str_batch_i}_{str_optim_idx}.pickle')
        # # ここでポーズGT・ESTも保存し、レンダリングに役立てる。
        # ##################################################
        
        return est_obj_pos_wrd, est_obj_axis_green_wrd, est_obj_axis_red_wrd, est_obj_scale, est_shape_code



    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            # Get batch info.
            batch_size, frame_clopped_mask, frame_clopped_distance_map, frame_clopped_rays_d_cam, gt_shape_code, frame_w2c, \
            frame_gt_obj_green_wrd, frame_gt_obj_red_wrd, frame_camera_pos_wrd, frame_gt_obj_pos_wrd, frame_gt_obj_scale_wrd, \
            ini_obj_pos, ini_obj_scale, ini_obj_axis_red_wrd, ini_obj_axis_green_wrd, ini_shape_code = self.preprocess(batch, mode='train')

            # Set frames
            frame_idx_list = list(range(self.itr_frame_num))
            current_frame_num = len(frame_idx_list)

            # Get current maps and camera inf.
            clopped_mask = frame_clopped_mask[:, frame_idx_list].reshape(-1, self.ddf_H, self.ddf_H).detach()
            clopped_distance_map = frame_clopped_distance_map[:, frame_idx_list].reshape(-1, self.ddf_H, self.ddf_H).detach()
            w2c = frame_w2c[:, frame_idx_list].reshape(-1, 3, 3).detach()
            cam_pos_wrd = frame_camera_pos_wrd[:, frame_idx_list].reshape(-1, 3).detach()
            rays_d_cam = frame_clopped_rays_d_cam[:, frame_idx_list].reshape(-1, self.ddf_H, self.ddf_H, 3).detach()


        ###################################
        #####     Start training      #####
        ###################################
        # Initialization.
        pre_obj_pos_wrd = ini_obj_pos.detach()
        pre_obj_scale_wrd = ini_obj_scale.detach()
        pre_obj_axis_green_wrd = ini_obj_axis_green_wrd.detach()
        pre_obj_axis_red_wrd = ini_obj_axis_red_wrd.detach()
        pre_shape_code = ini_shape_code.detach()
        # pre_obj_pos_wrd = frame_gt_obj_pos_wrd[:, 0].clone().detach()
        # pre_obj_scale_wrd = frame_gt_obj_scale_wrd[:, 0].clone().detach()
        # pre_obj_axis_red_wrd = frame_gt_obj_red_wrd[:, 0].clone().detach()
        # pre_obj_axis_green_wrd = frame_gt_obj_green_wrd[:, 0].clone().detach()
        # pre_shape_code = gt_shape_code.clone().detach()

        past_itr_log = {'itr_length': [], 'obs_d': [], 'est_d': [], 'obs_mask': [], 'est_mask': [], 'w2c': [], 'cam_pos_wrd': [], 'rays_d_cam': []}

        for optim_idx in range(self.optim_num):
            # Set optimizers
            opt = self.optimizers()
            opt.zero_grad()

            # Update.
            est_obj_pos_wrd, est_obj_axis_green_wrd, est_obj_axis_red_wrd, est_obj_scale, est_shape_code = \
                self.forward(batch_size, current_frame_num, clopped_mask, clopped_distance_map, past_itr_log, pre_obj_pos_wrd, pre_obj_axis_green_wrd, 
                pre_obj_axis_red_wrd, pre_obj_scale_wrd, pre_shape_code, w2c, cam_pos_wrd, rays_d_cam, batch_idx, optim_idx, 'train')

            # Cal loss against integrated estimations.
            loss_pos = self.mseloss(est_obj_pos_wrd, frame_gt_obj_pos_wrd[:, -1].detach())
            loss_axis_green = torch.mean(-self.cossim(est_obj_axis_green_wrd, frame_gt_obj_green_wrd[:, -1].detach()) + 1.)
            loss_axis_red = torch.mean(-self.cossim(est_obj_axis_red_wrd, frame_gt_obj_red_wrd[:, -1].detach()) + 1.)
            loss_scale = self.mseloss(est_obj_scale, frame_gt_obj_scale_wrd[:, -1].detach())
            loss_shape_code = self.mseloss(est_shape_code, gt_shape_code.detach())

            # Cal depth loss.
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


        # Return loss.
        return {'pos':   loss_pos.detach(), 
                'green':   loss_axis_green.detach(), 
                'red':   loss_axis_red.detach(), 
                'scale': loss_scale.detach(), 
                'shape': loss_shape_code.detach(), 
                'depth': loss_depth.detach(), 
                'total': loss.detach(), }



    def training_epoch_end(self, outputs):
        # Log loss.
        for key_i in ['pos', 'green', 'red', 'scale', 'shape', 'total']:
            avg_loss = torch.stack([x[key_i] for x in outputs]).mean()
            current_epoch = torch.tensor(self.current_epoch + 1., dtype=avg_loss.dtype)
            self.log_dict({f'train/loss_{key_i}': avg_loss, "step": current_epoch})
        # Save ckpt.
        if (self.current_epoch + 1) % self.save_interval == 0:
            ckpt_name = str(self.current_epoch + 1).zfill(10) + '.ckpt'
            ckpt_path = os.path.join(self.trainer.log_dir, 'checkpoints', ckpt_name)
            trainer.save_checkpoint(ckpt_path)



    def test_step(self, batch, batch_idx, step_mode='tes'):
        batch_size, frame_clopped_mask, frame_clopped_distance_map, frame_clopped_rays_d_cam, \
        gt_shape_code, frame_w2c, frame_gt_obj_green_wrd, frame_gt_obj_red_wrd, frame_camera_pos_wrd, \
        frame_gt_obj_pos_wrd, frame_gt_obj_scale_wrd, canonical_distance_map, canonical_camera_pos, \
        canonical_camera_rot, instance_id, path, ini_obj_pos, ini_obj_scale, ini_obj_axis_red_wrd, \
        ini_obj_axis_green_wrd,ini_shape_code, rand_seed, frame_gt_o2w = self.preprocess(batch, step_mode)


        ###################################
        #####     Start inference.    #####
        ###################################
        past_itr_log = {'itr_length': [], 'obs_d': [], 'est_d': [], 'obs_mask': [], 'est_mask': [], 'w2c': [], 'cam_pos_wrd': [], 'rays_d_cam': []}
        itr_err_list = {'pos':[], 'scale':[], 'red':[], 'green':[], 'shape':[], 'sec':[]}
        loss_list = {'total':[], 'pos':[], 'scale':[], 'red':[], 'green':[], 'shape':[]}
        
        for optim_idx in range(-1, self.optim_num):
            # Get frame index info.
            if self.optim_mode == 'whole':
                frame_idx_list = list(range(self.itr_frame_num))
            elif self.optim_mode == 'sequential':
                if optim_idx < 0:
                    frame_idx_list = [0]
                else:
                    frame_idx_list = list(range(0, optim_idx+1)) # list(range(0, optim_idx+2))
            current_frame_num = len(frame_idx_list)

            # Get current maps and camera infos.
            clopped_mask = frame_clopped_mask[:, frame_idx_list].reshape(-1, self.ddf_H, self.ddf_H).detach()
            clopped_distance_map = frame_clopped_distance_map[:, frame_idx_list].reshape(-1, self.ddf_H, self.ddf_H).detach()
            rays_d_cam = frame_clopped_rays_d_cam[:, frame_idx_list].reshape(-1, self.ddf_H, self.ddf_H, 3).detach()
            w2c = frame_w2c[:, frame_idx_list].reshape(-1, 3, 3).detach()
            cam_pos_wrd = frame_camera_pos_wrd[:, frame_idx_list].reshape(-1, 3).detach()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            ###################################
            ###   Perform initialization.   ###
            ###################################
            if optim_idx == -1:
                # Get init values.
                est_obj_pos_wrd = ini_obj_pos.detach()
                est_obj_scale = ini_obj_scale.detach()
                est_obj_axis_green_wrd = ini_obj_axis_green_wrd.detach()
                est_obj_axis_red_wrd = ini_obj_axis_red_wrd.detach()
                est_shape_code = ini_shape_code.detach()

            ###################################
            #####      Perform dfnet.     #####
            ###################################
            else:
                # Get update values.
                est_obj_pos_wrd, est_obj_axis_green_wrd, est_obj_axis_red_wrd, est_obj_scale, est_shape_code = \
                    self.forward(batch_size, current_frame_num, clopped_mask, clopped_distance_map, past_itr_log, pre_obj_pos_wrd, pre_obj_axis_green_wrd, 
                    pre_obj_axis_red_wrd, pre_obj_scale_wrd, pre_shape_code, w2c, cam_pos_wrd, rays_d_cam, batch_idx, optim_idx, 'val')

            # Save pre estimation.
            pre_obj_pos_wrd = est_obj_pos_wrd.clone().detach()
            pre_obj_axis_green_wrd = est_obj_axis_green_wrd.clone().detach()
            pre_obj_axis_red_wrd = est_obj_axis_red_wrd.clone().detach()
            pre_obj_scale_wrd = est_obj_scale.clone().detach()
            pre_shape_code = est_shape_code.clone().detach()
            # pre_obj_pos_wrd = frame_gt_obj_pos_wrd[:, 0].clone().detach() + \
            #     (1. - ((optim_idx+1) / self.optim_num)) * torch.sum((torch.tensor([0.3, 0, 0]).to(ini_obj_pos))[None, None, :].expand(batch_size, -1, -1)*frame_gt_o2w[:, 0], dim=-1)
            # # pre_obj_pos_wrd = frame_gt_obj_pos_wrd[:, 0].clone().detach()
            # pre_obj_axis_green_wrd = frame_gt_obj_green_wrd[:, 0].clone().detach()
            # pre_obj_axis_red_wrd = frame_gt_obj_red_wrd[:, 0].clone().detach()
            # pre_obj_scale_wrd = frame_gt_obj_scale_wrd[:, 0].clone().detach()
            # pre_shape_code = gt_shape_code.clone().detach()
            end.record()
            torch.cuda.synchronize()
            itr_err_list['sec'].append(start.elapsed_time(end) / 1000)

            # Cal err at each iteration.
            if step_mode=='tes' or optim_idx==(self.optim_num-1):
                itr_err_list['pos'].append(torch.abs(pre_obj_pos_wrd - frame_gt_obj_pos_wrd[:, 0]).mean(dim=-1))
                err_axis_green_cos_sim_i = self.cossim(pre_obj_axis_green_wrd, frame_gt_obj_green_wrd[:, 0]).clamp(min=self.cosssim_min, max=self.cosssim_max)
                itr_err_list['green'].append(torch.acos(err_axis_green_cos_sim_i) * 180 / torch.pi)
                err_axis_red_cos_sim_i = self.cossim(pre_obj_axis_red_wrd, frame_gt_obj_red_wrd[:, 0]).clamp(min=self.cosssim_min, max=self.cosssim_max)
                itr_err_list['red'].append(torch.acos(err_axis_red_cos_sim_i) * 180 / torch.pi)
                itr_err_list['scale'].append(100 * torch.abs(pre_obj_scale_wrd - frame_gt_obj_scale_wrd[:, 0]) / frame_gt_obj_scale_wrd[:, 0])
                depth_error = []
                for gt_distance_map, cam_pos_wrd, w2c in zip(canonical_distance_map.permute(1, 0, 2, 3), 
                                                            canonical_camera_pos.permute(1, 0, 2), 
                                                            canonical_camera_rot.permute(1, 0, 2, 3)):
                    rays_d_cam = self.canonical_rays_d_cam.expand(batch_size, -1, -1, -1)
                    _, est_distance_map = get_canonical_map(
                                                H = self.ddf_H, 
                                                cam_pos_wrd = cam_pos_wrd, 
                                                rays_d_cam = rays_d_cam, 
                                                w2c = w2c, 
                                                input_lat_vec = pre_shape_code, 
                                                ddf = self.ddf, )
                    depth_error.append(torch.abs(gt_distance_map-est_distance_map).mean(dim=-1).mean(dim=-1))
                    # check_map_torch(torch.abs(gt_distance_map-est_distance_map).reshape(-1, 128), 'tes.png')

                itr_err_list['shape'].append(torch.stack(depth_error, dim=-1).mean(dim=-1))


        # Cal loss to monitor performance.
        loss_list['pos'] = self.mseloss(est_obj_pos_wrd, frame_gt_obj_pos_wrd[:, 0]).detach()
        loss_list['green'] = torch.mean(-self.cossim(est_obj_axis_green_wrd, frame_gt_obj_green_wrd[:, 0]) + 1.).detach()
        loss_list['red'] = torch.mean(-self.cossim(est_obj_axis_red_wrd, frame_gt_obj_red_wrd[:, 0]) + 1.).detach()
        loss_list['scale'] = self.mseloss(est_obj_scale, frame_gt_obj_scale_wrd[:, 0]).detach()
        loss_list['shape'] = self.mseloss(est_shape_code, gt_shape_code).detach()
        loss_list['total'] = (self.L_p * loss_list['pos'] + self.L_a * loss_list['green'] + self.L_a * loss_list['red'] + \
                              self.L_s * loss_list['scale'] + self.L_c * loss_list['shape']).detach()


        # Return.
        return {'loss_list': loss_list, 
                'itr_err_list': itr_err_list, 
                'path': np.array(path), 
                'rand_P_seed': rand_seed['rand_P_seed'], 
                'rand_S_seed': rand_seed['rand_S_seed'], 
                'randn_theta_seed': rand_seed['randn_theta_seed'], 
                'randn_axis_idx': rand_seed['randn_axis_idx'], }



    def test_epoch_end(self, outputs):
        log_dict = {}
        sec_start, avg_start, med_start = -1, 1, 8
        log_txt = ['???_avg_log = {', 
                   '    \'pos\'   : np.array([', # ]), 
                   '    \'green\' : np.array([', # ]), 
                   '    \'red\'   : np.array([', # ]), 
                   '    \'scale\' : np.array([', # ]), 
                   '    \'shape\' : np.array([', # ]), 
                   '    }', 
                   '???_med_log = {', 
                   '    \'pos\'   : np.array([', # ]), 
                   '    \'green\' : np.array([', # ]), 
                   '    \'red\'   : np.array([', # ]), 
                   '    \'scale\' : np.array([', # ]), 
                   '    \'shape\' : np.array([', # ]), 
                   '    }', 
                   '???_sec_log = np.array([', # ])
                   ]

        itr_sec = np.stack([x['itr_err_list']['sec'] for x in outputs], axis=0)
        itr_sec_txt = (', ').join([str(n) for n in itr_sec[:-1].mean(0).tolist()])
        log_txt[sec_start] += itr_sec_txt + '])'
        for idx, key_i in enumerate(['pos', 'green', 'red', 'scale', 'shape']):
            itr_err = torch.cat([torch.stack(x['itr_err_list'][key_i], dim=-1) for x in outputs], dim=0)
            itr_err = itr_err.to('cpu').detach().numpy().copy()
            itr_err_med = np.median(itr_err, axis=0).tolist()
            itr_err_avg = np.mean(itr_err, axis=0).tolist()
            log_dict[key_i] = itr_err[:, -1]
            log_txt[avg_start+idx] += (', ').join([str(n) for n in itr_err_avg]) + ']), '
            log_txt[med_start+idx] += (', ').join([str(n) for n in itr_err_med]) + ']), '
            with open(self.test_log_path, 'a') as file:
                file.write(f'{key_i}, {str(itr_err_avg[-1])}, {str(itr_err_med[-1])}\n')


        # Check logs.
        with open(self.test_log_path, 'a') as file:
            for txt in log_txt:
                file.write(txt + '\n')
        log_dict['path'] = np.concatenate([x['path'] for x in outputs])
        log_dict['rand_P_seed'] = np.concatenate([x['rand_P_seed'] for x in outputs])
        log_dict['rand_S_seed'] = np.concatenate([x['rand_S_seed'] for x in outputs])
        log_dict['randn_theta_seed'] = np.concatenate([x['randn_theta_seed'] for x in outputs])
        log_dict['randn_axis_idx'] = np.concatenate([x['randn_axis_idx'] for x in outputs])
        pickle_dump(log_dict, self.test_log_path.split('.txt')[0] + '_error.pickle')



    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx, step_mode='val')



    def validation_epoch_end(self, outputs):
        for key_i in {'pos', 'green', 'red', 'scale', 'shape', 'total'}:
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



    def on_train_epoch_start(self):
        self.ddf.eval()
        self.rays_d_cam = self.rays_d_cam.to(self.df_net.device)
        np.random.seed(0 + self.current_epoch)
        # self.backbone_encoder.eval()
        # self.backbone_decoder.eval()



    def on_validation_epoch_start(self):
        self.rays_d_cam = self.rays_d_cam.to(self.df_net.device)
        self.canonical_rays_d_cam = self.canonical_rays_d_cam.to(self.df_net.device)



    def on_test_epoch_start(self):
        self.rays_d_cam = self.rays_d_cam.to(self.df_net.device)
        self.canonical_rays_d_cam = self.canonical_rays_d_cam.to(self.df_net.device)



    def configure_optimizers(self):
        # optimizer = torch.optim.Adam([
        #     {"params": self.df_net.parameters()},
        # ], lr=self.lr, betas=(0.9, 0.999),)
        # return optimizer

        print('#####   SCRACH   #####')
        optimizer = torch.optim.Adam(
            list(self.df_net.parameters()) + list(self.backbone_encoder.parameters()), 
        lr=self.lr, betas=(0.9, 0.999),)
        return optimizer



if __name__=='__main__':
    # Get args
    args = get_args()
    args.gpu_num = torch.cuda.device_count() # log used gpu num.
    args.check_val_every_n_epoch = args.save_interval
    ckpt_base_dir = 'lightning_logs/DeepTaR/chair/'
    val_model_name = args.expname.split('/')[-1] + args.exp_version
    if args.model_ckpt_path=='non':
        args.model_ckpt_path = f'{ckpt_base_dir}/{val_model_name}/checkpoints/{str(args.val_model_epoch).zfill(10)}.ckpt'
    if args.initnet_ckpt_path=='non':
        args.initnet_ckpt_path = f'{ckpt_base_dir}/{args.init_net_name}/checkpoints/{str(args.init_net_epoch).zfill(10)}.ckpt'


    if args.code_mode == 'TRAIN':
        # Set trainer.
        logger = pl.loggers.TensorBoardLogger(save_dir=os.getcwd(), version=f'{args.expname}{args.exp_version}',name='lightning_logs')
        trainer = pl.Trainer(
            gpus=args.gpu_num, 
            strategy=DDPPlugin(find_unused_parameters=True), #=False), 
            logger=logger,
            max_epochs=args.N_epoch, 
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            )

        # Save config files.
        os.makedirs(os.path.join('lightning_logs', f'{args.expname}{args.exp_version}'), exist_ok=True)
        f = os.path.join('lightning_logs', f'{args.expname}{args.exp_version}', 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        if args.config is not None:
            f = os.path.join('lightning_logs', f'{args.expname}{args.exp_version}', 'config.txt')
            with open(f, 'w') as file:
                file.write(open(args.config, 'r').read())

        # Create dataloader
        print('start dataloader.ini !')
        def seed_worker(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        train_generator = torch.Generator().manual_seed(7)
        train_dataset = TaR_dataset(args, 'train', args.train_instance_list_txt, args.train_data_dir, False)
        train_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.N_batch, num_workers=8, shuffle=True, worker_init_fn=seed_worker, generator=train_generator)
        
        val_generator = torch.Generator().manual_seed(8)
        val_dataset = TaR_dataset(args, 'val', args.val_instance_list_txt, args.val_data_dir, args.val_N_scenes)
        val_dataloader = data_utils.DataLoader(val_dataset, batch_size=args.N_batch, num_workers=8, shuffle=False, generator=val_generator)

        # Set ddf.
        ddf = DDF(args)
        ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args)
        ddf.eval()

        # Set back bone.
        backbone = backbone_encoder_decoder(args)
        if args.backbone_training_strategy in {'finetune', 'fixed'}:
            backbone = backbone.load_from_checkpoint(checkpoint_path=args.backbone_model_path, args=args)

        # Set model.
        ckpt_path = None
        model = original_optimizer(args, {'ddf':ddf, 'backbone_encoder':backbone.encoder_2dcnn, 'backbone_decoder':backbone.decoder_2dcnn})
        ##############################
        # backbone_model_path = 'BackBone/lightning_logs/03001627/randn/list0/all_wNorm/checkpoints/0000000100.ckpt'
        # backbone = backbone_encoder_decoder(args)
        # backbone = backbone.load_from_checkpoint(checkpoint_path=backbone_model_path, args=args)
        # model.backbone_encoder = backbone.encoder_2dcnn
        # model.backbone_decoder = backbone.decoder_2dcnn
        ##############################

        # Start training.
        model.use_init_net = False
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, datamodule=None, ckpt_path=ckpt_path)


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
        
        # Set back bone.
        backbone = backbone_encoder_decoder(args)

        # Set model.
        ckpt_path = None
        model = original_optimizer(args, {'ddf':ddf, 'backbone_encoder':backbone.encoder_2dcnn, 'backbone_decoder':backbone.decoder_2dcnn})
        model = model.load_from_checkpoint(
            checkpoint_path=args.model_ckpt_path, 
            args=args, 
            pretrain_models={'ddf':ddf, 'backbone_encoder':backbone.encoder_2dcnn, 'backbone_decoder':backbone.decoder_2dcnn}
            )
        ##############################
        # backbone = backbone.load_from_checkpoint(checkpoint_path=args.backbone_model_path, args=args)
        # model.backbone_encoder = backbone.encoder_2dcnn
        # model.backbone_decoder = backbone.decoder_2dcnn
        ##############################

        # Setting model.
        model.use_init_net = False
        model.start_frame_idx = 0
        model.half_lambda_max = 0
        model.only_init_net = False
        model.use_deep_optimizer = True
        model.use_adam_optimizer = not(model.use_deep_optimizer)
        model.optim_num = args.optim_num

        # # Save logs.
        # import datetime
        # dt_now = datetime.datetime.now()
        # time_log = dt_now.strftime('%Y_%m_%d_%H_%M_%S')

        # time_log += '_' + args.model_ckpt_path.split('/')[-3] + '_epo' + args.model_ckpt_path.split('/')[-1].split('.')[0]
        # os.mkdir('./txt/experiments/log/' + time_log)
        # file_name = './txt/experiments/log/' + time_log + '/log.txt'
        # model.test_log_path = file_name
        # with open(file_name, 'a') as file:
        #     file.write('time_log : ' + time_log + '\n')
        #     file.write('ckpt_path : ' + args.model_ckpt_path + '\n')
        #     if model.use_init_net:
        #         file.write('ini_ckpt_path : ' + args.initnet_ckpt_path + '\n')
        #     else:
        #         file.write('ini_ckpt_path : ' + 'non' + '\n')
        #         file.write('rand_P_range : ' + str(model.rand_P_range) + '\n')
        #         file.write('rand_S_range : ' + str(model.rand_S_range) + '\n')
        #         file.write('rand_R_range : ' + str(model.rand_R_range) + '\n')
        #         file.write('random_axis_num : ' + str(model.random_axis_num) + '\n')
        #         file.write('rand_Z_sigma : ' + str(model.rand_Z_sigma) + '\n')
        #     file.write('test_instance_list_txt : ' + str(args.test_instance_list_txt) + '\n')
        #     file.write('test_N_views : ' + str(args.test_N_views) + '\n')
        #     file.write('start_frame_idx : ' + str(model.start_frame_idx) + '\n')
        #     file.write('itr_frame_num : ' + str(model.itr_frame_num) + '\n')
        #     file.write('optim_num : ' + str(model.optim_num) + '\n')
        #     file.write('\n')

        # Test model.
        trainer = pl.Trainer(
            gpus=args.gpu_num, 
            strategy=DDPPlugin(find_unused_parameters=True), #=False), 
            enable_checkpointing = False,
            check_val_every_n_epoch = args.check_val_every_n_epoch,
            logger = pl.loggers.TensorBoardLogger(save_dir=os.getcwd(), version=f'val_trash', name='lightning_logs')
            )
        trainer.test(model, test_dataloader)
