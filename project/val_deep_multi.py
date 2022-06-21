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
from train_dfnet import *
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





class test_TaR(pl.LightningModule):

    def __init__(self, args, target_model):
        super().__init__()

        # Base configs
        self.fov = target_model.fov
        self.input_H = target_model.input_H
        self.input_W = target_model.input_W
        self.image_coord = target_model.image_coord
        self.ddf_H = target_model.ddf_H
        self.lr = target_model.lr
        self.rays_d_cam = target_model.rays_d_cam
        self.save_interval = target_model.save_interval
        self.model_params_dtype = target_model.model_params_dtype
        self.model_device = target_model.model_device
        self.train_optim_num = target_model.train_optim_num
        self.use_gru = target_model.use_gru
        self.use_depth_error = target_model.use_depth_error
        self.use_weighted_average = target_model.use_weighted_average
        self.start_frame_idx = target_model.start_frame_idx
        self.frame_sequence_num = target_model.frame_sequence_num
        self.half_lambda_max = target_model.half_lambda_max
        self.test_optim_num = target_model.test_optim_num
        self.test_mode = target_model.test_mode
        self.test_log_path = target_model.test_log_path
        self.only_init_net = target_model.only_init_net
        self.use_deep_optimizer = target_model.use_deep_optimizer
        self.use_adam_optimizer = target_model.use_adam_optimizer

        # Make model
        self.ddf = target_model.ddf
        self.init_net = target_model.init_net
        self.df_net = target_model.df_net

        # loss func.
        self.l1 = torch.nn.L1Loss()
        self.cossim = nn.CosineSimilarity(dim=-1)
        self.cosssim_min, self.cosssim_max = -1+1e-8, 1-1e-8



    # def training_step(self, batch, batch_idx):
    def test_step(self, batch, batch_idx):

        frame_mask, frame_distance_map, frame_camera_pos, frame_camera_rot, frame_obj_pos, frame_obj_rot, frame_obj_scale, \
            canonical_distance_map, canonical_camera_pos, canonical_camera_rot, instance_id, path = batch
        batch_size = len(instance_id)

        ###################################
        #####     Start Inference     #####
        ###################################
        # Set frame.
        start_frame_idx = self.start_frame_idx
        end_frame_idx = self.start_frame_idx + self.frame_sequence_num
        using_frame_num = self.frame_sequence_num

        # Clop distance map.
        raw_mask = frame_mask[:, start_frame_idx:end_frame_idx].reshape(-1, self.input_H, self.input_W)
        raw_distance_map = frame_distance_map[:, start_frame_idx:end_frame_idx].reshape(-1, self.input_H, self.input_W)
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
        o2w = frame_obj_rot[:, start_frame_idx:end_frame_idx].reshape(-1, 3, 3).to(torch.float)
        w2c = frame_camera_rot[:, start_frame_idx:end_frame_idx].reshape(-1, 3, 3).to(torch.float)
        o2c = torch.bmm(w2c, o2w).to(torch.float) # とりあえずこれを推論する
        gt_obj_axis_green_cam = o2c[:, :, 1] # Y
        gt_obj_axis_red_cam = o2c[:, :, 0] # X
        gt_axis_green_wrd = torch.sum(gt_obj_axis_green_cam[..., None, :]*w2c.permute(0, 2, 1), -1).reshape(batch_size, using_frame_num, 3)
        gt_axis_red_wrd = torch.sum(gt_obj_axis_red_cam[..., None, :]*w2c.permute(0, 2, 1), -1).reshape(batch_size, using_frame_num, 3)
        cam_pos_wrd = frame_camera_pos[:, start_frame_idx:end_frame_idx].reshape(-1, 3).to(torch.float)
        gt_obj_pos_wrd = frame_obj_pos[:, start_frame_idx:end_frame_idx].to(torch.float)
        gt_obj_scale = frame_obj_scale[:, start_frame_idx:end_frame_idx][:, :, None].to(torch.float)

        ###################################
        #####    Start Optim Step     #####
        ###################################
        check_optim_steps = []
        for optim_idx in range(self.test_optim_num):
            perform_init_est = optim_idx == 0

            # Estimating.
            with torch.no_grad():
                if perform_init_est:
                    inp = torch.stack([normalized_depth_map, clopped_mask], 1).detach()
                    est_obj_pos_cim, est_obj_axis_green_cam, est_obj_axis_red_cam, est_scale_cim, est_shape_code, pre_hidden_state = self.init_net(inp, bbox_info)
                    est_obj_pos_cam, est_obj_scale, cim2im_scale, im2cam_scale, bbox_center = diff2estimation(est_obj_pos_cim, est_scale_cim, bbox_list, avg_depth_map, self.fov, with_cim2cam_info=True)
                    est_obj_pos_wrd = torch.sum(est_obj_pos_cam[..., None, :]*w2c.permute(0, 2, 1), dim=-1) + cam_pos_wrd
                    est_obj_axis_green_wrd = torch.sum(est_obj_axis_green_cam[..., None, :]*w2c.permute(0, 2, 1), -1)
                    est_obj_axis_red_wrd = torch.sum(est_obj_axis_red_cam[..., None, :]*w2c.permute(0, 2, 1), -1)
                    # Get average.
                    ave_est_obj_pos_wrd = est_obj_pos_wrd.reshape(batch_size, using_frame_num, 3).mean(1)
                    ave_est_obj_scale = est_obj_scale.reshape(batch_size, using_frame_num, 1).mean(1)
                    ave_est_obj_axis_green_wrd = est_obj_axis_green_wrd.reshape(batch_size, using_frame_num, 3).mean(1)
                    ave_est_obj_axis_green_wrd = F.normalize(ave_est_obj_axis_green_wrd, dim=1)
                    ave_est_obj_axis_red_wrd = est_obj_axis_red_wrd.reshape(batch_size, using_frame_num, 3).mean(1)
                    ave_est_obj_axis_red_wrd = F.normalize(ave_est_obj_axis_red_wrd, dim=1)
                    ave_est_shape_code = est_shape_code.reshape(batch_size, using_frame_num, self.ddf.latent_size).mean(1)
                    # Expanding.                    
                    ave_est_obj_pos_wrd_frame = ave_est_obj_pos_wrd[:, None, :].expand(-1, using_frame_num, 3).reshape(-1, 3)
                    ave_est_obj_scale_frame = ave_est_obj_scale[:, None, :].expand(-1, using_frame_num, 1).reshape(-1, 1)
                    ave_est_obj_axis_green_wrd_frame = ave_est_obj_axis_green_wrd[:, None, :].expand(-1, using_frame_num, 3).reshape(-1, 3)
                    ave_est_obj_axis_red_wrd_frame = ave_est_obj_axis_red_wrd[:, None, :].expand(-1, using_frame_num, 3).reshape(-1, 3)
                    ave_est_shape_code_frame = ave_est_shape_code[:, None, :].expand(-1, using_frame_num, self.ddf.latent_size).reshape(-1, self.ddf.latent_size)
                elif not perform_init_est:
                    # Get inputs.
                    with torch.no_grad():
                        pre_obj_axis_green_cam = torch.sum(pre_obj_axis_green_wrd[..., None, :]*w2c, -1)
                        pre_obj_axis_red_cam = torch.sum(pre_obj_axis_red_wrd[..., None, :]*w2c, -1)
                        est_clopped_mask, est_clopped_distance_map = render_distance_map_from_axis(
                                                                        H = self.ddf_H, 
                                                                        obj_pos_wrd = pre_obj_pos_wrd, 
                                                                        axis_green = pre_obj_axis_green_cam, 
                                                                        axis_red = pre_obj_axis_red_cam, 
                                                                        obj_scale = pre_obj_scale[:, 0], 
                                                                        input_lat_vec = pre_shape_code, 
                                                                        cam_pos_wrd = cam_pos_wrd, 
                                                                        rays_d_cam = rays_d_cam, 
                                                                        w2c = w2c.detach(), 
                                                                        ddf = self.ddf, 
                                                                        with_invdistance_map = False, 
                                                                        )
                        pre_mask = est_clopped_mask.detach()
                        _, pre_depth_map, _ = get_normalized_depth_map(est_clopped_mask, est_clopped_distance_map, rays_d_cam, avg_depth_map)
                        pre_error = torch.abs(pre_depth_map - normalized_depth_map)
                        pre_error = pre_error.reshape(batch_size, using_frame_num, self.ddf_H, self.ddf_H).mean(dim=-1).mean(dim=-1)
                        pre_error = pre_error.mean(dim=-1)
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
                    # Convert cordinates.
                    diff_pos_wrd = torch.sum(diffcim2diffcam(diff_pos_cim, cim2im_scale, im2cam_scale)[..., None, :]*w2c.permute(0, 2, 1), -1)
                    diff_obj_axis_green_wrd = torch.sum(diff_obj_axis_green_cam[..., None, :]*w2c.permute(0, 2, 1), -1)
                    diff_obj_axis_red_wrd = torch.sum(diff_obj_axis_red_cam[..., None, :]*w2c.permute(0, 2, 1), -1)
                    # Get average.
                    ave_diff_pos_wrd = diff_pos_wrd.reshape(batch_size, using_frame_num, 3).mean(1)[:, None].expand(-1, using_frame_num, -1).reshape(batch_size*using_frame_num,-1)
                    ave_diff_obj_axis_green_wrd = diff_obj_axis_green_wrd.reshape(batch_size, using_frame_num, 3).mean(1)[:, None].expand(-1, using_frame_num, -1).reshape(batch_size*using_frame_num,-1)
                    ave_diff_obj_axis_red_wrd = diff_obj_axis_red_wrd.reshape(batch_size, using_frame_num, 3).mean(1)[:, None].expand(-1, using_frame_num, -1).reshape(batch_size*using_frame_num,-1)
                    ave_diff_scale = diff_scale_cim.reshape(batch_size, using_frame_num, 1).mean(1)[:, None].expand(-1, using_frame_num, -1).reshape(batch_size*using_frame_num,-1)
                    ave_diff_shape_code = diff_shape_code.reshape(batch_size, using_frame_num, self.ddf.latent_size).mean(1)[:, None].expand(-1, using_frame_num, -1).reshape(batch_size*using_frame_num,-1)
                    # Update pre estimations.
                    ave_est_obj_pos_wrd_frame = pre_obj_pos_wrd + ave_diff_pos_wrd
                    ave_est_obj_axis_green_wrd_frame = F.normalize(pre_obj_axis_green_wrd + ave_diff_obj_axis_green_wrd)
                    ave_est_obj_axis_red_wrd_frame = F.normalize(pre_obj_axis_red_wrd + ave_diff_obj_axis_red_wrd)
                    ave_est_obj_scale_frame = pre_obj_scale * ave_diff_scale
                    ave_est_shape_code_frame = pre_shape_code + ave_diff_shape_code

            ###################################
            #####    Start Lamda Step     #####
            ###################################
            for half_lambda_idx in range(self.half_lambda_max):
                # Get simulation results.
                # with torch.no_grad():
                ave_est_obj_axis_green_cam_frame = torch.sum(ave_est_obj_axis_green_wrd_frame[..., None, :]*w2c, -1)
                ave_est_obj_axis_red_cam_frame = torch.sum(ave_est_obj_axis_red_wrd_frame[..., None, :]*w2c, -1)
                ave_est_obj_pos_cam_frame = torch.sum((ave_est_obj_pos_wrd_frame - cam_pos_wrd)[..., None, :]*w2c, dim=-1)
                ave_est_obj_pos_cim_frame = torch.cat([
                                                (ave_est_obj_pos_cam_frame[:, :-1] / im2cam_scale[:, None] - bbox_center) / cim2im_scale[:, None], 
                                                (ave_est_obj_pos_cam_frame[:, -1] - avg_depth_map)[:, None]], dim=-1)
                ave_est_scale_cim_frame = ave_est_obj_scale_frame / (im2cam_scale[:, None] * cim2im_scale[:, None] * 2 * math.sqrt(2))
                est_mask, est_distance_map = render_distance_map_from_axis(
                                                            H = self.ddf_H, 
                                                            obj_pos_wrd = ave_est_obj_pos_wrd_frame, 
                                                            axis_green = ave_est_obj_axis_green_cam_frame, 
                                                            axis_red = ave_est_obj_axis_red_cam_frame, 
                                                            obj_scale = ave_est_obj_scale_frame[:, 0], 
                                                            input_lat_vec = ave_est_shape_code_frame, 
                                                            # obj_pos_wrd = gt_obj_pos_wrd, 
                                                            # axis_green = gt_obj_axis_green_cam, 
                                                            # axis_red = gt_obj_axis_red_cam, 
                                                            # obj_scale = gt_obj_scale[:, 0].to(est_obj_scale), 
                                                            # input_lat_vec = gt_shape_code, 
                                                            cam_pos_wrd = cam_pos_wrd, 
                                                            rays_d_cam = rays_d_cam, 
                                                            w2c = w2c.detach(), 
                                                            ddf = self.ddf, 
                                                            with_invdistance_map = False, 
                                                            )
                _, est_normalized_depth_map, _ = get_normalized_depth_map(
                                                    est_mask, est_distance_map, rays_d_cam, avg_depth_map, 
                                                    )
                error = torch.abs(est_normalized_depth_map - normalized_depth_map)
                error = error.reshape(batch_size, using_frame_num, self.ddf_H, self.ddf_H).mean(dim=-1).mean(dim=-1)
                error = error.mean(dim=-1)

                #############################################
                if optim_idx==4:
                    check_map = []
                    gt = normalized_depth_map
                    est = est_normalized_depth_map
                    for i in range(batch_size):
                        check_map.append(torch.cat([gt[i], est[i], torch.abs(gt[i]-est[i])], dim=0))
                    check_map_torch(torch.cat(check_map, dim=-1), f'tes_{optim_idx}.png')
                    import pdb; pdb.set_trace()
                #############################################

                # 最初のフレームの初期予測
                if perform_init_est:
                    # Get next inputs
                    pre_obj_pos_cim = ave_est_obj_pos_cim_frame.detach()
                    pre_obj_pos_wrd = ave_est_obj_pos_wrd_frame.detach()
                    pre_obj_scale_cim = ave_est_scale_cim_frame.detach()
                    pre_obj_scale = ave_est_obj_scale_frame.detach()
                    pre_obj_axis_green_wrd = ave_est_obj_axis_green_wrd_frame.detach()
                    pre_obj_axis_red_wrd = ave_est_obj_axis_red_wrd_frame.detach()
                    pre_shape_code = ave_est_shape_code_frame.detach()
                    # #########################
                    # gt = normalized_depth_map
                    # est = est_normalized_depth_map
                    # check_optim_steps.append(gt)
                    # check_optim_steps.append(torch.abs(gt-est))
                    # #########################
                    # 初期化ネットだけの性能を評価する場合
                    if self.only_init_net:
                        pre_mask = est_mask.detach()
                        pre_depth_map = est_normalized_depth_map.detach()
                        pre_error = error.detach()
                    break # 初期化の段階ではラムダステップは無し。

                else:
                    # Cal Error.
                    update_mask = (pre_error - error) > 0. #エラーが大きくなった場合、True
                    update_mask = update_mask[:, None].expand(-1, using_frame_num).reshape(batch_size*using_frame_num)
                    un_update_mask = torch.logical_not(update_mask)

                    # 更新により、エラーが全てのバッチで小さくなった or ラムダステップの最大まで行った
                    # -> 次の最適化ステップかフレームへ
                    decade_all_error = update_mask.all()
                    over_lamda_step = half_lambda_idx + 1 == self.half_lambda_max

                    if decade_all_error or over_lamda_step:
                        # Update values.
                        pre_obj_pos_cim[update_mask] = ave_est_obj_pos_cim_frame[update_mask].detach()
                        pre_obj_pos_wrd[update_mask] = ave_est_obj_pos_wrd_frame[update_mask].detach()
                        pre_obj_scale_cim[update_mask] = ave_est_scale_cim_frame[update_mask].detach()
                        pre_obj_scale[update_mask] = ave_est_obj_scale_frame[update_mask].detach()
                        pre_obj_axis_green_wrd[update_mask] = ave_est_obj_axis_green_wrd_frame[update_mask].detach()
                        pre_obj_axis_red_wrd[update_mask] = ave_est_obj_axis_red_wrd_frame[update_mask].detach()
                        pre_shape_code[update_mask] = ave_est_shape_code_frame[update_mask].detach()
                        break # ラムダステップ終了。

                    # 更新により、エラーが全てのバッチで小さくななかった
                    # -> ならなかったUpdateを半減させて再計算
                    else:
                        lamda_i = 1 / 2**(half_lambda_idx+1)
                        ave_est_obj_pos_wrd_frame[un_update_mask] = pre_obj_pos_wrd[un_update_mask] + lamda_i * ave_diff_pos_wrd[un_update_mask]
                        ave_est_obj_scale_frame[un_update_mask] = pre_obj_scale[un_update_mask] * (1. + lamda_i * (ave_diff_scale[un_update_mask] - 1.))
                        ave_est_obj_axis_green_wrd_frame[un_update_mask] = F.normalize(pre_obj_axis_green_wrd[un_update_mask] + lamda_i * ave_diff_obj_axis_green_wrd[un_update_mask])
                        ave_est_obj_axis_red_wrd_frame[un_update_mask] = F.normalize(pre_obj_axis_red_wrd[un_update_mask] + lamda_i * ave_diff_obj_axis_red_wrd[un_update_mask])
                        ave_est_shape_code_frame[un_update_mask] = pre_shape_code[un_update_mask] + lamda_i * ave_diff_shape_code[un_update_mask]
            
            if self.only_init_net:
                break

        est_obj_pos_wrd = pre_obj_pos_wrd.reshape(batch_size, using_frame_num, -1)[:, 0, :].clone()
        est_obj_scale = pre_obj_scale.reshape(batch_size, using_frame_num, -1)[:, 0, :].clone()
        est_obj_axis_green_wrd = pre_obj_axis_green_wrd.reshape(batch_size, using_frame_num, -1)[:, 0, :].clone()
        est_obj_axis_red_wrd = pre_obj_axis_red_wrd.reshape(batch_size, using_frame_num, -1)[:, 0, :].clone()
        est_shape_code = pre_shape_code.reshape(batch_size, using_frame_num, -1)[:, 0, :].clone()

        ###########################################################################
        #########################       check shape       #########################
        ###########################################################################
        depth_error = []
        for shape_i, (gt_distance_map, cam_pos_wrd, w2c) in enumerate(zip(canonical_distance_map.permute(1, 0, 2, 3), 
                                                                            canonical_camera_pos.permute(1, 0, 2), 
                                                                            canonical_camera_rot.permute(1, 0, 2, 3))):
            # Get inp.
            rays_d_cam = get_ray_direction(self.ddf_H, self.fov).expand(batch_size, -1, -1, -1).to(frame_camera_rot.device)
            est_obj_axis_green_cam = torch.sum(est_obj_axis_green_wrd[..., None, :]*w2c, -1)
            est_obj_axis_red_cam = torch.sum(est_obj_axis_red_wrd[..., None, :]*w2c, -1)

            # Get simulation results.
            with torch.no_grad():
                est_mask, est_distance_map = get_canonical_map(
                                                H = self.ddf_H, 
                                                cam_pos_wrd = cam_pos_wrd, 
                                                rays_d_cam = rays_d_cam, 
                                                w2c = w2c, 
                                                input_lat_vec = est_shape_code, 
                                                ddf = self.ddf, 
                                                )
            depth_error.append(torch.abs(gt_distance_map-est_distance_map).mean(dim=-1).mean(dim=-1))
        
            # ############################################
            # # Check map.
            # if shape_i == 3:
            #     check_map = []
            #     gt = gt_distance_map
            #     est = est_distance_map
            #     for i in range(batch_size):
            #         check_map.append(torch.cat([gt[i], est[i], torch.abs(gt[i]-est[i])], dim=0))
            #     check_map_torch(torch.cat(check_map, dim=-1), f'canonical_{batch_idx}.png')
            # ############################################

        # Cal err.
        err_pos = torch.abs(est_obj_pos_wrd - gt_obj_pos_wrd[:, 0]).mean(dim=-1)
        err_scale = torch.abs(1 - est_obj_scale[:, 0] / gt_obj_scale[:, 0, 0])
        err_axis_red_cos_sim = self.cossim(est_obj_axis_red_wrd, gt_axis_red_wrd[:, 0]).clamp(min=self.cosssim_min, max=self.cosssim_max)
        err_axis_red = torch.acos(err_axis_red_cos_sim) * 180 / torch.pi
        err_axis_green_cos_sim = self.cossim(est_obj_axis_green_wrd, gt_axis_green_wrd[:, 0]).clamp(min=self.cosssim_min, max=self.cosssim_max)
        err_axis_green = torch.acos(err_axis_green_cos_sim) * 180 / torch.pi
        depth_error = torch.stack(depth_error, dim=-1).mean(dim=-1)
        
        return {'err_pos':err_pos.detach(), 
                'err_scale': err_scale.detach(), 
                'err_axis_red': err_axis_red.detach(), 
                'err_axis_green':err_axis_green.detach(), 
                'depth_error':depth_error.detach(), 
                'path':np.array(path), 
                'pos':est_obj_pos_wrd.detach(), 
                'scale':est_obj_scale.detach(), 
                'axis_red':est_obj_axis_red_wrd.detach(), 
                'axis_green':est_obj_axis_green_wrd.detach(), 
                'shape_code':est_shape_code.detach()}



    def test_epoch_end(self, outputs):
        # Log loss.
        err_pos_list = torch.cat([x['err_pos'] for x in outputs], dim=0)
        avg_err_pos = err_pos_list.mean()
        err_scale_list = torch.cat([x['err_scale'] for x in outputs], dim=0)
        avg_err_scale = err_scale_list.mean()
        err_axis_red_list = torch.cat([x['err_axis_red'] for x in outputs], dim=0)
        avg_err_axis_red = err_axis_red_list.mean()
        err_axis_green_list = torch.cat([x['err_axis_green'] for x in outputs], dim=0)
        avg_err_axis_green = err_axis_green_list.mean()
        depth_error_list = torch.cat([x['depth_error'] for x in outputs], dim=0)
        avg_err_depth = depth_error_list.mean()
        path_list = np.concatenate([x['path'] for x in outputs])
        with open(self.test_log_path, 'a') as file:
            file.write('avg_err_pos : ' + str(avg_err_pos.item()) + '\n')
            file.write('avg_err_scale : ' + str(avg_err_scale.item()) + '\n')
            file.write('avg_err_axis_red : ' + str(avg_err_axis_red.item()) + '\n')
            file.write('avg_err_axis_green : ' + str(avg_err_axis_green.item()) + '\n')
            file.write('avg_err_depth : ' + str(avg_err_depth.item()) + '\n')


        err_pos_list = err_pos_list.to('cpu').detach().numpy().copy()
        err_scale_list = err_scale_list.to('cpu').detach().numpy().copy()
        err_axis_red_list = err_axis_red_list.to('cpu').detach().numpy().copy()
        err_axis_green_list = err_axis_green_list.to('cpu').detach().numpy().copy()
        depth_error_list = depth_error_list.to('cpu').detach().numpy().copy()
        log_dict = {'pos':err_pos_list, 
                    'scale':err_scale_list, 
                    'red':err_axis_red_list, 
                    'green':err_axis_green_list, 
                    'depth':depth_error_list, 
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

    # Create ddf.
    ddf = DDF(args)
    ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args)
    ddf.eval()

    # Create dfnet.
    df_net = TaR(args, ddf)
    df_net = df_net.load_from_checkpoint(
        checkpoint_path=args.model_ckpt_path, 
        args=args, 
        ddf=ddf
        )
    df_net.eval()

    # Setting model.
    model = df_net
    model.test_mode = 'average'
    model.start_frame_idx = 0
    model.frame_sequence_num = 3
    model.half_lambda_max = 3
    model.test_optim_num = 5
    if model.model_mode == 'only_init':
        model.only_init_net = True
        model.test_optim_num = 1
    else:
         model.only_init_net = False
    model.use_deep_optimizer = True
    model.use_adam_optimizer = not(model.use_deep_optimizer)
    model.use_weighted_average = False

    # Save logs.
    import datetime
    dt_now = datetime.datetime.now()
    time_log = dt_now.strftime('%Y_%m_%d_%H_%M_%S')

    os.mkdir('./txt/experiments/log/' + time_log)
    file_name = './txt/experiments/log/' + time_log + '/log.txt'
    model.test_log_path = file_name
    ckpt_path = args.model_ckpt_path
    with open(file_name, 'a') as file:
        file.write('script_name : ' + 'val deep multi' + '\n')
        file.write('time_log : ' + time_log + '\n')
        file.write('ckpt_path : ' + ckpt_path + '\n')
        file.write('val_N_views : ' + str(args.val_N_views) + '\n')
        file.write('val_instance_list_txt : ' + str(args.val_instance_list_txt) + '\n')
        file.write('\n')
        file.write('only_init_net : ' + str(model.only_init_net) + '\n')
        file.write('start_frame_idx : ' + str(model.start_frame_idx) + '\n')
        file.write('frame_sequence_num : ' + str(model.frame_sequence_num) + '\n')
        file.write('half_lambda_max : ' + str(model.half_lambda_max) + '\n')
        file.write('test_optim_num : ' + str(model.test_optim_num) + '\n')
        file.write('\n')

    # Test model.
    test_model = test_TaR(args, model)
    trainer = pl.Trainer(
        gpus=args.gpu_num, 
        strategy=DDPPlugin(find_unused_parameters=True), #=False), 
        enable_checkpointing = False,
        check_val_every_n_epoch = args.check_val_every_n_epoch,
        logger = pl.loggers.TensorBoardLogger(save_dir=os.getcwd(), version=f'val_trash', name='lightning_logs')
        )
    # trainer.fit(test_model, train_dataloaders=val_dataloader)
    trainer.test(test_model, val_dataloader)
