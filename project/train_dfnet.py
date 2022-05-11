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
from train_init_net import *
from DDF.train_pl import DDF

torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False

seed = 1
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
        self.optim_step_num = 5
        self.frame_num = args.frame_num
        self.model_params_dtype = False
        self.model_device = False

        # Make model
        self.ddf = ddf
        self.init_net = resnet_encoder(args, in_channel=2) #init_net
        self.use_gru = args.use_gru
        if self.use_gru:
            self.df_net = df_resnet_encoder_with_gru(args, in_channel=5)
        else:
            self.df_net = df_resnet_encoder(args, in_channel=5)

        # loss func.
        self.l1 = torch.nn.L1Loss()
        self.cossim = nn.CosineSimilarity(dim=-1)



    def training_step(self, batch, batch_idx):

        frame_rgb_map, frame_mask, frame_depth_map, frame_camera_pos, frame_camera_rot, frame_obj_rot, instance_id = batch
        batch_size = len(instance_id)
        
        loss_axis_green = []
        loss_axis_red = []
        loss_shape_code = []
        check_map_1 = []
        check_map_2 = []

        # for frame_idx in range(self.frame_num):
        frame_idx = random.randint(0, frame_rgb_map.shape[1]-1)
        for optim_idx in range(self.optim_step_num):

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
            if optim_idx < self.optim_step_num:
                with torch.no_grad():
                    # Get rotation matrix.
                    axis_green = est_axis_green
                    axis_red = est_axis_red
                    axis_blue = torch.cross(axis_red, axis_green, dim=-1)
                    axis_blue = F.normalize(axis_blue, dim=-1)
                    axis_red = torch.cross(axis_green, axis_blue, dim=-1)
                    est_o2c = torch.stack([axis_red, axis_green, axis_blue], dim=-1)
                
                    # Get rays direction.
                    rays_d_cam = self.rays_d_cam.expand(batch_size, -1, -1, -1).to(frame_camera_rot.device)
                    rays_d_obj = torch.sum(rays_d_cam[..., None, :]*est_o2c[..., None, None, :, :].permute(0, 1, 2, 4, 3), -1)
                
                    # Get rays origin.
                    cam_pos_wrd = frame_camera_pos[:, frame_idx]
                    obj_pos_wrd = torch.zeros_like(cam_pos_wrd)
                    est_o2w = torch.bmm(w2c.to(est_o2c.dtype).permute(0, 2, 1), est_o2c)
                    cam_pos_obj = torch.sum((cam_pos_wrd - obj_pos_wrd)[..., None, :] * est_o2w.permute(0, 2, 1), dim=-1) # = -obj_T_c2o
                    rays_o_obj = cam_pos_obj[:, None, None, :].expand(-1, self.H, self.H, -1)

                    # Get rays inputs.
                    rays_d = rays_d_obj
                    rays_o = rays_o_obj
                    input_lat_vec = est_shape_code

                    # Estimating.
                    est_invdepth_map = self.ddf.forward(rays_o, rays_d, input_lat_vec)
                    est_mask = est_invdepth_map > .5
                    est_depth_map = torch.zeros_like(est_invdepth_map)
                    est_depth_map[est_mask] = 1. / est_invdepth_map[est_mask]

                    # get next inputs
                    pre_axis_green = est_axis_green.detach()
                    pre_axis_red = est_axis_red.detach()
                    pre_shape_code = est_shape_code.detach()
                    pre_mask = est_mask.detach()
                    pre_depth_map = est_depth_map.detach()

        # Cal total loss.
        loss_axis_green = sum(loss_axis_green) / self.frame_num
        loss_axis_red = sum(loss_axis_red) / self.frame_num
        loss_shape_code = sum(loss_shape_code) / self.frame_num
        loss_axis = loss_axis_green + .5 * loss_axis_red
        loss = loss_axis + 1e2 * loss_shape_code
        
        if (self.current_epoch + 1) % 10 == 0 and batch_idx==0:
            for batch_idx in range(batch_size):
                check_map_i = torch.cat([depth_map[batch_idx], pre_depth_map[batch_idx]], dim=0)
                check_map_1.append(check_map_i)
            check_map_1 = torch.cat(check_map_1, dim=1)
            check_map(check_map_1, f'test_dfnet_png/check_map_{self.current_epoch + 1}.png', figsize=[10,2])
            check_map_1 = []
            # import pdb; pdb.set_trace()

        return {'loss': loss, 'loss_axis': loss_axis.detach(), 'loss_shape_code': loss_shape_code.detach()}



    def training_epoch_end(self, outputs):

        # Log loss.
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        current_epoch = torch.tensor(self.current_epoch + 1., dtype=avg_loss.dtype)
        self.log_dict({'train/total_loss': avg_loss, "step": current_epoch})

        avg_loss_axis = torch.stack([x['loss_axis'] for x in outputs]).mean()
        self.log_dict({'train/loss_axis': avg_loss_axis, "step": current_epoch})

        avg_loss_shape_code = torch.stack([x['loss_shape_code'] for x in outputs]).mean()
        self.log_dict({'train/loss_shape_code': avg_loss_shape_code, "step": current_epoch})

        # Save ckpt.
        if (self.current_epoch + 1) % self.save_interval == 0:
            ckpt_name = str(self.current_epoch + 1).zfill(10) + '.ckpt'
            ckpt_path = os.path.join(self.trainer.log_dir, 'checkpoints', ckpt_name)
            trainer.save_checkpoint(ckpt_path)



    def validation_step(self, batch, batch_idx):

        frame_rgb_map, frame_mask, frame_depth_map, frame_camera_pos, frame_camera_rot, frame_obj_rot, instance_id = batch
        batch_size = len(instance_id)
        
        # for frame_idx in range(self.frame_num):
        frame_idx = random.randint(0, frame_rgb_map.shape[1]-1)
        for optim_idx in range(self.optim_step_num):

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
                # # Check inputs.
                # check_map_inp = torch.cat([depth_map, mask, est_depth_map, est_mask, depth_map - est_depth_map], dim=2)
                # check_map(check_map_inp[0], 'input_for_dfn.png', figsize=[10,2])

            # Estimating depth map.
            if optim_idx < self.optim_step_num:
                with torch.no_grad():
                    # Get rotation matrix.
                    axis_green = est_axis_green
                    axis_red = est_axis_red
                    axis_blue = torch.cross(axis_red, axis_green, dim=-1)
                    axis_blue = F.normalize(axis_blue, dim=-1)
                    axis_red = torch.cross(axis_green, axis_blue, dim=-1)
                    est_o2c = torch.stack([axis_red, axis_green, axis_blue], dim=-1)
                
                    # Get rays direction.
                    rays_d_cam = self.rays_d_cam.expand(batch_size, -1, -1, -1).to(frame_camera_rot.device)
                    rays_d_obj = torch.sum(rays_d_cam[..., None, :]*est_o2c[..., None, None, :, :].permute(0, 1, 2, 4, 3), -1)
                
                    # Get rays origin.
                    cam_pos_wrd = frame_camera_pos[:, frame_idx]
                    obj_pos_wrd = torch.zeros_like(cam_pos_wrd)
                    est_o2w = torch.bmm(w2c.to(est_o2c.dtype).permute(0, 2, 1), est_o2c)
                    cam_pos_obj = torch.sum((cam_pos_wrd - obj_pos_wrd)[..., None, :] * est_o2w.permute(0, 2, 1), dim=-1) # = -obj_T_c2o
                    rays_o_obj = cam_pos_obj[:, None, None, :].expand(-1, self.H, self.H, -1)

                    # Get rays inputs.
                    rays_d = rays_d_obj
                    rays_o = rays_o_obj
                    input_lat_vec = est_shape_code

                    # Estimating.
                    est_invdepth_map = self.ddf.forward(rays_o, rays_d, input_lat_vec)
                    est_mask = est_invdepth_map > .5
                    est_depth_map = torch.zeros_like(est_invdepth_map)
                    est_depth_map[est_mask] = 1. / est_invdepth_map[est_mask]

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



    def test_step(self, batch, batch_idx):

        frame_rgb_map, frame_mask, frame_depth_map, frame_camera_pos, frame_camera_rot, frame_obj_rot, instance_id = batch
        batch_size = len(instance_id)
        
        ###########################################################################
        #########################        average          #########################
        ###########################################################################
        if self.test_mode == 'average':
            # 各フレームで独立して初期推定->最適化ステップを行い、それを平均する。



            est_axis_green_wrd = []
            est_axis_red_wrd = []
            est_shape_code = []
            
            
            
            # ランダムなあるフレームからframe_sequence_num個を取得
            frame_est_list = {'axis_green':[], 'axis_red':[], 'shape_code':[], 'error':[]}
            for frame_sequence_idx in range(self.frame_sequence_num):
                frame_idx = self.start_frame_idx + frame_sequence_idx

                for optim_idx in range(self.optim_step_num):

                    # Get ground truth.
                    o2w = batch_pi2rot_y(frame_obj_rot[:, frame_idx]).permute(0, 2, 1)
                    w2c = frame_camera_rot[:, frame_idx]
                    o2c = torch.bmm(w2c, o2w) # とりあえずこれを推論する
                    o2w = torch.bmm(w2c.permute(0, 2, 1), o2c)
                    o2o = torch.bmm(o2w.permute(0, 2, 1), o2w)
                    gt_axis_green_wrd = torch.sum(o2c[:, :, 1][..., None, :]*w2c.permute(0, 2, 1), -1) # Y
                    gt_axis_red_wrd = torch.sum(o2c[:, :, 0][..., None, :]*w2c.permute(0, 2, 1), -1) # X

                    # Get input.
                    mask = frame_mask[:, frame_idx]
                    depth_map = frame_depth_map[:, frame_idx]
                    
                    # Estimating.
                    if optim_idx == 0:
                        inp = torch.stack([depth_map, mask], 1)
                        if self.use_gru:
                            est_axis_green_cam_i, est_axis_red_cam_i, est_shape_code_i, feature_vec = self.init_net(inp, self.use_gru)
                            pre_hidden_state = feature_vec
                        else:
                            est_axis_green_cam_i, est_axis_red_cam_i, est_shape_code_i = self.init_net(inp, self.use_gru)
                    else:
                        inp = torch.stack([depth_map, mask, pre_depth_map, pre_mask, depth_map - pre_depth_map], 1)
                        if self.use_gru:
                            print('a')
                            diff_axis_green, diff_axis_red, diff_shape_code, feature_vec, pre_hidden_state = self.df_net(inp, pre_axis_green, pre_axis_red, pre_shape_code, pre_hidden_state)
                        else:
                            diff_axis_green, diff_axis_red, diff_shape_code = self.df_net(inp, pre_axis_green, pre_axis_red, pre_shape_code)
                        est_axis_green_cam_i = F.normalize(pre_axis_green + diff_axis_green, dim=-1)
                        est_axis_red_cam_i = F.normalize(pre_axis_red + diff_axis_red, dim=-1)
                        est_shape_code_i = pre_shape_code + diff_shape_code
                        print(f'frame_sequence_idx:{frame_sequence_idx}, frame_idx:{frame_idx}, optim_idx:{optim_idx}')

                    # 最適化ステップ内の、ラムダステップを実行
                    for half_lambda_idx in range(self.half_lambda_max):
                        # Get simulation results.
                        rays_d_cam = self.rays_d_cam.expand(batch_size, -1, -1, -1).to(frame_camera_rot.device)
                        obj_pos_wrd = torch.zeros(batch_size, 3, device=frame_camera_rot.device)
                        est_mask, est_depth_map = get_depth_map_from_axis(
                                                        H = self.H, 
                                                        axis_green = est_axis_green_cam_i, 
                                                        axis_red = est_axis_red_cam_i,
                                                        cam_pos_wrd = frame_camera_pos[:, frame_idx], 
                                                        obj_pos_wrd = obj_pos_wrd, 
                                                        rays_d_cam = rays_d_cam, 
                                                        w2c = w2c, 
                                                        input_lat_vec = est_shape_code_i, 
                                                        ddf = self.ddf, 
                                                        )

                        # 初期推定をセット、ラムダステップはなし
                        if optim_idx == 0:
                            # Get next inputs
                            pre_axis_green = est_axis_green_cam_i.detach()
                            pre_axis_red = est_axis_red_cam_i.detach()
                            pre_shape_code = est_shape_code_i.detach()
                            pre_mask = est_mask.detach()
                            pre_depth_map = est_depth_map.detach()
                            pre_error = torch.abs(pre_depth_map - depth_map).mean(dim=-1).mean(dim=-1)
                            break

                        # 最適化のラムダステップ
                        else:
                            # エラーを計算
                            error = torch.abs(est_depth_map - depth_map).mean(dim=-1).mean(dim=-1)
                            un_update_mask = (pre_error - error) < 0. #エラーが大きくなった場合、True
                            update_mask = torch.logical_not(un_update_mask)

                            # 更新により、エラーが全てのバッチで小さくなった
                            # ラムダステップの最大まで行った
                            # -> 次の最適化ステップ or フレームへ
                            decade_all_error = update_mask.all()
                            over_lamda_step = half_lambda_idx + 1 == self.half_lambda_max
                            last_optim_step = optim_idx + 1 == self.test_optim_num[frame_sequence_idx]
                            not_last_frame = frame_sequence_idx < self.frame_sequence_num - 1
                            go_next_frame = last_optim_step and not_last_frame

                            if decade_all_error or over_lamda_step:
                                # get next inputs
                                pre_axis_green[update_mask] = est_axis_green_cam_i[update_mask].detach()
                                pre_axis_red[update_mask] = est_axis_red_cam_i[update_mask].detach()
                                pre_shape_code[update_mask] = est_shape_code_i[update_mask].detach()
                                pre_mask[update_mask] = est_mask[update_mask].detach()
                                pre_depth_map[update_mask] = est_depth_map[update_mask].detach()
                                pre_error[update_mask] = error[update_mask].detach()
                                
                                # 全ての要素を更新出来なかった -> もうこれ以上の最適化ステップは意味がない
                                # ※　GRUの時は別
                                # ※　この先のフレームもやらない？
                                # if un_update_mask.all() and over_lamda_step:
                                #     early_stop = True

                                if last_optim_step:
                                    pre_axis_green_wrd = torch.sum(pre_axis_green[..., None, :]*w2c.permute(0, 2, 1), -1)
                                    pre_axis_red_wrd = torch.sum(pre_axis_red[..., None, :]*w2c.permute(0, 2, 1), -1)

                                    # Stack final estimation on this frame.
                                    if self.average_each_results:
                                        # 現フレームに対する推論結果をスタックする。
                                        frame_est_list['axis_green'].append(pre_axis_green_wrd.clone())
                                        frame_est_list['axis_red'].append(pre_axis_red_wrd.clone())
                                        frame_est_list['shape_code'].append(pre_shape_code.clone())
                                        frame_est_list['error'].append(pre_error) # 現フレームに対するエラーを見たい。
                                break

                            # 更新により、エラーが全てのバッチで小さくななかった
                            # -> ならなかったUpdateを半減させて再計算
                            else:
                                lamda_i = 1 / 2**(half_lambda_idx+1)
                                est_axis_green_cam_i[un_update_mask] = pre_axis_green[un_update_mask] + lamda_i * diff_axis_green[un_update_mask]
                                est_axis_green_cam_i = F.normalize(est_axis_green_cam_i, dim=-1)
                                est_axis_red_cam_i[un_update_mask] = pre_axis_red[un_update_mask] + lamda_i * diff_axis_red[un_update_mask]
                                est_axis_red_cam_i = F.normalize(est_axis_red_cam_i, dim=-1)
                                est_shape_code_i[un_update_mask] = pre_shape_code[un_update_mask] + lamda_i * diff_shape_code[un_update_mask]
            
            # Get average.
            if self.average_each_results:
                est_axis_green_wrd = get_weighted_average(
                                        target = torch.stack(frame_est_list['axis_green'], dim=1).detach(), 
                                        ratio = 1/torch.stack(frame_est_list['error'], dim=1).detach())
                est_axis_red_wrd = get_weighted_average(
                                        target = torch.stack(frame_est_list['axis_red'], dim=1).detach(), 
                                        ratio = 1/torch.stack(frame_est_list['error'], dim=1).detach())
                est_axis_green_wrd = F.normalize(est_axis_green_wrd, dim=1)
                est_axis_red_wrd = F.normalize(est_axis_red_wrd, dim=1)
                est_shape_code = get_weighted_average(
                                        target = torch.stack(frame_est_list['shape_code'], dim=1).detach(), 
                                        ratio = 1/torch.stack(frame_est_list['error'], dim=1).detach())
                ratio = 1/torch.stack(frame_est_list['error'], dim=1).detach()
                print((ratio / torch.sum(ratio, dim=1)[..., None])[0])
                print(torch.stack(frame_est_list['error'], dim=1)[0])
            elif not self.average_each_results:
                est_axis_green_wrd = torch.stack(frame_est_list['axis_green'], dim=1).mean(dim=1)
                est_axis_red_wrd = torch.stack(frame_est_list['axis_red'], dim=1).mean(dim=1)
                est_axis_green_wrd = F.normalize(est_axis_green_wrd, dim=1)
                est_axis_red_wrd = F.normalize(est_axis_red_wrd, dim=1)
                est_shape_code = torch.stack(frame_est_list['shape_code'], dim=1).mean(dim=1)



        ###########################################################################
        #########################      check result       #########################
        ###########################################################################
        with torch.no_grad():
            # Set random frames.
            frame_idx = -1 # random.randint(0, frame_rgb_map.shape[1]-1)
            mask = frame_mask[:, frame_idx]
            depth_map = frame_depth_map[:, frame_idx]

            # Get simulation results.
            w2c = frame_camera_rot[:, frame_idx]
            est_axis_green = torch.sum(est_axis_green_wrd[..., None, :]*w2c, -1)
            est_axis_red = torch.sum(est_axis_red_wrd[..., None, :]*w2c, -1)
            rays_d_cam = self.rays_d_cam.expand(batch_size, -1, -1, -1).to(frame_camera_rot.device)
            obj_pos_wrd = torch.zeros(batch_size, 3, device=frame_camera_rot.device)
            est_mask, est_depth_map = get_depth_map_from_axis(
                                            H = self.H, 
                                            axis_green = est_axis_green, 
                                            axis_red = est_axis_red,
                                            cam_pos_wrd = frame_camera_pos[:, frame_idx], 
                                            obj_pos_wrd = obj_pos_wrd, 
                                            rays_d_cam = rays_d_cam, 
                                            w2c = w2c, 
                                            input_lat_vec = est_shape_code_i, 
                                            ddf = self.ddf, 
                                            )
                    
            # # Check depth map.
            # check_map_1 = []
            # for batch_i in range(batch_size):
            #     check_map_i = torch.cat([depth_map[batch_i], est_depth_map[batch_i]], dim=0)
            #     check_map_1.append(check_map_i)
            # check_map_1 = torch.cat(check_map_1, dim=1)
            # check_map(check_map_1, f'check_batch_{str(batch_idx).zfill(5)}.png', figsize=[10,2])
            # check_map_1 = []

        # Cal err.
        err_axis_green = torch.mean(-self.cossim(est_axis_green_wrd, gt_axis_green_wrd) + 1.)
        err_axis_red = torch.mean(-self.cossim(est_axis_red_wrd, gt_axis_red_wrd) + 1.)
        err_depth = F.mse_loss(est_depth_map, depth_map)

        return {'err_axis_green': err_axis_green.detach(), 'err_axis_red': err_axis_red.detach(), 'err_depth': err_depth.detach()}



    def test_epoch_end(self, outputs):
        # Log loss.
        avg_err_axis_green = torch.stack([x['err_axis_green'] for x in outputs]).mean()
        avg_err_axis_red = torch.stack([x['err_axis_red'] for x in outputs]).mean()
        avg_err_depth = torch.stack([x['err_depth'] for x in outputs]).mean()

        with open(self.test_log_path, 'a') as file:
            file.write('avg_err_axis_green : ' + str(avg_err_axis_green.item()) + '\n')
            file.write('avg_err_axis_red : ' + str(avg_err_axis_red.item()) + '\n')
            file.write('avg_err_depth : ' + str(avg_err_depth.item()) + '\n')
        


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
    args.use_gru = True
    model = TaR(args, ddf)
    
    # Get ckpts path.
    ckpt_dir = os.path.join('lightning_logs', f'{args.expname}_{args.exp_version}', 'checkpoints/*')
    ckpt_path_list = sorted(glob.glob(ckpt_dir))

    # Load ckpt and start training.
    # if len(ckpt_path_list) == 0:
    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=val_dataloader, 
        datamodule=None, 
        ckpt_path=None
        )

    # # elif len(ckpt_path_list) > 0:
    # #     latest_ckpt_path = ckpt_path_list[-1]
    # #     print('\n', f'+++ Reloading from {latest_ckpt_path} +++ \n')
    # ckpt_path = './lightning_logs/DeepTaR/chair/test_dfnet_withx/checkpoints/0000003200.ckpt'
    # model = TaR(args, ddf)
    # trainer.fit(
    #     model=model, 
    #     train_dataloaders=train_dataloader, 
    #     val_dataloaders=val_dataloader, 
    #     datamodule=None, 
    #     ckpt_path=ckpt_path # latest_ckpt_path
    #     )
