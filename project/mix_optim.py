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
from train_init_net import *
from train_dfnet import *
from train_frame_dfnet import *
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





def test_model(model, dataloader):

    err_axis_green = []
    err_axis_red = []
    err_depth = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        _, frame_mask, frame_depth_map, frame_camera_pos, frame_camera_rot, frame_obj_rot, instance_id = batch
        batch_size = len(instance_id)

        ###########################################################################
        #########################        average          #########################
        ###########################################################################
        if model.test_mode == 'average':
            # ランダムなあるフレームからframe_sequence_num個を取得
            frame_est_list = {'axis_green':[], 'axis_red':[], 'shape_code':[], 'error':[]}
            for frame_sequence_idx in range(model.frame_sequence_num):
                frame_idx = model.start_frame_idx + frame_sequence_idx

                for optim_idx in range(model.optim_step_num):

                    with torch.no_grad():
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
                        invdepth_map = torch.zeros_like(depth_map)
                        invdepth_map[mask] = 1. / depth_map[mask]
                    
                    # Estimating initial or update values.
                    use_deep_optimizer = True
                    if optim_idx == 0:
                        with torch.no_grad():
                            inp = torch.stack([depth_map, mask], 1)
                            if model.use_gru:
                                est_axis_green_cam_i, est_axis_red_cam_i, est_shape_code_i, feature_vec = model.init_net(inp, model.use_gru)
                                pre_hidden_state = feature_vec
                            else:
                                est_axis_green_cam_i, est_axis_red_cam_i, est_shape_code_i = model.init_net(inp, model.use_gru)
                    elif use_deep_optimizer:
                        with torch.no_grad():
                            inp = torch.stack([depth_map, mask, pre_depth_map, pre_mask, depth_map - pre_depth_map], 1)
                            if model.use_gru:
                                print('a')
                                diff_axis_green, diff_axis_red, diff_shape_code, feature_vec, pre_hidden_state = model.df_net(inp, pre_axis_green, pre_axis_red, pre_shape_code, pre_hidden_state)
                            else:
                                diff_axis_green, diff_axis_red, diff_shape_code = model.df_net(inp, pre_axis_green, pre_axis_red, pre_shape_code)
                            est_axis_green_cam_i = F.normalize(pre_axis_green + diff_axis_green, dim=-1)
                            est_axis_red_cam_i = F.normalize(pre_axis_red + diff_axis_red, dim=-1)
                            est_shape_code_i = pre_shape_code + diff_shape_code
                            print(f'frame_sequence_idx:{frame_sequence_idx}, frame_idx:{frame_idx}, optim_idx:{optim_idx}')
                    # else:
                    #     # Set variables with requires_grad = True.
                    #     axis_green_for_cal_grads = pre_axis_green.detach().clone()
                    #     axis_red_for_cal_grads = pre_axis_red.detach().clone()
                    #     shape_code_for_cal_grads = pre_shape_code.detach().clone()
                    #     axis_green_for_cal_grads.requires_grad = True
                    #     axis_red_for_cal_grads.requires_grad = True
                    #     shape_code_for_cal_grads.requires_grad = True

                    #     params = [axis_green_for_cal_grads, 
                    #               axis_red_for_cal_grads, 
                    #               shape_code_for_cal_grads]
                    #     lr = 0.01
                    #     optimizer = torch.optim.Adam(params, lr)

                    #     print('start')

                    #     model.grad_optim_max = 3
                    #     for grad_optim_idx in range(model.grad_optim_max):
                    #         optimizer.zero_grad()
                    #         rays_d_cam = model.rays_d_cam.expand(batch_size, -1, -1, -1).to(frame_camera_rot.device)
                    #         obj_pos_wrd = torch.zeros(batch_size, 3, device=frame_camera_rot.device)
                    #         est_invdepth_map, est_mask, est_depth_map = get_depth_map_from_axis(
                    #                                                         H = model.H, 
                    #                                                         axis_green = axis_green_for_cal_grads, 
                    #                                                         axis_red = axis_red_for_cal_grads,
                    #                                                         cam_pos_wrd = frame_camera_pos[:, frame_idx].detach(), 
                    #                                                         obj_pos_wrd = obj_pos_wrd.detach(), 
                    #                                                         rays_d_cam = rays_d_cam.detach(), 
                    #                                                         w2c = w2c.detach(), 
                    #                                                         input_lat_vec = shape_code_for_cal_grads, 
                    #                                                         ddf = model.ddf, 
                    #                                                         with_invdepth_map = True, 
                    #                                                         )
                    #         energy = torch.abs(est_invdepth_map - invdepth_map.detach()).mean(dim=-1).mean(dim=-1)
                    #         energy.backward()
                    #         optimizer.step()

                    #         print(energy)

                    #     pre_axis_green_wrd = torch.sum(pre_axis_green[..., None, :]*w2c.permute(0, 2, 1), -1)
                    #     pre_axis_red_wrd = torch.sum(pre_axis_red[..., None, :]*w2c.permute(0, 2, 1), -1)
                    #     # Stack final estimation on this frame.
                    #     if model.average_each_results:
                    #         # 現フレームに対する推論結果をスタックする。
                    #         frame_est_list['axis_green'].append(pre_axis_green_wrd.clone())
                    #         frame_est_list['axis_red'].append(pre_axis_red_wrd.clone())
                    #         frame_est_list['shape_code'].append(pre_shape_code.clone())
                    #         frame_est_list['error'].append(energy)
                    #     break

                    # with torch.no_grad():
                    #     # Get next inputs
                    #     pre_axis_green = est_axis_green_cam_i.detach()
                    #     pre_axis_red = est_axis_red_cam_i.detach()
                    #     pre_shape_code = est_shape_code_i.detach()
                    #     pre_axis_green_wrd = torch.sum(pre_axis_green[..., None, :]*w2c.permute(0, 2, 1), -1)
                    #     pre_axis_red_wrd = torch.sum(pre_axis_red[..., None, :]*w2c.permute(0, 2, 1), -1)
                    #     frame_est_list['axis_green'].append(pre_axis_green_wrd.clone())
                    #     frame_est_list['axis_red'].append(pre_axis_red_wrd.clone())
                    #     frame_est_list['shape_code'].append(pre_shape_code.clone())
                    #     model.average_each_results = False

                    # 最適化ステップ内の、ラムダステップを実行
                    with torch.no_grad():
                        for half_lambda_idx in range(model.half_lambda_max):
                            print(half_lambda_idx)
                            # Get simulation results.
                            rays_d_cam = model.rays_d_cam.expand(batch_size, -1, -1, -1).to(frame_camera_rot.device)
                            obj_pos_wrd = torch.zeros(batch_size, 3, device=frame_camera_rot.device)
                            est_mask, est_depth_map = get_depth_map_from_axis(
                                                            H = model.H, 
                                                            axis_green = est_axis_green_cam_i, 
                                                            axis_red = est_axis_red_cam_i,
                                                            cam_pos_wrd = frame_camera_pos[:, frame_idx], 
                                                            obj_pos_wrd = obj_pos_wrd, 
                                                            rays_d_cam = rays_d_cam, 
                                                            w2c = w2c, 
                                                            input_lat_vec = est_shape_code_i, 
                                                            ddf = model.ddf, 
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
                                over_lamda_step = half_lambda_idx + 1 == model.half_lambda_max
                                last_optim_step = optim_idx + 1 == model.test_optim_num[frame_sequence_idx]
                                not_last_frame = frame_sequence_idx < model.frame_sequence_num - 1
                                go_next_frame = last_optim_step and not_last_frame

                                if decade_all_error or over_lamda_step:
                                    # get next inputs
                                    pre_axis_green[update_mask] = est_axis_green_cam_i[update_mask].detach()
                                    pre_axis_red[update_mask] = est_axis_red_cam_i[update_mask].detach()
                                    pre_shape_code[update_mask] = est_shape_code_i[update_mask].detach()
                                    pre_mask[update_mask] = est_mask[update_mask].detach()
                                    pre_depth_map[update_mask] = est_depth_map[update_mask].detach()
                                    pre_error[update_mask] = error[update_mask].detach()

                                    if last_optim_step:
                                        # Get world rotation.
                                        pre_axis_green_wrd = torch.sum(pre_axis_green[..., None, :]*w2c.permute(0, 2, 1), -1)
                                        pre_axis_red_wrd = torch.sum(pre_axis_red[..., None, :]*w2c.permute(0, 2, 1), -1)
                                        # Stack final estimation on this frame.
                                        if model.average_each_results:
                                            # 現フレームに対する推論結果をスタックする。
                                            frame_est_list['axis_green'].append(pre_axis_green_wrd.clone())
                                            frame_est_list['axis_red'].append(pre_axis_red_wrd.clone())
                                            frame_est_list['shape_code'].append(pre_shape_code.clone())
                                            frame_est_list['error'].append(pre_error)
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
            if model.average_each_results:
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
            elif not model.average_each_results:
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
            rays_d_cam = model.rays_d_cam.expand(batch_size, -1, -1, -1).to(frame_camera_rot.device)
            obj_pos_wrd = torch.zeros(batch_size, 3, device=frame_camera_rot.device)
            est_mask, est_depth_map = get_depth_map_from_axis(
                                            H = model.H, 
                                            axis_green = est_axis_green, 
                                            axis_red = est_axis_red,
                                            cam_pos_wrd = frame_camera_pos[:, frame_idx], 
                                            obj_pos_wrd = obj_pos_wrd, 
                                            rays_d_cam = rays_d_cam, 
                                            w2c = w2c, 
                                            input_lat_vec = est_shape_code_i, 
                                            ddf = model.ddf, 
                                            )
                    
            # Check depth map.
            check_map_1 = []
            for batch_i in range(batch_size):
                check_map_i = torch.cat([depth_map[batch_i], est_depth_map[batch_i]], dim=0)
                check_map_1.append(check_map_i)
            check_map_1 = torch.cat(check_map_1, dim=1)
            check_map(check_map_1, f'check_batch_{str(batch_idx).zfill(5)}.png', figsize=[10,2])
            check_map_1 = []

        # Cal err.
        err_axis_green_i = torch.mean(-model.cossim(est_axis_green_wrd, gt_axis_green_wrd) + 1.)
        err_axis_red_i = torch.mean(-model.cossim(est_axis_red_wrd, gt_axis_red_wrd) + 1.)
        err_depth_i = F.mse_loss(est_depth_map, depth_map)
        err_axis_green.append(err_axis_green_i.detach())
        err_axis_red.append(err_axis_red_i.detach())
        err_depth.append(err_depth_i.detach())

    avg_err_axis_green = torch.stack(err_axis_green).mean()
    avg_err_axis_red = torch.stack(err_axis_red).mean()
    avg_err_depth = torch.stack(err_depth).mean()
    with open(model.test_log_path, 'a') as file:
        file.write('avg_err_axis_green : ' + str(avg_err_axis_green.item()) + '\n')
        file.write('avg_err_axis_red : ' + str(avg_err_axis_red.item()) + '\n')
        file.write('avg_err_depth : ' + str(avg_err_depth.item()) + '\n')
    
    return 0





if __name__=='__main__':
    # Get args
    args = get_args()
    args.gpu_num = torch.cuda.device_count() # log used gpu num.

    # Create dataloader.
    val_dataset = TaR_testset(
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

    # Create ddf.
    ddf = DDF(args)
    ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args)
    ddf.eval()

    # Create dfnet.
    args.use_gru = True
    df_net = TaR(args, ddf)
    checkpoint_path = './lightning_logs/DeepTaR/chair/test_dfnet_gru/checkpoints/0000003200.ckpt'
    df_net = df_net.load_from_checkpoint(
        checkpoint_path=checkpoint_path, 
        args=args, 
        ddf=ddf
        )
    df_net.eval()

    # Setting model.
    model = df_net
    model.test_mode = 'average'
    model.average_each_results = True
    model.start_frame_idx = 0
    model.frame_sequence_num = 3
    model.half_lambda_max = 8
    if model.test_mode == 'average':
        model.test_optim_num = [5, 5, 5]
    if model.test_mode == 'sequence':
        model.test_optim_num = [5, 3, 2]

    # Save logs.
    import datetime
    dt_now = datetime.datetime.now()
    time_log = dt_now.strftime('%Y_%m_%d_%H_%M_%S')

    file_name = time_log + '.txt'
    ckpt_path = checkpoint_path
    with open(file_name, 'a') as file:
        file.write('time_log : ' + time_log + '\n')
        file.write('ckpt_path : ' + ckpt_path + '\n')
    
    # Test model.
    test_model(model, val_dataloader)
