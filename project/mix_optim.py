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





class test_TaR(pl.LightningModule):

    def __init__(self, args, target_model):
        super().__init__()

        # Base configs
        self.H = args.H
        self.fov = args.fov
        self.lr = args.lr
        self.rays_d_cam = get_ray_direction(self.H, self.fov)
        self.save_interval = args.save_interval
        self.test_mode = target_model.test_mode
        self.use_weighted_average = target_model.use_weighted_average
        self.start_frame_idx = target_model.start_frame_idx
        self.frame_sequence_num = target_model.frame_sequence_num
        self.half_lambda_max = target_model.half_lambda_max
        self.test_optim_num = target_model.test_optim_num
        self.test_mode = target_model.test_mode
        self.test_log_path = target_model.test_log_path

        # Make model
        self.ddf = target_model.ddf
        self.init_net = target_model.init_net
        self.use_gru = target_model.use_gru
        self.df_net = target_model.df_net

        # loss func.
        self.l1 = torch.nn.L1Loss()
        self.cossim = nn.CosineSimilarity(dim=-1)



    def test_step(self, batch, batch_idx):

        frame_rgb_map, frame_mask, frame_depth_map, frame_camera_pos, frame_camera_rot, frame_obj_rot, instance_id = batch
        batch_size = len(instance_id)
        frame_est_list = {'axis_green':[], 'axis_red':[], 'shape_code':[], 'error':[]}

        # ランダムなあるフレームからframe_sequence_num個を取得
        for frame_sequence_idx in range(self.frame_sequence_num):
            frame_idx = self.start_frame_idx + frame_sequence_idx

            for optim_idx in range(self.test_optim_num[frame_sequence_idx]):

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
                if self.test_mode == 'average':
                    perform_init_est = optim_idx == 0
                elif self.test_mode == 'sequence':
                    perform_init_est = optim_idx == 0 and frame_sequence_idx==0

                if perform_init_est:
                    inp = torch.stack([depth_map, mask], 1)
                    if self.use_gru:
                        est_axis_green_cam_i, est_axis_red_cam_i, est_shape_code_i, feature_vec = self.init_net(inp, self.use_gru)
                        pre_hidden_state = feature_vec
                    else:
                        est_axis_green_cam_i, est_axis_red_cam_i, est_shape_code_i = self.init_net(inp, self.use_gru)
                else:
                    inp = torch.stack([depth_map, mask, pre_depth_map, pre_mask, depth_map - pre_depth_map], 1)
                    if self.use_gru:
                        print('use gru model')
                        diff_axis_green, diff_axis_red, diff_shape_code, feature_vec, pre_hidden_state = self.df_net(inp, pre_axis_green, pre_axis_red, pre_shape_code, pre_hidden_state)
                    else:
                        diff_axis_green, diff_axis_red, diff_shape_code = self.df_net(inp, pre_axis_green, pre_axis_red, pre_shape_code)
                    est_axis_green_cam_i = F.normalize(pre_axis_green + diff_axis_green, dim=-1)
                    est_axis_red_cam_i = F.normalize(pre_axis_red + diff_axis_red, dim=-1)
                    est_shape_code_i = pre_shape_code + diff_shape_code
                    # Check inputs.
                    print(f'frame_sequence_idx:{frame_sequence_idx}, frame_idx:{frame_idx}, optim_idx:{optim_idx}')
                    # check_map_inp = torch.cat([
                    #     depth_map, 
                    #     pre_depth_map, 
                    #     torch.abs(depth_map - pre_depth_map)], dim=2)
                    # check_map(check_map_inp[0], f'input_frame_{frame_sequence_idx}_opt_{optim_idx}.png', figsize=[10,2])

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

                    # 最初のフレームの初期予測
                    # 最適化のラムダステップはなく、そのまま次の最適化ステップへ
                    if perform_init_est:
                        # Get next inputs
                        pre_axis_green = est_axis_green_cam_i.detach()
                        pre_axis_red = est_axis_red_cam_i.detach()
                        pre_shape_code = est_shape_code_i.detach()
                        pre_mask = est_mask.detach()
                        pre_depth_map = est_depth_map.detach()
                        pre_error = torch.abs(pre_depth_map - depth_map).mean(dim=-1).mean(dim=-1)

                        # check_map_1 = []
                        # for batch_idx in range(batch_size):
                        #     check_map_i = torch.cat([depth_map[batch_idx], est_depth_map[batch_idx]], dim=0)
                        #     check_map_1.append(check_map_i)
                        # check_map_1 = torch.cat(check_map_1, dim=1)
                        # check_map(check_map_1, 'check_map_1.png', figsize=[10,2])
                        # check_map_1 = []
                        # print('ccc')
                        # import pdb; pdb.set_trace()
                        break

                    # 最適化のラムダステップ
                    else:
                        # エラーを計算
                        error = torch.abs(est_depth_map - depth_map).mean(dim=-1).mean(dim=-1)
                        un_update_mask = (pre_error - error) < 0. #エラーが大きくなった場合、True
                        update_mask = torch.logical_not(un_update_mask)
                        # print(f'optim_idx:{optim_idx}')
                        # print(f'half_lambda_idx:{half_lambda_idx}')
                        # print(f'un_update_mask:{un_update_mask}')
                        # print(f'pre_error:{pre_error}')
                        # print(f'error:{error}')

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

                            # 現フレームに対する最適化が最終ステップ
                            # -> 推定結果を保存する。
                            if last_optim_step:
                                pre_axis_green_wrd = torch.sum(pre_axis_green[..., None, :]*w2c.permute(0, 2, 1), -1)
                                pre_axis_red_wrd = torch.sum(pre_axis_red[..., None, :]*w2c.permute(0, 2, 1), -1)

                                # 現フレームに対する推論結果をスタックする。
                                frame_est_list['axis_green'].append(pre_axis_green_wrd.clone())
                                frame_est_list['axis_red'].append(pre_axis_red_wrd.clone())
                                frame_est_list['shape_code'].append(pre_shape_code.clone())
                                frame_est_list['error'].append(pre_error) # 現フレームに対するエラーを見たい。

                                # 現フレームまでの推論結果を平均し、それを元に次フレームを予測。
                                if self.use_weighted_average:
                                    pre_axis_green_wrd = get_weighted_average(
                                                            target = torch.stack(frame_est_list['axis_green'], dim=1).detach(), 
                                                            ratio = 1/torch.stack(frame_est_list['error'], dim=1).detach())
                                    pre_axis_red_wrd = get_weighted_average(
                                                            target = torch.stack(frame_est_list['axis_red'], dim=1).detach(), 
                                                            ratio = 1/torch.stack(frame_est_list['error'], dim=1).detach())
                                    pre_axis_green_wrd = F.normalize(pre_axis_green_wrd, dim=1)
                                    pre_axis_red_wrd = F.normalize(pre_axis_red_wrd, dim=1)
                                    pre_shape_code = get_weighted_average(
                                                            target = torch.stack(frame_est_list['shape_code'], dim=1).detach(), 
                                                            ratio = 1/torch.stack(frame_est_list['error'], dim=1).detach())

                            # 最適化の最後には次のフレームを予測 -> カメラポーズが変化する。
                            if go_next_frame and self.test_mode == 'sequence':
                                # Set next frame.
                                next_frame_idx = frame_idx + 1
                                next_w2c = frame_camera_rot[:, next_frame_idx]
                                next_depth_map = frame_depth_map[:, next_frame_idx]
                                
                                # get next inputs : pre_*.
                                pre_axis_green = torch.sum(pre_axis_green_wrd[..., None, :]*next_w2c, -1)
                                pre_axis_green = pre_axis_green.to(pre_shape_code.dtype)
                                pre_axis_red = torch.sum(pre_axis_red_wrd[..., None, :]*next_w2c, -1)
                                pre_axis_red = pre_axis_red.to(pre_shape_code.dtype)
                                pre_shape_code = pre_shape_code.detach()

                                # Estimate next frame.
                                rays_d_cam = self.rays_d_cam.expand(batch_size, -1, -1, -1).to(frame_camera_rot.device)
                                obj_pos_wrd = torch.zeros(batch_size, 3, device=frame_camera_rot.device)
                                pre_mask, pre_depth_map = get_depth_map_from_axis(
                                                                H = self.H, 
                                                                axis_green = pre_axis_green, 
                                                                axis_red = pre_axis_red,
                                                                cam_pos_wrd = frame_camera_pos[:, next_frame_idx], 
                                                                obj_pos_wrd = obj_pos_wrd, 
                                                                rays_d_cam = rays_d_cam, 
                                                                w2c = next_w2c, 
                                                                input_lat_vec = pre_shape_code, 
                                                                ddf = self.ddf, 
                                                                )
                                
                                # Cal the current error of the next frame.
                                pre_error = torch.abs(pre_depth_map - next_depth_map).mean(dim=-1).mean(dim=-1)
                                # check_map_1 = []
                                # for batch_idx in range(batch_size):
                                #     check_map_i = torch.cat([next_depth_map[batch_idx], est_depth_map[batch_idx]], dim=0)
                                #     check_map_1.append(check_map_i)
                                # check_map_1 = torch.cat(check_map_1, dim=1)
                                # check_map(check_map_1, 'check_map_1.png', figsize=[10,2])
                                # print('aaa')
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
                            # check_map_1 = []
                            # for batch_idx in range(batch_size):
                            #     check_map_i = torch.cat([depth_map[batch_idx], est_depth_map[batch_idx]], dim=0)
                            #     check_map_1.append(check_map_i)
                            # check_map_1 = torch.cat(check_map_1, dim=1)
                            # check_map(check_map_1, 'check_map_1.png', figsize=[10,2])
                            # import pdb; pdb.set_trace()

        # 各フレームの結果を融合する
        if self.use_weighted_average:
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
        elif not self.use_weighted_average:
            if self.test_mode == 'average':
                est_axis_green_wrd = torch.stack(frame_est_list['axis_green'], dim=1).mean(dim=1)
                est_axis_red_wrd = torch.stack(frame_est_list['axis_red'], dim=1).mean(dim=1)
                est_axis_green_wrd = F.normalize(est_axis_green_wrd, dim=1)
                est_axis_red_wrd = F.normalize(est_axis_red_wrd, dim=1)
                est_shape_code = torch.stack(frame_est_list['shape_code'], dim=1).mean(dim=1)
            elif self.test_mode == 'sequence':
                est_axis_green_wrd = torch.sum(est_axis_green_cam_i[..., None, :]*w2c.permute(0, 2, 1), -1)
                est_axis_red_wrd = torch.sum(est_axis_red_cam_i[..., None, :]*w2c.permute(0, 2, 1), -1)
                est_axis_green_wrd = F.normalize(est_axis_green_wrd, dim=1)
                est_axis_red_wrd = F.normalize(est_axis_red_wrd, dim=1)
                est_shape_code = est_shape_code_i



        ###########################################################################
        #########################      check result       #########################
        ###########################################################################
        # Set random frames.
        frame_idx = -1 # 2
        # frame_idx = random.randint(0, frame_rgb_map.shape[1]-1)
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
                
        # Check depth map.
        check_map_1 = []
        for batch_i in range(batch_size):
            check_map_i = torch.cat([
                depth_map[batch_i], 
                est_depth_map[batch_i], 
                torch.abs(depth_map[batch_i]-est_depth_map[batch_i])], dim=0)
            check_map_1.append(check_map_i)
        check_map_1 = torch.cat(check_map_1, dim=1)
        check_map(check_map_1, f'check_batch_{str(batch_idx).zfill(5)}.png', figsize=[10,2])

        # Cal err.
        err_axis_green = torch.mean(-self.cossim(est_axis_green_wrd, gt_axis_green_wrd) + 1.)
        err_axis_red = torch.mean(-self.cossim(est_axis_red_wrd, gt_axis_red_wrd) + 1.)
        err_depth = self.l1(est_depth_map, depth_map)

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
    args.use_gru = False
    df_net = TaR(args, ddf)
    checkpoint_path = './lightning_logs/DeepTaR/chair/test_dfnet_withx/checkpoints/0000003200.ckpt'
    df_net = df_net.load_from_checkpoint(
        checkpoint_path=checkpoint_path, 
        args=args, 
        ddf=ddf
        )
    df_net.eval()

    # # Create init net.
    # init_net = TaR_init_only(args, ddf)
    # init_net = init_net.load_from_checkpoint(
    #     checkpoint_path='./lightning_logs/DeepTaR/chair/test_initnet_0/checkpoints/0000005000.ckpt', 
    #     args=args, 
    #     ddf=ddf
    #     ).model
    # df_net.init_net = init_net

    # Setting model.
    model = df_net
    model.test_mode = 'average'
    model.use_weighted_average = False
    model.start_frame_idx = 0
    model.frame_sequence_num = 3
    model.half_lambda_max = 3
    if model.test_mode == 'average':
        model.test_optim_num = [5, 5, 5]
    if model.test_mode == 'sequence':
        model.test_optim_num = [5, 3, 2]

    # Save logs.
    import datetime
    dt_now = datetime.datetime.now()
    time_log = dt_now.strftime('%Y_%m_%d_%H_%M_%S')

    file_name = time_log + '.txt'
    model.test_log_path = file_name
    ckpt_path = checkpoint_path
    with open(file_name, 'a') as file:
        file.write('time_log : ' + time_log + '\n')
        file.write('ckpt_path : ' + ckpt_path + '\n')
    
    # Test model.
    test_model = test_TaR(args, model)
    trainer = pl.Trainer(
        gpus=args.gpu_num, 
        strategy=DDPPlugin(find_unused_parameters=True), #=False), 
        enable_checkpointing = False,
        )
    trainer.test(test_model, val_dataloader)
