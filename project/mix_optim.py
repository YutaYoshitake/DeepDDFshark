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
        self.dynamic = target_model.dynamic
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
        self.frame_num = target_model.frame_num
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



    def training_step(self, batch, batch_idx):

        frame_mask, frame_distance_map, frame_camera_pos, frame_camera_rot, frame_obj_pos, frame_obj_rot, frame_obj_scale, instance_id = batch
        batch_size = len(instance_id)
        frame_est_list = {'pos_wrd':[], 'axis_green':[], 'axis_red':[], 'scale':[], 'shape_code':[], 'error':[]}

        # Set frame.
        for frame_sequence_idx in range(self.frame_sequence_num):
            frame_idx = self.start_frame_idx + frame_sequence_idx

            for optim_idx in range(self.test_optim_num[frame_sequence_idx]):

                # Clop distance map.
                raw_mask = frame_mask[:, frame_idx]
                raw_distance_map = frame_distance_map[:, frame_idx]
                clopped_mask, clopped_distance_map, bbox_list = clopping_distance_map(
                                                                    raw_mask, raw_distance_map, self.image_coord, self.input_H, self.input_W, self.ddf_H
                                                                    )
                gt_invdistance_map = torch.zeros_like(clopped_distance_map)
                gt_invdistance_map[clopped_mask] = 1. / clopped_distance_map[clopped_mask]

                # Get normalized depth map.
                rays_d_cam = get_clopped_rays_d_cam(self.ddf_H, self.fov, bbox_list).to(frame_camera_rot.device)
                clopped_depth_map, normalized_depth_map, avg_depth_map = get_normalized_depth_map(
                                                                            clopped_mask, clopped_distance_map, rays_d_cam
                                                                            )

                # Get ground truth.
                o2w = frame_obj_rot[:, frame_idx]
                w2c = frame_camera_rot[:, frame_idx]
                o2c = torch.bmm(w2c, o2w) # とりあえずこれを推論する
                gt_axis_green_cam = o2c[:, :, 1] # Y
                gt_axis_red_cam = o2c[:, :, 0] # X
                cam_pos_wrd = frame_camera_pos[:, frame_idx]
                gt_obj_pos_wrd = frame_obj_pos[:, frame_idx]
                gt_obj_scale = frame_obj_scale[:, frame_idx][:, None]

                # Estimating.
                if self.test_mode == 'average':
                    perform_init_est = optim_idx == 0
                elif self.test_mode == 'sequence':
                    perform_init_est = optim_idx == 0 and frame_sequence_idx==0

                # Get input.
                bbox_info = torch.cat([bbox_list.reshape(-1, 4), bbox_list.mean(1), avg_depth_map.to('cpu')[:, None]], dim=-1)
                if perform_init_est:
                    inp = torch.stack([normalized_depth_map, clopped_mask], 1)
                    est_x_cim, est_y_cim, est_z_diff, est_axis_green_cam, est_axis_red_cam, est_scale_diff, est_shape_code, pre_hidden_state = self.init_net(inp, bbox_info.to(inp))
                    est_obj_pos_cam, est_obj_scale, im2cam_scale = diff2estimation(est_x_cim, est_y_cim, est_z_diff, est_scale_diff, bbox_list, avg_depth_map, self.fov)
                est_obj_pos_wrd = torch.sum(est_obj_pos_cam[..., None, :]*w2c.permute(0, 2, 1), dim=-1) + cam_pos_wrd

                # Get simulation results.
                est_invdistance_map, est_mask, est_distance_map = render_distance_map_from_axis(
                                                H = self.ddf_H, 
                                                obj_pos_wrd = est_obj_pos_wrd, # gt_obj_pos_wrd, 
                                                axis_green = est_axis_green_cam, # gt_axis_green_cam, 
                                                axis_red = est_axis_red_cam, # gt_axis_red_cam, 
                                                obj_scale = est_obj_scale[:, 0], # gt_obj_scale[:, 0].to(est_obj_scale), 
                                                input_lat_vec = est_shape_code, # gt_shape_code, 
                                                cam_pos_wrd = cam_pos_wrd, 
                                                rays_d_cam = rays_d_cam, 
                                                w2c = w2c.detach(), 
                                                ddf = self.ddf, 
                                                with_invdistance_map = True, 
                                                )
                est_depth_map, est_normalized_depth_map, _ = get_normalized_depth_map(
                                                                est_mask, est_distance_map, rays_d_cam, avg_depth_map, 
                                                                )

                # 最初のフレームの初期予測
                # 最適化のラムダステップはなく、そのまま次の最適化ステップへ
                if perform_init_est:
                    # Get next inputs
                    pre_obj_pos_wrd = est_obj_pos_wrd.detach()
                    pre_obj_scale = est_obj_scale.detach()
                    pre_axis_green_cam = est_axis_green_cam.detach()
                    pre_axis_red_cam = est_axis_red_cam.detach()
                    pre_shape_code = est_shape_code.detach()
                    pre_mask = est_mask.detach()
                    pre_depth_map = est_normalized_depth_map.detach()
                    pre_error = torch.abs(pre_depth_map - normalized_depth_map).mean(dim=-1).mean(dim=-1)

                    # Check pre results.
                    check_map = []
                    gt = normalized_depth_map
                    est = pre_depth_map
                    for i in range(batch_size):
                        check_map.append(torch.cat([gt[i], est[i], torch.abs(gt[i]-est[i])], dim=0))
                    check_map_torch(torch.cat(check_map, dim=-1))
                    
                    # 初期化ネットだけの性能を評価する場合。
                    if self.only_init_net:
                        # 現フレームに対する推論結果をスタックする。
                        pre_axis_green_wrd = torch.sum(pre_axis_green_cam[..., None, :]*w2c.permute(0, 2, 1), -1)
                        pre_axis_red_wrd = torch.sum(pre_axis_red_cam[..., None, :]*w2c.permute(0, 2, 1), -1)
                        frame_est_list['pos_wrd'].append(pre_obj_pos_wrd.clone())
                        frame_est_list['scale'].append(pre_obj_scale.clone())
                        frame_est_list['axis_green'].append(pre_axis_green_wrd.clone())
                        frame_est_list['axis_red'].append(pre_axis_red_wrd.clone())
                        frame_est_list['shape_code'].append(pre_shape_code.clone())
                        frame_est_list['error'].append(pre_error) # 現フレームに対するエラーを見たい。
                        break
        
        est_obj_pos_wrd = get_weighted_average(
                                target = torch.stack(frame_est_list['pos_wrd'], dim=1).detach(), 
                                ratio = 1/torch.stack(frame_est_list['error'], dim=1).detach())
        est_obj_scale = get_weighted_average(
                                target = torch.stack(frame_est_list['scale'], dim=1).detach(), 
                                ratio = 1/torch.stack(frame_est_list['error'], dim=1).detach())
        est_axis_green_wrd = get_weighted_average(
                                target = torch.stack(frame_est_list['axis_green'], dim=1).detach(), 
                                ratio = 1/torch.stack(frame_est_list['error'], dim=1).detach())
        est_axis_green_wrd = F.normalize(est_axis_green_wrd, dim=1)
        est_axis_red_wrd = get_weighted_average(
                                target = torch.stack(frame_est_list['axis_red'], dim=1).detach(), 
                                ratio = 1/torch.stack(frame_est_list['error'], dim=1).detach())
        est_axis_red_wrd = F.normalize(est_axis_red_wrd, dim=1)
        est_shape_code = get_weighted_average(
                                target = torch.stack(frame_est_list['shape_code'], dim=1).detach(), 
                                ratio = 1/torch.stack(frame_est_list['error'], dim=1).detach())


        # est_obj_pos_wrd = frame_est_list['pos_wrd']
        # est_obj_scale = frame_est_list['scale']
        # est_axis_green_wrd = frame_est_list['axis_green']
        # est_axis_red_wrd = frame_est_list['axis_red']
        # est_axis_green_wrd = F.normalize(est_axis_green_wrd, dim=1)
        # est_axis_red_wrd = F.normalize(est_axis_red_wrd, dim=1)
        # est_shape_code = frame_est_list['shape_code'][0]



        # Cal err.
        err_pos = F.mse_loss(est_obj_pos_wrd, gt_obj_pos_wrd)
        err_scale = torch.mean(est_obj_scale / gt_obj_scale.to(est_obj_scale))
        err_axis_red = torch.mean(-self.cossim(est_axis_red_cam, torch.sum(gt_axis_red_cam[..., None, :]*w2c.permute(0, 2, 1), -1)) + 1.)



        ###########################################################################
        #########################      check result       #########################
        ###########################################################################
        # Set random frames.
        frame_idx = -1 # random.randint(0, frame_rgb_map.shape[1]-1)
        raw_mask = frame_mask[:, frame_idx]
        raw_distance_map = frame_distance_map[:, frame_idx]

        # Get camera poses.
        w2c = frame_camera_rot[:, frame_idx]
        cam_pos_wrd = frame_camera_pos[:, frame_idx]
        rays_d_cam = get_ray_direction(self.ddf_H, self.fov).expand(batch_size, -1, -1, -1).to(frame_camera_rot.device)
        
        # Get simulation results.
        est_axis_green_cam = torch.sum(est_axis_green_wrd[..., None, :]*w2c, -1)
        est_axis_red_cam = torch.sum(est_axis_red_wrd[..., None, :]*w2c, -1)
        est_invdistance_map, est_mask, est_distance_map = render_distance_map_from_axis(
                                        H = self.ddf_H, 
                                        obj_pos_wrd = est_obj_pos_wrd, # gt_obj_pos_wrd, 
                                        axis_green = est_axis_green_cam, # gt_axis_green_cam, 
                                        axis_red = est_axis_red_cam, # gt_axis_red_cam, 
                                        obj_scale = est_obj_scale[:, 0], # gt_obj_scale[:, 0].to(est_obj_scale), 
                                        input_lat_vec = est_shape_code, # gt_shape_code, 
                                        cam_pos_wrd = cam_pos_wrd, 
                                        rays_d_cam = rays_d_cam, 
                                        w2c = w2c.detach(), 
                                        ddf = self.ddf, 
                                        with_invdistance_map = True, 
                                        )

        check_map = []
        gt = raw_distance_map
        est = est_distance_map
        for i in range(batch_size):
            check_map.append(torch.cat([gt[i], est[i], torch.abs(gt[i]-est[i])], dim=0))
        check_map_torch(torch.cat(check_map, dim=-1))
        import pdb; pdb.set_trace()

        return {'err_pos':err_pos.detach(), 'err_scale': err_scale.detach(), 'err_axis_red': err_axis_red.detach()}



    # def test_epoch_end(self, outputs):
    #     # Log loss.
    #     avg_err_axis_green = torch.stack([x['err_axis_green'] for x in outputs]).mean()
    #     avg_err_axis_red = torch.stack([x['err_axis_red'] for x in outputs]).mean()
    #     avg_err_depth = torch.stack([x['err_depth'] for x in outputs]).mean()

    #     with open(self.test_log_path, 'a') as file:
    #         file.write('avg_err_axis_green : ' + str(avg_err_axis_green.item()) + '\n')
    #         file.write('avg_err_axis_red : ' + str(avg_err_axis_red.item()) + '\n')
    #         file.write('avg_err_depth : ' + str(avg_err_depth.item()) + '\n')



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
    # if args.xxx=='a':
    args.use_gru = False
    df_net = TaR(args, ddf)
    checkpoint_path = './lightning_logs/DeepTaR/chair/dfnet_optN3/checkpoints/0000001150.ckpt'
    df_net = df_net.load_from_checkpoint(
        checkpoint_path=checkpoint_path, 
        args=args, 
        ddf=ddf
        )
    df_net.eval()

    # # Create init net.
    # init_net = TaR_init_only(args, ddf)
    # checkpoint_path='./lightning_logs/DeepTaR/chair/initnet_first/checkpoints/0000001000.ckpt'
    # init_net = init_net.load_from_checkpoint(
    #     checkpoint_path=checkpoint_path, 
    #     args=args, 
    #     ddf=ddf
    #     ).model
    # df_net.init_net = init_net
    df_net.only_init_net = True

    # Setting model.
    model = df_net
    model.test_mode = 'average'
    model.start_frame_idx = 0
    model.frame_sequence_num = 1
    model.half_lambda_max = 8
    if model.test_mode == 'average':
        model.test_optim_num = [5, 5, 5]
    if model.test_mode == 'sequence':
        model.test_optim_num = [5, 3, 2]
    if model.only_init_net:
        model.test_mode = 'average'
        model.test_optim_num = [1, 1, 1, 1, 1, 1, 1]
    model.use_deep_optimizer = False
    model.use_adam_optimizer = True
    model.use_weighted_average = True

    # Save logs.
    import datetime
    dt_now = datetime.datetime.now()
    time_log = dt_now.strftime('%Y_%m_%d_%H_%M_%S')

    file_name = './txt/experiments/log/' + time_log + '.txt'
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
        check_val_every_n_epoch = args.check_val_every_n_epoch,
        logger = pl.loggers.TensorBoardLogger(save_dir=os.getcwd(), version=f'val_trash', name='lightning_logs')
        )
    trainer.fit(test_model, train_dataloaders=val_dataloader)
    trainer.test(test_model, val_dataloader)
