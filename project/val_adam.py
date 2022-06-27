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





class adam_optimizer(pl.LightningModule):

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
        self.optim_num = args.frame_sequence_num
        self.itr_frame_num = 3

        # Make model
        self.ddf = ddf
        from train_ori import initializer, deep_optimizer
        self.init_net = initializer(args, in_channel=2) #init_net
        self.df_net = deep_optimizer(args, in_channel=5)
        self.adam_step_ratio = model.adam_step_ratio
        self.grad_optim_max = model.grad_optim_max
        self.shape_code_reg = model.shape_code_reg

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


    def test_step(self, batch, batch_idx):
        batch_size, frame_raw_invdistance_map, frame_clopped_mask, \
        frame_clopped_distance_map, frame_bbox_list, frame_rays_d_cam, \
        frame_clopped_depth_map, frame_normalized_depth_map, frame_avg_depth_map, frame_bbox_info, \
        frame_w2c, frame_gt_obj_axis_green_wrd, frame_gt_obj_axis_red_wrd, \
        frame_camera_pos, frame_obj_pos, frame_obj_scale, \
        canonical_distance_map, canonical_camera_pos, canonical_camera_rot, instance_id, path = self.preprocess(batch, mode='val')


        ###################################
        #####     Start training      #####
        ###################################
        # Set frames
        frame_idx_list = list(range(self.frame_sequence_num)) # 初期化は全フレームに対する結果から行う
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
        #####     Perform initnet.    #####
        ###################################
        # print('ini')
        inp = torch.stack([normalized_depth_map, clopped_mask], 1).detach()
        est_obj_pos_cim, est_obj_axis_green_cam, est_obj_axis_red_cam, est_scale_cim, est_shape_code, _ = self.init_net(inp, bbox_info)
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
        est_obj_pos_wrd = est_obj_pos_wrd.reshape(batch_size, -1, 3).mean(1)
        est_obj_scale = est_obj_scale.reshape(batch_size, -1, 1).mean(1)
        est_obj_axis_green_wrd = est_obj_axis_green_wrd.reshape(batch_size, -1, 3).mean(1)
        est_obj_axis_red_wrd = est_obj_axis_red_wrd.reshape(batch_size, -1, 3).mean(1)
        est_shape_code = est_shape_code.reshape(batch_size, -1, self.ddf.latent_size).mean(1)

        # Get scale lists.
        cim2im_scale_list = cim2im_scale.reshape(batch_size, self.frame_sequence_num)
        im2cam_scale_list = im2cam_scale.reshape(batch_size, self.frame_sequence_num)
        bbox_center_list = bbox_center.reshape(batch_size, self.frame_sequence_num, 2)

        ###################################
        #####      Perform adam.     #####
        ###################################
        # Set variables with grad = True.
        import pdb; pdb.set_trace()
        torch.set_grad_enabled(True)
        obj_pos_wrd_optim = est_obj_pos_wrd.detach().clone()
        obj_scale_optim = est_obj_scale.detach().clone()
        obj_axis_green_wrd_optim = est_obj_axis_green_wrd.detach().clone()
        obj_axis_red_wrd_optim = est_obj_axis_red_wrd.detach().clone()
        shape_code_optim = est_shape_code.detach().clone()
        obj_pos_wrd_optim.requires_grad = True
        obj_scale_optim.requires_grad = True
        obj_axis_green_wrd_optim.requires_grad = True
        obj_axis_red_wrd_optim.requires_grad = True
        shape_code_optim.requires_grad = True

        # Set optimizer.
        params = [obj_pos_wrd_optim, obj_scale_optim, obj_axis_green_wrd_optim, obj_axis_red_wrd_optim, shape_code_optim]
        optimizer = torch.optim.Adam(params, self.adam_step_ratio)

        for grad_optim_idx in range(self.grad_optim_max):
            # Set frames
            frame_idx_list = random.sample(list(range(self.frame_sequence_num)), self.itr_frame_num) # それ以外ではランダムに抽出
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

            # Get input for each frame.
            pos_wrd_frame = obj_pos_wrd_optim[:, None, :].expand(-1, opt_frame_num, 3).reshape(-1, 3)
            scale_frame = obj_scale_optim[:, None, :].expand(-1, opt_frame_num, 1).reshape(-1, 1)
            axis_green_frame = torch.sum(F.normalize(obj_axis_green_wrd_optim, dim=1)[:, None, :].expand(-1, opt_frame_num, 3).reshape(-1, 3)[..., None, :]*w2c, -1)
            axis_red_frame = torch.sum(F.normalize(obj_axis_red_wrd_optim, dim=1)[:, None, :].expand(-1, opt_frame_num, 3).reshape(-1, 3)[..., None, :]*w2c, -1)
            shape_code_frame = shape_code_optim[:, None, :].expand(-1, opt_frame_num, self.ddf.latent_size).reshape(-1, self.ddf.latent_size)
            
            rays_d_cam = self.rays_d_cam.expand(batch_size*opt_frame_num, -1, -1, -1).to(w2c)
            est_invdistance_map, est_mask = render_distance_map_from_axis(
                                                    H = self.ddf_H, 
                                                    obj_pos_wrd = pos_wrd_frame, 
                                                    obj_scale = scale_frame[:, 0], 
                                                    axis_green = axis_green_frame, 
                                                    axis_red = axis_red_frame, 
                                                    cam_pos_wrd = cam_pos_wrd.detach(), 
                                                    rays_d_cam = rays_d_cam.detach(),  
                                                    w2c = w2c.detach(), 
                                                    input_lat_vec = shape_code_frame, 
                                                    ddf = self.ddf, 
                                                    with_invdistance_map = True, 
                                                    )
            energy = self.l1(est_invdistance_map, raw_invdistance_map.detach()) # + self.shape_code_reg * torch.norm(shape_code_optim) # 0.5?
            energy.backward()
            optimizer.step()

            ###################################
            # if grad_optim_idx%10==0:
            # if grad_optim_idx==30:
            check_map = []
            gt = raw_invdistance_map
            est = est_invdistance_map
            for i in range(batch_size*opt_frame_num):
                check_map.append(torch.cat([gt[i], est[i], torch.abs(gt[i]-est[i])], dim=0))
            # check_map_torch(torch.cat(check_map, dim=-1), f'opt_{grad_optim_idx}.png')
            check_map_torch(torch.cat(check_map, dim=-1), f'tes.png')
            import pdb; pdb.set_trace()
            ###################################

        torch.set_grad_enabled(False)
        pre_obj_pos_wrd = obj_pos_wrd_optim.detach().clone()
        pre_obj_scale = obj_scale_optim.detach().clone()
        pre_obj_axis_green_wrd = F.normalize(obj_axis_green_wrd_optim, dim=1).detach().clone()
        pre_obj_axis_red_wrd = F.normalize(obj_axis_red_wrd_optim, dim=1).detach().clone()
        pre_shape_code = shape_code_optim.detach().clone()


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
            import pdb; pdb.set_trace()


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
                'pos':est_obj_pos_wrd.detach(), 
                'scale':est_obj_scale.detach(), 
                'axis_red':est_obj_axis_red_wrd.detach(), 
                'axis_green':est_obj_axis_green_wrd.detach(), 
                'shape_code':est_shape_code.detach()}



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

    # Set models and Start training.
    ddf = DDF(args)
    ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args)
    ddf.eval()
    model = adam_optimizer(args, ddf)
    model = model.load_from_checkpoint(checkpoint_path=args.model_ckpt_path, args=args, ddf=ddf)
    ###########################################################################
    from train_pro import progressive_optimizer
    model_ = progressive_optimizer(args, ddf)
    init_net_skpt = 'lightning_logs/DeepTaR/chair/progressive_list0_0621/checkpoints/0000000500.ckpt'
    model_ = model_.load_from_checkpoint(checkpoint_path=init_net_skpt, args=args, ddf=ddf)
    model.init_net = model_.init_net
    del model_
    ###########################################################################

    # Setting model.
    model.start_frame_idx = 0
    model.half_lambda_max = 3
    model.grad_optim_max = 50
    model.shape_code_reg = 0.0
    model.adam_step_ratio = 0.01
    model.model_mode = args.model_mode
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
        file.write('start_frame_idx : ' + str(model.start_frame_idx) + '\n')
        file.write('frame_sequence_num : ' + str(model.frame_sequence_num) + '\n')
        file.write('grad_optim_max : ' + str(model.grad_optim_max) + '\n')
        file.write('shape_code_reg : ' + str(model.shape_code_reg) + '\n')
        file.write('adam_step_ratio : ' + str(model.adam_step_ratio) + '\n')
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
    