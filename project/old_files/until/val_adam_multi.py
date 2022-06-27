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
        self.model_params_dtype = target_model.model_params_dtype
        self.model_device = target_model.model_device
        self.use_depth_error = target_model.use_depth_error
        self.start_frame_idx = target_model.start_frame_idx
        self.frame_sequence_num = target_model.frame_sequence_num
        self.test_log_path = target_model.test_log_path
        self.adam_step_ratio = target_model.adam_step_ratio
        self.grad_optim_max = target_model.grad_optim_max
        self.shape_code_reg = target_model.shape_code_reg

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

        ###################################
        list_for_vis = []
        ###################################

        frame_mask, frame_distance_map, frame_camera_pos, frame_camera_rot, frame_obj_pos, frame_obj_rot, frame_obj_scale, \
            canonical_distance_map, canonical_camera_pos, canonical_camera_rot, instance_id, path = batch
        batch_size = len(instance_id)

        ###########################################################################
        #########################           EST           #########################
        ###########################################################################
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
        raw_invdistance_map = torch.zeros_like(raw_distance_map)
        raw_invdistance_map[raw_mask] = 1. / raw_distance_map[raw_mask]

        # Get normalized depth map.
        rays_d_cam = get_clopped_rays_d_cam(self.ddf_H, self.fov, bbox_list).to(frame_camera_rot.device)
        clopped_depth_map, normalized_depth_map, avg_depth_map = get_normalized_depth_map(
                                                                    clopped_mask, clopped_distance_map, rays_d_cam
                                                                    )

        # Get ground truth.
        o2w = frame_obj_rot[:, start_frame_idx:end_frame_idx].reshape(-1, 3, 3)
        w2c = frame_camera_rot[:, start_frame_idx:end_frame_idx].reshape(-1, 3, 3)
        o2c = torch.bmm(w2c, o2w) # とりあえずこれを推論する
        gt_obj_axis_green_cam = o2c[:, :, 1] # Y
        gt_obj_axis_red_cam = o2c[:, :, 0] # X
        gt_axis_green_wrd = torch.sum(gt_obj_axis_green_cam[..., None, :]*w2c.permute(0, 2, 1), -1).reshape(batch_size, using_frame_num, 3)
        gt_axis_red_wrd = torch.sum(gt_obj_axis_red_cam[..., None, :]*w2c.permute(0, 2, 1), -1).reshape(batch_size, using_frame_num, 3)
        cam_pos_wrd = frame_camera_pos[:, start_frame_idx:end_frame_idx].reshape(-1, 3)
        gt_obj_pos_wrd = frame_obj_pos[:, start_frame_idx:end_frame_idx]
        gt_obj_scale = frame_obj_scale[:, start_frame_idx:end_frame_idx][:, :, None]

        # initial estimation.
        bbox_info = torch.cat([bbox_list.reshape(-1, 4), bbox_list.mean(1), avg_depth_map.to('cpu')[:, None]], dim=-1)
        inp = torch.stack([normalized_depth_map, clopped_mask], 1)
        est_obj_pos_cim, est_obj_axis_green_cam, est_obj_axis_red_cam, est_scale_cim, est_shape_code, pre_hidden_state = self.init_net(inp, bbox_info.to(inp))
        est_obj_pos_cam, est_obj_scale = diff2estimation(est_obj_pos_cim, est_scale_cim, bbox_list, avg_depth_map, self.fov)
        est_obj_pos_wrd = torch.sum(est_obj_pos_cam[..., None, :]*w2c.permute(0, 2, 1), dim=-1) + cam_pos_wrd
        est_obj_axis_green_wrd = torch.sum(est_obj_axis_green_cam[..., None, :]*w2c.permute(0, 2, 1), -1)
        est_obj_axis_red_wrd = torch.sum(est_obj_axis_red_cam[..., None, :]*w2c.permute(0, 2, 1), -1)

        # Set variables with grad = True.
        torch.set_grad_enabled(True)
        obj_pos_wrd_optim = est_obj_pos_wrd.reshape(batch_size, using_frame_num, 3).mean(1).detach().clone()
        obj_scale_optim = est_obj_scale.reshape(batch_size, using_frame_num, 1).mean(1).detach().clone()
        obj_axis_green_wrd_optim = est_obj_axis_green_wrd.reshape(batch_size, using_frame_num, 3).mean(1).detach().clone()
        obj_axis_red_wrd_optim = est_obj_axis_red_wrd.reshape(batch_size, using_frame_num, 3).mean(1).detach().clone()
        shape_code_optim = est_shape_code.reshape(batch_size, using_frame_num, self.ddf.latent_size).mean(1).detach().clone()
        obj_pos_wrd_optim.requires_grad = True
        obj_scale_optim.requires_grad = True
        obj_axis_green_wrd_optim.requires_grad = True
        obj_axis_red_wrd_optim.requires_grad = True
        shape_code_optim.requires_grad = True

        # Set optimizer.
        params = [obj_pos_wrd_optim, obj_scale_optim, obj_axis_green_wrd_optim, obj_axis_red_wrd_optim, shape_code_optim]
        optimizer = torch.optim.Adam(params, self.adam_step_ratio)

        # Optimizing.
        for grad_optim_idx in range(self.grad_optim_max):
            optimizer.zero_grad()

            # Get input for each frame.
            pos_wrd_frame = obj_pos_wrd_optim[:, None, :].expand(-1, using_frame_num, 3).reshape(-1, 3)
            scale_frame = obj_scale_optim[:, None, :].expand(-1, using_frame_num, 1).reshape(-1, 1)
            axis_green_frame = torch.sum(F.normalize(obj_axis_green_wrd_optim, dim=1)[:, None, :].expand(-1, using_frame_num, 3).reshape(-1, 3)[..., None, :]*w2c, -1)
            axis_red_frame = torch.sum(F.normalize(obj_axis_red_wrd_optim, dim=1)[:, None, :].expand(-1, using_frame_num, 3).reshape(-1, 3)[..., None, :]*w2c, -1)
            shape_code_frame = shape_code_optim[:, None, :].expand(-1, using_frame_num, self.ddf.latent_size).reshape(-1, self.ddf.latent_size)
            
            rays_d_cam = self.rays_d_cam.expand(batch_size*using_frame_num, -1, -1, -1).to(frame_camera_rot.device)
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

            # ###################################
            # if grad_optim_idx%10==0:
            if grad_optim_idx==30:
                check_map = []
                gt = raw_invdistance_map
                est = est_invdistance_map
                for i in range(batch_size*using_frame_num):
                    check_map.append(torch.cat([gt[i], est[i], torch.abs(gt[i]-est[i])], dim=0))
                # check_map_torch(torch.cat(check_map, dim=-1), f'opt_{grad_optim_idx}.png')
                check_map_torch(torch.cat(check_map, dim=-1), f'tes.png')
                import pdb; pdb.set_trace()
            # ###################################
            # # f grad_optim_idx%10==0:
            # if grad_optim_idx in [0, 10, 30, 60, 100]:
            #     if grad_optim_idx == 0:
            #         check_map = []
            #         for i in range(batch_size*using_frame_num):
            #             check_map.append(raw_invdistance_map[i])
            #         check_map = torch.cat(check_map, dim=0)
            #         list_for_vis.append(check_map)
            #     check_map = []
            #     check_map_tensor = torch.abs(raw_invdistance_map - est_invdistance_map)
            #     for i in range(batch_size*using_frame_num):
            #         check_map.append(check_map_tensor[i])
            #     check_map = torch.cat(check_map, dim=0)
            #     list_for_vis.append(check_map)
            # ###################################
            # print(f'optim_idx : {grad_optim_idx}, energy : {energy.item()}')

        torch.set_grad_enabled(False)
        est_obj_pos_wrd = obj_pos_wrd_optim.detach().clone()
        est_obj_scale = obj_scale_optim.detach().clone()
        est_obj_axis_green_wrd = F.normalize(obj_axis_green_wrd_optim, dim=1).detach().clone()
        est_obj_axis_red_wrd = F.normalize(obj_axis_red_wrd_optim, dim=1).detach().clone()
        est_shape_code = shape_code_optim.detach().clone()

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
            est_mask, est_distance_map = get_canonical_map(
                                            H = self.ddf_H, 
                                            cam_pos_wrd = cam_pos_wrd, 
                                            rays_d_cam = rays_d_cam, 
                                            w2c = w2c, 
                                            input_lat_vec = est_shape_code, 
                                            ddf = self.ddf, 
                                            )
            depth_error.append(torch.abs(gt_distance_map-est_distance_map).mean(dim=-1).mean(dim=-1))
        
            # #############################################
            # if shape_i==3:
            #     check_map = []
            #     # check_map_tensor = torch.abs(gt_distance_map-est_distance_map)
            #     for i in range(batch_size*using_frame_num):
            #         if i == 0:
            #             check_map.append(gt_distance_map[0])
            #         elif i == 1:
            #             check_map.append(est_distance_map[0])
            #         else:
            #             check_map.append(torch.zeros_like(check_map_tensor[0]))
            #     check_map = torch.cat(check_map, dim=0)
            #     list_for_vis.append(check_map)
            #     list_for_vis = torch.cat(list_for_vis, dim=-1)
            #     name = path[0].split('/')[0]
            #     check_map_torch(list_for_vis, f'{name}.png')
            #     # check_map_torch(list_for_vis, f'sample_images/{name}.png')
            #     import pdb; pdb.set_trace()
            # #############################################
            # # # Check map.
            # # check_map = []
            # # gt = gt_distance_map
            # # est = est_distance_map
            # # for i in range(batch_size):
            # #     check_map.append(torch.cat([gt[i], est[i], torch.abs(gt[i]-est[i])], dim=0))
            # # check_map_torch(torch.cat(check_map, dim=-1), f'canonical_map_{shape_i}.png')
            # # #############################################

        # Cal err.
        err_pos = torch.abs(est_obj_pos_wrd - gt_obj_pos_wrd[:, 0]).mean(dim=-1)
        err_scale = torch.abs(1 - est_obj_scale[:, 0] / gt_obj_scale[:, 0, 0])
        err_axis_red = torch.acos(self.cossim(est_obj_axis_red_wrd, gt_axis_red_wrd[:, 0]).clamp(min=self.cosssim_min, max=self.cosssim_max)) * 180 / torch.pi
        err_axis_green = torch.acos(self.cossim(est_obj_axis_green_wrd, gt_axis_green_wrd[:, 0]).clamp(min=self.cosssim_min, max=self.cosssim_max)) * 180 / torch.pi
        depth_error = torch.stack(depth_error, dim=-1).mean(dim=-1)
        # import pdb; pdb.set_trace()

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
    model.only_init_net = False
    model.start_frame_idx = 0
    model.frame_sequence_num = 3
    model.test_optim_num = 2
    model.grad_optim_max = 50
    model.shape_code_reg = 0.0
    model.adam_step_ratio = 0.01

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
