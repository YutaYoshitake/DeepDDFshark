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





class resnet_encoder(pl.LightningModule):

    def __init__(self, args, in_channel=2):
        super().__init__()

        self.encoder = ResNet50(args, in_channel=in_channel)
        self.fc_axis_green = nn.Sequential(
                nn.Linear(2048, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 3), 
                )
        self.fc_axis_red = nn.Sequential(
                nn.Linear(2048, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 3), 
                )
        self.fc_shape_code = nn.Sequential(
                nn.Linear(2048, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, args.latent_size), 
                )

    
    def forward(self, inp):
        batch_size = inp.shape[0]
        x = self.encoder(inp)
        x = x.reshape(batch_size, -1)

        x_green = self.fc_axis_green(x)
        axis_green = F.normalize(x_green, dim=-1)
        x_red = self.fc_axis_red(x)
        axis_red = F.normalize(x_red, dim=-1)

        shape_code = self.fc_shape_code(x)
        return axis_green, axis_red, shape_code





class df_resnet_encoder(pl.LightningModule):

    def __init__(self, args, in_channel=2):
        super().__init__()

        self.encoder = ResNet50(args, in_channel=in_channel)
        self.fc_axis_green = nn.Sequential(
                nn.Linear(2048, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 3), 
                )
        self.fc_axis_red = nn.Sequential(
                nn.Linear(2048, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 3), 
                )
        self.fc_shape_code = nn.Sequential(
                nn.Linear(2048, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, args.latent_size), 
                )

    
    def forward(self, inp):
        batch_size = inp.shape[0]
        x = self.encoder(inp)
        x = x.reshape(batch_size, -1)

        diff_axis_green = self.fc_axis_green(x)
        diff_axis_red = self.fc_axis_red(x)
        diff_shape_code = self.fc_shape_code(x)

        return diff_axis_green, diff_axis_red, diff_shape_code





class TaR_init_only(pl.LightningModule):

    def __init__(self, args, ddf):
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
        self.model = resnet_encoder(args)

        # loss func.
        self.cossim = nn.CosineSimilarity(dim=-1)



    def forward(self):
        return 0



    def training_step(self, batch, batch_idx):

        with torch.no_grad():
            frame_rgb_map, frame_mask, frame_depth_map, frame_camera_pos, frame_camera_rot, frame_obj_rot, instance_id = batch
            frame_idx = 2 # random.randint(0, frame_rgb_map.shape[1]-1) # ランダムなフレームを選択


            # Get ground truth.
            batch_size = len(instance_id)
            instance_idx = [self.ddf_instance_list.index(instance_id_i) for instance_id_i in instance_id]
            gt_shape_code = self.ddf.lat_vecs(torch.tensor(instance_idx, device=self.ddf.device)).detach()

            # #########################
            # #########################
            # frame_idx = 0
            # data_dict = pickle_load('dataset/dugon/test_moving_camera/test_moving_camera_train_data/1a6f615e8b1b5ae4dbbc9440457e303e/00001.pickle')
            # frame_mask = torch.from_numpy(data_dict['mask'].astype(np.float32)).clone().to(frame_rgb_map.device)[None].expand(batch_size, -1, -1, -1)
            # frame_depth_map = torch.from_numpy(data_dict['depth_map'].astype(np.float32)).clone().to(frame_rgb_map.device)[None].expand(batch_size, -1, -1, -1)
            # frame_camera_pos = torch.from_numpy(data_dict['camera_pos'].astype(np.float32)).clone().to(frame_rgb_map.device)[None].expand(batch_size, -1, -1)
            # frame_camera_rot = torch.from_numpy(data_dict['camera_rot'].astype(np.float32)).clone().to(frame_rgb_map.device)[None].expand(batch_size, -1, -1, -1)
            # frame_obj_rot = torch.from_numpy(data_dict['obj_rot'].astype(np.float32)).clone().to(frame_rgb_map.device)[None].expand(batch_size, -1, -1)
            # instance_id = ['1a6f615e8b1b5ae4dbbc9440457e303e'] * batch_size
            # #########################
            # #########################

            o2w = batch_pi2rot_y(frame_obj_rot[:, frame_idx]).permute(0, 2, 1)
            w2c = frame_camera_rot[:, frame_idx]
            o2c = torch.bmm(w2c, o2w) # とりあえずこれを推論する
            o2w = torch.bmm(w2c.permute(0, 2, 1), o2c)
            o2o = torch.bmm(o2w.permute(0, 2, 1), o2w)
            gt_axis_green = o2c[:, :, 1] # Y
            gt_axis_red = o2c[:, :, 0] # X

            cam_pos_wrd = frame_camera_pos[:, frame_idx]
            obj_pos_wrd = torch.zeros_like(cam_pos_wrd)
            obj_pos_cam = torch.sum((obj_pos_wrd - cam_pos_wrd)[..., None, :] * w2c, dim=-1)
            cam_pos_obj = torch.sum((cam_pos_wrd - obj_pos_wrd)[..., None, :] * o2w.permute(0, 2, 1), dim=-1) # = -obj_T_c2o
            
            obj_T_c2w = torch.sum(-obj_pos_cam[..., None, :]*w2c.permute(0, 2, 1), -1)
            obj_T_c2o = torch.sum(obj_T_c2w[..., None, :]*o2w.permute(0, 2, 1), -1)
            
            rays_o_wrd = cam_pos_wrd[:, None, None, :].expand(-1, self.H, self.H, -1)
            rays_o_obj = cam_pos_obj[:, None, None, :].expand(-1, self.H, self.H, -1)
            rays_d_cam = self.rays_d_cam.expand(batch_size, -1, -1, -1).to(frame_camera_rot.device)
            rays_d_wrd = torch.sum(rays_d_cam[..., None, :]*w2c[..., None, None, :, :].permute(0, 1, 2, 4, 3), -1)
            rays_d_obj = torch.sum(rays_d_cam[..., None, :]*o2c[..., None, None, :, :].permute(0, 1, 2, 4, 3), -1)
            
            # Get input.
            mask = frame_mask[:, frame_idx]
            depth_map = frame_depth_map[:, frame_idx]
        

        inp = torch.stack([depth_map, mask], 1)
        est_axis_green, est_axis_red, est_shape_code = self.model(inp)

        # Cal loss.
        loss_axis_green = torch.mean(-self.cossim(est_axis_green, gt_axis_green) + 1.)
        loss_axis_red = torch.mean(-self.cossim(est_axis_red, gt_axis_red) + 1.)
        loss_shape_code = F.mse_loss(est_shape_code, gt_shape_code)

        # Cal total loss.
        loss_axis = loss_axis_green + .5 * loss_axis_red
        loss = loss_axis + 1e2 * loss_shape_code

        # import pdb; pdb.set_trace()






        # Check depth map.
        # if (self.current_epoch + 1) % 10 == 0 and batch_idx==0:
        result_map = []
        with torch.no_grad():
            for batch_idx in range(batch_size):
                # # batch_idx = 0
                # rays_o = rays_o_obj[batch_idx].unsqueeze(0)
                # rays_d = rays_d_obj[batch_idx].unsqueeze(0)

                # input_lat_vec = gt_shape_code[batch_idx]
                # gt_pose_gt_shape_code_invdepth_map = self.ddf.forward(rays_o, rays_d, input_lat_vec)
                # gt_pose_gt_shape_code_mask = gt_pose_gt_shape_code_invdepth_map > .5
                # gt_pose_gt_shape_code_depth_map = torch.zeros_like(gt_pose_gt_shape_code_invdepth_map)
                # gt_pose_gt_shape_code_depth_map[gt_pose_gt_shape_code_mask] = 1. / gt_pose_gt_shape_code_invdepth_map[gt_pose_gt_shape_code_mask]

                # input_lat_vec = est_shape_code[batch_idx]
                # gt_pose_est_shape_code_invdepth_map = self.ddf.forward(rays_o, rays_d, input_lat_vec)
                # gt_pose_est_shape_code_mask = gt_pose_est_shape_code_invdepth_map > .5
                # gt_pose_est_shape_code_depth_map = torch.zeros_like(gt_pose_est_shape_code_invdepth_map)
                # gt_pose_est_shape_code_depth_map[gt_pose_est_shape_code_mask] = 1. / gt_pose_est_shape_code_invdepth_map[gt_pose_est_shape_code_mask]

                # result_map_1 = torch.cat([frame_depth_map[batch_idx, frame_idx], gt_pose_gt_shape_code_depth_map[0], gt_pose_est_shape_code_depth_map[0]], dim=1)

                # axis_green = est_axis_green
                # axis_red = est_axis_red
                # axis_blue = torch.cross(axis_red, axis_green, dim=-1)
                # axis_blue = F.normalize(axis_blue, dim=-1)
                # axis_red = torch.cross(axis_green, axis_blue, dim=-1)
                # est_o2c = torch.stack([axis_red, axis_green, axis_blue], dim=-1)
                # rays_d_obj = torch.sum(rays_d_cam[..., None, :]*est_o2c[..., None, None, :, :].permute(0, 1, 2, 4, 3), -1)
                # rays_d = rays_d_obj[batch_idx].unsqueeze(0)

                # input_lat_vec = gt_shape_code[batch_idx]
                # est_pose_gt_shape_invdepth_map = self.ddf.forward(rays_o, rays_d, input_lat_vec)
                # est_pose_gt_shape_mask = est_pose_gt_shape_invdepth_map > .5
                # est_pose_gt_shape_depth_map = torch.zeros_like(est_pose_gt_shape_invdepth_map)
                # est_pose_gt_shape_depth_map[est_pose_gt_shape_mask] = 1. / est_pose_gt_shape_invdepth_map[est_pose_gt_shape_mask]

                # input_lat_vec = est_shape_code[batch_idx]
                # est_pose_est_shape_code_invdepth_map = self.ddf.forward(rays_o, rays_d, input_lat_vec)
                # est_pose_est_shape_code_mask = est_pose_est_shape_code_invdepth_map > .5
                # est_pose_est_shape_code_depth_map = torch.zeros_like(est_pose_est_shape_code_invdepth_map)
                # est_pose_est_shape_code_depth_map[est_pose_est_shape_code_mask] = 1. / est_pose_est_shape_code_invdepth_map[est_pose_est_shape_code_mask]

                # dummy_result = torch.zeros_like(frame_depth_map[batch_idx, frame_idx])
                # result_map_2 = torch.cat([dummy_result, est_pose_gt_shape_depth_map[0], est_pose_est_shape_code_depth_map[0]], dim=1)
                
                # result_map = torch.cat([result_map_1, result_map_2], dim=0)
                # check_map(result_map, 'initnet_results/train/' + str(batch_idx + 1).zfill(10) + '.png', figsize=[10,8])
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

                # for batch_idx in range(batch_size):
                # batch_idx = 0
                rays_d = rays_d_obj[batch_idx].unsqueeze(0)
                rays_o = rays_o_obj[batch_idx].unsqueeze(0)
                input_lat_vec = est_shape_code[batch_idx]

                est_invdepth_map = self.ddf.forward(rays_o, rays_d, input_lat_vec)
                est_mask = est_invdepth_map > .5
                est_depth_map = torch.zeros_like(est_invdepth_map)
                est_depth_map[est_mask] = 1. / est_invdepth_map[est_mask]

                result_map_i = torch.cat([depth_map[batch_idx], est_depth_map[0]], dim=0)
                result_map.append(result_map_i)
                
                if batch_idx > 10:
                    break

            result_map = torch.cat(result_map, dim=1)
            check_map(result_map, 'tes_ini.png', figsize=[10,2])

            import pdb; pdb.set_trace()



        # # Check point cloud.
        # batch_idx = 0
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")

        # # rays_d = rays_d_obj
        # # obj_T = cam_pos_obj
        # # rays_d = rays_d_wrd
        # # obj_T = obj_T_c2w
        # rays_d = rays_d_cam
        # obj_T = -obj_pos_cam
        # point = (depth_map[batch_idx][..., None] * rays_d[batch_idx])[mask[batch_idx]] + obj_T[batch_idx]
        # point = point.to('cpu').detach().numpy().copy()
        # ax.scatter(point[::3, 0], point[::3, 1], point[::3, 2], marker="o", linestyle='None', c='m', s=0.05)

        # # obj_R = o2o[0]
        # # obj_R = o2w[0]
        # # obj_R = o2c[0]
        # # axis_green = obj_R[:, 1].to('cpu').detach().numpy().copy() # Y
        # # axis_red = obj_R[:, 0].to('cpu').detach().numpy().copy() # X
        # axis_green = gt_axis_green[0].to('cpu').detach().numpy().copy() # Y
        # axis_red = gt_axis_red[0].to('cpu').detach().numpy().copy() # minus_X
        # axis_blue = np.cross(axis_red, axis_green) # Z
        # x, y, z = 0.0, 0.0, 0.0
        # color_list = ['green', 'red', 'blue']
        # for i, axis in enumerate([axis_green, axis_red, axis_blue]):
        #     u, v, w = 0.3*axis[0].item(), 0.3*axis[1].item(), 0.3*axis[2].item()
        #     ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.1, color=color_list[i])

        # ax.view_init(elev=0, azim=-90)
        # fig.savefig("point_cloud_cam.png")
        # plt.close()

        # import pdb; pdb.set_trace()



        # # Check point clouds in all frames.
        # batch_idx = 0
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")

        # for frame_idx in range(5):
        #     # Get depth map.
        #     mask = frame_mask[:, frame_idx]
        #     depth_map = frame_depth_map[:, frame_idx]
        #     # Repaoring pos and w2c.
        #     pos = frame_camera_pos[:, frame_idx]
        #     w2c = frame_camera_rot[:, frame_idx]
        #     # Get rays.
        #     rays_d_cam = self.rays_d_cam.expand(batch_size, -1, -1, -1).to(frame_camera_rot.device)
        #     rays_d_wrd = torch.sum(rays_d_cam[..., None, :]*w2c[..., None, None, :, :].permute(0, 1, 2, 4, 3), -1)
        #     point = (depth_map[batch_idx][..., None] * rays_d_wrd[batch_idx])[mask[batch_idx]] + pos[batch_idx]
        #     point = point.to('cpu').detach().numpy().copy()
        #     ax.scatter(point[::3, 0], point[::3, 1], point[::3, 2], marker="o", linestyle='None', c='m', s=0.05)

        # ax.view_init(elev=0, azim=0)
        # fig.savefig("tes.png")
        # plt.close()

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

        with torch.no_grad():
            frame_rgb_map, frame_mask, frame_depth_map, frame_camera_pos, frame_camera_rot, frame_obj_rot, instance_id = batch
            batch_size = len(instance_id)

            # Get ground truth.
            frame_idx = random.randint(0, frame_rgb_map.shape[1]-1) # ランダムなフレームを選択

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
        

        inp = torch.stack([depth_map, mask], 1)
        est_axis_green, est_axis_red, est_shape_code = self.model(inp)

        # Cal err.
        err_axis_green = torch.mean(-self.cossim(est_axis_green, gt_axis_green) + 1.)
        err_axis_red = torch.mean(-self.cossim(est_axis_red, gt_axis_red) + 1.)


        with torch.no_grad():
            for batch_idx in range(batch_size):
                # batch_idx = 0
                cam_pos_wrd = frame_camera_pos[:, frame_idx]
                obj_pos_wrd = torch.zeros_like(cam_pos_wrd)
                obj_pos_cam = torch.sum((obj_pos_wrd - cam_pos_wrd)[..., None, :] * w2c, dim=-1)
                cam_pos_obj = torch.sum((cam_pos_wrd - obj_pos_wrd)[..., None, :] * o2w.permute(0, 2, 1), dim=-1) # = -obj_T_c2o
                
                rays_o_wrd = cam_pos_wrd[:, None, None, :].expand(-1, self.H, self.H, -1)
                rays_o_obj = cam_pos_obj[:, None, None, :].expand(-1, self.H, self.H, -1)
                rays_d_cam = self.rays_d_cam.expand(batch_size, -1, -1, -1).to(frame_camera_rot.device)
                rays_d_wrd = torch.sum(rays_d_cam[..., None, :]*w2c[..., None, None, :, :].permute(0, 1, 2, 4, 3), -1)
                rays_d_obj = torch.sum(rays_d_cam[..., None, :]*o2c[..., None, None, :, :].permute(0, 1, 2, 4, 3), -1)
                rays_o = rays_o_obj[batch_idx].unsqueeze(0)
                rays_d = rays_d_obj[batch_idx].unsqueeze(0)

                input_lat_vec = est_shape_code[batch_idx]
                gt_pose_est_shape_code_invdepth_map = self.ddf.forward(rays_o, rays_d, input_lat_vec)
                gt_pose_est_shape_code_mask = gt_pose_est_shape_code_invdepth_map > .5
                gt_pose_est_shape_code_depth_map = torch.zeros_like(gt_pose_est_shape_code_invdepth_map)
                gt_pose_est_shape_code_depth_map[gt_pose_est_shape_code_mask] = 1. / gt_pose_est_shape_code_invdepth_map[gt_pose_est_shape_code_mask]

                axis_green = est_axis_green
                axis_red = est_axis_red
                axis_blue = torch.cross(axis_red, axis_green, dim=-1)
                axis_blue = F.normalize(axis_blue, dim=-1)
                axis_red = torch.cross(axis_green, axis_blue, dim=-1)
                est_o2c = torch.stack([axis_red, axis_green, axis_blue], dim=-1)
                rays_d_obj = torch.sum(rays_d_cam[..., None, :]*est_o2c[..., None, None, :, :].permute(0, 1, 2, 4, 3), -1)
                rays_d = rays_d_obj[batch_idx].unsqueeze(0)
                
                est_o2w = torch.bmm(w2c.to(est_o2c.dtype).permute(0, 2, 1), est_o2c)
                cam_pos_obj = torch.sum((cam_pos_wrd - obj_pos_wrd)[..., None, :] * est_o2w.permute(0, 2, 1), dim=-1) # = -obj_T_c2o
                rays_o_obj = cam_pos_obj[:, None, None, :].expand(-1, self.H, self.H, -1)
                rays_o = rays_o_obj[batch_idx].unsqueeze(0)

                input_lat_vec = est_shape_code[batch_idx]
                est_pose_est_shape_code_invdepth_map = self.ddf.forward(rays_o, rays_d, input_lat_vec)
                est_pose_est_shape_code_mask = est_pose_est_shape_code_invdepth_map > .5
                est_pose_est_shape_code_depth_map = torch.zeros_like(est_pose_est_shape_code_invdepth_map)
                est_pose_est_shape_code_depth_map[est_pose_est_shape_code_mask] = 1. / est_pose_est_shape_code_invdepth_map[est_pose_est_shape_code_mask]

                result_map = torch.cat([frame_depth_map[batch_idx, frame_idx], gt_pose_est_shape_code_depth_map[0], est_pose_est_shape_code_depth_map[0]], dim=1)
                check_map(result_map, 'initnet_results/val/' + str(batch_idx + 1).zfill(10) + '.png', figsize=[10,5])



        return {'err_axis_green': err_axis_green.detach(), 'err_axis_red': err_axis_red.detach()}



    def validation_epoch_end(self, outputs):
        # Log loss.
        avg_err_axis_green = torch.stack([x['err_axis_green'] for x in outputs]).mean()
        current_epoch = torch.tensor(self.current_epoch + 1., dtype=avg_err_axis_green.dtype)
        self.log_dict({'validation/err_axis_green': avg_err_axis_green, "step": current_epoch})

        avg_err_axis_red = torch.stack([x['err_axis_red'] for x in outputs]).mean()
        self.log_dict({'validation/err_axis_red': avg_err_axis_red, "step": current_epoch})



    def test_step(self, batch, batch_idx):

        est_axis_green_wrd = []
        est_axis_red_wrd = []
        est_shape_code = []
        
        for frame_idx in range(self.frame_num):
            with torch.no_grad():
                frame_rgb_map, frame_mask, frame_depth_map, frame_camera_pos, frame_camera_rot, frame_obj_rot, instance_id = batch
                batch_size = len(instance_id)

                # Get ground truth.
                # frame_idx = random.randint(0, frame_rgb_map.shape[1]-1) # ランダムなフレームを選択

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
            
            inp = torch.stack([depth_map, mask], 1)
            est_axis_green_cam_i, est_axis_red_cam_i, est_shape_code_i = self.model(inp)
            est_axis_green_wrd_i = torch.sum(est_axis_green_cam_i[..., None, :]*w2c.permute(0, 2, 1), -1)
            est_axis_red_wrd_i = torch.sum(est_axis_red_cam_i[..., None, :]*w2c.permute(0, 2, 1), -1)
            est_axis_green_wrd.append(est_axis_green_wrd_i)
            est_axis_red_wrd.append(est_axis_red_wrd_i)
            est_shape_code.append(est_shape_code_i)
        
        # Get average.
        est_axis_green_wrd = torch.stack(est_axis_green_wrd, dim=1).mean(dim=1)
        est_axis_red_wrd = torch.stack(est_axis_red_wrd, dim=1).mean(dim=1)
        est_axis_green_wrd = F.normalize(est_axis_green_wrd, dim=1)
        est_axis_red_wrd = F.normalize(est_axis_red_wrd, dim=1)
        est_shape_code = torch.stack(est_shape_code, dim=1).mean(dim=1)

        with torch.no_grad():
            # Get rotation matrix.
            w2c = frame_camera_rot[:, frame_idx]
            est_axis_green = torch.sum(est_axis_green_wrd[..., None, :]*w2c, -1)
            est_axis_red = torch.sum(est_axis_red_wrd[..., None, :]*w2c, -1)
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
                    
            # Check depth map.
            check_map_1 = []
            for batch_idx in range(batch_size):
                check_map_i = torch.cat([depth_map[batch_idx], est_depth_map[batch_idx]], dim=0)
                check_map_1.append(check_map_i)
            check_map_1 = torch.cat(check_map_1, dim=1)
            check_map(check_map_1, 'check_map_1.png', figsize=[10,2])
            check_map_1 = []

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
            {"params": self.model.parameters()},
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
        shuffle=True
        )

    # Get ddf.
    ddf = DDF(args)
    ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args)
    ddf.eval()
    
    # Get ckpts path.
    ckpt_dir = os.path.join('lightning_logs', f'{args.expname}_{args.exp_version}', 'checkpoints/*')
    ckpt_path_list = sorted(glob.glob(ckpt_dir))

    # Load ckpt and start training.
    if len(ckpt_path_list) == 0:
        model = TaR_init_only(args, ddf)
        trainer.fit(
            model=model, 
            train_dataloaders=train_dataloader, 
            val_dataloaders=val_dataloader, 
            datamodule=None, 
            ckpt_path=None
            )

    elif len(ckpt_path_list) > 0:
        latest_ckpt_path = ckpt_path_list[-1]
        print('\n', f'+++ Reloading from {latest_ckpt_path} +++ \n')
        model = TaR_init_only(args, ddf)
        trainer.fit(
            model=model, 
            train_dataloaders=train_dataloader, 
            val_dataloaders=val_dataloader, 
            datamodule=None, 
            ckpt_path=latest_ckpt_path
            )
