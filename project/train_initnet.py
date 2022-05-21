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

    
    def forward(self, inp, use_gru=False):
        batch_size = inp.shape[0]
        x = self.encoder(inp)
        x = x.reshape(batch_size, -1)

        x_green = self.fc_axis_green(x)
        axis_green = F.normalize(x_green, dim=-1)
        x_red = self.fc_axis_red(x)
        axis_red = F.normalize(x_red, dim=-1)
        z_diff = 0. * torch.ones_like(axis_green)[..., 0]
        x_cim = 0. * torch.ones_like(axis_green)[..., 0]
        y_cim = 0. * torch.ones_like(axis_green)[..., 0]
        scale_diff = torch.ones_like(axis_green)[..., 0]

        shape_code = self.fc_shape_code(x)

        if use_gru:
            return x_cim, y_cim, z_diff, axis_green, axis_red, scale_diff, shape_code, x
        else:
            return x_cim, y_cim, z_diff, axis_green, axis_red, scale_diff, shape_code





class df_resnet_encoder(pl.LightningModule):

    def __init__(self, args, in_channel=2):
        super().__init__()

        self.encoder = ResNet50(args, in_channel=in_channel)
        self.fc_axis_green = nn.Sequential(
                nn.Linear(2048 + 3, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 3), 
                )
        self.fc_axis_red = nn.Sequential(
                nn.Linear(2048 + 3, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 3), 
                )
        self.fc_shape_code = nn.Sequential(
                nn.Linear(2048 + args.latent_size, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, args.latent_size), 
                )

    
    def forward(self, inp, pre_axis_green, pre_axis_red, pre_shape_code):
        batch_size = inp.shape[0]
        x = self.encoder(inp)
        x = x.reshape(batch_size, -1)

        axis_green_x = torch.cat([x, pre_axis_green], dim=-1)
        diff_axis_green = self.fc_axis_green(axis_green_x)
        axis_red_x = torch.cat([x, pre_axis_red], dim=-1)
        diff_axis_red = self.fc_axis_red(axis_red_x)
        shape_code_x = torch.cat([x, pre_shape_code], dim=-1)
        diff_shape_code = self.fc_shape_code(shape_code_x)

        return diff_axis_green, diff_axis_red, diff_shape_code





class df_resnet_encoder_with_gru(pl.LightningModule):

    def __init__(self, args, in_channel=2):
        super().__init__()

        self.encoder = ResNet50(args, in_channel=in_channel)
        self.fc_axis_green = nn.Sequential(
                nn.Linear(2048 + 3, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 3), 
                )
        self.fc_axis_red = nn.Sequential(
                nn.Linear(2048 + 3, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 3), 
                )
        self.fc_shape_code = nn.Sequential(
                nn.Linear(2048 + args.latent_size, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, args.latent_size), 
                )
        self.gru = nn.GRU(input_size=2048, hidden_size=2048) # 試し
    
    def forward(self, inp, pre_axis_green, pre_axis_red, pre_shape_code, h_0):
        batch_size = inp.shape[0]
        
        x = self.encoder(inp)
        x = x.reshape(batch_size, -1)

        x, post_h = self.gru(x.unsqueeze(0), h_0.unsqueeze(0))
        x = x.reshape(batch_size, -1)
        h_1 = post_h.reshape(batch_size, -1)

        axis_green_x = torch.cat([x, pre_axis_green], dim=-1)
        diff_axis_green = self.fc_axis_green(axis_green_x)
        axis_red_x = torch.cat([x, pre_axis_red], dim=-1)
        diff_axis_red = self.fc_axis_red(axis_red_x)
        shape_code_x = torch.cat([x, pre_shape_code], dim=-1)
        diff_shape_code = self.fc_shape_code(shape_code_x)
        
        return diff_axis_green, diff_axis_red, diff_shape_code, x, h_1





def render_distance_map_from_axis(
        H, 
        obj_pos_wrd, 
        axis_green, 
        axis_red,
        obj_scale, 
        cam_pos_wrd, 
        rays_d_cam, 
        w2c, 
        input_lat_vec, 
        ddf, 
        with_invdistance_map = False, 
    ):
    # Get rotation matrix.
    axis_blue = torch.cross(axis_red, axis_green, dim=-1)
    axis_blue = F.normalize(axis_blue, dim=-1)
    orthogonal_axis_red = torch.cross(axis_green, axis_blue, dim=-1)
    o2c = torch.stack([orthogonal_axis_red, axis_green, axis_blue], dim=-1)

    # Get rays direction.
    rays_d_obj = torch.sum(rays_d_cam[..., None, :]*o2c[..., None, None, :, :].permute(0, 1, 2, 4, 3), -1)

    # Get rays origin.
    o2w = torch.bmm(w2c.to(o2c.dtype).permute(0, 2, 1), o2c)
    cam_pos_obj = torch.sum((cam_pos_wrd - obj_pos_wrd)[..., None, :] * o2w.permute(0, 2, 1), dim=-1) / obj_scale
    rays_o_obj = cam_pos_obj[:, None, None, :].expand(-1, H, H, -1)

    # Get rays inputs.
    rays_d = rays_d_obj
    rays_o = rays_o_obj

    # Estimating.
    est_invdistance_map_obj_scale, negative_D_mask, whitch_t_used = ddf.forward_from_far(rays_o, rays_d, input_lat_vec)
    est_invdistance_map = est_invdistance_map_obj_scale / obj_scale

    # Get distance map.
    mask_under_border = 1 / (cam_pos_obj.norm(dim=-1) + obj_scale * ddf.radius)
    est_mask = [map_i > border_i for map_i, border_i in zip(est_invdistance_map, mask_under_border)]
    est_mask = torch.stack(est_mask, dim=0)
    est_distance_map = torch.zeros_like(est_invdistance_map)
    est_distance_map[est_mask] = 1. / est_invdistance_map[est_mask]

    if with_invdistance_map:
        return est_invdistance_map, est_mask, est_distance_map
    else:
        return est_mask, est_distance_map





def clopping_distance_map(mask, distance_map, image_coord, input_H, input_W, ddf_H, bbox_list='not_given'):
    # Get bbox.
    if bbox_list == 'not_given':
        bbox_list = []
        for mask_i in mask:
            max_y, max_x = image_coord[mask_i].max(dim=0).values
            min_y, min_x = image_coord[mask_i].min(dim=0).values
            bbox_list.append(torch.tensor([[max_x, max_y], [min_x, min_y]]))
        bbox_list = torch.stack(bbox_list, dim=0)
        
        # 正方形でClop.
        bbox_H_xy = torch.stack([bbox_list[:, 0, 0] - bbox_list[:, 1, 0], # H_x
                                    bbox_list[:, 0, 1] - bbox_list[:, 1, 1]] # H_y
                                    , dim=-1)
        bbox_H = bbox_H_xy.max(dim=-1).values # BBoxのxy幅の内、大きい方で揃える
        diff_bbox_H = (bbox_H[:, None] - bbox_H_xy) / 2
        bbox_list = bbox_list + torch.stack([diff_bbox_H, -diff_bbox_H], dim=-2) # maxには足りない分を足し、minからは引く

        # BBoxが画像からはみ出た場合、収まるように戻す
        border = torch.tensor([[input_W-1, input_H-1], [0, 0]])[None]
        outside = border - bbox_list
        outside[:, 0][outside[:, 0] > .0] = 0. # 値が負ならMaxがはみ出た -> ずれを引く
        outside[:, 1][outside[:, 1] < .0] = 0. # 値が正ならMinがはみ出た -> ずれを足す
        bbox_list = bbox_list + outside.sum(dim=-2)[:, None, :]
        bbox_list = bbox_list / 128 - 1 # change range [-1, 1]

    # Clop
    mask = mask[:, None] # (N, dummy_C, H, W)
    distance_map = distance_map[:, None] # (N, dummy_C, H, W)

    coord_x = batch_linspace(bbox_list[:, 1, 0], bbox_list[:, 0, 0], ddf_H)
    coord_x = coord_x[:, None, :].expand(-1, ddf_H, -1)
    coord_y = batch_linspace(bbox_list[:, 1, 1], bbox_list[:, 0, 1], ddf_H)
    coord_y = coord_y[:, :, None].expand(-1, -1, ddf_H)
    sampling_coord = torch.stack([coord_x, coord_y], dim=-1).to(distance_map.device)
    cloped_mask = F.grid_sample(
                        mask.to(sampling_coord.dtype), 
                        sampling_coord, 
                        align_corners=True)[:, 0] > 0.99 # (N, H, W)
    cloped_distance_map = F.grid_sample(
                        distance_map.to(sampling_coord.dtype), 
                        sampling_coord, 
                        align_corners=True)[:, 0] # (N, H, W)
    return cloped_mask, cloped_distance_map, bbox_list





def get_normalized_depth_map(mask, distance_map, rays_d_cam, avg_depth_map=False):
    # Convert to depth map.
    depth_map = rays_d_cam[..., -1] * distance_map

    # Get average.
    if not avg_depth_map:
        avg_depth_map = torch.tensor(
            [depth_map_i[mask_i].mean() for mask_i, depth_map_i in zip(mask, depth_map)]
            , device=depth_map.device) # 物体の存在しているピクセルで平均を取る。

    # Normalizing.
    normalized_depth_map = depth_map - avg_depth_map[:, None, None]
    normalized_depth_map[torch.logical_not(mask)] = 0. # 物体の存在しているピクセルを正規化
    return depth_map, normalized_depth_map, avg_depth_map





def get_clopped_rays_d_cam(size, fov, bbox_list):
    bbox_list_ = 0.5 * bbox_list
    fov = torch.deg2rad(torch.tensor(fov, dtype=torch.float))
    x_coord = batch_linspace(torch.tan(fov*bbox_list_[:, 1, 0]), torch.tan(fov*bbox_list_[:, 0, 0]), size)
    x_coord = x_coord[:, None, :].expand(-1, size, -1)
    y_coord = batch_linspace(torch.tan(fov*bbox_list_[:, 1, 1]), torch.tan(fov*bbox_list_[:, 0, 1]), size)
    y_coord = y_coord[:, :, None].expand(-1, -1, size)
    rays_d_cam = torch.stack([x_coord, y_coord, torch.ones_like(x_coord)], dim=-1)
    rays_d_cam = F.normalize(rays_d_cam, dim=-1)
    return rays_d_cam # H, W, 3:xyz





def batch_linspace(start, end, step):
    raw = torch.linspace(0, 1, step)[None, :]
    return (end - start)[:, None] * raw + start[:, None]





class TaR_init_only(pl.LightningModule):

    def __init__(self, args, ddf):
        super().__init__()

        # Base configs
        self.dynamic = args.dynamic
        self.fov = args.fov
        self.input_H = 256
        self.input_W = 256
        self.x_coord = torch.arange(0, self.input_W)[None, :].expand(self.input_H, -1)
        self.y_coord = torch.arange(0, self.input_H)[:, None].expand(-1, self.input_W)
        self.image_coord = torch.stack([self.x_coord.T, self.x_coord], dim=-1) # [H, W, (Y and X)]
        self.ddf_H = 256

        self.lr = args.lr
        self.rays_d_cam = get_ray_direction(self.ddf_H, self.fov)
        self.ddf_instance_list = []
        with open(args.ddf_instance_list_txt, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                self.ddf_instance_list.append(line.rstrip('\n'))
        self.save_interval = args.save_interval
        self.optim_step_num = 3 # 5
        self.frame_num = args.frame_num
        self.model_params_dtype = False
        self.model_device = False

        # Make model
        self.ddf = ddf
        self.model = resnet_encoder(args)

        # loss func.
        self.cossim = nn.CosineSimilarity(dim=-1)



    def training_step(self, batch, batch_idx):
        frame_mask, frame_distance_map, frame_camera_pos, frame_camera_rot, frame_obj_pos, frame_obj_rot, frame_obj_scale, instance_id = batch
        batch_size = len(instance_id)
        instance_idx = [self.ddf_instance_list.index(instance_id_i) for instance_id_i in instance_id]
        gt_shape_code = self.ddf.lat_vecs(torch.tensor(instance_idx, device=self.ddf.device)).detach()
        frame_idx = random.randint(0, frame_mask.shape[1]-1) # ランダムなフレームを選択

        with torch.no_grad():
            # Clop distance map.
            raw_mask = frame_mask[:, frame_idx]
            raw_distance_map = frame_distance_map[:, frame_idx]
            clopped_mask, clopped_distance_map, bbox_list = clopping_distance_map(
                                                                raw_mask, 
                                                                raw_distance_map, 
                                                                self.image_coord, 
                                                                self.input_H, 
                                                                self.input_W, 
                                                                self.ddf_H
                                                                )

            # Get normalized depth map.
            rays_d_cam = get_clopped_rays_d_cam(self.ddf_H, self.fov, bbox_list).to(frame_camera_rot.device)
            clopped_depth_map, normalized_depth_map, avg_depth_map = get_normalized_depth_map(
                                                                        clopped_mask, 
                                                                        clopped_distance_map, 
                                                                        rays_d_cam
                                                                        )

            # Get ground truth.
            o2w = frame_obj_rot[:, frame_idx]
            w2c = frame_camera_rot[:, frame_idx]
            o2c = torch.bmm(w2c, o2w) # とりあえずこれを推論する
            o2w = torch.bmm(w2c.permute(0, 2, 1), o2c)
            o2o = torch.bmm(o2w.permute(0, 2, 1), o2w)
            gt_axis_green = o2c[:, :, 1] # Y
            gt_axis_red = o2c[:, :, 0] # X

            cam_pos_wrd = frame_camera_pos[:, frame_idx]
            obj_pos_wrd = frame_obj_pos[:, frame_idx]



        # Estimating.
        inp = torch.stack([clopped_distance_map, clopped_mask], 1)
        est_x_cim, est_y_cim, est_z_diff, est_axis_green, est_axis_red, est_scale_diff, est_shape_code = self.model(inp)
        # est_x_cim, est_y_cim : クロップされた画像座標（[-1, 1]で定義）における物体中心の予測
        # est_z_diff : デプス画像の正則に用いた平均から、物体中心がどれだけズレているか？

        def get_obj_pos_from_est(x_cim, y_cim, z_diff, bbox_list, mask, z_map, fov):
            bbox_list = bbox_list.to(x_cim)
            fov = torch.deg2rad(torch.tensor(fov, dtype=torch.float))
            xy_cim = torch.stack([x_cim, y_cim], dim=-1)
            bbox_hight = bbox_list[:, 0, 1] - bbox_list[:, 1, 1]
            bbox_center = bbox_list.mean(1)
            central_z = [(z_map_i[mask_i].max() + z_map_i[mask_i].min()) / 2 for mask_i, z_map_i, in zip(mask, z_map)]
            central_z = torch.tensor(central_z).to(z_diff)

            # clopした深度マップでの予測物体中心(x, y)をカメラ座標系における(x, y)に変換
            cim2im_scale = (bbox_hight) / 2 # clopしたBBoxの高さ÷画像の高さ２
            # z_at_xycim = F.grid_sample(z_map[:, None], xy_cim[:, None, None, :], align_corners=True)[:, 0, 0, 0] # ゼロなら？？
            # im2cam_scale = z_at_xycim * torch.tan(fov/2) # 中心のDepth（z軸の値）×torch.tan(fov/2)
            im2cam_scale = central_z * torch.tan(fov/2) # 中心のDepth（z軸の値）×torch.tan(fov/2)
            xy_im = cim2im_scale * xy_cim + bbox_center # 元画像における物体中心
            xy_cam = im2cam_scale * xy_im

            # 正規化深度画像での物体中心zをカメラ座標系におけるzに変換
            z_cam = z_diff + central_z
            return torch.cat([xy_cam, z_cam[..., None]], dim=-1), bbox_center

        est_obj_pos_cam, bbox_center = get_obj_pos_from_est(est_x_cim, est_y_cim, est_z_diff, bbox_list, clopped_mask, clopped_depth_map, self.fov)
        obj_pos_cam = torch.sum((obj_pos_wrd-cam_pos_wrd)[..., None, :]*w2c, dim=-1)
        est_obj_pos_cam[:, -1] = obj_pos_cam[:, -1]
        est_obj_pos_wrd = torch.sum(est_obj_pos_cam[..., None, :]*w2c.permute(0, 2, 1), dim=-1) + cam_pos_wrd

        # Cal loss.
        loss_axis_green = torch.mean(-self.cossim(est_axis_green, gt_axis_green) + 1.)
        loss_axis_red = torch.mean(-self.cossim(est_axis_red, gt_axis_red) + 1.)
        loss_shape_code = F.mse_loss(est_shape_code, gt_shape_code)

        # Cal total loss.
        loss_axis = loss_axis_green + .5 * loss_axis_red
        loss = loss_axis + 1e2 * loss_shape_code

        # Check distance map.
        with torch.no_grad():
            rays_d_cam = self.rays_d_cam.expand(batch_size, -1, -1, -1).to(frame_camera_rot.device)
            est_invdistance_map, est_mask, est_distance_map = render_distance_map_from_axis(
                                                                    H = self.ddf_H, 
                                                                    obj_pos_wrd = obj_pos_wrd, 
                                                                    axis_green = gt_axis_green, 
                                                                    axis_red = gt_axis_red, 
                                                                    obj_scale = 0.5, 
                                                                    cam_pos_wrd = frame_camera_pos[:, frame_idx].detach(), 
                                                                    rays_d_cam = rays_d_cam,  
                                                                    w2c = w2c.detach(), 
                                                                    input_lat_vec = gt_shape_code, 
                                                                    ddf = self.ddf, 
                                                                    with_invdistance_map = True, 
                                                                    )
            clopped_est_mask, clopped_est_distance_map, _ = clopping_distance_map(
                                                                est_mask, 
                                                                est_distance_map, 
                                                                self.image_coord, 
                                                                self.input_H, 
                                                                self.input_W, 
                                                                self.ddf_H, 
                                                                bbox_list
                                                                )
                
            
            # Check point cloud.
            batch_idx = 0
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            rays_d_cam = get_clopped_rays_d_cam(self.ddf_H, self.fov, bbox_list).to(frame_camera_rot.device)
            rays_d = rays_d_cam
            depth_map = clopped_distance_map
            mask = clopped_mask
            point = (depth_map[batch_idx][..., None] * rays_d[batch_idx])[mask[batch_idx]]
            point = point.to('cpu').detach().numpy().copy()
            ax.scatter(point[::3, 0], point[::3, 1], point[::3, 2], marker="o", linestyle='None', c='m', s=0.05)
            ax.scatter(point[::3, 0], point[::3, 1], point[::3, 2], marker="o", linestyle='None', c='m', s=0.05)
            depth_map = clopped_est_distance_map
            mask = clopped_est_mask
            point = (depth_map[batch_idx][..., None] * rays_d[batch_idx])[mask[batch_idx]]
            point = point.to('cpu').detach().numpy().copy()
            ax.scatter(point[::3, 0], point[::3, 1], point[::3, 2], marker="o", linestyle='None', c='c', s=0.05)
            ax.scatter(point[::3, 0], point[::3, 1], point[::3, 2], marker="o", linestyle='None', c='c', s=0.05)

            ax.view_init(elev=0, azim=0)
            fig.savefig("point_cloud_cam.png")
            plt.close()
            import pdb; pdb.set_trace()


            # Plotを作成
            fig = pylab.figure(figsize=(20, 8))
            # BBoxをピクセル座標へ
            bbox_list = 128 * (bbox_list.to('cpu').detach().numpy().copy() + 1)
            bbox_center = 128 * (bbox_center.to('cpu').detach().numpy().copy() + 1)
            bbox = np.concatenate([bbox_list, bbox_center[:, None, :]], axis=1)
            bbox_1 = bbox[0]
            bbox_2 = bbox[1]
            # 元画像
            ax_1 = fig.add_subplot(2, 5, 1)
            ax_1.scatter(bbox_1[:, 0], bbox_1[:, 1], c='red', s=20)
            ax_1.imshow(raw_distance_map[0].to('cpu').detach().numpy().copy())
            ax_2 = fig.add_subplot(2, 5, 6)
            ax_2.scatter(bbox_2[:, 0], bbox_2[:, 1], c='red', s=20)
            ax_2.imshow(raw_distance_map[1].to('cpu').detach().numpy().copy())
            # クロップした観測画像
            ax_3 = fig.add_subplot(2, 5, 2)
            ax_3.imshow(clopped_distance_map[0].to('cpu').detach().numpy().copy())
            ax_4 = fig.add_subplot(2, 5, 7)
            ax_4.imshow(clopped_distance_map[1].to('cpu').detach().numpy().copy())
            # 元画像の予測
            ax_5 = fig.add_subplot(2, 5, 3)
            ax_5.scatter(bbox_1[:, 0], bbox_1[:, 1], c='red', s=20)
            ax_5.imshow(est_distance_map[0].to('cpu').detach().numpy().copy())
            ax_6 = fig.add_subplot(2, 5, 8)
            ax_6.scatter(bbox_2[:, 0], bbox_2[:, 1], c='red', s=20)
            ax_6.imshow(est_distance_map[1].to('cpu').detach().numpy().copy())
            # クロップした画像の予測
            ax_7 = fig.add_subplot(2, 5, 4)
            ax_7.imshow(clopped_est_distance_map[0].to('cpu').detach().numpy().copy())
            ax_8 = fig.add_subplot(2, 5, 9)
            ax_8.imshow(clopped_est_distance_map[1].to('cpu').detach().numpy().copy())
            # 誤差
            clopped_error = torch.abs(clopped_distance_map - clopped_est_distance_map)
            ax_9 = fig.add_subplot(2, 5, 5)
            ax_9.imshow(clopped_error[0].to('cpu').detach().numpy().copy())
            ax_10 = fig.add_subplot(2, 5, 10)
            ax_10.imshow(clopped_error[1].to('cpu').detach().numpy().copy())
            # 画像を保存
            fig.savefig("tes.png", dpi=300)
            pylab.close()

        import pdb; pdb.set_trace()

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
    # val_dataset = TaR_dataset(
    #     args, 
    #     args.val_instance_list_txt, 
    #     args.val_data_dir, 
    #     args.val_N_views
    #     )
    # val_dataloader = data_utils.DataLoader(
    #     val_dataset, 
    #     batch_size=args.N_batch, 
    #     num_workers=args.num_workers, 
    #     drop_last=False, 
    #     shuffle=True
    #     )

    # Get ddf.
    ddf = DDF(args)
    ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args)
    ddf.eval()
    
    # Get ckpts path.
    ckpt_dir = os.path.join('lightning_logs', f'{args.expname}_{args.exp_version}', 'checkpoints/*')
    ckpt_path_list = sorted(glob.glob(ckpt_dir))

    # Load ckpt and start training.
    # if len(ckpt_path_list) == 0:
    model = TaR_init_only(args, ddf)
    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader, 
        # val_dataloaders=val_dataloader, 
        # datamodule=None, 
        # ckpt_path=None
        )

    # elif len(ckpt_path_list) > 0:
    #     latest_ckpt_path = ckpt_path_list[-1]
    #     print('\n', f'+++ Reloading from {latest_ckpt_path} +++ \n')
    #     model = TaR_init_only(args, ddf)
    #     trainer.fit(
    #         model=model, 
    #         train_dataloaders=train_dataloader, 
    #         val_dataloaders=val_dataloader, 
    #         datamodule=None, 
    #         ckpt_path=latest_ckpt_path
    #         )
