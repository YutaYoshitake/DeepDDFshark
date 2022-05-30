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

        self.backbone_encoder = nn.Sequential(
                ResNet50(args, in_channel=in_channel), 
                )
        self.backbone_fc = nn.Sequential(
                nn.Linear(2048 + 7, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, 512), nn.LeakyReLU(0.2),
                )
        self.fc_axis_green = nn.Sequential(
                nn.Linear(512, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 3), 
                )
        self.fc_axis_red = nn.Sequential(
                nn.Linear(512, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 3), 
                )
        self.fc_shape_code = nn.Sequential(
                nn.Linear(512, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, args.latent_size), 
                )
        self.fc_pos = nn.Sequential(
                nn.Linear(512, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 3), 
                )
        self.fc_scale = nn.Sequential(
                nn.Linear(512, 256), nn.LeakyReLU(0.2), # nn.Linear(518, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 1), 
                )
        self.use_gru = args.use_gru

    
    def forward(self, inp, bbox_info):
        # Backbone.
        x = self.backbone_encoder(inp)
        x = x.reshape(inp.shape[0], -1)
        x = self.backbone_fc(torch.cat([x, bbox_info], dim=-1))

        # Get pose.
        x_pos = self.fc_pos(x)
        x_cim = x_pos[:, 0]
        y_cim = x_pos[:, 1]
        z_diff = x_pos[:, 2]

        # Get axis.
        x_green = self.fc_axis_green(x)
        axis_green = F.normalize(x_green, dim=-1)
        x_red = self.fc_axis_red(x)
        axis_red = F.normalize(x_red, dim=-1)

        # Get scale.
        x_scale = self.fc_scale(x)
        scale_diff = x_scale + torch.ones_like(x_scale)

        # Get shape code.
        shape_code = self.fc_shape_code(x)

        if self.use_gru:
            return x_cim, y_cim, z_diff, axis_green, axis_red, scale_diff, shape_code, x
        else:
            return x_cim, y_cim, z_diff, axis_green, axis_red, scale_diff, shape_code, 0





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
    cam_pos_obj = torch.sum((cam_pos_wrd - obj_pos_wrd)[..., None, :] * o2w.permute(0, 2, 1), dim=-1) / obj_scale[:, None]
    rays_o_obj = cam_pos_obj[:, None, None, :].expand(-1, H, H, -1)

    # Get rays inputs.
    rays_d = rays_d_obj
    rays_o = rays_o_obj

    # Estimating.
    est_invdistance_map_obj_scale, negative_D_mask, whitch_t_used = ddf.forward_from_far(rays_o, rays_d, input_lat_vec)
    est_invdistance_map = est_invdistance_map_obj_scale / obj_scale[:, None, None]

    # Get distance map.
    mask_under_border = 1 / (cam_pos_obj.norm(dim=-1) + 1.0 * obj_scale * ddf.radius) # 良いのか...？
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





def get_normalized_depth_map(mask, distance_map, rays_d_cam, avg_depth_map='not_given'):
    # Convert to depth map.
    depth_map = rays_d_cam[..., -1] * distance_map

    # Get average.
    if avg_depth_map=='not_given':
        # avg_depth_map = torch.tensor(
        #     [depth_map_i[mask_i].mean() for mask_i, depth_map_i in zip(mask, depth_map)]
        #     , device=depth_map.device) # 物体の存在しているピクセルで平均を取る。
        top_n = 10 # top_nからtop_n//2までの平均を取る
        avg_depth_map = torch.stack([
                            torch.topk(depth_map_i[mask_i], top_n).values[top_n//2:].mean() - torch.topk(-depth_map_i[mask_i], top_n).values[top_n//2:].mean() 
                            for mask_i, depth_map_i in zip(mask, depth_map)], dim=0)
        avg_depth_map = avg_depth_map / 2

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





def diff2estimation(x_cim, y_cim, z_diff, scale_diff, bbox_list, avg_z_map, fov, canonical_bbox_diagonal=1.0):
    # Get Bbox info.
    bbox_list = bbox_list.to(x_cim)
    fov = torch.deg2rad(torch.tensor(fov, dtype=torch.float))
    xy_cim = torch.stack([x_cim, y_cim], dim=-1)
    bbox_hight = bbox_list[:, 0, 1] - bbox_list[:, 1, 1]
    bbox_center = bbox_list.mean(1)

    # clopした深度マップでの予測物体中心(x, y)をカメラ座標系における(x, y)に変換
    cim2im_scale = (bbox_hight) / 2 # clopしたBBoxの高さ÷画像の高さ２
    im2cam_scale = avg_z_map * torch.tan(fov/2) # 中心のDepth（z軸の値）×torch.tan(fov/2)
    xy_im = cim2im_scale[:, None] * xy_cim + bbox_center # 元画像における物体中心
    xy_cam = im2cam_scale[:, None] * xy_im

    # 正規化深度画像での物体中心zをカメラ座標系におけるzに変換
    z_cam = z_diff + avg_z_map

    # clopしたBBoxの対角
    clopping_bbox_diagonal = 2 * math.sqrt(2)

    # clopしたBBoxの対角とカノニカルBBoxの対角の比を変換
    scale = scale_diff * im2cam_scale[:, None] * cim2im_scale[:, None] * clopping_bbox_diagonal / canonical_bbox_diagonal
    return torch.cat([xy_cam, z_cam[..., None]], dim=-1), scale, im2cam_scale





def diffcim2diffcam(diff_x_cim, diff_y_cim, diff_z_diff, bbox_list, avg_z_map, fov):
    # Get Bbox info.
    bbox_list = bbox_list.to(diff_x_cim)
    fov = torch.deg2rad(torch.tensor(fov, dtype=torch.float))
    diff_xy_cim = torch.stack([diff_x_cim, diff_y_cim], dim=-1)
    bbox_hight = bbox_list[:, 0, 1] - bbox_list[:, 1, 1]

    # clopした深度マップでの予測物体中心(x, y)をカメラ座標系における(x, y)に変換
    cim2im_scale = (bbox_hight) / 2 # clopしたBBoxの高さ÷画像の高さ２
    im2cam_scale = avg_z_map * torch.tan(fov/2) # 中心のDepth（z軸の値）×torch.tan(fov/2)
    diff_xy_im = cim2im_scale[:, None] * diff_xy_cim # 元画像における物体中心
    diff_xy_cam = im2cam_scale[:, None] * diff_xy_im
    return torch.cat([diff_xy_cam, diff_z_diff[:, None]], dim=-1)





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

        # Get ground truth.
        instance_idx = [self.ddf_instance_list.index(instance_id_i) for instance_id_i in instance_id]
        gt_shape_code = self.ddf.lat_vecs(torch.tensor(instance_idx, device=self.ddf.device)).detach()

        # Set frame.
        frame_idx = random.randint(0, frame_mask.shape[1]-1) # ランダムなフレームを選択

        with torch.no_grad():
            # Clop distance map.
            raw_mask = frame_mask[:, frame_idx]
            raw_distance_map = frame_distance_map[:, frame_idx]
            clopped_mask, clopped_distance_map, bbox_list = clopping_distance_map(
                                                                raw_mask, raw_distance_map, self.image_coord, self.input_H, self.input_W, self.ddf_H
                                                                )

            # Get normalized depth map.
            rays_d_cam = get_clopped_rays_d_cam(self.ddf_H, self.fov, bbox_list).to(frame_camera_rot.device)
            clopped_depth_map, normalized_depth_map, avg_depth_map = get_normalized_depth_map(
                                                                        clopped_mask, clopped_distance_map, rays_d_cam
                                                                        )

            # Get ground truth.
            o2w = frame_obj_rot[:, frame_idx]
            w2c = frame_camera_rot[:, frame_idx]
            o2c = torch.bmm(w2c, o2w) # とりあえずこれを推論する
            gt_axis_green = o2c[:, :, 1] # Y
            gt_axis_red = o2c[:, :, 0] # X
            cam_pos_wrd = frame_camera_pos[:, frame_idx]
            gt_obj_pos_wrd = frame_obj_pos[:, frame_idx]
            gt_obj_scale = frame_obj_scale[:, frame_idx]
        

        # Get input.
        inp = torch.stack([normalized_depth_map, clopped_mask], 1)
        bbox_info = torch.cat([bbox_list.reshape(-1, 4), bbox_list.mean(1), avg_depth_map.to('cpu')[:, None]], dim=-1).to(inp)
        # axis_info = torch.cat([gt_axis_green, gt_axis_red], dim=-1).to(inp)

        # Estimating.
        # est_x_cim, est_y_cim : クロップされた画像座標（[-1, 1]で定義）における物体中心の予測, 
        # est_z_diff : デプス画像の正則に用いた平均から、物体中心がどれだけズレているか？, 
        # est_axis_green : カメラ座標系での物体の上方向, 
        # est_axis_red : カメラ座標系での物体の右方向, 
        # est_scale_diff : Clopping-BBoxの対角と物体のカノニカルBBoxの対角がどれくらいずれているか, 
        est_x_cim, est_y_cim, est_z_diff, est_axis_green, est_axis_red, est_scale_diff, est_shape_code, _ = self.model(inp, bbox_info) # self.model(inp, bbox_info, axis_info)
        est_obj_pos_cam, est_obj_scale, im2cam_scale = diff2estimation(est_x_cim, est_y_cim, est_z_diff, est_scale_diff, bbox_list, avg_depth_map, self.fov)
        est_obj_pos_wrd = torch.sum(est_obj_pos_cam[..., None, :]*w2c.permute(0, 2, 1), dim=-1) + cam_pos_wrd


        # Cal loss.
        loss_pos = F.mse_loss(est_obj_pos_wrd, gt_obj_pos_wrd)
        loss_scale = F.mse_loss(est_obj_scale, gt_obj_scale.to(est_obj_scale))
        loss_axis_green = torch.mean(-self.cossim(est_axis_green, gt_axis_green) + 1.)
        loss_axis_red = torch.mean(-self.cossim(est_axis_red, gt_axis_red) + 1.)
        loss_shape_code = F.mse_loss(est_shape_code, gt_shape_code)

        # Cal total loss.
        loss = 1e1 * loss_pos + 1e1 * loss_scale + loss_axis_green + loss_axis_red + 1e1 * loss_shape_code

        # Check distance map.
        if (self.current_epoch+1)//10==0 and batch_idx==0:
            with torch.no_grad():
                rays_d_cam = self.rays_d_cam.expand(2, -1, -1, -1).to(frame_camera_rot.device)
                est_invdistance_map, est_mask, est_distance_map = render_distance_map_from_axis(
                                                                        H = self.ddf_H, 
                                                                        obj_pos_wrd = est_obj_pos_wrd[:2], 
                                                                        axis_green = est_axis_green[:2], 
                                                                        axis_red = est_axis_red[:2], 
                                                                        obj_scale = est_obj_scale[:2], 
                                                                        cam_pos_wrd = cam_pos_wrd[:2], 
                                                                        rays_d_cam = rays_d_cam,  
                                                                        w2c = w2c[:2].detach(), 
                                                                        input_lat_vec = est_shape_code[:2], 
                                                                        ddf = self.ddf, 
                                                                        with_invdistance_map = True, 
                                                                        )
                clopped_est_mask, clopped_est_distance_map, _ = clopping_distance_map(
                                                                    est_mask, est_distance_map, self.image_coord, self.input_H, self.input_W, self.ddf_H, bbox_list[:2]
                                                                    )

                # Plotを作成
                gt_obj_pos_cam = torch.sum((gt_obj_pos_wrd-cam_pos_wrd)[..., None, :]*w2c, dim=-1)
                fig = pylab.figure(figsize=(20, 8))
                # BBoxをピクセル座標へ
                bbox_list = 128 * (bbox_list.to('cpu').detach().numpy().copy() + 1)
                bbox_center = bbox_list.mean(1)
                obj_pos_cam = 128 * (gt_obj_pos_cam / im2cam_scale[:, None] + 1).to('cpu').detach().numpy().copy()
                obj_pos_cam_ = 128 * (est_obj_pos_cam / im2cam_scale[:, None] + 1).to('cpu').detach().numpy().copy()
                # bbox_center = 128 * (bbox_center.to('cpu').detach().numpy().copy() + 1)
                bbox = np.concatenate([bbox_list, bbox_center[:, None, :], obj_pos_cam[:, None, :2], obj_pos_cam_[:, None, :2]], axis=1)
                # bbox = np.concatenate([bbox_list, obj_pos_cam[:, None, :2]], axis=1)
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
                clopped_error = torch.abs(clopped_distance_map[:2] - clopped_est_distance_map)
                ax_9 = fig.add_subplot(2, 5, 5)
                ax_9.imshow(clopped_error[0].to('cpu').detach().numpy().copy())
                ax_10 = fig.add_subplot(2, 5, 10)
                ax_10.imshow(clopped_error[1].to('cpu').detach().numpy().copy())
                # 画像を保存
                fig.savefig(f"sample_images/initnet_first_test/epo_{str(self.current_epoch).zfill(5)}.png", dpi=300)
                # fig.savefig(f"tes.png", dpi=300)
                pylab.close()

        return {'loss': loss, 'loss_pos':loss_pos.detach(), 'loss_scale': loss_scale.detach(), 'loss_axis_red': loss_axis_red.detach(), 'loss_shape_code': loss_shape_code.detach()}



    def training_epoch_end(self, outputs):

        # Log loss.
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        current_epoch = torch.tensor(self.current_epoch + 1., dtype=avg_loss.dtype)
        self.log_dict({'train/total_loss': avg_loss, "step": current_epoch})

        avg_loss_pos = torch.stack([x['loss_pos'] for x in outputs]).mean()
        self.log_dict({'train/loss_pos': avg_loss_pos, "step": current_epoch})

        avg_loss_scale = torch.stack([x['loss_scale'] for x in outputs]).mean()
        self.log_dict({'train/loss_scale': avg_loss_scale, "step": current_epoch})

        avg_loss_axis_red = torch.stack([x['loss_axis_red'] for x in outputs]).mean()
        self.log_dict({'train/loss_axis_red': avg_loss_axis_red, "step": current_epoch})

        avg_loss_shape_code = torch.stack([x['loss_shape_code'] for x in outputs]).mean()
        self.log_dict({'train/loss_shape_code': avg_loss_shape_code, "step": current_epoch})

        # Save ckpt.
        if (self.current_epoch + 1) % self.save_interval == 0:
            ckpt_name = str(self.current_epoch + 1).zfill(10) + '.ckpt'
            ckpt_path = os.path.join(self.trainer.log_dir, 'checkpoints', ckpt_name)
            trainer.save_checkpoint(ckpt_path)



    def validation_step(self, batch, batch_idx):
        frame_mask, frame_distance_map, frame_camera_pos, frame_camera_rot, frame_obj_pos, frame_obj_rot, frame_obj_scale, instance_id = batch
        batch_size = len(instance_id)

        # Set frame.
        frame_idx = random.randint(0, frame_mask.shape[1]-1) # ランダムなフレームを選択

        with torch.no_grad():
            # Clop distance map.
            raw_mask = frame_mask[:, frame_idx]
            raw_distance_map = frame_distance_map[:, frame_idx]
            clopped_mask, clopped_distance_map, bbox_list = clopping_distance_map(
                                                                raw_mask, raw_distance_map, self.image_coord, self.input_H, self.input_W, self.ddf_H
                                                                )

            # Get normalized depth map.
            rays_d_cam = get_clopped_rays_d_cam(self.ddf_H, self.fov, bbox_list).to(frame_camera_rot.device)
            clopped_depth_map, normalized_depth_map, avg_depth_map = get_normalized_depth_map(
                                                                        clopped_mask, clopped_distance_map, rays_d_cam
                                                                        )

            # Get ground truth.
            o2w = frame_obj_rot[:, frame_idx]
            w2c = frame_camera_rot[:, frame_idx]
            o2c = torch.bmm(w2c, o2w) # とりあえずこれを推論する
            gt_axis_green = o2c[:, :, 1] # Y
            gt_axis_red = o2c[:, :, 0] # X
            cam_pos_wrd = frame_camera_pos[:, frame_idx]
            gt_obj_pos_wrd = frame_obj_pos[:, frame_idx]
            gt_obj_scale = frame_obj_scale[:, frame_idx]
        

        # Get input.
        inp = torch.stack([normalized_depth_map, clopped_mask], 1)
        bbox_info = torch.cat([bbox_list.reshape(-1, 4), bbox_list.mean(1), avg_depth_map.to('cpu')[:, None]], dim=-1).to(inp)
        # axis_info = torch.cat([gt_axis_green, gt_axis_red], dim=-1).to(inp)

        # Estimating.
        # est_x_cim, est_y_cim : クロップされた画像座標（[-1, 1]で定義）における物体中心の予測, 
        # est_z_diff : デプス画像の正則に用いた平均から、物体中心がどれだけズレているか？, 
        # est_axis_green : カメラ座標系での物体の上方向, 
        # est_axis_red : カメラ座標系での物体の右方向, 
        # est_scale_diff : Clopping-BBoxの対角と物体のカノニカルBBoxの対角がどれくらいずれているか, 
        est_x_cim, est_y_cim, est_z_diff, est_axis_green, est_axis_red, est_scale_diff, est_shape_code, _ = self.model(inp, bbox_info) # self.model(inp, bbox_info, axis_info)
        est_obj_pos_cam, est_obj_scale, im2cam_scale = diff2estimation(est_x_cim, est_y_cim, est_z_diff, est_scale_diff, bbox_list, avg_depth_map, self.fov)
        est_obj_pos_wrd = torch.sum(est_obj_pos_cam[..., None, :]*w2c.permute(0, 2, 1), dim=-1) + cam_pos_wrd


        # Cal err.
        err_pos = F.mse_loss(est_obj_pos_wrd, gt_obj_pos_wrd)
        err_scale = torch.mean(est_obj_scale / gt_obj_scale.to(est_obj_scale))
        err_axis_red = torch.mean(-self.cossim(est_axis_red, gt_axis_red) + 1.)

        return {'err_pos':err_pos.detach(), 'err_scale': err_scale.detach(), 'err_axis_red': err_axis_red.detach()}



    def validation_epoch_end(self, outputs):
        # Log err.
        avg_err_axis_red = torch.stack([x['err_axis_red'] for x in outputs]).mean()
        current_epoch = torch.tensor(self.current_epoch + 1., dtype=avg_err_axis_red.dtype)
        self.log_dict({'validation/err_axis_red': avg_err_axis_red, "step": current_epoch})

        avg_err_scale = torch.stack([x['err_scale'] for x in outputs]).mean()
        self.log_dict({'validation/err_scale': avg_err_scale, "step": current_epoch})

        avg_err_pos = torch.stack([x['err_pos'] for x in outputs]).mean()
        self.log_dict({'validation/err_pos': avg_err_pos, "step": current_epoch})




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
        'train', 
        args.train_instance_list_txt, 
        args.train_data_dir, 
        args.train_N_views, 
        )
    train_dataloader = data_utils.DataLoader(
        train_dataset, 
        batch_size=args.N_batch, 
        num_workers=args.num_workers, 
        drop_last=False, 
        shuffle=True, 
        )
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
        val_dataloaders=val_dataloader, 
        datamodule=None, 
        ckpt_path=None
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
