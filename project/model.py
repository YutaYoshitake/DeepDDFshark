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
from typing import Optional, Any, Union, Callable
import copy
import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from torch.autograd import grad
import torch.utils.data as data
from torch import Tensor
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
from model import *
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



class initializer(pl.LightningModule):

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
                nn.Linear(512, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 1), nn.Softplus(beta=.7), 
                )
        # self.fc_weight = nn.Sequential(
        #         nn.Linear(512, 256), nn.LeakyReLU(0.2),
        #         nn.Linear(256, 256), nn.LeakyReLU(0.2),
        #         nn.Linear(256, 1), nn.Sigmoid(), 
        #         )

    
    def forward(self, inp, bbox_info):
        # Backbone.
        x = self.backbone_encoder(inp)
        x = x.reshape(inp.shape[0], -1)
        x = self.backbone_fc(torch.cat([x, bbox_info], dim=-1))

        # Get pose.
        x_pos = self.fc_pos(x)

        # Get axis.
        x_green = self.fc_axis_green(x)
        axis_green = F.normalize(x_green, dim=-1)
        x_red = self.fc_axis_red(x)
        axis_red = F.normalize(x_red, dim=-1)

        # Get scale.
        scale_diff = self.fc_scale(x) + 1e-5 # Prevent scale=0.

        # Get shape code.
        shape_code = self.fc_shape_code(x)

        return x_pos, axis_green, axis_red, scale_diff, shape_code, 0





class deep_optimizer(pl.LightningModule):

    def __init__(
        self, 
        input_type = 'depth', 
        hidden_dim = 512, 
        latent_size = 256, 
        integrate_mode = 'average', 
        output_diff_coordinate = 'obj', 
        ):
        super().__init__()

        # Config.
        self.integrate_mode = integrate_mode
        self.output_diff_coordinate = output_diff_coordinate
        self.input_type = input_type
        if self.input_type == 'osmap':
            in_channel = 11
            conv1d_inp_dim = 2048
        elif self.input_type == 'depth':
            in_channel = 5
            conv1d_inp_dim = 2048 + 3
            x = torch.linspace(-1, 1, steps=16)[None, :].expand(16, -1)
            y = torch.linspace(-1, 1, steps=16)[:, None].expand(-1, 16)
            self.sample_grid =  torch.stack([x, y], dim=-1)[None, :, :, :]
        
        # Back Bone.
        self.hidden_dim = hidden_dim
        self.backbone = ResNet50_wo_dilation(in_channel=in_channel, gpu_num=1)
        self.conv_1d = nn.Conv2d(conv1d_inp_dim, hidden_dim, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        x = torch.linspace(-1, 1, steps=16)[None, :].expand(16, -1)
        y = torch.linspace(-1, 1, steps=16)[:, None].expand(-1, 16)
        self.sample_grid =  torch.stack([x, y], dim=-1)[None, :, :, :]

        # Head MLP.
        self.latent_size = latent_size
        self.fc_pos = nn.Sequential(
                nn.Linear(hidden_dim + 3, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 3))
        self.fc_axis_green = nn.Sequential(
                nn.Linear(hidden_dim + 3, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 3))
        self.fc_axis_red = nn.Sequential(
                nn.Linear(hidden_dim + 3, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 3))
        self.fc_scale = nn.Sequential(
                nn.Linear(hidden_dim + 1, 256), nn.LeakyReLU(0.2), 
                nn.Linear(256, 1), nn.Softplus(beta=.7))
        self.fc_shape_code = nn.Sequential(
                nn.Linear(hidden_dim + self.latent_size, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, self.latent_size))


    def forward(self, inp, pre_pos_wrd, pre_green_wrd, pre_red_wrd, pre_scale_wrd, pre_shape_code, # pre is [batch*seq, ?]
        cim2im_scale, im2cam_scale, bbox_center, avg_depth_map, w2c, cam_pos_wrd, rays_d_cam, pre_o2w, model_mode = 'train'):
        batch_size, seq_len, cha_num, H, W = inp.shape
        
        # Backbone.
        inp = inp.reshape(batch_size*seq_len, cha_num, H, W)
        x = self.backbone(inp) # torch.Size([batch*seq, 2048, 16, 16])
        if self.input_type == 'depth':
            sample_grid = self.sample_grid.expand(batch_size*seq_len, -1, -1, -1).to(rays_d_cam)
            sampled_rays_d = F.grid_sample(rays_d_cam.permute(0, 3, 1, 2), sample_grid, align_corners=True)
            sampled_rays_d = F.normalize(sampled_rays_d, dim=1) # 特徴量のカメラ座標系でのRay方向
            x = torch.cat([x, sampled_rays_d], dim=1)
        x = self.conv_1d(x) # torch.Size([batch*seq, 512, 16, 16])
        x = self.avgpool(x) # torch.Size([batch*seq, 2048, 1, 1])
        x = x.reshape(batch_size*seq_len, self.hidden_dim)

        # Estimate diff.
        if self.output_diff_coordinate == 'img':
            # Make pre_est.
            pre_green_cam = torch.sum(pre_green_wrd[..., None, :]*w2c, -1)
            pre_red_cam = torch.sum(pre_red_wrd[..., None, :]*w2c, -1)
            pre_pos_cam = torch.sum((pre_pos_wrd - cam_pos_wrd)[..., None, :]*w2c, dim=-1)
            pre_pos_cim = torch.cat([
                            (pre_pos_cam[:, :-1] / im2cam_scale[:, None] - bbox_center) / cim2im_scale[:, None], 
                            (pre_pos_cam[:, -1] - avg_depth_map)[:, None]], dim=-1)
            pre_scale_cim = pre_scale_wrd / (im2cam_scale[:, None] * cim2im_scale[:, None] * 2 * math.sqrt(2))
            
            # Head.
            diff_pos_cim = self.fc_pos(torch.cat([x, pre_pos_cim.detach()], dim=-1))
            diff_green_cam = self.fc_axis_green(torch.cat([x, pre_green_cam.detach()], dim=-1))
            diff_red_cam = self.fc_axis_red(torch.cat([x, pre_red_cam.detach()], dim=-1))
            diff_scale = self.fc_scale(torch.cat([x, pre_scale_cim.detach()], dim=-1)) + 1e-5 # Prevent scale=0.
            diff_shape_code = self.fc_shape_code(torch.cat([x, pre_shape_code.detach()], dim=-1))

            # Convert cordinates.
            diff_pos_cam = diffcim2diffcam(diff_pos_cim, cim2im_scale, im2cam_scale)
            diff_pos_wrd = torch.sum(diff_pos_cam[..., None, :]*w2c.permute(0, 2, 1), -1)
            diff_green_wrd = torch.sum(diff_green_cam[..., None, :]*w2c.permute(0, 2, 1), -1)
            diff_red_wrd = torch.sum(diff_red_cam[..., None, :]*w2c.permute(0, 2, 1), -1)

        elif self.output_diff_coordinate == 'obj':
            # Make pre_est.
            pre_green_obj = torch.tensor([[0.0, 1.0, 0.0]]).expand(batch_size*seq_len, -1).to(x)
            pre_red_obj = torch.tensor([[1.0, 0.0, 0.0]]).expand(batch_size*seq_len, -1).to(x)
            pre_pos_obj = torch.tensor([[0.0, 0.0, 0.0]]).expand(batch_size*seq_len, -1).to(x)
            pre_scale_obj = torch.tensor([[1.0]]).expand(batch_size*seq_len, -1).to(x)

            # Head.
            diff_pos_obj = self.fc_pos(torch.cat([x, pre_pos_obj.detach()], dim=-1))
            diff_green_obj = self.fc_axis_green(torch.cat([x, pre_green_obj.detach()], dim=-1))
            diff_red_obj = self.fc_axis_red(torch.cat([x, pre_red_obj.detach()], dim=-1))
            diff_scale = self.fc_scale(torch.cat([x, pre_scale_obj.detach()], dim=-1)) + 1e-5 # Prevent scale=0.
            diff_shape_code = self.fc_shape_code(torch.cat([x, pre_shape_code.detach()], dim=-1))

            # Convert cordinates.
            diff_pos_wrd = torch.sum(diff_pos_obj[..., None, :]*pre_o2w, -1) * pre_scale_wrd
            diff_green_wrd = torch.sum(diff_green_obj[..., None, :]*pre_o2w, -1) * pre_scale_wrd
            diff_red_wrd = torch.sum(diff_red_obj[..., None, :]*pre_o2w, -1) * pre_scale_wrd

        # Reshape update to [Seq, batch, dim].
        diff_pos_wrd = diff_pos_wrd.reshape(batch_size, -1, 3).permute(1, 0, 2)
        diff_green_wrd = diff_green_wrd.reshape(batch_size, -1, 3).permute(1, 0, 2)
        diff_red_wrd = diff_red_wrd.reshape(batch_size, -1, 3).permute(1, 0, 2)
        diff_scale = diff_scale.reshape(batch_size, -1, 1).permute(1, 0, 2)
        diff_shape_code = diff_shape_code.reshape(batch_size, -1, self.latent_size).permute(1, 0, 2)

        # Get integrated update.
        if (self.integrate_mode == 'average') or (model_mode=='val'):
            diff_pos_wrd = diff_pos_wrd.mean(0)
            diff_green_wrd = diff_green_wrd.mean(0)
            diff_red_wrd = diff_red_wrd.mean(0)
            diff_scale = diff_scale.mean(0)
            diff_shape_code = diff_shape_code.mean(0)

        return diff_pos_wrd, diff_green_wrd, diff_red_wrd, diff_scale, diff_shape_code





class optimize_former(pl.LightningModule):

    def __init__(
        self, 
        input_type = 'depth', 
        hidden_dim = 512, 
        num_encoder_layers = 6, 
        split_into_patch = True, 
        latent_size = 256, 
        position_dim = 7, 
        positional_encoding_mode = 'add', 
        ):
        super().__init__()

        # Back Bone.
        self.input_type = input_type
        if self.input_type == 'osmap':
            in_channel = 11
            conv1d_inp_dim = 2048
        elif self.input_type == 'depth':
            in_channel = 5
            conv1d_inp_dim = 2048 + 3
            x = torch.linspace(-1, 1, steps=16)[None, :].expand(16, -1)
            y = torch.linspace(-1, 1, steps=16)[:, None].expand(-1, 16)
            self.sample_grid =  torch.stack([x, y], dim=-1)[None, :, :, :]
        self.positional_encoding_mode = positional_encoding_mode
        if self.positional_encoding_mode == 'add':
            self.back_bone_dim = hidden_dim
        elif self.positional_encoding_mode == 'cat':
            self.back_bone_dim = hidden_dim - position_dim
        elif self.positional_encoding_mode == 'non':
            self.back_bone_dim = hidden_dim
        self.split_into_patch = split_into_patch
        self.backbone = ResNet50_wo_dilation(in_channel=in_channel, gpu_num=1)
        self.conv_1d = nn.Conv2d(conv1d_inp_dim, self.back_bone_dim, 1)
        if self.split_into_patch:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        # Transformer.
        self.hidden_dim = hidden_dim
        if self.positional_encoding_mode == 'add':
            self.positional_encoding_mlp = nn.Linear(position_dim, hidden_dim)
            nn.init.xavier_uniform_(self.positional_encoding_mlp.weight)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_encoder_layers)
        self.cls_token = nn.Parameter(torch.empty(1, 1, hidden_dim)) # torch.Size([dummy_batch, dummy_seq, hidden_dim])
        nn.init.xavier_uniform_(self.cls_token)

        # Head MLP.
        self.latent_size = latent_size
        self.fc_pos = nn.Sequential(
                nn.Linear(hidden_dim + 3, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 3))
        self.fc_axis_green = nn.Sequential(
                nn.Linear(hidden_dim + 3, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 3))
        self.fc_axis_red = nn.Sequential(
                nn.Linear(hidden_dim + 3, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 3))
        self.fc_scale = nn.Sequential(
                nn.Linear(hidden_dim + 1, 256), nn.LeakyReLU(0.2), 
                nn.Linear(256, 1), nn.Softplus(beta=.7))
        self.fc_shape_code = nn.Sequential(
                nn.Linear(hidden_dim + self.latent_size, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, self.latent_size))


    def forward(self, inp, rays_d_cam, pre_scale_wrd, pre_shape_code, pre_o2w, positional_encoding_target=False):
        batch_size, seq_len, cha_num, H, W = inp.shape
        
        # Backbone.
        inp = inp.reshape(batch_size*seq_len, cha_num, H, W)
        x = self.backbone(inp) # torch.Size([batch*seq, 2048, 16, 16])
        if self.input_type == 'depth':
            sample_grid = self.sample_grid.expand(batch_size*seq_len, -1, -1, -1).to(rays_d_cam)
            sampled_rays_d = F.grid_sample(rays_d_cam.permute(0, 3, 1, 2), sample_grid, align_corners=True)
            sampled_rays_d = F.normalize(sampled_rays_d, dim=1) # 特徴量のカメラ座標系でのRay方向
            x = torch.cat([x, sampled_rays_d], dim=1)
        x = self.conv_1d(x) # torch.Size([batch*seq, 512, 16, 16])

        # Transformer.
        if self.split_into_patch:
            x = self.avgpool(x) # torch.Size([batch*seq, 2048, 1, 1])
            x = x.reshape(batch_size, seq_len, self.back_bone_dim) # torch.Size([batch, seq, hidden_dim])
        if self.positional_encoding_mode == 'add':
            positional_encoding_vec = self.positional_encoding_mlp(positional_encoding_target)
            x = x + positional_encoding_vec
        elif self.positional_encoding_mode == 'cat':
            x = torch.cat([positional_encoding_target, x], dim=-1)
        x = torch.cat([self.cls_token.expand(batch_size, -1, -1), x], dim=1)
        x = x.permute(1, 0, 2) # torch.Size([seq+1, batch, hidden_dim])
        x = self.transformer_encoder(x) # torch.Size([seq+1, batch, hidden_dim])
        x = x[0] # get cls token. #  # torch.Size([batch, hidden_dim])

        # Make pre_est.
        pre_green_obj = torch.tensor([[0.0, 1.0, 0.0]]).expand(batch_size, -1).to(x)
        pre_red_obj = torch.tensor([[1.0, 0.0, 0.0]]).expand(batch_size, -1).to(x)
        pre_pos_obj = torch.tensor([[0.0, 0.0, 0.0]]).expand(batch_size, -1).to(x)
        pre_scale_obj = torch.tensor([[1.0]]).expand(batch_size, -1).to(x)

        # Head.
        diff_pos_obj = self.fc_pos(torch.cat([x, pre_pos_obj.detach()], dim=-1))
        diff_green_obj = self.fc_axis_green(torch.cat([x, pre_green_obj.detach()], dim=-1))
        diff_red_obj = self.fc_axis_red(torch.cat([x, pre_red_obj.detach()], dim=-1))
        diff_scale = self.fc_scale(torch.cat([x, pre_scale_obj.detach()], dim=-1)) + 1e-5 # Prevent scale=0.
        diff_shape_code = self.fc_shape_code(torch.cat([x, pre_shape_code.detach()], dim=-1))

        # Convert cordinates.
        diff_pos_wrd = torch.sum(diff_pos_obj[..., None, :]*pre_o2w, -1) * pre_scale_wrd
        diff_green_wrd = torch.sum(diff_green_obj[..., None, :]*pre_o2w, -1) * pre_scale_wrd
        diff_red_wrd = torch.sum(diff_red_obj[..., None, :]*pre_o2w, -1) * pre_scale_wrd

        return diff_pos_wrd, diff_green_wrd, diff_red_wrd, diff_scale, diff_shape_code
