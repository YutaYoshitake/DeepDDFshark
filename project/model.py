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
import einops

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





class optimize_former(pl.LightningModule):

    def __init__(
        self, 
        transformer_model = 'pytorch', 
        input_type = 'depth', 
        split_into_patch = 'non', 
        hidden_dim = 512, 
        num_encoder_layers = 6, 
        dim_feedforward = 2048, 
        latent_size = 256, 
        position_dim = 7, 
        num_head = 8, 
        dropout = 0.1, 
        positional_encoding_mode = 'non', 
        integration_mode = 'average', 
        encoder_norm_type = 'LayerNorm', 
        reset_transformer_params = False, 
        loss_timing = 'after_mean', 
        ):
        super().__init__()

        # Back Bone.
        self.input_type = input_type
        if self.input_type == 'osmap':
            in_channel = 11
            conv1d_inp_dim = 2048
        self.back_bone_dim = hidden_dim
        self.backbone = ResNet50_wo_dilation(in_channel=in_channel, gpu_num=1)
        self.conv_1d = nn.Conv2d(conv1d_inp_dim, self.back_bone_dim, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        # Transformer.
        self.integration_mode = integration_mode
        if not integration_mode in {'cnn_only_1', 'cnn_only_2'}:
            ##################################################
            encoder_norm = nn.LayerNorm(hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_head, dim_feedforward=dim_feedforward, dropout=dropout, activation="relu")
            self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm) # , 1, encoder_norm)
            # encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=num_head, dim_feedforward=dim_feedforward, dropout=dropout, activation="relu")
            # self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm) # , 1, encoder_norm)
            ##################################################
            # self.encoder = nn.MultiheadAttention(hidden_dim, num_head, dropout=0.0)
            ##################################################
            # encoder_layer = TransformerEncoderLayer_woNorm(d_model=hidden_dim, nhead=num_head, dim_feedforward=dim_feedforward, dropout=0.0, activation="relu")
            # self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, None) # , 1, encoder_norm)
            ##################################################
        self.loss_timing = loss_timing


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
                nn.Linear(hidden_dim + self.latent_size, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, self.latent_size))
                # nn.Linear(hidden_dim + self.latent_size, 512), nn.LeakyReLU(0.2),
                # nn.Linear(512, self.latent_size))
        # import pdb; pdb.set_trace()


    def forward(self, inp, rays_d_cam, pre_scale_wrd, pre_shape_code, pre_o2w, 
        inp_pre_scale_wrd, inp_pre_shape_code, inp_pre_o2w, positional_encoding_target=False, model_mode='train'):
        batch_size, seq_len, cha_num, H, W = inp.shape
        
        # Backbone.
        inp = inp.reshape(batch_size*seq_len, cha_num, H, W)
        x = self.backbone(inp) # torch.Size([batch*seq, 2048, 16, 16])
        x = self.conv_1d(x) # torch.Size([batch*seq, 512, 16, 16])
        
        # Transformer.
        x = self.avgpool(x) # torch.Size([batch*seq, 2048, 1, 1])
        x = x.reshape(batch_size, seq_len, self.back_bone_dim) # torch.Size([batch, seq, hidden_dim])
        if not self.integration_mode in {'cnn_only_1', 'cnn_only_2'}:
            x = x.permute(1, 0, 2) # torch.Size([seq, batch, hidden_dim])
            if self.loss_timing=='after_mean' or model_mode=='val':
                ##################################################
                x = self.encoder(x).mean(0) # torch.Size([seq, batch, hidden_dim]), get mean.
                ##################################################
                # x = (x + self.encoder(x, x, x)[0]).mean(0)
                ##################################################
                # x = x.mean(0)
                ##################################################
            else:
                import pdb; pdb.set_trace()
                x = self.encoder(x) # torch.Size([seq, batch, hidden_dim])
        # elif self.integration_mode == 'cnn_only_2':
        #     x = x.mean(1)
        elif self.integration_mode == 'cnn_only_1':
            x = x.reshape(batch_size*seq_len, self.back_bone_dim)

        if self.integration_mode == 'cnn_only_1':
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
            diff_shape_code = self.fc_shape_code(torch.cat([x, inp_pre_shape_code.detach()], dim=-1))
            # Convert cordinates.
            diff_pos_wrd = torch.sum(diff_pos_obj[..., None, :]*inp_pre_o2w, -1) * inp_pre_scale_wrd
            diff_green_wrd = torch.sum(diff_green_obj[..., None, :]*inp_pre_o2w, -1) * inp_pre_scale_wrd
            diff_red_wrd = torch.sum(diff_red_obj[..., None, :]*inp_pre_o2w, -1) * inp_pre_scale_wrd
            # Reshape update to [Seq, batch, dim].
            diff_pos_wrd = diff_pos_wrd.reshape(batch_size, -1, 3).permute(1, 0, 2)
            diff_green_wrd = diff_green_wrd.reshape(batch_size, -1, 3).permute(1, 0, 2)
            diff_red_wrd = diff_red_wrd.reshape(batch_size, -1, 3).permute(1, 0, 2)
            diff_scale = diff_scale.reshape(batch_size, -1, 1).permute(1, 0, 2)
            diff_shape_code = diff_shape_code.reshape(batch_size, -1, self.latent_size).permute(1, 0, 2)
            # Get integrated update.
            if self.loss_timing=='after_mean': # or model_mode=='val':
            # if self.loss_timing=='after_mean' or model_mode=='val':
                diff_pos_wrd = diff_pos_wrd.mean(0)
                diff_green_wrd = diff_green_wrd.mean(0)
                diff_red_wrd = diff_red_wrd.mean(0)
                diff_scale = diff_scale.mean(0)
                diff_shape_code = diff_shape_code.mean(0)
        else:
            if self.loss_timing=='after_mean' or model_mode=='val':
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
            else:
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
                diff_shape_code = self.fc_shape_code(torch.cat([x, inp_pre_shape_code.detach()], dim=-1))
                # Convert cordinates.
                diff_pos_wrd = torch.sum(diff_pos_obj[..., None, :]*inp_pre_o2w, -1) * inp_pre_scale_wrd
                diff_green_wrd = torch.sum(diff_green_obj[..., None, :]*inp_pre_o2w, -1) * inp_pre_scale_wrd
                diff_red_wrd = torch.sum(diff_red_obj[..., None, :]*inp_pre_o2w, -1) * inp_pre_scale_wrd
                # Reshape update to [Seq, batch, dim].
                diff_pos_wrd = diff_pos_wrd.reshape(batch_size, -1, 3).permute(1, 0, 2)
                diff_green_wrd = diff_green_wrd.reshape(batch_size, -1, 3).permute(1, 0, 2)
                diff_red_wrd = diff_red_wrd.reshape(batch_size, -1, 3).permute(1, 0, 2)
                diff_scale = diff_scale.reshape(batch_size, -1, 1).permute(1, 0, 2)
                diff_shape_code = diff_shape_code.reshape(batch_size, -1, self.latent_size).permute(1, 0, 2)

        return diff_pos_wrd, diff_green_wrd, diff_red_wrd, diff_scale, diff_shape_code


    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)





class TransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output



class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2, self.attn_output_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                                        key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src



def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])





class TransformerEncoderLayer_woNorm(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer_woNorm, self).__init__()
        ##################################################
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0)
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.linear2 = nn.Linear(dim_feedforward, d_model)
        ##################################################
        dropout = 0.1
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        ##################################################
        self.activation = _get_activation_fn(activation)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2, self.attn_output_weights = self.self_attn(src, src, src)
        ##################################################
        # src = src + src2
        # src2 = self.linear2(self.activation(self.linear1(src)))
        # return src + src2
        ##################################################
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return src + self.dropout2(src2)





# class Attention(nn.Module):

#     def __init__(self, dim, n_head, head_dim, dropout=0.):
#         super().__init__()
#         self.n_head = n_head
#         inner_dim = n_head * head_dim
#         self.to_q = nn.Linear(dim, inner_dim, bias=False)
#         self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
#         self.scale = head_dim ** -0.5
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout),
#         )

#     def forward(self, fr, to=None):
#         if to is None:
#             to = fr
#         q = self.to_q(fr)
#         k, v = self.to_kv(to).chunk(2, dim=-1)
#         q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=self.n_head), [q, k, v])

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#         attn = F.softmax(dots, dim=-1) # b h n n
#         out = torch.matmul(attn, v)
#         out = einops.rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)


# class FeedForward(nn.Module):

#     def __init__(self, dim, ff_dim, dropout=0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, ff_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(ff_dim, dim),
#             nn.Dropout(dropout),
#         )

#     def forward(self, x):
#         return self.net(x)


# class PreNorm(nn.Module):

#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn

#     def forward(self, x):
#         return self.fn(self.norm(x))


# class TransformerEncoderInr(nn.Module):

#     def __init__(self, dim, depth, n_head, head_dim, ff_dim, dropout=0.):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 PreNorm(dim, Attention(dim, n_head, head_dim, dropout=dropout)),
#                 PreNorm(dim, FeedForward(dim, ff_dim, dropout=dropout)),
#             ]))

#     def forward(self, x):
#         for norm_attn, norm_ff in self.layers:
#             x = x + norm_attn(x)
#             x = x + norm_ff(x)
#         return x





# if __name__=='__main__':

#     batch_size = 2
#     seq_len = 5
#     cha_num = 11
#     H = 256
#     W = 256

#     mmm = torch.ones(1, requires_grad=True)
#     inp = mmm * torch.ones(batch_size, seq_len, cha_num, H, W)

#     df_net = optimize_former(input_type='osmap', positional_encoding_mode='non')
#     # df_net = deep_optimizer(input_type='osmap', output_diff_coordinate='obj')


#     pre_scale = torch.ones(batch_size, 1)
#     pre_shape_code = torch.ones(batch_size, 256)
#     pre_o2w = torch.ones(batch_size, 3, 3)
#     inp_pre_scale = pre_scale[:, None, :].expand(-1, seq_len, -1).reshape(-1, 1)
#     inp_pre_shape_code = pre_shape_code[:, None, :].expand(-1, seq_len, -1).reshape(-1, 256)
#     inp_pre_o2w = pre_o2w[:, None, :, :].expand(-1, seq_len, -1, -1).reshape(-1, 3, 3)
#     diff_pos_wrd, diff_obj_axis_green_wrd, diff_obj_axis_red_wrd, diff_scale, diff_shape_code \
#         = df_net(inp, 0, pre_scale, pre_shape_code, pre_o2w, 0)
#     # diff_pos_wrd, diff_obj_axis_green_wrd, diff_obj_axis_red_wrd, diff_scale, diff_shape_code \
#     #     = df_net(inp, 0, 0, 0, inp_pre_scale, inp_pre_shape_code, 0, 0, 0, 0, 0, 0, 0, inp_pre_o2w, 'train')
    
#     loss = torch.norm(diff_pos_wrd)
#     loss.backward()
#     import pdb; pdb.set_trace()