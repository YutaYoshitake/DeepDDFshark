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

    def __init__(self, args, in_channel=5):
        super().__init__()
        self.code_mode = args.code_mode

        self.backbone_encoder = nn.Sequential(
                ResNet50(args, in_channel=in_channel), 
                )
        self.backbone_fc = nn.Sequential(
                nn.Linear(2048 + 7, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, 512), nn.LeakyReLU(0.2),
                )
        self.fc_axis_green = nn.Sequential(
                nn.Linear(512 + 3, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 3), 
                )
        self.fc_axis_red = nn.Sequential(
                nn.Linear(512 + 3, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 3), 
                )
        self.fc_shape_code = nn.Sequential(
                nn.Linear(512 + args.latent_size, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, args.latent_size), 
                )
        self.fc_pos = nn.Sequential(
                nn.Linear(512 + 3, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 3), 
                )
        self.fc_scale = nn.Sequential(
                nn.Linear(512 + 1, 256), nn.LeakyReLU(0.2), 
                nn.Linear(256, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 1), nn.Softplus(beta=.7), 
                )
        
        self.integrate_mode = args.integrate_mode

        if self.integrate_mode == 'sha_v0':
            self.integrater = mha_integrator_v0(d_model=512, nhead=1, add_zero_attn=False, update_keys=['update_value'], use_nn_linear = False)

        elif self.integrate_mode == 'sha_v1':
            self.integrater = mha_integrator_v0(d_model=512, nhead=1, add_zero_attn=True, update_keys=['update_value'], use_nn_linear = False)

        elif self.integrate_mode == 'sha_v3':
            self.integrater = mha_integrator_v0(d_model=512, nhead=1, add_zero_attn=False, update_keys=['update_value'], use_nn_linear = True)

        elif self.integrate_mode == 'sha_v2':
            self.integrater = mha_integrator_v0(d_model=512, nhead=1, add_zero_attn=False, update_keys=['P', 'z'], use_nn_linear = False)

        elif self.integrate_mode == 'sha_v4':
            self.integrater = mha_integrator_v0(d_model=512, nhead=1, add_zero_attn=False, update_keys=['update_value'], use_nn_linear = False)

        elif self.integrate_mode == 'mha_v0':
            self.integrater = mha_integrator_v0(d_model=512, heads_integrate_mode='average')

        elif self.integrate_mode == 'mha_v1':
            self.integrater = mha_integrator_v0(d_model=512, heads_integrate_mode='learnable_weighted_average')

        elif self.integrate_mode == 'mha_v2':
            self.integrater = mha_integrator_v0(d_model=512, heads_integrate_mode='average')

        elif self.integrate_mode == 'transformer_v0':
            self.integrater = atten_integrator_v0(args, d_model=512, use_integrated_q=True)
        
        elif self.integrate_mode in {'transformer_v1', 'transformer_v2'}:
            self.integrater = atten_integrator_v0(args, d_model=512)


    def forward(self, inp, bbox_info, pre_pos, pre_axis_green, pre_axis_red, pre_scale, pre_shape_code, with_x=False):
        # Backbone.
        x = self.backbone_encoder(inp)
        x = x.reshape(inp.shape[0], -1)
        x = self.backbone_fc(torch.cat([x, bbox_info], dim=-1))

        # Get pose diff.
        diff_pos = self.fc_pos(torch.cat([x, pre_pos], dim=-1))
        # diff_pos = torch.zeros_like(pre_pos)

        # Get axis diff.
        diff_axis_green = self.fc_axis_green(torch.cat([x, pre_axis_green], dim=-1))
        diff_axis_red = self.fc_axis_red(torch.cat([x, pre_axis_red], dim=-1))

        # Get scale diff.
        diff_scale_cim = self.fc_scale(torch.cat([x, pre_scale], dim=-1)) + 1e-5 # Prevent scale=0.

        # Get shape code diff.
        diff_shape_code = self.fc_shape_code(torch.cat([x, pre_shape_code], dim=-1))

        if not with_x:
            return diff_pos, diff_axis_green, diff_axis_red, diff_scale_cim, diff_shape_code

        elif with_x:
            return diff_pos, diff_axis_green, diff_axis_red, diff_scale_cim, diff_shape_code, x.clone()


    def integrate_update(
            self, 
            inp, 
            bbox_info, 
            pre_obj_pos_cim, 
            pre_obj_axis_green_cam, 
            pre_obj_axis_red_cam, 
            pre_obj_scale_cim, 
            pre_shape_code, 
            cim2im_scale, 
            im2cam_scale, 
            w2c, 
            latent_size, 
            opt_frame_num, 
            batch_size, 
            ):

        # Estimate diff.
        diff_pos_cim, diff_obj_axis_green_cam, diff_obj_axis_red_cam, diff_scale, diff_shape_code, feature = self.forward(
                                                                                                                inp = inp, 
                                                                                                                bbox_info = bbox_info, 
                                                                                                                pre_pos = pre_obj_pos_cim, 
                                                                                                                pre_axis_green = pre_obj_axis_green_cam, 
                                                                                                                pre_axis_red = pre_obj_axis_red_cam, 
                                                                                                                pre_scale = pre_obj_scale_cim, 
                                                                                                                pre_shape_code = pre_shape_code, 
                                                                                                                with_x=True)

        # Convert cordinates.
        diff_pos_cam = diffcim2diffcam(diff_pos_cim, cim2im_scale, im2cam_scale)
        diff_pos_wrd = torch.sum(diff_pos_cam[..., None, :]*w2c.permute(0, 2, 1), -1)
        diff_obj_axis_green_wrd = torch.sum(diff_obj_axis_green_cam[..., None, :]*w2c.permute(0, 2, 1), -1)
        diff_obj_axis_red_wrd = torch.sum(diff_obj_axis_red_cam[..., None, :]*w2c.permute(0, 2, 1), -1)

        # Get integrated update.
        if self.integrate_mode in {'average', 'average_but_loss_against_before'}:
            diff_pos_wrd = diff_pos_wrd.reshape(batch_size, -1, 3).permute(1, 0, 2)
            diff_scale = diff_scale.reshape(batch_size, -1, 1).permute(1, 0, 2)
            diff_obj_axis_green_wrd = diff_obj_axis_green_wrd.reshape(batch_size, -1, 3).permute(1, 0, 2)
            diff_obj_axis_red_wrd = diff_obj_axis_red_wrd.reshape(batch_size, -1, 3).permute(1, 0, 2)
            diff_shape_code = diff_shape_code.reshape(batch_size, -1, latent_size).permute(1, 0, 2)
            if (self.integrate_mode == 'average') or (self.code_mode=='VAL'):
                diff_pos_wrd = diff_pos_wrd.mean(0)
                diff_scale = diff_scale.mean(0)
                diff_obj_axis_green_wrd = diff_obj_axis_green_wrd.mean(0)
                diff_obj_axis_red_wrd = diff_obj_axis_red_wrd.mean(0)
                diff_shape_code = diff_shape_code.mean(0)

        elif self.integrater.update_keys==['update_value']:
            pre_update = torch.cat([diff_pos_wrd, diff_scale, diff_obj_axis_green_wrd, diff_obj_axis_red_wrd, diff_shape_code], dim=-1)
            pre_update = pre_update.reshape(batch_size, opt_frame_num, 10+latent_size).permute(1, 0, 2) # [sequence, batch_size, embed_dim]
            pre_update_dict = {'update_value':pre_update}
        
            tgt = feature.clone().reshape(batch_size, opt_frame_num, -1).permute(1, 0, 2) # [sequence, batch_size, embed_dim]
            memory = feature.clone().reshape(batch_size, opt_frame_num, -1).permute(1, 0, 2) # [sequence, batch_size, embed_dim]
            integrated_update_dict, raw_update_dict, attn_weight_dict = self.integrater(tgt, memory, pre_update_dict)

            if (self.integrate_mode in {'sha_v4', 'mha_v2'}) and self.code_mode=='TRAIN':
                diff_pos_wrd = raw_update_dict['update_value'][..., :3]
                diff_scale = raw_update_dict['update_value'][..., 3].unsqueeze(-1)
                diff_obj_axis_green_wrd = raw_update_dict['update_value'][..., 4:7]
                diff_obj_axis_red_wrd = raw_update_dict['update_value'][..., 7:10]
                diff_shape_code = raw_update_dict['update_value'][..., 10:]
            else:
                diff_pos_wrd = integrated_update_dict['update_value'][:, :3]
                diff_scale = integrated_update_dict['update_value'][:, 3].unsqueeze(-1)
                diff_obj_axis_green_wrd = integrated_update_dict['update_value'][:, 4:7]
                diff_obj_axis_red_wrd = integrated_update_dict['update_value'][:, 7:10]
                diff_shape_code = integrated_update_dict['update_value'][:, 10:]

        elif self.integrater.update_keys==['P', 'z']:
            pre_update_P = torch.cat([diff_pos_wrd, diff_scale, diff_obj_axis_green_wrd, diff_obj_axis_red_wrd], dim=-1)
            pre_update_P = pre_update_P.reshape(batch_size, opt_frame_num, 10).permute(1, 0, 2) # [sequence, batch_size, embed_dim]
            pre_update_z = diff_shape_code.reshape(batch_size, opt_frame_num, latent_size).permute(1, 0, 2) # [sequence, batch_size, embed_dim]
            pre_update_dict = {'P':pre_update_P, 'z':pre_update_z}
        
            tgt = feature.clone().reshape(batch_size, opt_frame_num, -1).permute(1, 0, 2) # [sequence, batch_size, embed_dim]
            memory = feature.clone().reshape(batch_size, opt_frame_num, -1).permute(1, 0, 2) # [sequence, batch_size, embed_dim]
            integrated_update_dict, _, _ = self.integrater(tgt, memory, pre_update_dict)

            diff_pos_wrd = integrated_update_dict['P'][:, :3]
            diff_scale = integrated_update_dict['P'][:, 3].unsqueeze(-1)
            diff_obj_axis_green_wrd = integrated_update_dict['P'][:, 4:7]
            diff_obj_axis_red_wrd = integrated_update_dict['P'][:, 7:]
            diff_shape_code = integrated_update_dict['z']

        return diff_pos_wrd, diff_scale, diff_obj_axis_green_wrd, diff_obj_axis_red_wrd, diff_shape_code





class mha_integrator_v0(pl.LightningModule):
    # k個の更新値をバリューとして受け取り、それを統合する1層のMulti Head Attention層

    def __init__(
        self, 
        d_model, 
        nhead = 8, 
        update_keys = ['update_value'], 
        trans_integrate_mode = 'average', 
        add_zero_attn = False, 
        heads_integrate_mode = 'average', 
        use_nn_linear = False, 
        ):
        super(mha_integrator_v0, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.update_keys = update_keys
        self.trans_integrate_mode = trans_integrate_mode

        self.use_nn_linear = use_nn_linear
        if self.update_keys == ['update_value']:
            if use_nn_linear:
                print('nn_Liner')
                self.W_k = nn.Linear(d_model, d_model)
                self.W_q = nn.Linear(d_model, d_model)
                nn.init.xavier_uniform_(self.W_k.weight)
                nn.init.xavier_uniform_(self.W_q.weight)
            else:
                self.W_k = nn.Parameter(torch.empty(nhead, d_model, self.d_k))
                self.W_q = nn.Parameter(torch.empty(nhead, d_model, self.d_k))
                nn.init.xavier_uniform_(self.W_q)
                nn.init.xavier_uniform_(self.W_k)
        else:
            if use_nn_linear:
                self.W_k = nn.ModuleDict({})
                self.W_q = nn.ModuleDict({})
                for key_i in update_keys:
                    self.W_k[key_i] = nn.Linear(d_model, d_model)
                    self.W_q[key_i] = nn.Linear(d_model, d_model)
                    nn.init.xavier_uniform_(self.W_k[key_i].weight)
                    nn.init.xavier_uniform_(self.W_q[key_i].weight)
            else:
                self.W_k = nn.ParameterDict({})
                self.W_q = nn.ParameterDict({})
                for key_i in update_keys:
                    self.W_k[key_i] = nn.Parameter(torch.empty(nhead, d_model, self.d_k))
                    self.W_q[key_i] = nn.Parameter(torch.empty(nhead, d_model, self.d_k))
                    nn.init.xavier_uniform_(self.W_q[key_i])
                    nn.init.xavier_uniform_(self.W_k[key_i])

        self.add_zero_attn = add_zero_attn

        self.heads_integrate_mode = heads_integrate_mode
        if self.heads_integrate_mode == 'learnable_weighted_average':
            self.heads_integrate_weight = nn.Parameter(torch.ones(nhead, 1, 1, 1))
        


    def forward(
        self,
        inp_q: torch.Tensor,
        inp_k: torch.Tensor,
        update,
        ) -> torch.Tensor:

        att_update = {}
        att_weight = {}
        integrated_update = {}

        for key_i in self.update_keys:
            q = inp_q.clone().permute(1, 0, 2)
            k = inp_k.clone().permute(1, 0, 2)
            v = update[key_i].permute(1, 0, 2) # [batch_size, seq_len, update_dim]
            batch_size, seq_len = q.size(0), q.size(1)
            v_dim = v.size(-1)

            if self.use_nn_linear:
                q = self.W_q(q).reshape(batch_size, seq_len, self.nhead, self.d_k).permute(2, 0, 1, 3)
                k = self.W_k(k).reshape(batch_size, seq_len, self.nhead, self.d_k).permute(2, 0, 1, 3)
                v = v.repeat(self.nhead, 1, 1, 1)  # head, batch_size, seq_len, d_model
            else:
                q = q.repeat(self.nhead, 1, 1, 1)  # head, batch_size, seq_len, d_model
                k = k.repeat(self.nhead, 1, 1, 1)  # head, batch_size, seq_len, d_model
                v = v.repeat(self.nhead, 1, 1, 1)  # head, batch_size, seq_len, d_model
                if self.update_keys == ['update_value']:
                    q = torch.einsum("hijk,hkl->hijl", (q, self.W_q)) # head, batch_size, seq_len, d_k
                    k = torch.einsum("hijk,hkl->hijl", (k, self.W_k)) # head, batch_size, seq_len, d_k
                else:
                    q = torch.einsum("hijk,hkl->hijl", (q, self.W_q[key_i])) # head, batch_size, seq_len, d_k
                    k = torch.einsum("hijk,hkl->hijl", (k, self.W_k[key_i])) # head, batch_size, seq_len, d_k
            
            q = q.contiguous().view(self.nhead * batch_size, seq_len, self.d_k) # [batch_size, opt_frame_num, d_model]
            k = k.contiguous().view(self.nhead * batch_size, seq_len, self.d_k)
            v = v.contiguous().view(self.nhead * batch_size, seq_len, v_dim)

            if self.add_zero_attn:
                k = torch.cat([k, torch.zeros((self.nhead * batch_size, 1, self.d_k)).to(k)], dim=1)
                v = torch.cat([v, torch.zeros((self.nhead * batch_size, 1, v_dim)).to(v)], dim=1)
            k_size = k.size(1)
            scalar = np.sqrt(self.d_k)

            attention_weight = torch.matmul(q, torch.transpose(k, 1, 2)) / scalar # Q*X^T / (D^0.5)
            attention_weight = F.softmax(attention_weight, dim=2) # get attention_weight
            x = torch.matmul(attention_weight, v) # attention_weight * X
            x = x.view(self.nhead, batch_size, seq_len, v_dim)
            attention_weight = attention_weight.view(self.nhead, batch_size, seq_len, k_size)

            if self.heads_integrate_mode == 'average':
                x = x.mean(dim=0)
                attention_weight = attention_weight.mean(dim=0)
            if self.heads_integrate_mode == 'learnable_weighted_average':
                weights = F.softmax(self.heads_integrate_weight, dim=0)
                x = torch.sum(weights*x, dim=0)
                attention_weight = torch.sum(weights*attention_weight, dim=0)

            att_update[key_i] = x.permute(1, 0, 2)
            att_weight[key_i] = attention_weight.permute(1, 0, 2)

            if self.trans_integrate_mode=='average':
                integrated_update[key_i] = att_update[key_i].mean(0)

        return integrated_update, att_update, att_weight





class atten_integrator_v0(pl.LightningModule):
    # k個の更新値をバリューとして受け取り、それを統合する1層のTransfoemr

    def __init__(
        self, 
        args, 
        d_model, 
        nhead=False, 
        dropout=0.1, 
        activation="relu", 
        update_keys=['update_value'], 
        trans_integrate_mode='average', 
        use_integrated_q = False, 
        use_self_attn=False, ):
        super(atten_integrator_v0, self).__init__()

        # self.use_self_attn = use_self_attn
        # if self.use_self_attn:
        #     self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        #     self.norm1 = nn.LayerNorm(d_model)
        #     self.dropout1 = nn.Dropout(dropout)

        self.use_integrated_q = use_integrated_q
        if self.use_integrated_q:
            self.integrater_q = nn.Sequential(
                                    nn.Linear(args.itr_frame_num * d_model, d_model), nn.LeakyReLU(0.2),
                                    nn.Linear(d_model, d_model), nn.LeakyReLU(0.2),
                                    nn.Linear(d_model, d_model), nn.LeakyReLU(0.2),
                                    )
            # self.integrater_q = nn.Sequential(
            #                         nn.Linear(args.itr_frame_num * d_model, d_model), nn.LeakyReLU(0.2),
            #                         )

        self.update_keys = update_keys
        self.linear_q = nn.ModuleDict({})
        self.linear_k = nn.ModuleDict({})
        for key in update_keys:
            self.linear_q[key] = nn.Linear(d_model, d_model)
            self.linear_k[key] = nn.Linear(d_model, d_model)
        self.trans_integrate_mode = trans_integrate_mode


    def forward(self, tgt: Tensor, memory: Tensor, update, 
                tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        # tgt : [dec_sequence, batch_size, embed_dim]
        # memory : [enc_sequence, batch_size, embed_dim]
        x = tgt
        batch_size = tgt.shape[1]

        # if self.use_self_attn:
        #     x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
        if self.use_integrated_q:
            x = x.permute(1, 0, 2).reshape(batch_size, -1)
            x = self.integrater_q(x).unsqueeze(0) # [1, batch_size, embed_dim]
        
        att_update, att_weight = self.update_attn(x, memory, update)
        
        update = {}
        for key_i in self.update_keys:
            if self.trans_integrate_mode=='average':
                update[key_i] = att_update[key_i].mean(0)

        return update, att_update, att_weight


    def update_attn(
        self,
        inp_q: torch.Tensor,
        inp_k: torch.Tensor,
        update: torch.Tensor,
        ) -> torch.Tensor:
        
        if not self.update_keys == list(update.keys()):
            print('update key eror')
            sys.exit()
        
        att_update = {}
        att_weight = {}
        
        for key_i in self.update_keys:
            q = self.linear_q[key_i](inp_q.permute(1, 0, 2)) # [batch_size, opt_frame_num, d_model]
            k = self.linear_k[key_i](inp_k.permute(1, 0, 2)) # [batch_size, opt_frame_num, d_model]
            v = update[key_i].permute(1, 0, 2) # [batch_size, opt_frame_num, update_dim]

            d_k = k.shape[-1]
            scalar = np.sqrt(d_k)
            attention_weight = torch.matmul(q, torch.transpose(k, 1, 2)) / scalar # Q*X^T / (D^0.5)
            attention_weight = nn.functional.softmax(attention_weight, dim=2) # Attention weightを計算
            x = torch.matmul(attention_weight, v) # (Attention weight) * X により重み付け.

            att_update[key_i] = x.permute(1, 0, 2)
            att_weight[key_i] = attention_weight.permute(1, 0, 2)

        return att_update, att_weight # [opt_frame_num, batch_size, update_dim]


    # # self-attention block
    # def _sa_block(self, x: Tensor,
    #               attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
    #     x = self.self_attn(x, x, x,
    #                        attn_mask=attn_mask,
    #                        key_padding_mask=key_padding_mask, 
    #                        need_weights=False)[0]
    #     return self.dropout1(x)





class vanilla_TransformerDecoder(pl.LightningModule):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(vanilla_TransformerDecoder, self).__init__()

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm


def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
    
    output = tgt

    for mod in self.layers:
        output = mod(output, memory, tgt_mask=tgt_mask,
                        memory_mask=memory_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask)

    if self.norm is not None:
        output = self.norm(output)

    return output





def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])





def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu





class vanilla_TransformerDecoderLayer(pl.LightningModule):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(vanilla_TransformerDecoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        x = tgt
        x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
        x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
        x = self.norm3(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)





# if __name__=='__main__':

#     batch_size = 32
#     opt_frame_num = 5
#     decoder_layer = transintegrator_single_v1(d_model=512, nhead=8) #, update_keys=['P', 'z'])
#     memory = torch.rand(opt_frame_num, batch_size, 512) # From ResNet50 ?
#     update_i = {'update_value':torch.rand(opt_frame_num, batch_size, 263)} #, 'z':torch.rand(opt_frame_num, batch_size, 256)} # update_i
#     tgt = torch.rand(opt_frame_num, batch_size, 512) # From ResNet18 ?
#     update, weight = decoder_layer(tgt, memory, update_i)

#     import pdb; pdb.set_trace()


    # # Get args
    # args = get_args()
    # args.gpu_num = 0 # log used gpu num.

    # model = ResNet18(args, 5)
    # dummy_image = torch.zeros([10, 5, 256, 256])

#     import pdb; pdb.set_trace()
