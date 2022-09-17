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
        main_layers_name = 'only_mlp', 
        optnet_InOut_type = 'encinObs_decinEst_decoutDif', 
        input_type = 'depth', 
        num_encoder_layers = 2, 
        num_decoder_layers = 1, 
        hidden_dim = 256, 
        num_head = 8, 
        dim_feedforward = 1024, 
        latent_size = 256, 
        position_dim = 7, 
        dropout = 0.1, 
        positional_encoding_mode = 'non', 
        ):
        super(optimize_former, self).__init__()

        # Main layers.
        self.main_layers_name = main_layers_name
        if main_layers_name=='autoreg':
            self.optnet_InOut_type = optnet_InOut_type.split('_')
            self.main_layers = auto_regressive_model(num_encoder_layers=num_encoder_layers, 
                                                     num_decoder_layers=num_decoder_layers, 
                                                     enc_in_dim=512, 
                                                     dec_in_dim=512, 
                                                     num_head=num_head, 
                                                     hidden_dim=hidden_dim, 
                                                     dim_feedforward=dim_feedforward, )
        elif main_layers_name=='encoder_model':
            self.main_layers = encoder_model(inp_embed_dim=3*512, #512, 
                                             num_encoder_layers=num_encoder_layers, 
                                             num_head=num_head, 
                                             hidden_dim=hidden_dim, 
                                             dim_feedforward=dim_feedforward, )
        elif main_layers_name=='only_mlp':
            self.main_layers = only_mlp(inp_embed_dim=3*512, 
                                        hidden_dim=hidden_dim, 
                                        mlp_layers_num=num_encoder_layers, 
                                        mlp_hidden_dim=1024, 
                                        dropout=0.1, )

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
                nn.Linear(256, 1), nn.Softplus(beta=0.7))
        self.fc_shape_code = nn.Sequential(
                nn.Linear(hidden_dim + self.latent_size, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, self.latent_size))
        # import pdb; pdb.set_trace()


    def forward(self, obs_embed, est_embed, dif_embed, past_itr_length, 
        inp_pre_obj_pos_wrd, inp_pre_obj_green_wrd, inp_pre_obj_red_wrd, inp_pre_obj_scale_wrd, inp_pre_obj_shape_code, 
        pre_pos_wrd, pre_green_wrd, pre_red_wrd, pre_scale_wrd, pre_shape_code):

        if self.main_layers_name in {'autoreg'}:
            if self.optnet_InOut_type[0] == 'encinObs':
                inp_enc = obs_embed.permute(1, 0, 2)
            if self.optnet_InOut_type[1] == 'decinEst':
                inp_dec = {'embed': est_embed.permute(1, 0, 2), 'length': past_itr_length}
            elif self.optnet_InOut_type[1] == 'decinDif':
                inp_dec = {'embed': dif_embed.permute(1, 0, 2), 'length': past_itr_length}

            x = self.main_layers(inp_enc, inp_dec)[-past_itr_length[-1]:, :, :].mean(0)

            if self.optnet_InOut_type[2] == 'decoutDif':
                diff_pos_wrd = self.fc_pos(torch.cat([x, pre_pos_wrd.detach()], dim=-1))
                diff_green_wrd = self.fc_axis_green(torch.cat([x, pre_green_wrd.detach()], dim=-1))
                diff_red_wrd = self.fc_axis_red(torch.cat([x, pre_red_wrd.detach()], dim=-1))
                diff_scale_wrd = self.fc_scale(torch.cat([x, pre_scale_wrd.detach()], dim=-1)) + 1e-5
                diff_shape_code = self.fc_shape_code(torch.cat([x, pre_shape_code.detach()], dim=-1))

                est_pos_wrd = pre_pos_wrd.detach() + diff_pos_wrd
                est_green_wrd = F.normalize(pre_green_wrd.detach() + diff_green_wrd, dim=-1)
                est_red_wrd = F.normalize(pre_red_wrd.detach() + diff_red_wrd, dim=-1)
                est_scale_wrd = pre_scale_wrd.detach() * diff_scale_wrd
                est_shape_code = pre_shape_code.detach() + diff_shape_code
            
            elif self.optnet_InOut_type[2] == 'decoutEst':
                est_pos_wrd = self.fc_pos(torch.cat([x, pre_pos_wrd.detach()], dim=-1))
                est_green_wrd = F.normalize(self.fc_axis_green(torch.cat([x, pre_green_wrd.detach()], dim=-1)))
                est_red_wrd = F.normalize(self.fc_axis_red(torch.cat([x, pre_red_wrd.detach()], dim=-1)))
                est_scale_wrd = self.fc_scale(torch.cat([x, pre_scale_wrd.detach()], dim=-1)) + 1e-5
                est_shape_code = self.fc_shape_code(torch.cat([x, pre_shape_code.detach()], dim=-1))

        else:
            if self.main_layers_name in {'encoder_model'}:
                inp = torch.cat([obs_embed, est_embed, dif_embed], dim=2).permute(1, 0, 2) # obs_embed.permute(1, 0, 2) # [seq, batch, inp_embed_dim]
                x = self.main_layers(inp).mean(0) # [batch, hidden_dim]
                diff_pos_wrd = self.fc_pos(torch.cat([x, pre_pos_wrd.detach()], dim=-1))
                diff_green_wrd = self.fc_axis_green(torch.cat([x, pre_green_wrd.detach()], dim=-1))
                diff_red_wrd = self.fc_axis_red(torch.cat([x, pre_red_wrd.detach()], dim=-1))
                diff_scale_wrd = self.fc_scale(torch.cat([x, pre_scale_wrd.detach()], dim=-1)) + 1e-5
                diff_shape_code = self.fc_shape_code(torch.cat([x, pre_shape_code.detach()], dim=-1))

            elif self.main_layers_name in {'only_mlp'}:
                inp = torch.cat([obs_embed, est_embed, dif_embed], dim=2).permute(1, 0, 2) # [seq, batch, inp_embed_dim]
                x = self.main_layers(inp) # [seq, batch, hidden_dim]
                diff_pos_wrd = self.fc_pos(torch.cat([x, inp_pre_obj_pos_wrd.permute(1, 0, 2).detach()], dim=-1)).mean(0)
                diff_green_wrd = self.fc_axis_green(torch.cat([x, inp_pre_obj_green_wrd.permute(1, 0, 2).detach()], dim=-1)).mean(0)
                diff_red_wrd = self.fc_axis_red(torch.cat([x, inp_pre_obj_red_wrd.permute(1, 0, 2).detach()], dim=-1)).mean(0)
                diff_scale_wrd = self.fc_scale(torch.cat([x, inp_pre_obj_scale_wrd.permute(1, 0, 2).detach()], dim=-1)).mean(0) + 1e-5
                diff_shape_code = self.fc_shape_code(torch.cat([x, inp_pre_obj_shape_code.permute(1, 0, 2).detach()], dim=-1)).mean(0)

            # Get updated estimations.
            est_pos_wrd = pre_pos_wrd.detach() + diff_pos_wrd
            est_green_wrd = F.normalize(pre_green_wrd.detach() + diff_green_wrd, dim=-1)
            est_red_wrd = F.normalize(pre_red_wrd.detach() + diff_red_wrd, dim=-1)
            est_scale_wrd = pre_scale_wrd.detach() * diff_scale_wrd
            est_shape_code = pre_shape_code.detach() + diff_shape_code

        return est_pos_wrd, est_green_wrd, est_red_wrd, est_scale_wrd, est_shape_code



class encoder_model(pl.LightningModule):

    def __init__(
        self, 
        inp_embed_dim=3*512, 
        num_encoder_layers=3, 
        num_head=8, 
        hidden_dim=256, 
        dim_feedforward=1024, 
        ):
        super(encoder_model, self).__init__()

        self.align_mlp = nn.Linear(inp_embed_dim, hidden_dim)
        encoder_layer = TransformerEncoderLayer_woNorm(d_model=hidden_dim, nhead=num_head, dim_feedforward=dim_feedforward, dropout=0.0, activation="relu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, None) # , 1, encoder_norm)

    def forward(self, inp):
        x = self.align_mlp(inp)
        return self.encoder(x)



def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu



class only_mlp(pl.LightningModule):

    def __init__(
        self, 
        inp_embed_dim=3*512, 
        hidden_dim=256, 
        mlp_layers_num=5, 
        mlp_hidden_dim=1024, 
        dropout=0.1, 
        ):
        super(only_mlp, self).__init__()

        self.mlp_layers = nn.Sequential()
        if mlp_layers_num == 0:
            self.mlp_layers = nn.Linear(inp_embed_dim, hidden_dim)
        else:
            for mlp_layer_idx in range(mlp_layers_num):
                if mlp_layer_idx == 0:
                    inp_dim, out_dim = inp_embed_dim, mlp_hidden_dim
                elif mlp_layer_idx == mlp_layers_num-1:
                    inp_dim, out_dim = mlp_hidden_dim, hidden_dim
                else:
                    inp_dim, out_dim = mlp_hidden_dim, mlp_hidden_dim
                self.mlp_layers.add_module(f'fc_{str(mlp_layer_idx).zfill(3)}', nn.Linear(inp_dim, out_dim))
                self.mlp_layers.add_module(f'act_{str(mlp_layer_idx).zfill(3)}', nn.ReLU())
                self.mlp_layers.add_module(f'drp_{str(mlp_layer_idx).zfill(3)}', nn.Dropout(dropout))

    def forward(self, inp):
        return self.mlp_layers(inp)



class auto_regressive_model(pl.LightningModule):
    def __init__(
        self, 
        num_encoder_layers=2, 
        num_decoder_layers=1, 
        enc_in_dim=512, 
        dec_in_dim=512, 
        num_head=8, 
        hidden_dim=256, 
        dim_feedforward=1024, 
        ):
        super(auto_regressive_model, self).__init__()

        # Encoder.
        self.encoder = encoder_model(inp_embed_dim=512, num_encoder_layers=num_encoder_layers, num_head=num_head, hidden_dim=hidden_dim, dim_feedforward=dim_feedforward)
        
        # Decoder.
        decoder_layer = TransformerDecoderLayer_woNorm(d_model=hidden_dim, nhead=num_head, dim_feedforward=dim_feedforward, dropout=0.0, activation="relu")
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, None)

        self.align_mlp = nn.Linear(512, hidden_dim)
        self.ie = IterationEncoding(hidden_dim)
        self._reset_parameters()


    def forward(self, src, tgt_dict):
        memory = self.encoder(src) # [seq_e, batch, inp_embed_dim] -> [seq_e, batch, hidden_dim]
        tgt = tgt_dict['embed'] # [seq_d=frame*itr_num, batch, inp_embed_dim]
        tgt = self.align_mlp(tgt) # [seq_d, batch, hidden_dim]
        tgt = self.ie(tgt, tgt_dict['length']) # [seq_d, batch, hidden_dim]
        out = self.decoder(tgt, memory)# [seq_d, batch, hidden_dim]
        return out


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



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



class TransformerDecoderLayer_woNorm(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer_woNorm, self).__init__()
        
        dropout = 0.1
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt2, self.self_attn_weights = self.self_attn(tgt, tgt, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt2, self.mha_attn_weights = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        return tgt + self.dropout3(tgt2)



class IterationEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, 1, d_model) # max_len, frame, batch, dim
        pe[:, 0, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, length):
        pe = [self.pe[itr_idx].expand(length_i, -1, -1) for itr_idx, length_i in enumerate(length)]
        pe = torch.cat(pe, dim=0)
        x = x + pe
        return x



# if __name__=='__main__':

#     model = auto_regressive_model()
#     itr_log_max = 5
#     obs = torch.rand((3, 8, 512))
#     est = {}
#     est['length'] = [10, 5, 10]
#     est['embed'] = [torch.rand((10, 8, 512)), torch.rand((5, 8, 512)), torch.rand((10, 8, 512))]
#     est['length'] = est['length'][:itr_log_max]
#     est['embed'] = est['embed'][:itr_log_max]
#     model(obs, est)
