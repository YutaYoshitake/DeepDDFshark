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

from often_use import *
from resnet import *
from fixup_model.transformer_fixup import TransformerEncoder_FixUp, TransformerDecoder_FixUp





class optimize_former(pl.LightningModule):
    def __init__(
        self, 
        main_layers_name = 'onlymlp', 
        input_type = 'depth', 
        num_encoder_layers = 2, 
        num_decoder_layers = 1, 
        hidden_dim = 256, 
        num_head = 8, 
        dim_feedforward = 1024, 
        mlp_hidden_dim = 1024, 
        latent_size = 256, 
        positional_encoding_mode = 'non', 
        add_conf = None, 
        enc_in_dim=3*512, 
        dec_in_dim=2*512, 
        itr_frame_num=1, 
        dropout=0.1, 
        ):
        super(optimize_former, self).__init__()

        # Main layers.
        self.add_conf = add_conf
        self.main_layers_name = main_layers_name
        self.use_past_map = (main_layers_name == 'autoreg') or (add_conf in {'onlydec', 'onlydecv2', 'onlydecv3', 'exp22v2'})
        self.padding_embeddings = add_conf in {'exp22v2', 'onlydecv3'}
        if main_layers_name=='encoder':
            self.main_layers = encoder_model(inp_embed_dim=enc_in_dim, #512, 
                                             num_encoder_layers=num_encoder_layers, 
                                             num_head=num_head, 
                                             hidden_dim=hidden_dim, 
                                             dim_feedforward=dim_feedforward, 
                                             add_conf=add_conf, 
                                             dropout=dropout, )
        elif main_layers_name=='autoreg':
            self.inp_itr_num = 3
            self.dec_inp = 'dif_obs'
            enc_in_dim = 3*512
            dec_in_dim = 3*512
            self.main_layers = auto_regressive_model(num_encoder_layers=num_encoder_layers, 
                                                     num_decoder_layers=num_decoder_layers, 
                                                     enc_in_dim=enc_in_dim, 
                                                     dec_in_dim=dec_in_dim, 
                                                     num_head=num_head, 
                                                     hidden_dim=hidden_dim, 
                                                     dim_feedforward=dim_feedforward, 
                                                     add_conf=add_conf, 
                                                     dropout=dropout, 
                                                     itr_frame_num=itr_frame_num, )
        elif main_layers_name=='onlymlp':
            self.main_layers = nn.Sequential()
            for mlp_layer_idx in range(num_encoder_layers):
                inp_dim = out_dim = mlp_hidden_dim
                if mlp_layer_idx == 0:
                    inp_dim = enc_in_dim
                elif mlp_layer_idx == num_encoder_layers-1:
                    out_dim = hidden_dim
                self.main_layers.add_module(f'fc_{str(mlp_layer_idx).zfill(3)}', nn.Linear(inp_dim, out_dim))
                self.main_layers.add_module(f'act_{str(mlp_layer_idx).zfill(3)}', nn.ReLU())
                self.main_layers.add_module(f'drp_{str(mlp_layer_idx).zfill(3)}', nn.Dropout(dropout))

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


    def forward(self, obs_embed, est_embed, dif_embed, tra_embed, past_itr_length, optim_idx, inp_pre_obj_pos_wrd, 
        inp_pre_obj_green_wrd, inp_pre_obj_red_wrd, inp_pre_o2w, inp_pre_obj_scale_wrd, inp_pre_obj_shape_code, 
        pre_pos_wrd, pre_green_wrd, pre_red_wrd, pre_o2w, pre_scale_wrd, pre_shape_code, decoder=None):
        batch, seq, _ = obs_embed.shape
        current_frame_num = past_itr_length[-1]

        if self.main_layers_name in {'autoreg', 'encoder'}:
            if self.main_layers_name == 'encoder':
                # Make inputs.
                inp = torch.cat([obs_embed, est_embed, dif_embed], dim=-1).permute(1, 0, 2) # [seq, batch, inp_embed_dim]
                # Main layer.
                x = self.main_layers(inp, past_itr_length, decoder) # [batch, hidden_dim]
            elif self.main_layers_name == 'autoreg':
                # Make inputs.
                inp_enc = torch.cat([obs_embed[:, -current_frame_num:, :], 
                                     est_embed[:, -current_frame_num:, :], 
                                     dif_embed[:, -current_frame_num:, :]], dim=-1).permute(1, 0, 2)
                if optim_idx > 0:
                    dec_inp_frame_len_list = past_itr_length[-(self.inp_itr_num+1):-1]
                    dec_inp_end = sum(dec_inp_frame_len_list) + current_frame_num
                    if self.dec_inp == 'dif_obs':
                        inp_dec = torch.cat([obs_embed[:, -dec_inp_end:-current_frame_num, :], 
                                             est_embed[:, -dec_inp_end:-current_frame_num, :], 
                                             dif_embed[:, -dec_inp_end:-current_frame_num, :]], dim=-1).permute(1, 0, 2)
                    elif self.dec_inp == 'dif_est':
                        inp_dec = torch.cat([torch.cat([est_embed[:, -current_frame_num:, :][:, :view_num, :] for view_num in dec_inp_frame_len_list], dim=1), 
                                             est_embed[:, -dec_inp_end:-current_frame_num, :], 
                                             tra_embed[:, -dec_inp_end+current_frame_num:, :]], dim=-1).permute(1, 0, 2)
                    elif self.dec_inp == 'dif_obs_est':
                        inp_dec = torch.cat([est_embed[:, -dec_inp_end:-current_frame_num, :], 
                                             dif_embed[:, -dec_inp_end:-current_frame_num, :], 
                                             tra_embed[:, -dec_inp_end+current_frame_num:, :]], dim=-1).permute(1, 0, 2)
                else:
                    dec_inp_frame_len_list = past_itr_length
                    inp_dec = None
                # Main layer.
                x = self.main_layers(inp_enc, inp_dec, dec_inp_frame_len_list, decoder) # [batch, hidden_dim]
            
            # Make pre_est.
            pre_green_obj = torch.tensor([[0.0, 1.0, 0.0]]).expand(batch, -1).to(x)
            pre_red_obj = torch.tensor([[1.0, 0.0, 0.0]]).expand(batch, -1).to(x)
            pre_pos_obj = torch.tensor([[0.0, 0.0, 0.0]]).expand(batch, -1).to(x)
            pre_scale_obj = torch.tensor([[1.0]]).expand(batch, -1).to(x)
            # Head.
            diff_pos_obj = self.fc_pos(torch.cat([x, pre_pos_obj.detach()], dim=-1))
            diff_green_obj = self.fc_axis_green(torch.cat([x, pre_green_obj.detach()], dim=-1))
            diff_red_obj = self.fc_axis_red(torch.cat([x, pre_red_obj.detach()], dim=-1))
            diff_scale_wrd = self.fc_scale(torch.cat([x, pre_scale_obj.detach()], dim=-1)) + 1e-5 # Prevent scale=0.
            diff_shape_code = self.fc_shape_code(torch.cat([x, pre_shape_code.detach()], dim=-1))
            # Convert cordinates.
            diff_pos_wrd = torch.sum(diff_pos_obj[..., None, :]*pre_o2w, -1) * pre_scale_wrd
            diff_green_wrd = torch.sum(diff_green_obj[..., None, :]*pre_o2w, -1) * pre_scale_wrd
            diff_red_wrd = torch.sum(diff_red_obj[..., None, :]*pre_o2w, -1) * pre_scale_wrd

        elif self.main_layers_name == 'onlymlp':
            # Make inputs.
            inp = torch.cat([obs_embed, est_embed, dif_embed], dim=2) # [batch, seq, inp_embed_dim]
            # Main layer.
            x = self.main_layers(inp) # [batch, seq, inp_embed_dim]
            # Make pre_est.
            pre_green_obj = torch.tensor([[[0.0, 1.0, 0.0]]]).expand(batch, seq, 3).to(x)
            pre_red_obj = torch.tensor([[[1.0, 0.0, 0.0]]]).expand(batch, seq, 3).to(x)
            pre_pos_obj = torch.tensor([[[0.0, 0.0, 0.0]]]).expand(batch, seq, 3).to(x)
            pre_scale_obj = torch.tensor([[[1.0]]]).expand(batch, seq, 1).to(x)
            # Head MLP.
            diff_pos_obj = self.fc_pos(torch.cat([x, pre_pos_obj.detach()], dim=-1))
            diff_green_obj = self.fc_axis_green(torch.cat([x, pre_green_obj.detach()], dim=-1))
            diff_red_obj = self.fc_axis_red(torch.cat([x, pre_red_obj.detach()], dim=-1))
            diff_scale_wrd = self.fc_scale(torch.cat([x, pre_scale_obj.detach()], dim=-1)) + 1e-5 # Prevent scale=0.
            diff_shape_code = self.fc_shape_code(torch.cat([x, inp_pre_obj_shape_code.detach()], dim=-1))
            # Convert cordinates.
            diff_pos_wrd = (torch.sum(diff_pos_obj[..., None, :]*inp_pre_o2w, -1)*inp_pre_obj_scale_wrd).mean(1)
            diff_green_wrd = (torch.sum(diff_green_obj[..., None, :]*inp_pre_o2w, -1)*inp_pre_obj_scale_wrd).mean(1)
            diff_red_wrd = (torch.sum(diff_red_obj[..., None, :]*inp_pre_o2w, -1)*inp_pre_obj_scale_wrd).mean(1)
            diff_scale_wrd = diff_scale_wrd.mean(1)
            diff_shape_code = diff_shape_code.mean(1)

        # Get updated estimations.
        est_pos_wrd = pre_pos_wrd.detach() + diff_pos_wrd
        est_green_wrd = F.normalize(pre_green_wrd.detach() + diff_green_wrd, dim=-1)
        est_red_wrd = F.normalize(pre_red_wrd.detach() + diff_red_wrd, dim=-1)
        est_scale_wrd = pre_scale_wrd.detach() * diff_scale_wrd
        est_shape_code = pre_shape_code.detach() + diff_shape_code
        return est_pos_wrd, est_green_wrd, est_red_wrd, est_scale_wrd, est_shape_code



class auto_regressive_model(pl.LightningModule):
    def __init__(
        self, 
        num_encoder_layers=2, 
        num_decoder_layers=2, 
        enc_in_dim=512, 
        dec_in_dim=512, 
        num_head=8, 
        hidden_dim=256, 
        dim_feedforward=1024, 
        add_conf='Nothing', 
        dropout=0.1, 
        itr_frame_num=1, 
        ):
        super(auto_regressive_model, self).__init__()

        self.encoder = TransformerEncoder_FixUp(
                            encoder_layers=num_encoder_layers, 
                            d_model=hidden_dim, 
                            nhead=num_head, 
                            dim_feedforward=dim_feedforward, 
                            dropout=dropout, 
                            activation='relu', 
                            T_Fixup=add_conf=='T_Fixup')
        self.decoder = TransformerDecoder_FixUp(
                            decoder_layers=num_decoder_layers, 
                            d_model=hidden_dim, 
                            nhead=num_head, 
                            dim_feedforward=dim_feedforward, 
                            dropout=dropout, 
                            activation='relu', 
                            T_Fixup=add_conf=='T_Fixup')
        print('##########   ENCODER   ##########')
        print(self.encoder.layers[0].fc1.weight[:2, :5])
        print(self.encoder.layers[1].fc2.weight[:2, :5])
        print(self.encoder.layers[0].self_attn.v_proj.weight[:2, :5])
        print('##########   DECODER   ##########')
        print(self.decoder.layers[0].fc1.weight[:2, :5])
        print(self.decoder.layers[0].self_attn.v_proj.weight[:2, :5])
        print(self.decoder.layers[1].encoder_attn.v_proj.weight[:2, :5])
        print('##########     END     ##########')

        self.add_conf = add_conf
        self.enc_align_mlp = nn.Linear(enc_in_dim, hidden_dim)
        self.align_mlp = nn.Linear(dec_in_dim, hidden_dim)
        self.ie = IterationEncoding(hidden_dim)
        init_inp = torch.normal(mean=0.0, std=hidden_dim**(-1/2), size=(itr_frame_num, 1, hidden_dim))
        if add_conf=='T_Fixup':
            init_inp *= (9 * num_decoder_layers) ** (- 1. / 4.) 
        self.init_inp = nn.Parameter(init_inp)

    def forward(self, src, tgt, tgt_frame_len, decoder=None):
        # check_map_torch(decoder(src[:, 0,    0: 512][:, :, None, None])[:, -1].reshape(-1, 128)>0, 'src_obs.png')
        # check_map_torch(decoder(src[:, 0,  512:1024][:, :, None, None])[:, -1].reshape(-1, 128)>0, 'src_est.png')
        # check_map_torch(decoder(src[:, 0, 1024:1536][:, :, None, None])[:, -1].reshape(-1, 128)>0, 'src_dif.png')
        # if not tgt is None:
        #     check_map_torch(decoder(tgt[:, 0,    0: 512][:, :, None, None])[:, -1].reshape(-1, 128)>0, 'tgt_obs.png')
        #     check_map_torch(decoder(tgt[:, 0,  512:1024][:, :, None, None])[:, -1].reshape(-1, 128)>0, 'tgt_est.png')
        #     check_map_torch(decoder(tgt[:, 0, 1024:1536][:, :, None, None])[:, -1].reshape(-1, 128)>0, 'tgt_dif.png')

        # encorder part.
        src = self.enc_align_mlp(src) # [seq_e, batch, inp_embed_dim] -> [seq_e, batch, hidden_dim]
        memory = self.encoder(src)

        # decorder part.
        if tgt is None:
            tgt = self.init_inp[:src.shape[0], :, :]
            tgt = tgt.expand(-1, src.shape[1], -1)
            if len(tgt_frame_len) > 1:
                tgt = self.ie(tgt, tgt_frame_len) # [seq_d, batch, hidden_dim]
        else:
            tgt = self.align_mlp(tgt) # [seq_d, batch, inp_embed_dim] -> [seq_d, batch, hidden_dim]
        out = self.decoder(tgt=tgt, memory=memory)
        out = out[-tgt_frame_len[-1]:, :, :].mean(0)
        return out



class encoder_model(pl.LightningModule):

    def __init__(
        self, 
        inp_embed_dim=3*512, 
        num_encoder_layers=3, 
        num_head=8, 
        hidden_dim=256, 
        dim_feedforward=1024, 
        add_conf='Nothing', 
        return_rawvec=False, 
        dropout=0.1, 
        ):
        super(encoder_model, self).__init__()

        self.num_head = num_head
        self.hidden_dim = hidden_dim
        self.return_rawvec = return_rawvec
        self.align_mlp = nn.Linear(inp_embed_dim, hidden_dim)
        T_Fixup = add_conf in {'T_Fixup', 'onlydec', 'onlydecv2', 'onlydecv3'}
        self.encoder = TransformerEncoder_FixUp(
                            encoder_layers=num_encoder_layers, 
                            d_model=hidden_dim, 
                            nhead=num_head, 
                            dim_feedforward=dim_feedforward, 
                            dropout=dropout, 
                            activation='relu', 
                            two_mha=add_conf=='onlydecv3', 
                            T_Fixup =T_Fixup, )
        # encoder_layer = TransformerEncoderLayer_woNorm(d_model=hidden_dim, 
        #                     nhead=num_head, 
        #                     dim_feedforward=dim_feedforward, 
        #                     dropout=0.1, 
        #                     activation="relu")
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, None) 
        if add_conf in {'exp22', 'exp22v2'}:
            self.pos_encoder = PositionalEncoding(d_model=hidden_dim, dropout=0.0, max_len=16)
            self.temporal_encoder = TransformerEncoder_FixUp(
                                        encoder_layers=num_encoder_layers, 
                                        d_model=hidden_dim, 
                                        nhead=num_head, 
                                        dim_feedforward=dim_feedforward, 
                                        dropout=dropout, 
                                        activation='relu', )
        elif self.add_conf in {'onlydec', 'onlydecv2', 'onlydecv3'}:
            self.itr_encoder = IterationEncoding(hidden_dim)
        print('##########   ENCODER   ##########')
        print(self.encoder.layers[0].fc1.weight[:2, :5])
        print(self.encoder.layers[1].fc2.weight[:2, :5])
        print(self.encoder.layers[0].self_attn.v_proj.weight[:2, :5])
        print('##########     END     ##########')

    def forward(self, inp, past_itr_length): #, decoder=None):

        if self.add_conf in {'onlydec', 'onlydecv2', 'onlydecv3'}:
            if self.add_conf == 'onlydecv3':
                batch, itr, seq, dim = inp.shape
                x = self.align_mlp(inp)   # [batch, itr, seq, dim]
                x = x.permute(1, 2, 0, 3) # [itr, seq, batch, dim]
                ##############################
                x = self.itr_encoder(x, [seq] * itr)

                seq_mask = torch.full((itr*batch, past_itr_length[-1]), True).to(x.device) # [itr*batch, seq]
                itr_mask = torch.full((seq*batch, itr), True).to(x.device) # [itr*batch, seq]
                for itr_idx in range(itr):
                    seq_mask[batch*(itr_idx):batch*(itr_idx + 1), :past_itr_length[itr_idx]] = False
                    itr_mask[:batch*(past_itr_length[itr_idx]), itr_idx] = False

                x = self.encoder(x, seq_padding_mask=seq_mask, itr_padding_mask=itr_mask) # [itr, seq, batch, dim]
                ##############################
                x = x.mean(1) # [itr, batch, dim]
                return x[-1] # [batch, dim]

            else:
                batch, itr_and_seq, dim = inp.shape
                itr = len(past_itr_length)
                frame_idx = torch.cat([torch.arange(length) for length in past_itr_length], dim=0)[..., None].expand(-1, itr_and_seq)
                x = self.align_mlp(inp)   # [batch, itr_and_seq, dim]
                x = x.permute(1, 0, 2) # [itr_and_seq, batch, dim]
                # frame_idx_ = torch.tile(torch.arange(seq), (itr, ))[..., None].expand(-1, itr*seq)
                # flame_length = [seq] * itr
                # inp = inp.reshape(itr_and_seq, batch, 512*3)

                if self.add_conf == 'onlydecv2':
                    frame_mask = frame_idx != frame_idx.transpose(-2, -1)
                    itr_mask = torch.full((sum(past_itr_length), sum(past_itr_length)), True)
                    for itr_idx, frame_num in enumerate(past_itr_length):
                        itr_start = sum(past_itr_length[:itr_idx])
                        itr_end = sum(past_itr_length[:itr_idx+1])
                        itr_mask[itr_start:itr_end, itr_start:itr_end] = False
                    atten_mask = torch.logical_and(itr_mask, frame_mask).to(x.device)
                elif self.add_conf == 'onlydec':
                    atten_mask = None
                # check_map_torch(decoder(inp[:, 0, 512:1024][:, :, None, None])[:, -1].reshape(-1, 128), 'tes.png')
                # check_map_torch(decoder(inp[:, 0, 512:1024][:, :, None, None])[torch.logical_not(atten_mask[0]), -1].reshape(-1, 128), 'tes.png')

                # check_map_torch(self.itr_encoder(torch.zeros_like(x), past_itr_length)[:, 0], 'tes.png')
                x = self.itr_encoder(x, past_itr_length)
                x = self.encoder(x, atten_mask=atten_mask) # [itr*seq, batch, dim]
                # x = x.reshape(itr, seq, batch, self.hidden_dim) # [itr, seq, batch, dim]
                # # inp = inp.reshape(itr, seq, batch, 512*3)
                # x = x.mean(1) # [itr, batch, dim]
                return x[-past_itr_length[-1]:, :, :].mean(0) # [batch, dim]

        elif self.add_conf in {'exp22v2'}:
            batch, itr, seq, dim = inp.shape
            x = self.align_mlp(inp)   # [batch, itr, seq, dim]
            x = x.permute(2, 1, 0, 3) # [seq, itr, batch, dim]
            x = x.reshape(seq, itr*batch, self.hidden_dim) # [seq, itr*batch, dim]
            #########################
            itr_mask = torch.full((itr*batch, past_itr_length[-1]), True).to(x.device) # [itr*batch, seq, seq]
            for itr_idx in range(itr):
                itr_mask[batch*(itr_idx):batch*(itr_idx + 1), :past_itr_length[itr_idx]] = False
            #########################
            x = self.encoder(x, src_key_padding_mask=itr_mask) # [seq, itr*batch, dim]
            itr_mask_float = torch.logical_not(itr_mask).to(x)
            itr_mask_float = itr_mask_float / itr_mask_float.sum(dim=-1)[:, None]
            itr_mask_float = itr_mask_float.T
            x = (itr_mask_float[..., None] * x).sum(dim=0) # x = x.mean(0) # [itr*batch, dim]
            # x_ = self.encoder(x) # [seq, itr*batch, dim]
            # x_ = x_.mean(0)
            #########################
            x = x.reshape(itr, batch, self.hidden_dim) # [itr, batch, dim]
            x = self.pos_encoder(x) # [itr, batch, dim]
            x = self.temporal_encoder(x) # [itr, batch, dim]
            return x[-1] # [batch, dim]

        else:
            first_itr = len(past_itr_length) == 1
            x = self.align_mlp(inp)
            if self.add_conf in {'T_Fixup'}:
                x = self.encoder(x, None)
            if self.add_conf in {'fix_norm'}:
                x = self.encoder(x, first_itr)
            else:
                x = self.encoder(x)

            if self.return_rawvec:
                return x
            x = x.mean(0)

            if self.add_conf == 'exp22':
                if first_itr:
                    self.temporal_x_list = []
                self.temporal_x_list.append(x.clone().detach())
                
                temporal_x = torch.stack(self.temporal_x_list[:-1] + [x], dim=0)

                temporal_x = self.pos_encoder(temporal_x)
                temporal_x = self.temporal_encoder(temporal_x)[-1]
                return temporal_x
            else:
                return x



class onlymlp(pl.LightningModule):

    def __init__(
        self, 
        inp_embed_dim=3*512, 
        hidden_dim=256, 
        mlp_layers_num=5, 
        mlp_hidden_dim=1024, 
        dropout=0.1, 
        ):
        super(onlymlp, self).__init__()

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





class optimize_former_old(pl.LightningModule):

    def __init__(
        # self, 
        # input_type = 'depth', 
        # split_into_patch = 'non', 
        # hidden_dim = 512, 
        # num_encoder_layers = 6, 
        # dim_feedforward = 2048, 
        # latent_size = 256, 
        # position_dim = 7, 
        # num_head = 8, 
        # dropout = 0.1, 
        # positional_encoding_mode = 'non', 
        # integration_mode = 'average', 
        # encoder_norm_type = 'LayerNorm', 
        # reset_transformer_params = False, 
        # loss_timing = 'after_mean', 
        self, 
        main_layers_name = 'onlymlp', 
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
        add_conf = None, 
        ):
        super().__init__()

        # Back Bone.
        self.input_type = input_type
        self.backbone_type = 'ResNet18_wo_dilation' # 'ResNet50_wo_dilation'
        if self.input_type == 'osmap':
            in_channel = 11
            if self.backbone_type == 'ResNet18_wo_dilation':
                conv1d_inp_dim = 512
                self.backbone = ResNet18_wo_dilation(in_channel=in_channel, gpu_num=1)
            elif self.backbone_type == 'ResNet50_wo_dilation':
                conv1d_inp_dim = 2048
                self.backbone = ResNet50_wo_dilation(in_channel=in_channel, gpu_num=1)
        self.back_bone_dim = hidden_dim
        self.conv_1d = nn.Conv2d(conv1d_inp_dim, self.back_bone_dim, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        # Transformer.
        self.integration_mode = main_layers_name
        if self.backbone_type == 'ResNet18_wo_dilation' and self.integration_mode == 'cnn_only_1':
            self.encoder = nn.Sequential(nn.Linear(hidden_dim, 1024), nn.ReLU(0.2),
                                         nn.Linear(1024, 1024), nn.ReLU(0.2),
                                         nn.Linear(1024, hidden_dim))
        if self.integration_mode == 'encoder':
            encoder_layer = TransformerEncoderLayer_woNorm(d_model=hidden_dim, nhead=num_head, dim_feedforward=dim_feedforward, dropout=0.0, activation="relu")
            self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, None) # , 1, encoder_norm)
        
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


    # def forward(self, inp, rays_d_cam, pre_scale_wrd, pre_shape_code, pre_o2w, 
    #     inp_pre_scale_wrd, inp_pre_shape_code, inp_pre_o2w, positional_encoding_target=False, model_mode='train'):
    def forward(self, inp, obs_embed, est_embed, dif_embed, past_itr_length, first_itr, 
        inp_pre_obj_pos_wrd, inp_pre_obj_green_wrd, inp_pre_obj_red_wrd, inp_pre_obj_scale_wrd, inp_pre_obj_shape_code, 
        pre_pos_wrd, pre_green_wrd, pre_red_wrd, pre_scale_wrd, pre_shape_code):

        batch_size, seq_len, cha_num, H, W = inp.shape
        
        # Backbone.
        inp = inp.reshape(batch*seq, cha_num, H, W)
        x = self.backbone(inp) # torch.Size([batch*seq, 2048, 16, 16])
        x = self.conv_1d(x) # torch.Size([batch*seq, 512, 16, 16])
        
        # Transformer.
        x = self.avgpool(x) # torch.Size([batch*seq, 2048, 1, 1])
        x = x.reshape(batch_size, seq_len, self.back_bone_dim) # torch.Size([batch, seq, hidden_dim])
        if self.integration_mode == 'encoder':
            x = x.permute(1, 0, 2) # torch.Size([seq, batch, hidden_dim])
            x = self.encoder(x).mean(0) # torch.Size([seq, batch, hidden_dim]), get mean.
        elif self.integration_mode == 'cnn_only_1':
            # if self.backbone_type == 'ResNet18_wo_dilation':
            x = self.encoder(x)

        if self.integration_mode == 'cnn_only_1':
            diff_pos_wrd = self.fc_pos(torch.cat([x, inp_pre_obj_pos_wrd.detach()], dim=-1)).permute(1, 0, 2).mean(0)
            diff_green_wrd = self.fc_axis_green(torch.cat([x, inp_pre_obj_green_wrd.detach()], dim=-1)).permute(1, 0, 2).mean(0)
            diff_red_wrd = self.fc_axis_red(torch.cat([x, inp_pre_obj_red_wrd.detach()], dim=-1)).permute(1, 0, 2).mean(0)
            diff_scale_wrd = self.fc_scale(torch.cat([x, inp_pre_obj_scale_wrd.detach()], dim=-1)).permute(1, 0, 2).mean(0) + 1e-5
            diff_shape_code = self.fc_shape_code(torch.cat([x, inp_pre_obj_shape_code.detach()], dim=-1)).permute(1, 0, 2).mean(0)
            # # Make pre_est.
            # pre_green_obj = torch.tensor([[0.0, 1.0, 0.0]]).expand(batch*seq, -1).to(x)
            # pre_red_obj = torch.tensor([[1.0, 0.0, 0.0]]).expand(batch*seq, -1).to(x)
            # pre_pos_obj = torch.tensor([[0.0, 0.0, 0.0]]).expand(batch*seq, -1).to(x)
            # pre_scale_obj = torch.tensor([[1.0]]).expand(batch*seq, -1).to(x)
            # # Head.
            # diff_pos_obj = self.fc_pos(torch.cat([x, pre_pos_obj.detach()], dim=-1))
            # diff_green_obj = self.fc_axis_green(torch.cat([x, pre_green_obj.detach()], dim=-1))
            # diff_red_obj = self.fc_axis_red(torch.cat([x, pre_red_obj.detach()], dim=-1))
            # diff_scale = self.fc_scale(torch.cat([x, pre_scale_obj.detach()], dim=-1)) + 1e-5 # Prevent scale=0.
            # diff_shape_code = self.fc_shape_code(torch.cat([x, inp_pre_shape_code.detach()], dim=-1))
            # # Convert cordinates.
            # diff_pos_wrd = torch.sum(diff_pos_obj[..., None, :]*inp_pre_o2w, -1) * inp_pre_scale_wrd
            # diff_green_wrd = torch.sum(diff_green_obj[..., None, :]*inp_pre_o2w, -1) * inp_pre_scale_wrd
            # diff_red_wrd = torch.sum(diff_red_obj[..., None, :]*inp_pre_o2w, -1) * inp_pre_scale_wrd
            # # Reshape update to [Seq, batch, dim].
            # diff_pos_wrd = diff_pos_wrd.reshape(batch_size, -1, 3).permute(1, 0, 2)
            # diff_green_wrd = diff_green_wrd.reshape(batch_size, -1, 3).permute(1, 0, 2)
            # diff_red_wrd = diff_red_wrd.reshape(batch_size, -1, 3).permute(1, 0, 2)
            # diff_scale = diff_scale.reshape(batch_size, -1, 1).permute(1, 0, 2)
            # diff_shape_code = diff_shape_code.reshape(batch_size, -1, self.latent_size).permute(1, 0, 2)
            # # Get integrated update.
            # diff_pos_wrd = diff_pos_wrd.mean(0)
            # diff_green_wrd = diff_green_wrd.mean(0)
            # diff_red_wrd = diff_red_wrd.mean(0)
            # diff_scale = diff_scale.mean(0)
            # diff_shape_code = diff_shape_code.mean(0)

        elif self.integration_mode == 'encoder':
            # Get updated estimations.
            diff_pos_wrd = self.fc_pos(torch.cat([x, pre_pos_wrd.detach()], dim=-1))
            diff_green_wrd = self.fc_axis_green(torch.cat([x, pre_green_wrd.detach()], dim=-1))
            diff_red_wrd = self.fc_axis_red(torch.cat([x, pre_red_wrd.detach()], dim=-1))
            diff_scale_wrd = self.fc_scale(torch.cat([x, pre_scale_wrd.detach()], dim=-1)) + 1e-5
            diff_shape_code = self.fc_shape_code(torch.cat([x, pre_shape_code.detach()], dim=-1))
        
        # Get updated estimations.
        est_pos_wrd = pre_pos_wrd.detach() + diff_pos_wrd
        est_green_wrd = F.normalize(pre_green_wrd.detach() + diff_green_wrd, dim=-1)
        est_red_wrd = F.normalize(pre_red_wrd.detach() + diff_red_wrd, dim=-1)
        est_scale_wrd = pre_scale_wrd.detach() * diff_scale_wrd
        est_shape_code = pre_shape_code.detach() + diff_shape_code
        return est_pos_wrd, est_green_wrd, est_red_wrd, est_scale_wrd, est_shape_code
        #     # Make pre_est.
        #     pre_green_obj = torch.tensor([[0.0, 1.0, 0.0]]).expand(batch_size, -1).to(x)
        #     pre_red_obj = torch.tensor([[1.0, 0.0, 0.0]]).expand(batch_size, -1).to(x)
        #     pre_pos_obj = torch.tensor([[0.0, 0.0, 0.0]]).expand(batch_size, -1).to(x)
        #     pre_scale_obj = torch.tensor([[1.0]]).expand(batch_size, -1).to(x)
        #     # Head.
        #     diff_pos_obj = self.fc_pos(torch.cat([x, pre_pos_obj.detach()], dim=-1))
        #     diff_green_obj = self.fc_axis_green(torch.cat([x, pre_green_obj.detach()], dim=-1))
        #     diff_red_obj = self.fc_axis_red(torch.cat([x, pre_red_obj.detach()], dim=-1))
        #     diff_scale = self.fc_scale(torch.cat([x, pre_scale_obj.detach()], dim=-1)) + 1e-5 # Prevent scale=0.
        #     diff_shape_code = self.fc_shape_code(torch.cat([x, pre_shape_code.detach()], dim=-1))
        #     # Convert cordinates.
        #     diff_pos_wrd = torch.sum(diff_pos_obj[..., None, :]*pre_o2w, -1) * pre_scale_wrd
        #     diff_green_wrd = torch.sum(diff_green_obj[..., None, :]*pre_o2w, -1) * pre_scale_wrd
        #     diff_red_wrd = torch.sum(diff_red_obj[..., None, :]*pre_o2w, -1) * pre_scale_wrd

        # return diff_pos_wrd, diff_green_wrd, diff_red_wrd, diff_scale, diff_shape_code


    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



if __name__=='__main__':
    e, d = TransformerEncoder_FixUp(), TransformerDecoder_FixUp()
    inp = torch.randn(20, 32, 256)
    e(inp, True)