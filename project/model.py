import os
import pdb
import sys
from turtle import pd
import numpy as np
import random
import glob
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
        view_selection = 'simultaneous', 
        main_layers_name = 'onlymlp', 
        add_conf = None, 
        use_attn_mask='no', 
        layer_wise_attention = 'no', 
        use_cls='no', 
        num_encoder_layers = 2, 
        num_decoder_layers = 1, 
        hidden_dim = 256, 
        num_head = 8, 
        dim_feedforward = 1024, 
        mlp_hidden_dim = 1024, 
        latent_size = 256, 
        positional_encoding_mode = 'no', 
        enc_in_dim=3*512, 
        dec_in_dim=2*512, 
        dropout=0.1, 
        total_obs_num=1, 
        itr_per_frame=1, 
        total_itr=1, 
        inp_itr_num=1, 
        dec_inp_type='dif_obs', ):
        super(optimize_former, self).__init__()

        # Main layers.
        self.add_conf = add_conf
        self.main_layers_name = main_layers_name
        self.use_past_map = (main_layers_name == 'autoreg') or (add_conf in {'onlydec', 'onlydecv2', 'onlydecv3', 'exp22v2'})
        self.padding_embeddings = add_conf in {'exp22v2', 'onlydecv3'}
        if main_layers_name=='autoreg':
            self.inp_itr_num = inp_itr_num
            self.dec_inp_type = dec_inp_type
            enc_in_dim = 3*512
            dec_in_dim = 5*512 if self.dec_inp_type == 'dif_obs_est' else 3*512
            self.main_layers = auto_regressive_model(view_selection=view_selection, 
                                                     use_attn_mask=use_attn_mask, 
                                                     total_itr=total_itr, 
                                                     max_frame_num=total_obs_num, 
                                                     itr_per_frame=itr_per_frame, 
                                                     inp_itr_num=inp_itr_num, 
                                                     add_conf=add_conf, 
                                                     layer_wise_attention=layer_wise_attention, 
                                                     use_cls=use_cls, 
                                                     num_encoder_layers=num_encoder_layers, 
                                                     num_decoder_layers=num_decoder_layers, 
                                                     enc_in_dim=enc_in_dim, 
                                                     dec_in_dim=dec_in_dim, 
                                                     num_head=num_head, 
                                                     hidden_dim=hidden_dim, 
                                                     dim_feedforward=dim_feedforward, 
                                                     dropout=dropout, )
        elif main_layers_name=='encoder':
            self.main_layers = encoder_model(inp_embed_dim=enc_in_dim, #512, 
                                             num_encoder_layers=num_encoder_layers, 
                                             num_head=num_head, 
                                             hidden_dim=hidden_dim, 
                                             dim_feedforward=dim_feedforward, 
                                             add_conf=add_conf, 
                                             dropout=dropout, 
                                             view_selection=view_selection, 
                                             total_itr=total_itr, 
                                             max_frame_num=total_obs_num, 
                                             inp_itr_num=inp_itr_num, 
                                             itr_per_frame=itr_per_frame, )
        elif main_layers_name=='onlymlp':
            self.main_layers = nn.Sequential()
            for mlp_layer_idx in range(num_encoder_layers):
                inp_dim = mlp_hidden_dim
                out_dim = mlp_hidden_dim
                if mlp_layer_idx == 0:
                    inp_dim = enc_in_dim
                if mlp_layer_idx == num_encoder_layers-1:
                    out_dim = hidden_dim
                self.main_layers.add_module(f'fc_{str(mlp_layer_idx).zfill(3)}', nn.Linear(inp_dim, out_dim))
                self.main_layers.add_module(f'act_{str(mlp_layer_idx).zfill(3)}', nn.ReLU())
                if mlp_layer_idx != num_encoder_layers-1:
                    self.main_layers.add_module(f'drp_{str(mlp_layer_idx).zfill(3)}', nn.Dropout(dropout))
        self.positional_encoding_mode = positional_encoding_mode
        if positional_encoding_mode == 'yes':
            self.learnable_pe = nn.Linear(7, hidden_dim)

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
        inp_pre_obj_green_wrd, inp_pre_obj_red_wrd, inp_pre_o2w, inp_pre_obj_scale_wrd, inp_pre_obj_shape_code, pre_pos_wrd, 
        pre_green_wrd, pre_red_wrd, pre_o2w, pre_scale_wrd, pre_shape_code, pe_target, print_debug=False, image_decoder=None):
        batch, _, _ = obs_embed.shape
        current_frame_num = past_itr_length[-1]
        if (self.positional_encoding_mode=='yes') and (pe_target is not None):
            pe_target = self.learnable_pe(pe_target)

        if self.main_layers_name == 'onlymlp':
            # Make inputs.
            inp = torch.cat([obs_embed[:, -current_frame_num:, :], 
                             est_embed[:, -current_frame_num:, :], 
                             dif_embed[:, -current_frame_num:, :]], dim=-1) # [batch, seq, inp_embed_dim]
            # check_map_torch(image_decoder(inp[0, :,    0: 512][:, :, None, None])[:, -1].reshape(-1, 128)>0, 'src_obs.png')
            # check_map_torch(image_decoder(inp[0, :,  512:1024][:, :, None, None])[:, -1].reshape(-1, 128)>0, 'src_est.png')
            # check_map_torch(image_decoder(inp[0, :, 1024:1536][:, :, None, None])[:, -1].reshape(-1, 128)>0, 'src_dif.png')
            # Main layer.
            x = self.main_layers(inp) # [batch, seq, inp_embed_dim]
            if print_debug:
                print(f'---   mlp_inp   --- : {inp.shape}')
                print(f'---   mlp_out   --- : {x.shape}') # [batch, seq, inp_embed_dim]
            
            # Make pre_est.
            pre_green_obj = torch.tensor([[[0.0, 1.0, 0.0]]]).expand(batch, current_frame_num, 3).to(x)
            pre_red_obj = torch.tensor([[[1.0, 0.0, 0.0]]]).expand(batch, current_frame_num, 3).to(x)
            pre_pos_obj = torch.tensor([[[0.0, 0.0, 0.0]]]).expand(batch, current_frame_num, 3).to(x)
            pre_scale_obj = torch.tensor([[[1.0]]]).expand(batch, current_frame_num, 1).to(x)
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

        elif self.main_layers_name in {'autoreg', 'encoder'}:
            # Make inputs.
            if self.main_layers_name == 'encoder':
                inp = torch.cat([obs_embed, 
                                 est_embed, 
                                 dif_embed], dim=-1).permute(1, 0, 2)
            elif self.main_layers_name == 'autoreg':
                # Make inputs.
                inp_enc = torch.cat([obs_embed[:, -current_frame_num:, :], 
                                     est_embed[:, -current_frame_num:, :], 
                                     dif_embed[:, -current_frame_num:, :]], dim=-1).permute(1, 0, 2)
                if optim_idx > 0:
                    dec_inp_frame_len_list = past_itr_length[-(self.inp_itr_num+1):-1]
                    dec_inp_end = sum(dec_inp_frame_len_list) + current_frame_num
                    if self.dec_inp_type == 'dif_obs_est':
                        if print_debug:
                            print(f'--- dec_inp_typ --- : dif_obs_est')
                        inp_dec = torch.cat([dif_embed[:, -dec_inp_end:-current_frame_num, :], obs_embed[:, -current_frame_num:, :], # inp_dec = torch.cat([est_embed[:, -dec_inp_end:-current_frame_num, :], 
                                             tra_embed[:, -dec_inp_end+current_frame_num:, :], torch.cat([est_embed[:, sum(past_itr_length[:view_idx+1]):sum(past_itr_length[:view_idx+1])+past_itr_length[view_idx], :] for view_idx in range(max(0, optim_idx-self.inp_itr_num), optim_idx)], dim=1), #                      dif_embed[:, -dec_inp_end:-current_frame_num, :], 
                                             est_embed[:, -dec_inp_end:-current_frame_num, :], ], dim=-1).permute(1, 0, 2) # tra_embed[:, -dec_inp_end+current_frame_num:, :]], dim=-1).permute(1, 0, 2)
                    elif self.dec_inp_type == 'dif_obs':
                        if print_debug:
                            print(f'--- dec_inp_typ --- : dif_obs')
                        inp_dec = torch.cat([obs_embed[:, -dec_inp_end:-current_frame_num, :], 
                                             est_embed[:, -dec_inp_end:-current_frame_num, :], 
                                             dif_embed[:, -dec_inp_end:-current_frame_num, :]], dim=-1).permute(1, 0, 2)
                    elif self.dec_inp_type == 'dif_est':
                        if print_debug:
                            print(f'--- dec_inp_typ --- : dif_est')
                        inp_dec = torch.cat([torch.cat([est_embed[:, sum(past_itr_length[:view_idx+1]):sum(past_itr_length[:view_idx+1])+past_itr_length[view_idx], :] for view_idx in range(max(0, optim_idx-self.inp_itr_num), optim_idx)], dim=1), 
                                             est_embed[:, -dec_inp_end:-current_frame_num, :], 
                                             tra_embed[:, -sum(dec_inp_frame_len_list):, :]], dim=-1).permute(1, 0, 2)
                        # inp_dec = torch.cat([torch.cat([est_embed[:, -current_frame_num:, :][:, :view_num, :] for view_num in dec_inp_frame_len_list], dim=1), 
                        #                      est_embed[:, -dec_inp_end:-current_frame_num, :], 
                        #                      tra_embed[:, -sum(dec_inp_frame_len_list):, :]], dim=-1).permute(1, 0, 2)
                else:
                    dec_inp_frame_len_list = past_itr_length
                    inp_dec = None
            # Main layer.
            if self.main_layers_name == 'encoder':
                x = self.main_layers(inp, past_itr_length, pe_target, print_debug, image_decoder) # [batch, hidden_dim]
            elif self.main_layers_name == 'autoreg':
                x = self.main_layers(inp_enc, inp_dec, past_itr_length, pe_target, print_debug, image_decoder) # [batch, hidden_dim]

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
        view_selection='simultaneous', 
        use_attn_mask='no', 
        total_itr=1, 
        max_frame_num=1, 
        inp_itr_num=1, 
        itr_per_frame=1, 
        add_conf='Nothing', 
        layer_wise_attention='no', 
        use_cls='no', 
        num_encoder_layers=2, 
        num_decoder_layers=2, 
        enc_in_dim=512, 
        dec_in_dim=512, 
        num_head=8, 
        hidden_dim=256, 
        dim_feedforward=1024, 
        dropout=0.1, 
        ):
        super(auto_regressive_model, self).__init__()

        self.encoder = TransformerEncoder_FixUp(
                            encoder_layers=num_encoder_layers, 
                            d_model=hidden_dim, 
                            nhead=num_head, 
                            dim_feedforward=dim_feedforward, 
                            dropout=dropout, 
                            T_Fixup=add_conf=='T_Fixup', 
                            layer_wise_attention=layer_wise_attention=='yes', )
        self.decoder = TransformerDecoder_FixUp(
                            decoder_layers=num_decoder_layers, 
                            d_model=hidden_dim, 
                            nhead=num_head, 
                            dim_feedforward=dim_feedforward, 
                            dropout=dropout, 
                            T_Fixup=add_conf=='T_Fixup', 
                            layer_wise_attention=layer_wise_attention=='yes', )
        print('##########   ENCODER   ##########')
        print(self.encoder.layers[0].fc1.weight[0, :5])
        print('##########   DECODER   ##########')
        print(self.decoder.layers[0].fc1.weight[0, :5])
        print('##########     END     ##########')

        self.add_conf = add_conf
        self.enc_align_mlp = nn.Linear(enc_in_dim, hidden_dim)
        self.align_mlp = nn.Linear(dec_in_dim, hidden_dim)
        if view_selection=='simultaneous':
            init_inp = torch.normal(mean=0.0, std=hidden_dim**(-1/2), size=(max_frame_num, 1, hidden_dim))
        elif view_selection=='sequential':
            init_inp = torch.normal(mean=0.0, std=hidden_dim**(-1/2), size=(1, 1, hidden_dim))
        if add_conf=='T_Fixup':
            init_inp *= (9 * num_decoder_layers) ** (- 1. / 4.) 
        self.init_inp = nn.Parameter(init_inp)
        self.inp_itr_num = inp_itr_num
        if self.inp_itr_num > 1:
            self.ie = IterationEncoding(hidden_dim)
        if use_attn_mask=='yes':
            if view_selection=='simultaneous':
                q_seq_len = max_frame_num * (total_itr - 1)
                frame_idx = torch.cat([torch.arange(max_frame_num)]*(total_itr - 1), dim=0)[..., None].expand(-1, q_seq_len)
                frame_mask = frame_idx != frame_idx.transpose(-2, -1)
                itr_idx = torch.arange((total_itr - 1))[:, None].expand(-1, max_frame_num).reshape(-1)[..., None].expand(-1, q_seq_len)
                itr_mask = itr_idx != itr_idx.transpose(-2, -1)
                total_atten_mask = torch.logical_and(itr_mask, frame_mask) # .to(torch.uint8)
                self.atten_mask = []
                for itr_idx in range(1, total_itr):
                    start = max(itr_idx - self.inp_itr_num, 0) * max_frame_num
                    end = itr_idx * max_frame_num
                    self.atten_mask.append(total_atten_mask[start:end, start:end])
            if view_selection=='sequential':
                q_seq_len = sum(range(1, max_frame_num + 1)) * itr_per_frame - max_frame_num
                frame_idx = torch.cat(
                                [torch.cat([torch.arange(frame_num)]*itr_per_frame, dim=0) for frame_num in range(1, max_frame_num+1)], 
                                dim=0)[:-max_frame_num, None].expand(-1, q_seq_len)
                frame_mask = frame_idx != frame_idx.transpose(-2, -1)
                itr_idx = torch.cat([torch.tensor([itr_id]*(itr_id//itr_per_frame+1)) for itr_id in range(total_itr-1)], dim=0)[..., None].expand(-1, q_seq_len)
                itr_mask = itr_idx != itr_idx.T
                total_atten_mask = torch.logical_and(itr_mask, frame_mask)
                self.atten_mask = []
                for itr_idx in range(0, total_itr-1):
                    start = sum([itr_idx_i//itr_per_frame + 1 for itr_idx_i in range(max(itr_idx-self.inp_itr_num+1, 0))])
                    end = sum([itr_idx_i//itr_per_frame + 1 for itr_idx_i in range(itr_idx+1)])
                    self.atten_mask.append(total_atten_mask[start:end, start:end])
            print('### w_atten_mask ###')
            print(self.atten_mask[-1].to(torch.uint8))
        elif use_attn_mask=='no':
            self.atten_mask = None
            print('### wo_atten_mask ###')
        self.use_cls = use_cls # 'yes'
        if self.use_cls=='yes':
            cls_token = torch.normal(mean=0.0, std=hidden_dim**(-1/2), size=(1, 1, hidden_dim))
            if add_conf=='T_Fixup':
                init_inp *= (9 * num_decoder_layers) ** (- 1. / 4.)
            self.cls_token = nn.Parameter(cls_token)


    def forward(self, src, tgt, past_itr_length, pe_target, print_debug=False, image_decoder=None):
        # check_map_torch(image_decoder(src[:, 0,    0: 512][:, :, None, None])[:, 1].reshape(-1, 128), 'src_obs.png')
        # check_map_torch(image_decoder(src[:, 0,  512:1024][:, :, None, None])[:, 1].reshape(-1, 128), 'src_est.png')
        # check_map_torch(image_decoder(src[:, 0, 1024:1536][:, :, None, None])[:, 1].reshape(-1, 128), 'src_dif.png')
        # if not tgt is None:
        #     check_map_torch(image_decoder(tgt[:, 0,    0: 512][:, :, None, None])[:, 1].reshape(-1, 128), 'tgt_obs.png')
        #     check_map_torch(image_decoder(tgt[:, 0,  512:1024][:, :, None, None])[:, 1].reshape(-1, 128), 'tgt_est.png')
        #     check_map_torch(image_decoder(tgt[:, 0, 1024:1536][:, :, None, None])[:, 1].reshape(-1, 128), 'tgt_dif.png')

        # encorder part.
        src = self.enc_align_mlp(src) # [seq_e, batch, inp_embed_dim] -> [seq_e, batch, hidden_dim]
        if pe_target is not None:
            src = src + pe_target[:, -past_itr_length[-1]:, :].permute(1, 0, 2)
        memory = self.encoder(src)
        if print_debug:
            print(f'---   ec_src    --- : {src.shape}')
            # print(f'---   ec_mem    --- : {memory.shape}')

        # decorder part.
        if tgt is not None:
            tgt = self.align_mlp(tgt) # [seq_d, batch, inp_embed_dim] -> [seq_d, batch, hidden_dim]
            if self.inp_itr_num > 1:
                tgt = self.ie(tgt, past_itr_length[-max(0, (self.inp_itr_num+1)):-1]) # [seq_d, batch, hidden_dim]
            atten_mask = None
            if self.atten_mask is not None:
                atten_mask = self.atten_mask[len(past_itr_length)-2].to(tgt.device)
            if pe_target is not None:
                pe_start = - sum(past_itr_length[-max(0, (self.inp_itr_num+1)):])
                pe_end = - past_itr_length[-1]
                tgt = tgt + pe_target[:, pe_start:pe_end, :].permute(1, 0, 2)
            if self.use_cls == 'yes':
                tgt = torch.cat([tgt, self.cls_token.expand(-1, src.shape[1], -1)], dim=0)
        else:
            atten_mask = None
            tgt = self.init_inp.expand(-1, src.shape[1], -1)
            if self.use_cls == 'yes':
                tgt = torch.cat([tgt, self.cls_token.expand(-1, src.shape[1], -1)], dim=0)
        out = self.decoder(tgt=tgt, memory=memory, atten_mask=atten_mask)
        if  self.use_cls == 'yes':
            if print_debug:
                print(f'---   dc_tgt    --- : {tgt.shape}')
                print(f'---   dc_out    --- : {out.shape}')
                print(f'--- dc_bf_mean  --- : cls_token')
            return out[-1]
        else:
            out_size = past_itr_length[-min(len(past_itr_length), 2)]
            if print_debug:
                print(f'---   dc_tgt    --- : {tgt.shape}')
                print(f'---   dc_out    --- : {out.shape}')
                print(f'--- dc_bf_mean  --- : {out[-out_size:, :, :].shape}')
            return out[-out_size:, :, :].mean(0)



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
        view_selection='simultaneous', 
        total_itr=1, 
        max_frame_num=1, 
        inp_itr_num=1, 
        itr_per_frame=1, 
        ):
        super(encoder_model, self).__init__()

        self.num_head = num_head
        self.hidden_dim = hidden_dim
        self.align_mlp = nn.Linear(inp_embed_dim, hidden_dim)
        self.add_conf = add_conf
        self.forward_mode = 'encoder'
        T_Fixup = add_conf in {'T_Fixup', 'exp22', 'exp22v2', 'onlydec', 'onlydecv2', 'onlydecv3'}
        self.encoder = TransformerEncoder_FixUp(
                            encoder_layers=num_encoder_layers, 
                            d_model=hidden_dim, 
                            nhead=num_head, 
                            dim_feedforward=dim_feedforward, 
                            dropout=dropout, 
                            activation='relu', 
                            two_mha=add_conf=='onlydecv3', 
                            T_Fixup =T_Fixup, )
        if add_conf in {'exp22', 'exp22v2'}:
            self.forward_mode = 'temporal'
            self.pos_encoder = PositionalEncoding(d_model=hidden_dim, dropout=0.0, max_len=16)
            self.temporal_encoder = TransformerEncoder_FixUp(
                                        encoder_layers=num_encoder_layers, 
                                        d_model=hidden_dim, 
                                        nhead=num_head, 
                                        dim_feedforward=dim_feedforward, 
                                        dropout=dropout, 
                                        activation='relu', )
        elif add_conf in {'onlydec', 'onlydecv2', 'onlydecv3'}:
            self.inp_itr_num = 3 # inp_itr_num
            self.forward_mode = add_conf
            if add_conf == 'onlydecv2':
                if view_selection=='simultaneous':
                    seq_len = max_frame_num * total_itr
                    frame_idx = torch.cat([torch.arange(max_frame_num)]*total_itr, dim=0)[..., None].expand(-1, seq_len)
                    frame_mask = frame_idx != frame_idx.transpose(-2, -1)
                    itr_idx = torch.arange(total_itr)[:, None].expand(-1, max_frame_num).reshape(-1)[..., None].expand(-1, seq_len)
                    itr_mask = itr_idx != itr_idx.transpose(-2, -1)
                    total_atten_mask = torch.logical_and(itr_mask, frame_mask)
                    self.atten_mask = []
                    for itr_idx in range(0, total_itr):
                        start = max(itr_idx + 1 - self.inp_itr_num, 0) * max_frame_num
                        end = (itr_idx + 1) * max_frame_num
                        self.atten_mask.append(total_atten_mask[start:end, start:end])
                if view_selection=='sequential':
                    seq_len = sum(range(1, max_frame_num + 1)) * itr_per_frame
                    frame_idx = torch.cat(
                                    [torch.cat([torch.arange(frame_num)]*itr_per_frame, dim=0) for frame_num in range(1, max_frame_num+1)], 
                                    dim=0)[..., None].expand(-1, seq_len)
                    frame_mask = frame_idx != frame_idx.transpose(-2, -1)
                    itr_idx = torch.cat([torch.tensor([itr_id]*(itr_id//itr_per_frame+1)) for itr_id in range(total_itr)], dim=0)[..., None].expand(-1, seq_len)
                    itr_mask = itr_idx != itr_idx.T
                    total_atten_mask = torch.logical_and(itr_mask, frame_mask)
                    self.atten_mask = []
                    for itr_idx in range(0, total_itr):
                        start = sum([itr_idx_i//itr_per_frame + 1 for itr_idx_i in range(max(itr_idx-self.inp_itr_num+1, 0))])
                        end = sum([itr_idx_i//itr_per_frame + 1 for itr_idx_i in range(itr_idx+1)])
                        self.atten_mask.append(total_atten_mask[start:end, start:end])
                print('### w_atten_mask ###')
                print(self.atten_mask[-1].to(torch.uint8))
            self.itr_encoder = IterationEncoding(hidden_dim)
        print('##########   ENCODER   ##########')
        print(self.encoder.layers[0].fc1.weight[0, :5])
        print('##########     END     ##########')

    def forward(self, inp, past_itr_length, pe_target, print_debug=False, image_decoder=None):
        # check_map_torch(image_decoder(inp[-past_itr_length[-1]:][:, 0,    0: 512][:, :, None, None])[:, -1].reshape(-1, 128)>0, 'src_obs.png')
        # check_map_torch(image_decoder(inp[-past_itr_length[-1]:][:, 0,  512:1024][:, :, None, None])[:, -1].reshape(-1, 128)>0, 'src_est.png')
        # check_map_torch(image_decoder(inp[-past_itr_length[-1]:][:, 0, 1024:1536][:, :, None, None])[:, -1].reshape(-1, 128)>0, 'src_dif.png')
        if self.forward_mode in {'onlydec', 'onlydecv2'}:
            inp_start = sum(past_itr_length[-self.inp_itr_num:])
            x = self.align_mlp(inp[-inp_start:])
            if pe_target is not None:
                x = x + pe_target[:, -inp_start:, :].permute(1, 0, 2)
            if self.forward_mode == 'onlydecv2':
                atten_mask = self.atten_mask[len(past_itr_length)-1].to(inp.device)
            else:
                atten_mask = None
            x = self.itr_encoder(x, past_itr_length[-self.inp_itr_num:])
            if print_debug:
                print(f'---   ec_inp   --- : {x.shape}')
            x = self.encoder(x, atten_mask)
            if print_debug:
                print(f'---   ec_out    --- : {x.shape}')
                print(f'--- out_bf_mean --- : {x[-past_itr_length[-1]:].shape}')
            x = x[-past_itr_length[-1]:].mean(0)
            return x
        
        elif self.forward_mode in {'onlydecv3'}:
            inp_start = sum(past_itr_length[-self.inp_itr_num:])
            x = self.align_mlp(inp[-inp_start:])
            if pe_target is not None:
                x = x + pe_target[:, -inp_start:, :].permute(1, 0, 2)
            x = self.itr_encoder(x, past_itr_length[-self.inp_itr_num:])
            start_itr = max(0, len(past_itr_length) - self.inp_itr_num)
            end_frame = len(past_itr_length)
            tmp_list = [x[sum(past_itr_length[start_itr:itr_idx]):sum(past_itr_length[start_itr:itr_idx+1])] for itr_idx in range(start_itr, end_frame)]
            x = nn.utils.rnn.pad_sequence(tmp_list, batch_first=True, padding_value=0.0) # [itr, seq, batch, dim] 
            if print_debug:
                print(f'---   ec_inp   --- : {x.shape}')
            x = self.encoder(x, seq_padding_mask=None, itr_padding_mask=None)
            if print_debug:
                print(f'---   ec_out    --- : {x.shape}')
                print(f'--- out_af_mean --- : {x.mean(1).shape}')
            x = x.mean(1)
            return x[-1]
        
        else:
            x = self.align_mlp(inp[-past_itr_length[-1]:])
            if print_debug:
                print(f'---   ec_inp   --- : {x.shape}')
            if pe_target is not None:
                x = x + pe_target[:, -past_itr_length[-1]:, :].permute(1, 0, 2)
            x = self.encoder(x)
            if print_debug:
                print(f'---   ec_out    --- : {x.shape}')
                print(f'--- out_bf_mean --- : {x.shape}')
            return x.mean(0)
