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

from parser import *
from often_use import *
# from dataset import *

torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False

# seed = 0
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)
# if device=='cuda':
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False





class DDF_decoder(pl.LightningModule):

    def __init__(self, args):
        super().__init__()

        self.cov_inp_size = 512
        self.decoder_fc = nn.Sequential(
                nn.Linear(args.latent_size, self.cov_inp_size), nn.LeakyReLU(0.2)
                )
        self.decoder_cov = nn.Sequential(
                nn.ConvTranspose3d(self.cov_inp_size, 256, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.ConvTranspose3d(256, 128, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.ConvTranspose3d(128, 64, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.ConvTranspose3d(64, 32, 4, 2, 1),
                nn.Softplus(),
                )
    
    def forward(self, inp):
        x = self.decoder_fc(inp)
        x = x.view(-1, self.cov_inp_size, 1, 1, 1)
        x = self.decoder_cov(x)
        return x





class DDF_latent_sampler(pl.LightningModule):

    def __init__(self, args):
        super().__init__()

        # Base configs
        self.H = args.ddf_H
        self.W = self.H
        self.fov = args.fov
        self.use_3d_code = args.use_3d_code
        self.only_latent = args.only_latent
        self.latent_size = args.latent_size
        self.latent_3d_size = args.latent_3d_size
        self.use_normal_loss = args.use_normal_loss

        # For sampling
        self.voxel_scale = args.voxel_scale
        sample_start = 1 - math.sqrt(3) * self.voxel_scale
        sample_end = 1 + math.sqrt(3) * self.voxel_scale
        self.sample_ind = torch.linspace(sample_start, sample_end, args.voxel_sample_num)[None, None, None, :, None] # to(torch.half)
        self.voxel_sample_num = args.voxel_sample_num
        self.voxel_ch_num = args.voxel_ch_num
        
        # Integrate config
        self.integrate_sampling_mode = args.integrate_sampling_mode
        # self.integrate_TransFormer_mode = args.integrate_TransFormer_mode
        # if self.integrate_sampling_mode=='TransFormer':
        #     # self.positional_encoding = PositionalEncoding(self.voxel_ch_num, 0.0, max_len=self.voxel_sample_num)
        #     self.time_series_model = nn.MultiheadAttention(embed_dim=self.voxel_ch_num, num_heads=args.num_heads, bias=True)
        if self.integrate_sampling_mode=='CAT':
            self.fc_inp_size = self.voxel_ch_num * self.voxel_sample_num
            self.fc = nn.Sequential(nn.Linear(self.fc_inp_size, self.latent_3d_size), nn.LeakyReLU(0.2))
        else:
            print('unknown integrate mode')
            sys.exit()
            
    
    def forward(self, lat_voxel, rays_d_wrd, rays_o, blur_mask=False, normal_mask=False, train=False):
        ####################################################################################################
        if train==False:
        ####################################################################################################
            # Sample features
            sample_point = 1 / self.voxel_scale * (rays_o[..., None, :] + rays_d_wrd[..., None, :] * self.sample_ind.to(rays_o.device))
            sample_point = sample_point.to(lat_voxel.dtype)
            sampled_lat_vec = F.grid_sample(lat_voxel, sample_point, padding_mode='border', align_corners=True).permute(0, 2, 3, 4, 1)
            valid = torch.prod(torch.gt(sample_point, -1.0) * torch.lt(sample_point, 1.0), dim=-1).byte().float()
            sampled_lat_vec = valid[..., None] * sampled_lat_vec # Padding outside voxel

            if self.integrate_sampling_mode=='CAT':
                if blur_mask == 'without_mask':
                    batch, H, W, _, _ = sampled_lat_vec.shape
                    return self.fc(sampled_lat_vec.reshape(batch, H, W, self.fc_inp_size))
                else:
                    batch, H, W = blur_mask.shape
                    return self.fc(sampled_lat_vec.reshape(batch, H, W, self.fc_inp_size)[blur_mask])
            
        ####################################################################################################
        if train==True:
        ####################################################################################################
            if self.use_normal_loss:
                # Sample features
                sampled_lat_vec = []
                sampled_lat_vec_r = []
                sampled_lat_vec_u = []
                total_blur_mask = 0
                total_normal_mask = 0
                for lat_voxel_i, rays_d_wrd_i, rays_o_i, blur_mask_i, normal_mask_i in zip(lat_voxel, rays_d_wrd, rays_o, blur_mask, normal_mask):
                    with torch.no_grad():
                        # Rays : [center, right, under], 3
                        rays_d_wrd_i = torch.cat([rays_d_wrd_i[0][blur_mask_i], rays_d_wrd_i[1][normal_mask_i], rays_d_wrd_i[2][normal_mask_i]], 0)
                        rays_o_i = torch.cat([rays_o_i[0][blur_mask_i], rays_o_i[1][normal_mask_i], rays_o_i[2][normal_mask_i]], 0)
                        # Get sample point
                        sample_point_i = 1 / self.voxel_scale * (rays_o_i[..., None, :] + rays_d_wrd_i[..., None, :] * self.sample_ind.to(rays_o_i.device)).detach().to(lat_voxel.dtype)
                        valid = torch.prod(torch.gt(sample_point_i, -1.0) * torch.lt(sample_point_i, 1.0), dim=-1).byte().detach().to(lat_voxel.dtype)
                    sampled_lat_vec_i = F.grid_sample(lat_voxel_i.unsqueeze(0), sample_point_i, padding_mode='border', align_corners=True).permute(0, 2, 3, 4, 1)
                    sampled_lat_vec_i = valid[..., None] * sampled_lat_vec_i # Padding outside voxel
                    # Append results.
                    num_blur_mask = torch.sum(blur_mask_i)
                    num_normal_mask = torch.sum(normal_mask_i)
                    total_blur_mask += num_blur_mask
                    total_normal_mask += num_normal_mask
                    sampled_lat_vec.append(sampled_lat_vec_i[0, 0, :num_blur_mask])
                    sampled_lat_vec_r.append(sampled_lat_vec_i[0, 0, num_blur_mask:num_blur_mask + num_normal_mask])
                    sampled_lat_vec_u.append(sampled_lat_vec_i[0, 0, num_blur_mask + num_normal_mask:])
                
                sampled_lat_vec = torch.cat(sampled_lat_vec)
                sampled_lat_vec_r = torch.cat(sampled_lat_vec_r)
                sampled_lat_vec_u = torch.cat(sampled_lat_vec_u)

                if self.integrate_sampling_mode=='CAT':
                    sampled_lat_vec = self.fc(sampled_lat_vec.reshape(-1, self.fc_inp_size))
                    sampled_lat_vec_r = self.fc(sampled_lat_vec_r.reshape(-1, self.fc_inp_size))
                    sampled_lat_vec_u = self.fc(sampled_lat_vec_u.reshape(-1, self.fc_inp_size))
                    return [sampled_lat_vec, sampled_lat_vec_r, sampled_lat_vec_u], total_blur_mask, total_normal_mask
            
            else:
                # Sample features
                sampled_lat_vec = []
                for lat_voxel_i, rays_d_wrd_i, rays_o_i, blur_mask_i in zip(lat_voxel, rays_d_wrd, rays_o, blur_mask):
                    with torch.no_grad():
                        rays_d_wrd_i = rays_d_wrd_i[blur_mask_i]
                        rays_o_i = rays_o_i[blur_mask_i]
                        # Get sample point
                        sample_point_i = 1 / self.voxel_scale * (rays_o_i[..., None, :] + rays_d_wrd_i[..., None, :] * self.sample_ind.to(rays_o_i.device)).detach().to(lat_voxel.dtype)
                        valid = torch.prod(torch.gt(sample_point_i, -1.0) * torch.lt(sample_point_i, 1.0), dim=-1).byte().detach().to(lat_voxel.dtype)
                    sampled_lat_vec_i = F.grid_sample(lat_voxel_i.unsqueeze(0), sample_point_i, padding_mode='border', align_corners=True).permute(0, 2, 3, 4, 1)
                    sampled_lat_vec_i = valid[..., None] * sampled_lat_vec_i # Padding outside voxel
                    # Append results.
                    sampled_lat_vec.append(sampled_lat_vec_i[0, 0, ...])

                sampled_lat_vec = torch.cat(sampled_lat_vec)

                if self.integrate_sampling_mode=='CAT':
                    sampled_lat_vec = self.fc(sampled_lat_vec.reshape(-1, self.fc_inp_size))
                    return [sampled_lat_vec]





# Model
class DDF_mlp(pl.LightningModule):

    def __init__(self, D, W, input_ch_pos, input_ch_dir, output_ch, skips=[4], input_ch_vec=0):
        super().__init__()

        self.D = D
        self.W = W
        self.input_ch_pos = input_ch_pos
        self.input_ch_dir = input_ch_dir
        self.skips = skips
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch_pos + input_ch_dir + input_ch_vec, W)] + [nn.Linear(W, W) if i not in self.skips 
            else nn.Linear(W + input_ch_pos + input_ch_dir + input_ch_vec, W) 
            for i in range(D-1)]
            )
        
        self.act = nn.LeakyReLU(0.1)
        self.views_linears = nn.ModuleList([nn.Linear(W, W//2)])
        self.feature_linear = nn.Linear(W, W)
        self.depth_linear = nn.Linear(W//2, output_ch)
        self.inverce_depth_normalization = nn.Softplus()

    def forward(self, x):
        h = x

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = self.act(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)

        h = self.feature_linear(h) # not input dir
        
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = self.act(h)

        inverce_depth = self.depth_linear(h)

        return self.inverce_depth_normalization(inverce_depth)





class DDF(pl.LightningModule):

    def __init__(self, args):
        super().__init__()

        # Base configs
        self.H = args.ddf_H
        self.W = self.H
        self.fov = args.fov
        self.same_instances = args.same_instances
        self.use_world_dir = True

        self.vec_lrate = args.vec_lrate
        self.model_lrate = args.model_lrate

        self.canonical_bbox_diagonal = 1.0
        self.use_3d_code = args.use_3d_code
        self.only_latent = args.only_latent
        self.latent_size = args.latent_size
        self.latent_3d_size = args.latent_3d_size

        # Latent vecs
        self.lat_vecs = torch.nn.Embedding(args.N_instances, self.latent_size, max_norm=1.0)
        torch.nn.init.normal_(self.lat_vecs.weight.data, 0.0, 1.0 / math.sqrt(self.latent_size))

        # Make model
        model_config = {
                        'netdepth' : args.netdepth,
                        'netwidth' : args.netwidth,
                        'output_ch' : 1,
                        'skips' : [4],
                        'mapping_size' : args.mapping_size,
                        'mapping_scale' : args.mapping_scale,
            }

        if self.use_3d_code:
            self.decoder = DDF_decoder(args)
            self.latent_sampler = DDF_latent_sampler(args)
            if args.only_latent:
                self.mlp = DDF_mlp(D=model_config['netdepth'], W=model_config['netwidth'],
                                input_ch_pos=0, output_ch=model_config['output_ch'], skips=model_config['skips'],
                                input_ch_dir=0, input_ch_vec=args.latent_3d_size)
            else:
                self.mlp = DDF_mlp(D=model_config['netdepth'], W=model_config['netwidth'],
                                input_ch_pos=3, output_ch=model_config['output_ch'], skips=model_config['skips'],
                                input_ch_dir=3, input_ch_vec=args.latent_3d_size)
        else:
            self.latent_sampler = False
            self.mlp = DDF_mlp(D=model_config['netdepth'], W=model_config['netwidth'],
                            input_ch_pos=3, output_ch=model_config['output_ch'], skips=model_config['skips'],
                            input_ch_dir=3, input_ch_vec=args.latent_size)

        # log config
        self.save_interval = args.save_interval
        self.log_image_interval = 10
        # self.test_path = args.test_path

        # loss func.
        self.mae = nn.L1Loss()
        self.code_reg_lambda = args.code_reg_lambda

        # normal
        self.use_normal_loss = args.use_normal_loss
        self.start_normal_loss = 1e4
        self.current_normal_loss = True
        self.rays_d_cam = get_ray_direction(args.ddf_H, args.fov, False)
        self.diff_rad = args.pixel_diff_ratio * torch.pi 
        self.rays_o_cam = torch.tensor([0., 0., -1])
        self.rot_r = Exp(torch.tensor([0., self.diff_rad, 0.]))
        self.rot_u = Exp(torch.tensor([self.diff_rad ,0., 0.]))

        # Model info
        self.model_params_dtype = False
        self.model_device = False

        # far point config
        self.origin = torch.zeros(3)
        self.radius = 1.



    def forward(self, rays_o, rays_d, input_lat_vec, blur_mask='without_mask'):

        # get latent vec
        lat_voxel = self.decoder(input_lat_vec)
        sampled_lat_vec = self.latent_sampler(lat_voxel, rays_d, rays_o, blur_mask)

        # get inp tensor
        if self.only_latent:
            inp = sampled_lat_vec
        else:
            inp = torch.cat([rays_o, rays_d, sampled_lat_vec], dim=-1)

        # estimate depth with an mlp.
        est_inverced_depth = self.mlp(inp).squeeze(-1)

        if not blur_mask=='without_mask':
            return est_inverced_depth
        else:
            batch_size, H, W, dim = rays_o.shape
            return est_inverced_depth.reshape(batch_size, H, W)



    def training_step(self, batch, batch_idx):

        with torch.no_grad():
            # Get input
            instance_id, pos, c2w, rays_d_cam, inverced_depth_map, blur_mask, gt_normal_map, normal_mask = batch

            # Train with only one instance
            if self.same_instances:
                instance_id = torch.zeros_like(instance_id).detach()

            # Get ray direction
            rays_o = pos[:, None, None, :].expand(-1, self.H, self.W, -1).detach()
            if not self.use_world_dir:
                print('Support for world coordinate system only.')
                sys.exit()
            rays_d_wrd = torch.sum(rays_d_cam[:, :, :, None, :] * c2w[:, None, None, :, :], -1).detach()

            # Sample rays for normal loss.
            if self.use_normal_loss:
                rot_r = self.rot_r.to(rays_d_cam.device, rays_d_cam.dtype)
                rays_d_cam_r = torch.sum(rays_d_cam[:, :, :, None, :] * rot_r[None, None, None, :, :], -1)
                rays_d_wrd_r = torch.sum(rays_d_cam_r[:, :, :, None, :] * c2w[:, None, None, :, :], -1)
                rot_u = self.rot_u.to(rays_d_cam.device, rays_d_cam.dtype)
                rays_d_cam_u = torch.sum(rays_d_cam[:, :, :, None, :] * rot_u[None, None, None, :, :], -1)
                rays_d_wrd_u = torch.sum(rays_d_cam_u[:, :, :, None, :] * c2w[:, None, None, :, :], -1)
                rays_d_cru = torch.stack([rays_d_wrd, rays_d_wrd_r, rays_d_wrd_u], 1).detach() # batch, normal, H, W, xyz
                rays_o_cru = rays_o[:, None, ...].expand(-1, 3, -1, -1, -1) # batch, normal, H, W, xyz

        # Get latent code
        if self.use_3d_code:
            input_lat_vec = self.lat_vecs(instance_id)
            lat_voxel = self.decoder(input_lat_vec)
            if self.use_normal_loss:
                sampled_lat_vec, num_blur_mask, num_normal_mask = self.latent_sampler(lat_voxel, rays_d_cru, rays_o_cru, blur_mask, normal_mask, train=True)
            else:
                sampled_lat_vec = self.latent_sampler(lat_voxel, rays_d_wrd, rays_o, blur_mask, train=True)
        else:
            sampled_lat_vec = [self.lat_vecs(instance_id)[:, None, None, :].expand(-1, self.H, self.W, -1)]

        # get inp tensor
        inp = []
        for i, sampled_lat_vec_i in enumerate(sampled_lat_vec): # [center, right, under]
            if self.only_latent:
                inp.append(sampled_lat_vec_i)
            else:
                if self.use_3d_code:
                    if i == 0:
                        inp.append(torch.cat([rays_o[blur_mask], rays_d_cru[:, i][blur_mask], sampled_lat_vec_i], dim=-1))
                    elif i > 0:
                        inp.append(torch.cat([rays_o[normal_mask], rays_d_cru[:, i][normal_mask], sampled_lat_vec_i], dim=-1))
                else:
                    inp.append(torch.cat([rays_o, rays_d_wrd, sampled_lat_vec_i], dim=-1))
        inp = torch.cat(inp, 0)
        
        # Estimate inverced depth
        est_inverced_depth_cru = self.mlp(inp).squeeze(-1)
        if self.use_normal_loss:
            est_inverced_depth = est_inverced_depth_cru[:num_blur_mask]
            est_inverced_depth_r = est_inverced_depth_cru[num_blur_mask:num_blur_mask+num_normal_mask]
            est_inverced_depth_u = est_inverced_depth_cru[num_blur_mask+num_normal_mask:]
        else:
            est_inverced_depth = est_inverced_depth_cru
            
        # Cal depth loss.
        depth_loss = self.mae(est_inverced_depth, inverced_depth_map[blur_mask])
        # depth_loss = F.mse_loss(est_inverced_depth, inverced_depth_map[blur_mask])

        # Cal latent reg.
        latent_vec_reg = torch.sum(torch.norm(input_lat_vec, dim=-1)) / input_lat_vec.shape[0]

        # Cal normal loss.
        if self.current_normal_loss:
            # Get masks.
            hit_obj_mask = torch.full_like(normal_mask, False, dtype=bool)
            hit_obj_mask[normal_mask] = (est_inverced_depth[normal_mask[blur_mask]] > .5) * (est_inverced_depth_r > .5) * (est_inverced_depth_u > .5)
            hit_obj_mask = hit_obj_mask.detach()
            # Get depth.
            est_depth = 1 / est_inverced_depth[hit_obj_mask[blur_mask]]
            est_depth_r = 1 / est_inverced_depth_r[hit_obj_mask[normal_mask]]
            est_depth_u = 1 / est_inverced_depth_u[hit_obj_mask[normal_mask]]
            # Get surface points.
            est_point = est_depth[..., None] * rays_d_wrd[hit_obj_mask].reshape(-1, 3) + rays_o[hit_obj_mask].reshape(-1, 3)
            est_point_r = est_depth_r[..., None] * rays_d_wrd_r[hit_obj_mask].reshape(-1, 3) + rays_o[hit_obj_mask].reshape(-1, 3)
            est_point_u = est_depth_u[..., None] * rays_d_wrd_u[hit_obj_mask].reshape(-1, 3) + rays_o[hit_obj_mask].reshape(-1, 3)
            # Calculate normals from exterior products.
            diff_from_right = est_point - est_point_r
            diff_from_under = est_point - est_point_u
            est_normal = F.normalize(torch.cross(diff_from_right, diff_from_under, dim=-1), dim=-1)
            # Calculate normal loss.
            normal_loss = self.mae(est_normal, gt_normal_map[hit_obj_mask])
            # normal_loss = F.mse_loss(est_normal, gt_normal_map[hit_obj_mask])

        # Cal latent reg.
        latent_vec_reg = torch.sum(torch.norm(input_lat_vec, dim=-1)) / input_lat_vec.shape[0]

        # Total los function.
        if self.current_normal_loss:
            loss = depth_loss + 0.01 * normal_loss + self.code_reg_lambda * min(1, self.current_epoch / 1000) * latent_vec_reg
            # loss = depth_loss + 0.0001 * normal_loss + self.code_reg_lambda * min(1, self.current_epoch / 1000) * latent_vec_reg
            if torch.isnan(loss):
                print('##################################################')
                print('####################depth#########################')
                print('##################################################')
                print(depth_loss)
                print(normal_loss)
                print('##################################################')
                print('####################depth#########################')
                print('##################################################')
                print(est_depth.min(), est_depth.max())
                print(est_depth_r.min(), est_depth_r.max())
                print(est_depth_u.min(), est_depth_u.max())
                print('#####################################################')
                print('####################invdepth#########################')
                print('#####################################################')
                print(est_inverced_depth.min(), est_inverced_depth.max())
                print(est_inverced_depth_r.min(), est_inverced_depth_r.max())
                print(est_inverced_depth_u.min(), est_inverced_depth_u.max())
                sys.exit()

        else:
            loss = depth_loss + self.code_reg_lambda * min(1, self.current_epoch / 1000) * latent_vec_reg
        
        # # log image
        # if batch_idx == 0:
        #     sample_img = torch.zeros_like(inverced_depth_map[0])
        #     sample_img[blur_mask[0]] = est_inverced_depth[:torch.sum(blur_mask[0])]
        #     sample_img = sample_img / sample_img.max()
        #     sample_img = torch.cat([sample_img, inverced_depth_map[0]], dim=0).unsqueeze(0)
        #     self.logger.experiment.add_image('train/estimated_depth', sample_img, 0)
        return loss



    def training_epoch_end(self, outputs):

        # Log loss.
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        current_epoch = torch.tensor(self.current_epoch + 1., dtype=avg_loss.dtype)
        self.log_dict({'train/total_loss': avg_loss, "step": current_epoch})

        # Save ckpt.
        if (self.current_epoch + 1) % self.save_interval == 0:
            ckpt_name = str(self.current_epoch + 1).zfill(10) + '.ckpt'
            ckpt_path = os.path.join(self.trainer.log_dir, 'checkpoints', ckpt_name)
            trainer.save_checkpoint(ckpt_path)

        # # Switch loss.
        # if self.use_normal_loss and self.current_epoch > self.start_normal_loss and not(self.current_normal_loss):
        #     self.current_normal_loss = True
        #     self.configure_optimizers() # Reset optimizer stats?



    # https://risalc.info/src/sphere-line-intersection.html を参考
    def forward_from_far(self, rays_o, rays_d, input_lat_vec, return_invdepth = True):
        # def depthmap_renderer_voxel(self, decoder, model, instance_id, lat_vecs, pos, c2w):
        origin = self.origin.to(rays_o)
        radius = self.radius

        if not self.use_world_dir:
            print('Can use only world dir.')

        D = torch.sum(rays_d * (rays_o - origin), dim=-1)**2 - (torch.sum((rays_o - origin)**2, dim=-1) - radius**2)
        negative_D_mask = D < 1e-12

        d_dot_o = torch.sum(rays_d * (rays_o - origin), dim=-1)
        D[negative_D_mask] = 1e-12
        sqrt_D = torch.sqrt(D)
        t_minus = - d_dot_o - sqrt_D
        t_plus = - d_dot_o + sqrt_D

        t_mask = torch.abs(t_plus) > torch.abs(t_minus)
        t = t_plus
        t[t_mask] = t_minus[t_mask]
        intersect_rays_o = rays_o + t_plus[..., None] * rays_d
        intersect_rays_o[t_mask] = (rays_o + t_minus[..., None] * rays_d)[t_mask]

        # Estimate inverced depth
        est_invdepth_rawmap = self.forward(intersect_rays_o, rays_d, input_lat_vec)
        est_invdepth_map = est_invdepth_rawmap / (1. + est_invdepth_rawmap * t.to(est_invdepth_rawmap))
        est_invdepth_map[negative_D_mask] = 0.

        if return_invdepth:
            return est_invdepth_map, negative_D_mask
    


    def get_normals(self, rays_o, rays_d_wrd, input_lat_vec, c2w, hit_obj_mask, est_inverced_depth=False):
        with torch.no_grad():
            rays_d_cam_r = torch.sum(self.rays_d_cam[:, :, :, None, :] * self.rot_r[None, None, None, :, :], -1)
            rays_d_cam_r = rays_d_cam_r.to(c2w.device).to(c2w.dtype)
            rays_d_wrd_r = torch.sum(rays_d_cam_r[:, :, :, None, :] * c2w[:, None, None, :, :], -1).detach()
            rays_d_cam_u = torch.sum(self.rays_d_cam[:, :, :, None, :] * self.rot_u[None, None, None, :, :], -1)
            rays_d_cam_u = rays_d_cam_u.to(c2w.device).to(c2w.dtype)
            rays_d_wrd_u = torch.sum(rays_d_cam_u[:, :, :, None, :] * c2w[:, None, None, :, :], -1).detach()

        est_inverced_depth_r = self(rays_o, rays_d_wrd_r, input_lat_vec, blur_mask=hit_obj_mask)
        est_inverced_depth_u = self(rays_o, rays_d_wrd_u, input_lat_vec, blur_mask=hit_obj_mask)

        est_depth = 1 / est_inverced_depth
        est_depth_r = 1 / est_inverced_depth_r
        est_depth_u = 1 / est_inverced_depth_u

        est_point = est_depth[..., None] * rays_d_wrd[hit_obj_mask].reshape(-1, 3) + rays_o[hit_obj_mask].reshape(-1, 3)
        est_point_r = est_depth_r[..., None] * rays_d_wrd_r[hit_obj_mask].reshape(-1, 3) + rays_o[hit_obj_mask].reshape(-1, 3)
        est_point_u = est_depth_u[..., None] * rays_d_wrd_u[hit_obj_mask].reshape(-1, 3) + rays_o[hit_obj_mask].reshape(-1, 3)

        diff_from_right = est_point - est_point_r
        diff_from_under = est_point - est_point_u
        est_normal = F.normalize(torch.cross(diff_from_right, diff_from_under, dim=-1), dim=-1)
        return est_normal



    def render_depth_map(self, pos, c2w, instance_id, H=False, inverced_depth_map=True):
        if not H:
            H = self.H
        W = H
        
        # get inputs
        rays_d = get_ray_direction(H, self.fov, c2w) # batch, H, W, 3:xyz
        rays_o = pos[:, None, None, :].expand(-1, H, W, -1)
        input_lat_vec = self.lat_vecs(instance_id)

        # Estimate depth.
        est_depth, mask, _  = self.forward_from_far(rays_o, rays_d, input_lat_vec, inverced_depth = inverced_depth_map)

        # Make depth map
        depth_map = torch.zeros_like(mask[0], dtype=est_depth.dtype, device=est_depth.device)
        depth_map[mask[0]] = est_depth

        return depth_map
    


    def check_model_info (self):
        self.model_params_dtype = list(self.mlp.parameters())[-1].dtype
        self.model_device = self.device



    def configure_optimizers(self):
        if self.use_3d_code:
            optimizer = torch.optim.Adam([
                {"params": self.lat_vecs.parameters(), "lr": self.vec_lrate},
                {"params": self.decoder.parameters()},
                {"params": self.latent_sampler.parameters()},
                {"params": self.mlp.parameters()},
            ], lr=self.model_lrate, betas=(0.9, 0.999),)
            # optimizer = torch.optim.SGD([
            #     {"params": self.lat_vecs.parameters(), "lr": self.vec_lrate},
            #     {"params": self.decoder.parameters()},
            #     {"params": self.latent_sampler.parameters()},
            #     {"params": self.mlp.parameters()},
            # ], lr=self.model_lrate)
        else:
            optimizer = torch.optim.Adam([
                {"params": self.lat_vecs.parameters(), "lr": self.vec_lrate},
                {"params": self.mlp.parameters()},
            ], lr=self.model_lrate, betas=(0.9, 0.999),)
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
        strategy=DDPPlugin(find_unused_parameters=False), 
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
    train_dataset = DDF_dataset(args, args.train_data_dir, args.N_views)
    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.N_batch, num_workers=args.num_workers, drop_last=False, shuffle=True)
    val_dataset = DDF_dataset(args, args.val_data_dir, args.N_val_views)
    val_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.N_batch, num_workers=args.num_workers, drop_last=False, shuffle=True)

    # For single instance.
    args.same_instances = False
    # if len(train_data_list) != 1 and train_data_list[0]==train_data_list[-1]:
    #     args.same_instances = True

    # Get ckpts path.
    ckpt_dir = os.path.join('lightning_logs', f'{args.expname}_{args.exp_version}', 'checkpoints/*')
    ckpt_path_list = sorted(glob.glob(ckpt_dir))

    # Load ckpt and start training.
    if len(ckpt_path_list) == 0:
        model = DDF(args)
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
        model = DDF(args)
        trainer.fit(
            model=model, 
            train_dataloaders=train_dataloader, 
            val_dataloaders=val_dataloader, 
            datamodule=None, 
            ckpt_path=latest_ckpt_path
            )
        # model = model.load_from_checkpoint(checkpoint_path=latest_ckpt_path, args=args)
        # trainer.fit(
        #     model=model, 
        #     train_dataloaders=train_dataloader, 
        #     val_dataloaders=val_dataloader, 
        #     datamodule=None, 
        #     ckpt_path=None
        #     )
