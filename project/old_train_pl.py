import os
import sys
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from chamferdist import ChamferDistance
from scipy.spatial.transform import Rotation as R

from parser import *
from often_use import *





class DDF_decoder(pl.LightningModule):

    def __init__(self, args):
        super().__init__()

        self.decoder_fc = nn.Sequential(
                nn.Linear(args.latent_size, 1024), nn.LeakyReLU(0.2)
                )
        self.cov_inp_size = 1024
        if not args.integrate_sampling_mode == 'Volume_Rendering':
            self.decoder_cov = nn.Sequential(
                    nn.ConvTranspose3d(1024, 512, 4, 2, 1), nn.LeakyReLU(0.2),
                    nn.ConvTranspose3d(512, 256, 4, 2, 1), nn.LeakyReLU(0.2),
                    nn.ConvTranspose3d(256, 128, 4, 2, 1), nn.LeakyReLU(0.2),
                    nn.ConvTranspose3d(128, 64, 4, 2, 1),
                    nn.Softplus(),
                    )
        elif args.integrate_sampling_mode == 'Volume_Rendering':
            self.decoder_cov = nn.Sequential(
                    nn.ConvTranspose3d(1024, 512, 4, 2, 1), nn.LeakyReLU(0.2),
                    nn.ConvTranspose3d(512, 256, 4, 2, 1), nn.LeakyReLU(0.2),
                    nn.ConvTranspose3d(256, 128, 4, 2, 1), nn.LeakyReLU(0.2),
                    nn.ConvTranspose3d(128, 64, 4, 2, 1),
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
        self.H = args.H
        self.W = args.W
        self.fov = args.fov
        self.use_3d_code = args.use_3d_code
        self.only_latent = args.only_latent
        self.latent_size = args.latent_size
        self.latent_3d_size = args.latent_3d_size

        # Decoder
        self.decoder = DDF_decoder(args)

        # For sampling
        self.voxel_scale = args.voxel_scale
        sample_start = 1 - math.sqrt(3) * self.voxel_scale
        sample_end = 1 + math.sqrt(3) * self.voxel_scale
        self.sample_ind = torch.linspace(sample_start, sample_end, args.voxel_sample_num)[None, None, None, :, None] # to(torch.half)

        # Sampling config
        self.voxel_sample_num = args.voxel_sample_num
        self.voxel_ch_num = args.voxel_ch_num

        # Integrate config
        self.integrate_sampling_mode = args.integrate_sampling_mode
        if self.integrate_sampling_mode=='TransFormer':
            self.positional_encoding = PositionalEncoding(self.voxel_ch_num, 0.0, max_len=self.voxel_sample_num)
            self.time_series_model = nn.MultiheadAttention(embed_dim=self.voxel_ch_num, num_heads=args.num_heads, bias=True)


    def forward(self, inp, rays_d_wrd=False, rays_o=False, blur_mask=False):
        # Decode voxel
        lat_voxel = self.decoder(inp)
        if self.integrate_sampling_mode == 'Volume_Rendering':
            lat_voxel = torch.sigmoid(lat_voxel)


        # Sample features
        sample_point = 1 / self.voxel_scale * (rays_o[:, :, :, None, :] + rays_d_wrd[:, :, :, None, :] * self.sample_ind.to(rays_o.device))
        sample_point = sample_point.to(lat_voxel.dtype) # OK ?
        sampled_lat_vec = F.grid_sample(lat_voxel, sample_point, padding_mode='border', align_corners=True).permute(0, 2, 3, 4, 1)
        valid = torch.prod(torch.gt(sample_point, -1.0) * torch.lt(sample_point, 1.0), dim=-1).byte().float()
        sampled_lat_vec = valid[..., None] * sampled_lat_vec # Padding outside voxel


        # # Integrate sampled features
        if self.integrate_sampling_mode == 'Volume_Rendering':
            sampled_lat_vec = sampled_lat_vec[blur_mask].permute(0, 2, 1).reshape(-1, self.voxel_sample_num) # N_rays*N_voxel_ch, N_samples

            raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
            z_vals = (self.voxel_scale * self.sample_ind).permute(0, 1, 2, 4, 3).reshape(-1, args.voxel_sample_num)
            dists = z_vals[..., 1:] - z_vals[..., :-1]
            dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)
            alpha = raw2alpha(sampled_lat_vec, dists.to(sampled_lat_vec.device)) # N_rays, N_voxel_ch, N_samples

            weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
            depth_feature_map = torch.sum(weights * z_vals.to(sampled_lat_vec.device), -1).reshape(-1, self.voxel_ch_num)
            return depth_feature_map

        if self.integrate_sampling_mode=='TransFormer':
            sampled_lat_vec = sampled_lat_vec[blur_mask].permute(1, 0, 2)
            sampled_lat_vec = self.positional_encoding(sampled_lat_vec)
            Q = K = V = sampled_lat_vec
            attn_output, attn_weights = self.time_series_model(Q, K, V)
            integrated_lat_vec = attn_output.permute(1, 2, 0).sum(-1)
            return integrated_lat_vec

        elif self.integrate_sampling_mode=='CAT':
            if blur_mask == 'without_mask':
                return sampled_lat_vec.reshape(-1, self.H, self.W, self.latent_3d_size)
            else:
                return sampled_lat_vec.reshape(-1, self.H, self.W, self.latent_3d_size)[blur_mask]





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





def get_ray_direction(size, fov, c2w=False):
    fov = torch.deg2rad(torch.tensor(fov, dtype=torch.float))
    x_coord = torch.linspace(-torch.tan(fov*.5), torch.tan(fov*.5), size)[None].expand(size, -1)
    y_coord = x_coord.T
    rays_d_cam = torch.stack([x_coord, y_coord, torch.ones_like(x_coord)], dim=2)
    rays_d_cam = F.normalize(rays_d_cam, dim=-1)
    if c2w is False:
        return rays_d_cam.detach() # H, W, 3:xyz
    else:
        rays_d_cam = rays_d_cam.unsqueeze(0).to(c2w.device).detach()
        rays_d_wrd = torch.sum(rays_d_cam[:, :, :, None, :] * c2w[:, None, None, :, :], -1)
        return rays_d_wrd # batch, H, W, 3:xyz





class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)





class DDF_dataset(data.Dataset):
    def __init__(self, data_list, args):
        self.data_list = data_list
        self.H = args.H
        self.W = args.W
        self.fov = args.fov
        self.N_views = args.N_views
        # axis_grid = torch.linspace(0, self.size-1, self.size)
        # self.coords = torch.stack(torch.meshgrid(axis_grid, axis_grid), -1).reshape([-1,2])
        self.use_normal = args.use_normal_loss + args.use_normal_data
        self.rays_d_cam = get_ray_direction(self.H, self.fov)

    def __getitem__(self, index):
        view_ind = random.randrange(0, self.N_views)
        path = os.path.join(self.data_list[index], str(view_ind).zfill(5))
        if not(os.path.exists(path+'_mask.pickle') and os.path.exists(path+'_pose.pickle')):
            print('Data doesnt exist')
            print(path)
            sys.exit()
        camera_info = pickle_load(path+'_pose.pickle')
        pos = camera_info['pos']
        c2w = camera_info['rot'].T

        depth_info = pickle_load(path+'_mask.pickle')
        inverced_depth = depth_info['inverced_depth']
        blur_mask = depth_info['blur_mask']
        if self.use_normal:
            if 'normal_map' not in depth_info.keys():
                print(path)
            normal_map = depth_info['normal_map']
            normal_mask = depth_info['normal_mask']
        else:
            normal_map = False
            normal_mask = False

        return index, pos, c2w, self.rays_d_cam, inverced_depth, blur_mask, normal_map, normal_mask

    def __len__(self):
        return len(self.data_list)





class DDF(pl.LightningModule):

    def __init__(self, args):
        super().__init__()

        # Base configs
        self.H = args.H
        self.W = args.W
        self.fov = args.fov
        self.same_instances = args.same_instances
        self.use_world_dir = args.use_world_dir

        self.vec_lrate = args.vec_lrate
        self.model_lrate = args.model_lrate

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
                        'multires_views' : args.multires_views
            }

        if self.use_3d_code:
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



    def get_inp(self, instance_id, rays_d_wrd, rays_o, blur_mask):
        if self.use_3d_code:
            input_lat_vec = self.lat_vecs(instance_id)
            sampled_lat_vec = self.latent_sampler(input_lat_vec, rays_d_wrd, rays_o, blur_mask)
        else:
            input_lat_vec = self.lat_vecs(instance_id)
            sampled_lat_vec = self.lat_vecs(instance_id)[:, None, None, :].expand(-1, self.H, self.W, -1)

        if self.only_latent:
            inp = sampled_lat_vec
        else:
            inp = torch.cat([rays_o[blur_mask], rays_d[blur_mask], sampled_lat_vec], dim=-1)

        return inp, input_lat_vec



    def get_inp_val(self, input_lat_vec, rays_d_wrd, rays_o):
        if self.use_3d_code:
            input_lat_vec = self.lat_vecs(instance_id)
            sampled_lat_vec = self.latent_sampler(input_lat_vec, rays_d_wrd, rays_o, blur_mask)
        else:
            input_lat_vec = self.lat_vecs(instance_id)
            sampled_lat_vec = self.lat_vecs(instance_id)[:, None, None, :].expand(-1, self.H, self.W, -1)

        if self.only_latent:
            inp = sampled_lat_vec
        else:
            inp = torch.cat([rays_o[blur_mask], rays_d[blur_mask], sampled_lat_vec], dim=-1)

        return inp, input_lat_vec



    def forward(self, input_tensor, rays_d_wrd=False, rays_o=False, blur_mask='without_mask'):
        import pdb; pdb.set_trace()
        return 0



    def training_step(self, batch, batch_idx):
        # Get input
        instance_id, pos, c2w, rays_d_cam, inverced_depth, blur_mask, gt_normal_map, normal_mask = batch
        current_batch_size = c2w.shape[0]
        rays_o = pos[:, None, None, :].expand(-1, self.H, self.W, -1).detach()

        # Train with only one instance
        if self.same_instances:
            instance_id = torch.zeros_like(instance_id)

        # Get ray direction
        if not self.use_world_dir:
            print('Support for world coordinate system only.')
            sys.exit()
        rays_d_wrd = torch.sum(rays_d_cam[:, :, :, None, :] * c2w[:, None, None, :, :], -1)

        # Make input
        inp, input_lat_vec = self.get_inp(instance_id, rays_d_wrd, rays_o, blur_mask)

        # Estimate inverced depth
        est_inverced_depth = self.mlp(inp).reshape(-1)

        # Cal depth loss.
        depth_loss = F.mse_loss(est_inverced_depth, inverced_depth[blur_mask].to(torch.float32))
        latent_vec_reg = torch.sum(torch.norm(input_lat_vec, dim=-1)) / input_lat_vec.shape[0]

        # Cal total loss
        if args.use_normal_loss:
            loss = depth_loss + 0.5 * normal_loss + args.code_reg_lambda * min(1, self.current_epoch / 1000) * latent_vec_reg
        else:
            loss = depth_loss + args.code_reg_lambda * min(1, self.current_epoch / 1000) * latent_vec_reg

        # log
        self.log('train/total_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def configure_optimizers(self):

        if self.use_3d_code:
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": self.lat_vecs.parameters(),
                        "lr": self.vec_lrate,
                        "betas": (0.9, 0.999),
                    },
                    {
                        "params": self.latent_sampler.parameters(),
                        "lr": self.model_lrate,
                        "betas": (0.9, 0.999),
                    },
                    {
                        "params": self.mlp.parameters(),
                        "lr": self.model_lrate,
                        "betas": (0.9, 0.999),
                    },
                ]
            )
        else:
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": model.parameters(),
                        "lr": self.model_lrate,
                        "betas": (0.9, 0.999),
                    },
                    {
                        "params": lat_vecs.parameters(),
                        "lr": self.vec_lrate,
                        "betas": (0.9, 0.999),
                    },
                ]
            )
        return optimizer



if __name__=='__main__':
    # atgs
    args = get_args()


    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    os.makedirs(os.path.join(basedir, expname, 'network'), exist_ok=True)
    os.makedirs(os.path.join(basedir, expname, 'network', 'old_models'), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())


    # Create dataloader
    train_data_list = []
    with open(args.instance_list_txt, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            train_data_list.append(os.path.join(args.datadir, line.rstrip('\n')))
    if len(train_data_list) != args.N_instances:
        print('Instance number is missmutching !')
    train_dataset = DDF_dataset(train_data_list, args)
    dataloader = data_utils.DataLoader(train_dataset, batch_size=args.N_batch, num_workers=args.num_workers)

    args.same_instances = False
    if len(train_data_list) != 1 and train_data_list[0]==train_data_list[-1]:
        args.same_instances = True


    # Set trainer.
    checkpoint = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(basedir, expname, 'network'),
            save_top_k=-1,
            every_n_epochs=args.save_interval,
        )
    trainer = pl.Trainer(callbacks=[checkpoint], gpus= torch.cuda.device_count(), accelerator="ddp", min_epochs=0, max_epochs=args.N_epoch, plugins=DDPPlugin(find_unused_parameters=False))


    # Get ckpts.
    ckpt_path_list = glob.glob(os.path.join(basedir, expname, 'network', '*.ckpt'))

    # Check file name : 'epoch=***-step=***.ckpt' type is correct.
    check_file_name = all([re.sub(r'[0-9]+', '', ckpt_path.split('/')[-1]) == 'epoch=-step=.ckpt' for ckpt_path in ckpt_path_list])
    if not check_file_name:
        print('Including unknown type file name.')
        sys.exit()

    if len(ckpt_path_list) == 0:
        model = DDF(args)
        trainer.fit(model, dataloader)

    elif len(ckpt_path_list) > 0:
        ckpt_step_list = [int(re.split('epoch=|-step=|.ckpt', ckpt_path)[2]) for ckpt_path in ckpt_path_list]

        if len(ckpt_step_list) != len(set(ckpt_step_list)):
            print('Including same step ckpts.')
            sys.exit()

        latest_ckpt_index = ckpt_step_list.index(max(ckpt_step_list))
        latest_ckpt_path = ckpt_path_list[latest_ckpt_index]
        print('\n', f'+++ Reloading from {latest_ckpt_path} +++ \n')

        model = DDF(args)
        model.load_from_checkpoint(checkpoint_path=latest_ckpt_path, args=args)
        trainer.fit(model, dataloader, ckpt_path=latest_ckpt_path)





##########################################################################################################################################################################################################################################################
##########################################################################################################################################################################################################################################################
##########################################################################################################################################################################################################################################################
##########################################################################################################################################################################################################################################################
##########################################################################################################################################################################################################################################################





import os
import pdb
import sys
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





class DDF_decoder(pl.LightningModule):

    def __init__(self, args):
        super().__init__()

        self.decoder_fc = nn.Sequential(
                nn.Linear(args.latent_size, 1024), nn.LeakyReLU(0.2)
                )
        self.cov_inp_size = 1024
        self.decoder_cov = nn.Sequential(
                nn.ConvTranspose3d(1024, 512, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.ConvTranspose3d(512, 256, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.ConvTranspose3d(256, 128, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.ConvTranspose3d(128, 64, 4, 2, 1),
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
        self.H = args.H
        self.W = args.W
        self.fov = args.fov
        self.use_3d_code = args.use_3d_code
        self.only_latent = args.only_latent
        self.latent_size = args.latent_size
        self.latent_3d_size = args.latent_3d_size

        # Decoder
        self.decoder = DDF_decoder(args)

        # For sampling
        self.voxel_scale = args.voxel_scale
        sample_start = 1 - math.sqrt(3) * self.voxel_scale
        sample_end = 1 + math.sqrt(3) * self.voxel_scale
        self.sample_ind = torch.linspace(sample_start, sample_end, args.voxel_sample_num)[None, None, None, :, None] # to(torch.half)
        
        # Sampling config
        self.voxel_sample_num = args.voxel_sample_num
        self.voxel_ch_num = args.voxel_ch_num
        
        # Integrate config
        self.integrate_sampling_mode = args.integrate_sampling_mode
        self.integrate_TransFormer_mode = args.integrate_TransFormer_mode
        if self.integrate_sampling_mode=='TransFormer':
            self.positional_encoding = PositionalEncoding(self.voxel_ch_num, 0.0, max_len=self.voxel_sample_num)
            self.time_series_model = nn.MultiheadAttention(embed_dim=self.voxel_ch_num, num_heads=args.num_heads, bias=True)
            
    
    def forward(self, inp, rays_d_wrd=False, rays_o=False, blur_mask=False):
        # Decode voxel
        lat_voxel = self.decoder(inp)

        # Sample features
        sample_point = 1 / self.voxel_scale * (rays_o[..., None, :] + rays_d_wrd[..., None, :] * self.sample_ind.to(rays_o.device))
        sample_point = sample_point.to(lat_voxel.dtype) # OK ?
        sampled_lat_vec = F.grid_sample(lat_voxel, sample_point, padding_mode='border', align_corners=True).permute(0, 2, 3, 4, 1)
        valid = torch.prod(torch.gt(sample_point, -1.0) * torch.lt(sample_point, 1.0), dim=-1).byte().float()
        sampled_lat_vec = valid[..., None] * sampled_lat_vec # Padding outside voxel

        # Integrate sampled features
        if self.integrate_sampling_mode=='TransFormer':
            if blur_mask == 'without_mask':
                batch, H, W, _, _ = sampled_lat_vec.shape
                sampled_lat_vec = sampled_lat_vec.reshape(-1, self.voxel_sample_num, self.voxel_ch_num).permute(1, 0, 2)
                sampled_lat_vec = self.positional_encoding(sampled_lat_vec)
                Q = K = V = sampled_lat_vec
                attn_output, attn_weights = self.time_series_model(Q, K, V)
                if  self.integrate_TransFormer_mode == 'tf_sum':
                    integrated_lat_vec = attn_output.permute(1, 2, 0).sum(-1)
                elif self.integrate_TransFormer_mode == 'tf_cat':
                    integrated_lat_vec = attn_output.permute(1, 2, 0).reshape(-1, self.latent_3d_size)
                return integrated_lat_vec.reshape(-1, H, W, self.latent_3d_size)
            else:
                sampled_lat_vec = sampled_lat_vec[blur_mask].permute(1, 0, 2)
                sampled_lat_vec = self.positional_encoding(sampled_lat_vec)
                Q = K = V = sampled_lat_vec
                attn_output, attn_weights = self.time_series_model(Q, K, V)
                if  self.integrate_TransFormer_mode == 'tf_sum':
                    integrated_lat_vec = attn_output.permute(1, 2, 0).sum(-1)
                elif self.integrate_TransFormer_mode == 'tf_cat':
                    integrated_lat_vec = attn_output.permute(1, 2, 0).reshape(-1, self.latent_3d_size)
                return integrated_lat_vec

        elif self.integrate_sampling_mode=='CAT':
            if blur_mask == 'without_mask':
                batch, H, W, _, _ = sampled_lat_vec.shape
                return sampled_lat_vec.reshape(batch, H, W, self.latent_3d_size)
            else:
                batch, H, W = blur_mask.shape
                return sampled_lat_vec.reshape(batch, H, W, self.latent_3d_size)[blur_mask]





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





class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)





class DDF_dataset(data.Dataset):
    def __init__(self, args, data_dir, N_views):
    
        self.data_list = []
        with open(args.instance_list_txt, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                self.data_list.append(os.path.join(data_dir, line.rstrip('\n')))

        self.H = args.H
        self.W = args.W
        self.fov = args.fov
        self.N_views = N_views
        self.use_normal = args.use_normal_loss + args.use_normal_data
        self.rays_d_cam = get_ray_direction(self.H, self.fov)

        self.random_sample_rays = args.random_sample_rays
        self.coords = np.arange(0, self.H**2)
        self.sample_ratio = args.sample_ratio # The ratio of total pixels used in this batch
        self.inside_true_ratio = args.inside_true_ratio # The ratio used in this batch inside the mask
        self.outside_true_ratio = args.outside_true_ratio # The ratio used in this batch outside the mask

    def __getitem__(self, index):
        view_ind = random.randrange(0, self.N_views)
        path = os.path.join(self.data_list[index], str(view_ind).zfill(5))
        if not(os.path.exists(path+'_mask.pickle') and os.path.exists(path+'_pose.pickle')):
            print('Data doesnt exist')
            print(path)
            sys.exit()
        camera_info = pickle_load(path+'_pose.pickle')
        pos = camera_info['pos']
        c2w = camera_info['rot'].T

        depth_info = pickle_load(path+'_mask.pickle')
        inverced_depth = depth_info['inverced_depth']
        blur_mask = depth_info['blur_mask']
        if self.use_normal:
            if 'normal_map' not in depth_info.keys():
                print(path)
            normal_map = depth_info['normal_map']
            normal_mask = depth_info['normal_mask']
        else:
            normal_map = False
            normal_mask = False

        if self.random_sample_rays:

            blur_mask = blur_mask.reshape(-1)
            coord_inside_mask = self.coords[blur_mask]
            coord_outside_mask = self.coords[np.logical_not(blur_mask)]

            inside_true_num = int(self.sample_ratio * self.inside_true_ratio * len(coord_inside_mask))
            inside_false_num = len(coord_inside_mask) - inside_true_num
            outside_true_num = int(self.sample_ratio * self.outside_true_ratio * len(coord_outside_mask))

            np.random.shuffle(coord_inside_mask)
            np.random.shuffle(coord_outside_mask)

            blur_mask[coord_inside_mask[:inside_false_num]] = False
            blur_mask[coord_outside_mask[:outside_true_num]] = True

            blur_mask = blur_mask.reshape(self.H, self.W)

        return index, pos, c2w, self.rays_d_cam, inverced_depth, blur_mask, normal_map, normal_mask

    def __len__(self):
        return len(self.data_list)





class DDF(pl.LightningModule):

    def __init__(self, args):
        super().__init__()

        # Base configs
        self.H = args.H
        self.W = args.W
        self.fov = args.fov
        self.same_instances = args.same_instances
        self.use_world_dir = args.use_world_dir

        self.vec_lrate = args.vec_lrate
        self.model_lrate = args.model_lrate

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
                        'multires_views' : args.multires_views
            }

        if self.use_3d_code:
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
        self.test_path = args.test_path

        # far point config
        self.origin = torch.zeros(3)
        self.radius = 1.

        # Model info
        self.model_params_dtype = False
        self.model_device = False



    def forward(self, rays_o, rays_d, input_lat_vec, blur_mask='without_mask'):

        # get latent vec
        sampled_lat_vec = self.latent_sampler(input_lat_vec, rays_d, rays_o, blur_mask)

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

        # Get input
        instance_id, pos, c2w, rays_d_cam, inverced_depth, blur_mask, gt_normal_map, normal_mask = batch
        rays_o = pos[:, None, None, :].expand(-1, self.H, self.W, -1).detach()

        # Train with only one instance
        if self.same_instances:
            instance_id = torch.zeros_like(instance_id)

        # Get ray direction
        if not self.use_world_dir:
            print('Support for world coordinate system only.')
            sys.exit()
        rays_d_wrd = torch.sum(rays_d_cam[:, :, :, None, :] * c2w[:, None, None, :, :], -1)

        # Get latent code
        input_lat_vec = self.lat_vecs(instance_id)
        if self.use_3d_code:
            sampled_lat_vec = self.latent_sampler(input_lat_vec, rays_d_wrd, rays_o, blur_mask)
        else:
            sampled_lat_vec = self.lat_vecs(instance_id)[:, None, None, :].expand(-1, self.H, self.W, -1)

        # Estimate inverced depth
        est_inverced_depth = self(rays_o, rays_d_wrd, input_lat_vec, blur_mask).reshape(-1)

        # Cal depth loss.
        depth_loss = F.mse_loss(est_inverced_depth, inverced_depth[blur_mask].to(est_inverced_depth.dtype))
        latent_vec_reg = torch.sum(torch.norm(input_lat_vec, dim=-1)) / input_lat_vec.shape[0]

        # Cal total loss
        if args.use_normal_loss:
            loss = depth_loss + 0.5 * normal_loss + args.code_reg_lambda * min(1, self.current_epoch / 1000) * latent_vec_reg
        else:
            loss = depth_loss + args.code_reg_lambda * min(1, self.current_epoch / 1000) * latent_vec_reg

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

        # # log image
        # if self.current_epoch % self.log_image_interval == 0:
        #     # Load path info
        #     instance_id = torch.zeros(1, dtype=torch.long).to(self.device)
        #     test_H = test_W = 256
        #     pos, c2w = path2posc2w(self.test_path, self) # returen as torchtensor

        #     # Get ray direction
        #     rays_o = pos[:, None, None, :].expand(-1, test_H, test_W, -1).detach()
        #     rays_d_wrd = get_ray_direction(test_H, args.fov, c2w)
            
        #     # Get latent code
        #     input_lat_vec = self.lat_vecs(instance_id)

        #     # Estimate inverced depth
        #     sample_img = self(rays_o, rays_d_wrd, input_lat_vec)[0].unsqueeze(0)

        #     # Load depth info
        #     inverced_depth, blur_mask = path2depthinfo(self.test_path, self, H=256)
        #     sample_img = torch.cat([sample_img, inverced_depth], dim=-1)

        #     # log image
        #     sample_img = sample_img / sample_img.max()
        #     self.logger.experiment.add_image('train/estimated_depth', sample_img, 0)



    def validation_step(self, batch, batch_idx):

        # Get input
        instance_id, pos, c2w, rays_d_cam, inverced_depth, blur_mask, gt_normal_map, normal_mask = batch
        rays_o = pos[:, None, None, :].expand(-1, self.H, self.W, -1).detach()

        # Train with only one instance
        if self.same_instances:
            instance_id = torch.zeros_like(instance_id)

        # Get ray direction
        if not self.use_world_dir:
            print('Support for world coordinate system only.')
            sys.exit()
        rays_d_wrd = torch.sum(rays_d_cam[:, :, :, None, :] * c2w[:, None, None, :, :], -1)

        # Get latent code
        input_lat_vec = self.lat_vecs(instance_id)
        if self.use_3d_code:
            sampled_lat_vec = self.latent_sampler(input_lat_vec, rays_d_wrd, rays_o, blur_mask)
        else:
            sampled_lat_vec = self.lat_vecs(instance_id)[:, None, None, :].expand(-1, self.H, self.W, -1)

        # Estimate inverced depth
        est_inverced_depth = self(rays_o, rays_d_wrd, input_lat_vec, blur_mask).reshape(-1)

        # Cal depth err.
        depth_err = F.mse_loss(est_inverced_depth, inverced_depth[blur_mask].to(est_inverced_depth.dtype))
        
        # log image
        sample_img = torch.zeros_like(blur_mask[0], dtype=est_inverced_depth.dtype, device=est_inverced_depth.device)
        sample_img[blur_mask[0]] = est_inverced_depth[:torch.sum(blur_mask[0])]
        sample_img = sample_img.unsqueeze(0)
        sample_img = sample_img / sample_img.max()
        self.logger.experiment.add_image('train/batch_sample_depth', sample_img, 0)
        
        # log image
        sample_img = torch.zeros_like(blur_mask[0], dtype=est_inverced_depth.dtype, device=est_inverced_depth.device)
        sample_img[blur_mask[0]] = est_inverced_depth[:torch.sum(blur_mask[0])]
        sample_img = sample_img.unsqueeze(0)
        sample_img = sample_img / sample_img.max()
        self.logger.experiment.add_image('validation/batch_sample_depth', sample_img, 0)

        return {'depth_err': depth_err}



    def validation_epoch_end(self, outputs):
        # Log loss.
        avg_depth_err = torch.stack([x['depth_err'] for x in outputs]).mean()
        current_epoch = torch.tensor(self.current_epoch + 1., dtype=avg_depth_err.dtype)
        self.log_dict({'validation/total_depth_err': avg_depth_err, "step": current_epoch})



    def forward_from_far(self, rays_o, rays_d, input_lat_vec, blur_mask='without_mask', inverced_depth = True):
        # def depthmap_renderer_voxel(args, decoder, model, instance_id, lat_vecs, pos, c2w):
        origin = self.origin.to(rays_o)
        radius = self.radius

        if not self.use_world_dir:
            print('Can use only world dir.')

        D = torch.sum(rays_d * (rays_o - origin), dim=-1)**2 - (torch.sum((rays_o - origin)**2, dim=-1) - radius**2)
        D_mask = D > 0
        masked_D = D[D_mask]
        masked_rays_d = rays_d[D_mask]
        masked_rays_o = rays_o[D_mask]


        t_minus = - torch.sum(masked_rays_d * (masked_rays_o - origin), dim=-1) - torch.sqrt(masked_D)
        t_plus = - torch.sum(masked_rays_d * (masked_rays_o - origin), dim=-1) + torch.sqrt(masked_D)

        rays_o_minus = masked_rays_o + t_minus[..., None] * masked_rays_d
        rays_o_plus = masked_rays_o + t_plus[..., None] * masked_rays_d
        t_mask = torch.abs(t_plus) > torch.abs(t_minus)

        if t_mask.all():
            masked_rays_o = rays_o_minus
            t = t_minus
            t_minus_used = True
        elif torch.logical_not(t_mask).all():
            masked_rays_o = rays_o_plus
            t = t_plus
            t_minus_used = False
        else:
            print('t sign err!')
            sys.exit()
        
        # Supported only for a single instance.
        if D_mask.shape[0] > 1:
            # num_D_mask = [torch.sum(D_mask_i) for D_mask_i in D_mask] # 最大sample数に合わせ後は零埋め？
            print('Too many instances.')
            sys.exit()

        # Make dummy batch.
        masked_rays_o = masked_rays_o[None, None] # batch=1, dummy_H=1, dummy_W=mask_True_num, 3
        masked_rays_d = masked_rays_d[None, None] # batch=1, dummy_H=1, dummy_W=mask_True_num, 3
        dummy_blur_mask = torch.ones_like(masked_rays_o[..., 0], dtype=torch.bool) # batch=1, dummy_H=1, sample=mask_True_num

        # Estimate inverced depth
        est_inverced_depth = self(masked_rays_o, masked_rays_d, input_lat_vec, dummy_blur_mask)

        # Return results.
        if inverced_depth:
            return est_inverced_depth, D_mask, t_minus_used
        else:
            est_mask = est_inverced_depth > .75
            hit_obj_mask = torch.zeros_like(D_mask)
            hit_obj_mask[D_mask] = est_mask
            est_depth = 1 / est_inverced_depth[est_mask] + t[est_mask]
            return est_depth, hit_obj_mask, t_minus_used



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
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": self.lat_vecs.parameters(),
                        "lr": self.vec_lrate,
                        "betas": (0.9, 0.999),
                    },
                    {
                        "params": self.latent_sampler.parameters(),
                        "lr": self.model_lrate,
                        "betas": (0.9, 0.999),
                    },
                    {
                        "params": self.mlp.parameters(),
                        "lr": self.model_lrate,
                        "betas": (0.9, 0.999),
                    },
                ]
            )
        else:
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": model.parameters(),
                        "lr": self.model_lrate,
                        "betas": (0.9, 0.999),
                    },
                    {
                        "params": lat_vecs.parameters(),
                        "lr": self.vec_lrate,
                        "betas": (0.9, 0.999),
                    },
                ]
            )
        return optimizer



if __name__=='__main__':
    # Get args
    args = get_args()
    args.gpu_num = torch.cuda.device_count() # log used gpu num.


    # Set trainer.
    logger = pl.loggers.TensorBoardLogger(
            save_dir=os.getcwd(),
            version=f'{args.expname}_v{args.exp_version}',
            name='lightning_logs'
        )
    trainer = pl.Trainer(
        gpus=args.gpu_num, 
        strategy=DDPPlugin(find_unused_parameters=False), 
        logger=logger,
        max_epochs=args.N_epoch, 
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        )
    

    # Save config files.
    os.makedirs(os.path.join('lightning_logs', f'{args.expname}_v{args.exp_version}'), exist_ok=True)
    f = os.path.join('lightning_logs', f'{args.expname}_v{args.exp_version}', 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join('lightning_logs', f'{args.expname}_v{args.exp_version}', 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())


    # Create dataloader
    train_dataset = DDF_dataset(args, args.train_data_dir, args.N_views)
    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.N_batch, num_workers=args.num_workers)
    val_dataset = DDF_dataset(args, args.val_data_dir, args.N_val_views)
    val_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.N_batch, num_workers=args.num_workers)

    # For single instance.
    args.same_instances = False
    # if len(train_data_list) != 1 and train_data_list[0]==train_data_list[-1]:
    #     args.same_instances = True

    # Get ckpts path.
    ckpt_dir = os.path.join('lightning_logs', f'{args.expname}_v{args.exp_version}', 'checkpoints/*')
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