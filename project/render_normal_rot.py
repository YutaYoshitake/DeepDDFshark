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
from train_pl import *

torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
if device=='cuda':
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





if __name__=='__main__':
    # Get args
    args = get_args()
    args.gpu_num = torch.cuda.device_count() # log used gpu num.
    args.same_instances = False

    # Get ckpts path.
    ckpt_dir = os.path.join('lightning_logs', f'{args.expname}_{args.exp_version}', 'checkpoints/*')
    ckpt_path_list = sorted(glob.glob(ckpt_dir))
    latest_ckpt_path = ckpt_path_list[-1]

    # Make model and load ckpt.
    model = DDF(args)
    model = model.load_from_checkpoint(checkpoint_path=latest_ckpt_path, args=args)

    # model to gpu.
    device=torch.device('cuda')
    model.to(device)

    # Get pos and c2w list for rot views.
    freq = 20
    lat_deg = 30
    pos_list, w2c_list = get_rot_views(lat_deg, freq, model)

    # Get instance id.
    instance_id = torch.zeros(1, dtype=torch.long).to(model.device)
    # instance_id = torch.full_like(instance_id, 5)



    # Create dir
    basedir = 'rendered_results/'
    expname = os.path.join(re.split('[./]', latest_ckpt_path)[2], re.split('[./]', latest_ckpt_path)[-2])
    instancename = (f'instance{instance_id.item()}')
    dir_path = os.path.join(basedir, expname, instancename)
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(os.path.join(dir_path, 'image'), exist_ok=True)
    os.makedirs(os.path.join(dir_path, 'depth_np'), exist_ok=True)



    # Render depth map.
    for i in tqdm(range(freq)):
        pos = pos_list[i].unsqueeze(0) # with dummy_batch
        c2w = w2c_list[i].T.unsqueeze(0) # with dummy_batch


        # get inputs
        rays_d_cam = get_ray_direction(args.H, args.fov, False) # batch, H, W, 3:xyz
        rays_d_wrd = get_ray_direction(args.H, args.fov, c2w).to(c2w.dtype).to(c2w.device) # batch, H, W, 3:xyz
        rays_o = pos[:, None, None, :].expand(-1, args.H, args.W, -1)
        input_lat_vec = model.lat_vecs(instance_id)

        # Get inverced depth map
        blur_mask = torch.full_like(rays_o[..., 0], True, dtype=torch.bool)
        est_inverced_depth = model(rays_o, rays_d_wrd, input_lat_vec, blur_mask=blur_mask)

        # Start
        hit_obj_mask = torch.full_like(blur_mask, False)
        hit_obj_mask[blur_mask] = est_inverced_depth > .5

        # get normal
        est_normal = model.get_normals(rays_o, rays_d_wrd, input_lat_vec, c2w, hit_obj_mask, est_inverced_depth[hit_obj_mask[blur_mask]])
        
        # get normal map
        est_normal_image = torch.zeros_like(rays_o)
        est_normal_image[hit_obj_mask] = (est_normal + 1) / 2

        # Make fig
        png_path = os.path.join(os.path.join(dir_path, 'image', str(i).zfill(5)+'.png'))
        check_map(est_normal_image[0], png_path)

        # # Save as np.
        # est_depth_map = est_depth_map.to('cpu').detach().numpy().copy()
        # np_path = os.path.join(dir_path, 'depth_np', str(i).zfill(5)+'.pickle')
        # pickle_dump(est_depth_map, np_path)
    
    # Print command for rot video.
    print('Command to make a video')
    print(f'$ ffmpeg -r 30 -i {dir_path}/image/%05d.png -vcodec libx264 -pix_fmt yuv420p -r 60 {dir_path}/normal.mp4')
