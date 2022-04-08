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





if __name__=='__main__':
    # Get args
    args = get_args()
    args.gpu_num = torch.cuda.device_count() # log used gpu num.
    args.same_instances = False

    # Get ckpts path.
    ckpt_dir = os.path.join('lightning_logs', f'{args.expname}_v{args.exp_version}', 'checkpoints/*')
    ckpt_path_list = sorted(glob.glob(ckpt_dir))
    latest_ckpt_path = ckpt_path_list[-1]

    # Make model and load ckpt.
    model = DDF(args)
    model = model.load_from_checkpoint(checkpoint_path=latest_ckpt_path, args=args)

    # model to gpu.
    device=torch.device('cuda')
    model.to(device)

    # Get pos and c2w list for rot views.
    freq = 200
    lat_deg = 30
    pos_list, w2c_list = get_rot_views(lat_deg, freq, model)

    # Get instance id.
    instance_id = torch.zeros(1, dtype=torch.long).to(model.device)
    instance_id = torch.full_like(instance_id, 5)



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

        # Get inverced depth map
        est_depth_map = model.render_depth_map(pos, c2w, instance_id, H=256, inverced_depth_map=True)
        
        # Make fig
        fig = pylab.figure(figsize=(5, 5))

        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('result')
        ax1.imshow(est_depth_map.to('cpu').detach().numpy().copy())

        ax1.xaxis.set_ticklabels([])
        ax1.yaxis.set_ticklabels([])
        png_path = os.path.join(os.path.join(dir_path, 'image', str(i).zfill(5)+'.png'))
        fig.savefig(png_path, dpi=200)
        pylab.close()

        # Save as np.
        est_depth_map = est_depth_map.to('cpu').detach().numpy().copy()
        np_path = os.path.join(dir_path, 'depth_np', str(i).zfill(5)+'.pickle')
        pickle_dump(est_depth_map, np_path)
    