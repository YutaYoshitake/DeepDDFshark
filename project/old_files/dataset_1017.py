# python dataset.py --config=/home/yyoshitake/works/DeepSDF/project/configs/list_0/randn/covertdata/chair_128to128.txt

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
import torchvision.transforms as T
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from chamferdist import ChamferDistance
from scipy.spatial.transform import Rotation as R

from parser_get_arg import *
from often_use import *
from DDF.train_pl import DDF

torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)
# if device=='cuda':
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False





class TaR_dataset(data.Dataset):
    def __init__(
        self, 
        args, 
        mode, 
        instance_list_txt, 
        data_dir, 
        N_views, 
        ):

        self.mode = mode
        self.N_views = N_views
        self.instance_path_list = []
        with open(instance_list_txt, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                if line not in {'clopped'}:
                    if self.mode in {'train', 'covert'}:
                        for view_ind in range(self.N_views):
                            self.instance_path_list.append(
                                os.path.join(data_dir, line.rstrip('\n'), f'{str(view_ind+1).zfill(5)}.pickle')
                                )
                    elif self.mode=='val':
                        for view_ind in range(self.N_views):
                            self.instance_path_list.append(
                                os.path.join(data_dir, line.rstrip('\n'), f'{str(view_ind+1).zfill(5)}.pickle')
                                )
        self.rand_idx_list = 'non'
        self.pre_clopped = 'clopped' in data_dir.split('/')
        self.canonical_path = args.canonical_data_path
        ###############################################################
        self.rand_idx_list = 'hi'
        pickle_path = '000_dec_is_bad_dec_is_bad.pickle'
        targets = pickle_load(pickle_path)
        self.instance_path_list = targets['path']
        self.instance_path_list = [
            '/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/kmean_list0/kmean0_randn/resolution128/clopped/'+instance_path
             for instance_path in self.instance_path_list]
        self.rand_P_seed = targets['rand_P_seed']
        self.rand_S_seed = targets['rand_S_seed']
        self.randn_theta_seed = targets['randn_theta_seed']
        self.randn_axis_idx = targets['randn_axis_idx']
        ###############################################################
        # self.instance_path_list = self.instance_path_list[:5]

    def __getitem__(self, index):
        path = self.instance_path_list[index]
        instance_id = self.instance_path_list[index].split('/')[-2] # [-1]
        data_dict = pickle_load(path)

        if self.mode=='val':
            log_path = '/'.join(path.split('/')[-2:])
            canonical_path = os.path.join(self.canonical_path, instance_id + '.pickle')
            canonical_data_dict = pickle_load(canonical_path)
            canonical_distance_map = canonical_data_dict['depth_map'].astype(np.float32)
            canonical_camera_pos = canonical_data_dict['camera_pos'].astype(np.float32)
            canonical_camera_rot = canonical_data_dict['camera_rot'].astype(np.float32)
            if self.rand_idx_list != 'non':
                rand_P_seed = self.rand_P_seed[index]
                rand_S_seed = self.rand_S_seed[index]
                randn_theta_seed = self.randn_theta_seed[index]
                randn_axis_idx = self.randn_axis_idx[index]
            else:
                rand_P_seed = rand_S_seed = randn_theta_seed = randn_axis_idx = 'non'
        else:
            log_path = canonical_distance_map = canonical_camera_pos = canonical_camera_rot = False
            rand_P_seed = rand_S_seed = randn_theta_seed = randn_axis_idx = 'non'


        if self.pre_clopped:
            clopped_mask = data_dict['clopped_obs']['mask']
            
            clopped_distance_map = data_dict['clopped_obs']['distance']
            camera_pos = data_dict['clopped_cam']['pos']
            w2c = data_dict['clopped_cam']['w2c']
            rays_d_cam = data_dict['clopped_cam']['rays_d_cam']
            obj_pos = data_dict['obj']['pos']
            gt_green_cam = data_dict['obj']['green_cam']
            gt_red_cam = data_dict['obj']['red_cam']
            gt_green_wrd = data_dict['obj']['green_wrd']
            gt_red_wrd = data_dict['obj']['red_wrd']
            obj_scale = data_dict['obj']['scale']
            gt_shape_code = data_dict['obj']['shape']
            gt_o2w = data_dict['obj']['o2w']

            return clopped_mask, clopped_distance_map, camera_pos, w2c, rays_d_cam, \
            obj_pos, gt_green_cam, gt_red_cam, gt_green_wrd, gt_red_wrd, \
            obj_scale, gt_shape_code, gt_o2w, instance_id, \
            canonical_distance_map, canonical_camera_pos, canonical_camera_rot, \
            log_path, rand_P_seed, rand_S_seed, randn_theta_seed, randn_axis_idx
            
        else:
            frame_mask = data_dict['mask']
            frame_distance_map = data_dict['depth_map'].astype(np.float32)
            frame_camera_pos = data_dict['camera_pos'].astype(np.float32)
            frame_camera_rot = data_dict['camera_rot'].astype(np.float32)
            frame_obj_pos = data_dict['obj_pos'].astype(np.float32)
            frame_obj_rot = data_dict['obj_rot'].astype(np.float32)
            frame_obj_scale = np.squeeze(data_dict['obj_scale'].astype(np.float32))
            
            if self.mode=='train':
                return frame_mask, frame_distance_map, frame_camera_pos, frame_camera_rot, frame_obj_pos, frame_obj_rot, frame_obj_scale, instance_id
            elif self.mode=='val':
                return frame_mask, frame_distance_map, frame_camera_pos, frame_camera_rot, frame_obj_pos, frame_obj_rot, frame_obj_scale, \
                canonical_distance_map, canonical_camera_pos, canonical_camera_rot, instance_id, log_path, rand_P_seed, rand_S_seed, randn_theta_seed, randn_axis_idx
            elif self.mode=='covert':
                return frame_mask, frame_distance_map, frame_camera_pos, frame_camera_rot, frame_obj_pos, frame_obj_rot, frame_obj_scale, instance_id, path

    def __len__(self):
        return len(self.instance_path_list)





if __name__=='__main__':
    # Get args
    args = get_args()
    args.gpu_num = torch.cuda.device_count() # log used gpu num.

    target_data_dir = args.train_data_dir
    train_N_views = max(args.train_N_views, args.val_N_views, args.test_N_views)
    instance_list = [data_path.split('/')[-1] for data_path in glob.glob(target_data_dir + '/*')]
    instance_list_txt = f'tmp_convert_data.txt'
    with open(instance_list_txt, 'a') as f:
        for instance_id in instance_list:
            f.write(instance_id+'\n')
    output_dir = ('/').join(target_data_dir.split('/')[:-1] + ['clopped'])
    os.makedirs(output_dir, exist_ok=True)
    for instance_i in instance_list:
        os.makedirs(os.path.join(output_dir, instance_i), exist_ok=True)
    
    # Create dataloader
    train_dataset = TaR_dataset(args, 'covert', instance_list_txt, target_data_dir, train_N_views)
    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.N_batch, num_workers=16, drop_last=False, shuffle=False)
    os.remove(instance_list_txt)


    #############################################
    #############################################
    #############################################
    # Set ddf.
    ddf = DDF(args)
    ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args)
    ddf.eval()

    ddf_instance_list = []
    with open(args.ddf_instance_list_txt, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            ddf_instance_list.append(line.rstrip('\n'))
    
    fov = args.fov
    inp_H = args.input_H
    inp_W = args.input_W
    ddf_H = args.ddf_H_W_during_dfnet
    x_coord = torch.arange(0, inp_W)[None, :].expand(inp_H, -1)
    y_coord = torch.arange(0, inp_H)[:, None].expand(-1, inp_W)
    image_coord = torch.stack([y_coord, x_coord], dim=-1) # [H, W, (Y and X)]
    rays_d_cam_origin_imgsize = get_ray_direction(ddf_H, fov)


    #############################################
    #############################################
    #############################################
    for batch in tqdm(train_dataloader):
        frame_mask, frame_distance_map, frame_camera_pos, frame_camera_rot, frame_obj_pos, frame_obj_rot, frame_obj_scale, instance_id, path = batch
        batch_size = len(frame_mask)
        o2w = frame_obj_rot.reshape(-1, 3, 3)
        w2c = frame_camera_rot.reshape(-1, 3, 3)
        o2c = torch.bmm(w2c, o2w)
        gt_obj_axis_green_cam = o2c[:, :, 1] # Y
        gt_obj_axis_red_cam = o2c[:, :, 0] # X
        gt_obj_axis_green_wrd = torch.sum(gt_obj_axis_green_cam[..., None, :]*w2c.permute(0, 2, 1), -1) # Y_w
        gt_obj_axis_red_wrd = torch.sum(gt_obj_axis_red_cam[..., None, :]*w2c.permute(0, 2, 1), -1) # X_w
        
        frame_w2c = w2c.reshape(batch_size, -1, 3, 3)
        frame_gt_obj_axis_green_cam = gt_obj_axis_green_cam.reshape(batch_size, -1, 3)
        frame_gt_obj_axis_red_cam = gt_obj_axis_red_cam.reshape(batch_size, -1, 3)
        frame_gt_obj_axis_green_wrd = gt_obj_axis_green_wrd.reshape(batch_size, -1, 3)
        frame_gt_obj_axis_red_wrd = gt_obj_axis_red_wrd.reshape(batch_size, -1, 3)

        # Clop distance map.
        instance_idx = [ddf_instance_list.index(instance_id_i) for instance_id_i in instance_id]
        gt_shape_code = ddf.lat_vecs(torch.tensor(instance_idx, device=ddf.device)).detach()

        # Clop distance map.
        raw_invdistance_map = torch.zeros_like(frame_distance_map)
        raw_invdistance_map[frame_mask] = 1. / frame_distance_map[frame_mask]
        clopped_mask, clopped_distance_map, bbox_list = clopping_distance_map(
                                                            frame_mask.reshape(-1, inp_H, inp_W), 
                                                            frame_distance_map.reshape(-1, inp_H, inp_W), 
                                                            image_coord, 
                                                            inp_H, 
                                                            inp_W, 
                                                            ddf_H)
        ###########################################################################
        # origin_map = frame_distance_map.reshape(-1, inp_H, inp_W)#[:, ::2, ::2]
        # origin_mask = frame_mask.reshape(-1, inp_H, inp_W)#[:, ::2, ::2]
        # map_list = [clopped_mask, clopped_distance_map, origin_map, origin_mask]
        # map_list = torch.cat([map_i for map_i in map_list], dim=-1)
        # check_map_torch(torch.cat([map_i for map_i in map_list], dim=-2), 'tes.png')
        ###########################################################################

        # Get normalized depth map.
        rays_d_cam = get_clopped_rays_d_cam(ddf_H, bbox_list, rays_d_cam_origin_imgsize).to(frame_camera_rot)
        clopped_depth_map, normalized_depth_map, avg_depth_map = get_normalized_depth_map(
                                                                    clopped_mask, 
                                                                    clopped_distance_map, 
                                                                    rays_d_cam)
        bbox_info = torch.cat([bbox_list.reshape(-1, 4), 
                                bbox_list.mean(1), 
                                avg_depth_map.to('cpu')[:, None]], dim=-1)
        
        ###########################################################################
        # est_clopped_mask, est_clopped_distance_map = render_distance_map_from_axis(
        #                                                 H             = ddf_H, 
        #                                                 obj_pos_wrd   = frame_obj_pos.reshape(-1, 3),
        #                                                 axis_green    = frame_gt_obj_axis_green_cam.reshape(-1, 3),
        #                                                 axis_red      = frame_gt_obj_axis_red_cam.reshape(-1, 3), 
        #                                                 obj_scale     = frame_obj_scale.reshape(-1), 
        #                                                 input_lat_vec = gt_shape_code[:, None, :].expand(-1, 5, -1).reshape(-1, 256), 
        #                                                 cam_pos_wrd   = frame_camera_pos.reshape(-1, 3), 
        #                                                 rays_d_cam    = rays_d_cam, 
        #                                                 w2c           = w2c, 
        #                                                 ddf           = ddf, 
        #                                                 with_invdistance_map = False)
        ###########################################################################
        # gt = clopped_distance_map
        # est = est_clopped_distance_map
        # check_map = torch.cat([gt, est, torch.abs(gt-est)], dim=-1)
        # check_map = torch.cat([map_i for map_i in check_map], dim=0)
        # check_map_torch(check_map, f'tes_ddf.png')
        # import pdb; pdb.set_trace()
        ###########################################################################


        # Reshaping maps.
        frame_clopped_mask = clopped_mask.reshape(batch_size, -1, ddf_H, ddf_H)
        frame_clopped_distance_map = clopped_distance_map.reshape(batch_size, -1, ddf_H, ddf_H)
        frame_rays_d_cam = rays_d_cam.reshape(batch_size, -1, ddf_H, ddf_H, 3)
        frame_bbox_info = bbox_info.reshape(batch_size, -1, 7).to(frame_camera_rot)
        # frame_clopped_depth_map = clopped_depth_map.reshape(batch_size, -1, self.ddf_H, self.ddf_H)
        # frame_normalized_depth_map = normalized_depth_map.reshape(batch_size, -1, self.ddf_H, self.ddf_H)
        # frame_avg_depth_map = avg_depth_map.reshape(batch_size, -1)
        # frame_bbox_list = bbox_list.reshape(batch_size, -1, 2, 2).to(frame_camera_rot)
        # cim2im_scale, im2cam_scale, bbox_center = get_clopping_infos(bbox_list, avg_depth_map, fov)
        # frame_cim2im_scale = cim2im_scale.reshape(batch_size, -1)
        # frame_im2cam_scale = im2cam_scale.reshape(batch_size, -1)
        # frame_bbox_center = bbox_center.reshape(batch_size, -1, 2)

        # Reshaping ground truth.
        frame_w2c = w2c.reshape(batch_size, -1, 3, 3)
        frame_gt_obj_axis_green_cam = gt_obj_axis_green_cam.reshape(batch_size, -1, 3)
        frame_gt_obj_axis_red_cam = gt_obj_axis_red_cam.reshape(batch_size, -1, 3)
        frame_gt_obj_axis_green_wrd = gt_obj_axis_green_wrd.reshape(batch_size, -1, 3)
        frame_gt_obj_axis_red_wrd = gt_obj_axis_red_wrd.reshape(batch_size, -1, 3)


        #############################################
        #############################################
        #############################################
        for batch_id in range(batch_size):
            clopped_data_dict = {'clopped_obs':{}, 'clopped_cam':{}, 'obj':{}}
            clopped_data_dict['clopped_obs']['mask'] = frame_clopped_mask[batch_id]
            clopped_data_dict['clopped_obs']['distance'] = frame_clopped_distance_map[batch_id]
            clopped_data_dict['clopped_cam']['pos'] = frame_camera_pos[batch_id]
            clopped_data_dict['clopped_cam']['w2c'] = frame_w2c[batch_id]
            clopped_data_dict['clopped_cam']['rays_d_cam'] = frame_rays_d_cam[batch_id]
            clopped_data_dict['obj']['pos'] = frame_obj_pos[batch_id]
            clopped_data_dict['obj']['green_cam'] = frame_gt_obj_axis_green_cam[batch_id]
            clopped_data_dict['obj']['red_cam'] = frame_gt_obj_axis_red_cam[batch_id]
            clopped_data_dict['obj']['green_wrd'] = frame_gt_obj_axis_green_wrd[batch_id]
            clopped_data_dict['obj']['red_wrd'] = frame_gt_obj_axis_red_wrd[batch_id]
            clopped_data_dict['obj']['scale'] = frame_obj_scale[batch_id]
            clopped_data_dict['obj']['shape'] = gt_shape_code[batch_id]
            clopped_data_dict['obj']['o2w'] = frame_obj_rot[batch_id]
            splitted_path = path[batch_id].split('/')
            clopped_path = '/'.join(splitted_path[:-3] + ['clopped'] + splitted_path[-2:])
            # if(os.path.isfile(clopped_path)):
            #     os.remove(clopped_path)
            pickle_dump(clopped_data_dict, clopped_path)
