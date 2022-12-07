import os
import pdb
import sys
import numpy as np
import random
import pylab
import glob
import math
import re
import pickle
import sys
import numpy as np
import torch
import torch.nn.functional as F

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





def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)





def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data





def polar2xyz(theta, phi, r=1):
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return x, y, z





def xyz2polar(x, y, z):
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.arccos(z/r)
    phi = torch.sgn(y) * torch.arccos(x/torch.sqrt(x**2+y**2))
    return r, theta, phi





def check_map(image, path = 'tes.png'):
    fig = pylab.figure(figsize=(5, 5))

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title('result')
    ax1.imshow(image.to('cpu').detach().numpy().copy())

    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])
    fig.savefig(path, dpi=300)
    pylab.close()





def vec2skew(v):
    """
    :param v:  (3, ) torch tensor
    :return:   (3, 3)
    """
    zero = torch.zeros(1, dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([ zero,    -v[2:3],   v[1:2]])  # (3, 1)
    skew_v1 = torch.cat([ v[2:3],   zero,    -v[0:1]])
    skew_v2 = torch.cat([-v[1:2],   v[0:1],   zero])
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
    return skew_v  # (3, 3)





def Exp(r):
    """so(3) vector to SO(3) matrix
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3)
    """
    skew_r = vec2skew(r)  # (3, 3)
    norm_r = r.norm() + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device)
    R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
    return R





def get_ray_direction(size, fov, c2w=False):
    fov = torch.deg2rad(torch.tensor(fov, dtype=torch.float))
    x_coord = torch.linspace(-torch.tan(fov*.5), torch.tan(fov*.5), size)[None].expand(size, -1)
    y_coord = x_coord.T
    rays_d_cam = torch.stack([x_coord, y_coord, torch.ones_like(x_coord)], dim=2)
    rays_d_cam = F.normalize(rays_d_cam, dim=-1)
    if c2w is False:
        return rays_d_cam.unsqueeze(0).detach() # H, W, 3:xyz
    else:
        rays_d_cam = rays_d_cam.unsqueeze(0).to(c2w.device).detach()
        rays_d_wrd = torch.sum(rays_d_cam[:, :, :, None, :] * c2w[:, None, None, :, :], -1)
        return rays_d_wrd # batch, H, W, 3:xyz





def path2posc2w(path, model):
    if not model.model_params_dtype:
        model.check_model_info()

    camera_info = pickle_load(path + '_pose.pickle')
    pos = camera_info['pos']
    c2w = camera_info['rot'].T
    pos = torch.from_numpy(pos).clone().to(model.model_params_dtype).unsqueeze(0).to(model.device) # batch, 3:xyz
    c2w = torch.from_numpy(c2w).clone().to(model.model_params_dtype).unsqueeze(0).to(model.device) # batch, 3*3:SO3

    return pos, c2w # returen as torchtensor





def path2depthinfo(path, model, with_depth = False, with_normal = False, H = 'unspecified'):
    if not model.model_params_dtype:
        model.check_model_info()

    depth_info = pickle_load(path+'_mask.pickle')
    inverced_depth_map = depth_info['inverced_depth']
    blur_mask = depth_info['blur_mask']
    inverced_depth_map = torch.from_numpy(inverced_depth_map).to(model.model_params_dtype).unsqueeze(0).to(model.device) # batch, H, W
    blur_mask = torch.from_numpy(blur_mask).unsqueeze(0).to(model.device) # batch, H, W

    if not H == 'unspecified':
        interval = inverced_depth_map.shape[1] // H
        inverced_depth_map = inverced_depth_map[:, ::interval, ::interval]
        blur_mask = blur_mask[:, ::interval, ::interval]
    else:
        interval = 1
    
    if with_depth:
        hit_obj_mask = inverced_depth_map > 0.
        depth_map = torch.zeros_like(inverced_depth_map)
        depth_map[hit_obj_mask] = 1. / inverced_depth_map[hit_obj_mask]

    if with_normal and 'normal_map' in depth_info.keys():
        normal_map = depth_info['normal_map'][::interval, ::interval]
        normal_mask = depth_info['normal_mask'][::interval, ::interval]
        normal_map = torch.from_numpy(normal_map).to(model.model_params_dtype).unsqueeze(0).to(model.device) # batch, H, W, 3
        normal_mask = torch.from_numpy(normal_mask).unsqueeze(0).to(model.device) # batch, H, W
        if with_depth:
            return inverced_depth_map, blur_mask, normal_map, normal_mask, depth_map, hit_obj_mask
        else:
            return inverced_depth_map, blur_mask, normal_map, normal_mask

    elif with_normal:
        print('Normal info does not exist in the file.')
        sys.exit()

    else:
        if with_depth:
            return inverced_depth_map, blur_mask, depth_map, hit_obj_mask
        else:
            return inverced_depth_map, blur_mask





def get_rot_views(lat_deg, freq, model):
    if not model.model_params_dtype:
        model.check_model_info()

    lat = np.deg2rad(lat_deg)
    y = np.sin(lat)
    xz_R = np.cos(lat)
    rad = np.linspace(0, 2*np.pi, freq+1)[:freq]

    pos_list = np.stack([xz_R*np.cos(rad), np.full_like(rad, y), xz_R*np.sin(rad)], axis=-1)
    up_list = np.tile(np.array([[0, 1, 0]]), (freq, 1))
    
    at_list = 0 - pos_list
    at_list = at_list/np.linalg.norm(at_list, axis=-1)[..., None]
    
    right_list = np.cross(at_list, up_list)
    right_list = right_list/np.linalg.norm(right_list, axis=-1)[..., None]
    
    up_list = np.cross(right_list, at_list)
    up_list = up_list/np.linalg.norm(up_list, axis=-1)[..., None]
    
    w2c_list = np.stack([right_list, -up_list, at_list], axis=-2)

    # Adjust list's type to the model.
    pos_list = torch.from_numpy(pos_list).clone().to(model.model_params_dtype).to(model.device)
    w2c_list = torch.from_numpy(w2c_list).clone().to(model.model_params_dtype).to(model.device)

    return pos_list, w2c_list
            




def txt2list(txt_file):
    result_list = []
    with open(txt_file, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            result_list.append(line.rstrip('\n'))
    return result_list





def list2txt(result_list, txt_file):
    with open(txt_file, 'a') as f:
        for result in result_list:
            f.write(result + '\n')