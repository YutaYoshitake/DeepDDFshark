import os
import sys
from contextlib import redirect_stdout
import random
import numpy as np
import pylab
import glob
import math
import pickle
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from scipy.spatial import cKDTree as KDTree
from scipy.linalg import logm
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False





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



def check_map_torch(image, path = 'tes.png', figsize=[10,10]):
    fig = pylab.figure(figsize=figsize)

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title('result')
    ax1.imshow(image.to('cpu').detach().numpy().copy())

    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])
    fig.savefig(path, dpi=300)
    pylab.close()



def check_map_np(image, path = 'tes.png', figsize=[10,10]):
    fig = pylab.figure(figsize=figsize)

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title('result')
    ax1.imshow(image)

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



def batch_linspace(start, end, step):
    raw = torch.linspace(0, 1, step)[None, :].to(start)
    return (end.to(start) - start)[:, None] * raw + start[:, None]



def axis2rotation(axis_green, axis_red, axis_blue=False):
    axis_blue = torch.cross(axis_red, axis_green, dim=-1)
    axis_blue = F.normalize(axis_blue, dim=-1)
    orthogonal_axis_red = torch.cross(axis_green, axis_blue, dim=-1)
    return torch.stack([orthogonal_axis_red, axis_green, axis_blue], dim=-1)



def render_distance_map_from_axis(
        H, 
        axis_green, 
        axis_red,
        obj_scale, 
        rays_d_cam, 
        input_lat_vec, 
        ddf, 
        cam_pos_wrd = 'non', 
        obj_pos_wrd = 'non', 
        w2c = 'non', 
        obj_pos_cam = 'not_given', 
        with_invdistance_map = False, 
    ):
    # Get rotation matrix.
    axis_blue = torch.cross(axis_red, axis_green, dim=-1)
    axis_blue = F.normalize(axis_blue, dim=-1)
    orthogonal_axis_red = torch.cross(axis_green, axis_blue, dim=-1)
    o2c = torch.stack([orthogonal_axis_red, axis_green, axis_blue], dim=-1)

    # Get rays direction.
    rays_d_obj = torch.sum(rays_d_cam[..., None, :]*o2c[..., None, None, :, :].permute(0, 1, 2, 4, 3), -1)

    # Get rays origin.
    if obj_pos_cam == 'not_given':
        obj_pos_cam = torch.sum((obj_pos_wrd - cam_pos_wrd)[..., None, :]*w2c, dim=-1)
    cam_pos_obj = - torch.sum(obj_pos_cam[..., None, :]*o2c.permute(0, 2, 1), dim=-1) / obj_scale[:, None]
    rays_o_obj = cam_pos_obj[:, None, None, :].expand(-1, H, H, -1)

    # Get rays inputs.
    rays_d = rays_d_obj
    rays_o = rays_o_obj

    # Estimating.
    est_invdistance_map_obj_scale, negative_D_mask = ddf.forward_from_far(rays_o, rays_d, input_lat_vec)
    est_invdistance_map = est_invdistance_map_obj_scale / obj_scale[:, None, None]

    # Get distance map.
    mask_under_border = 1 / (cam_pos_obj.norm(dim=-1) + .5 * obj_scale * ddf.radius) # 良いのか...？
    est_mask = [map_i > border_i for map_i, border_i in zip(est_invdistance_map, mask_under_border)]
    est_mask = torch.stack(est_mask, dim=0)
    est_distance_map = torch.zeros_like(est_invdistance_map)
    est_distance_map[est_mask] = 1. / est_invdistance_map[est_mask]

    if with_invdistance_map:
        return est_invdistance_map, est_mask, est_distance_map
    else:
        return est_mask, est_distance_map



def clopping_distance_map_from_bbox(ddf_H, bbox_list, rays_d_cam, mask, distance_map):
    # Reshape.
    batch, seq, _, _ = bbox_list.shape
    bbox_list = bbox_list.reshape(-1, 2, 2)
    mask = mask.reshape(-1, ddf_H, ddf_H)
    distance_map = distance_map.reshape(-1, ddf_H, ddf_H)

    mask = mask[:, None] # (N, dummy_C, H, W)
    distance_map = distance_map[:, None] # (N, dummy_C, H, W)
    coord_x = batch_linspace(bbox_list[:, 1, 0], bbox_list[:, 0, 0], ddf_H)
    coord_x = coord_x[:, None, :].expand(-1, ddf_H, -1)
    coord_y = batch_linspace(bbox_list[:, 1, 1], bbox_list[:, 0, 1], ddf_H)
    coord_y = coord_y[:, :, None].expand(-1, -1, ddf_H)
    sampling_coord = torch.stack([coord_x, coord_y], dim=-1).to(distance_map.device)
    cloped_mask = F.grid_sample(mask.to(sampling_coord.dtype), sampling_coord, align_corners=True)[:, 0] > 0.99 # (N, H, W)
    cloped_distance_map = F.grid_sample(
                        distance_map.to(sampling_coord.dtype), 
                        sampling_coord, 
                        align_corners=True)[:, 0] # (N, H, W)
    rays_d_cam = rays_d_cam.expand(bbox_list.shape[0], -1, -1, -1).permute(0, 3, 1, 2)
    rays_d_cam = F.grid_sample(
                        rays_d_cam.to(sampling_coord.dtype), 
                        sampling_coord, align_corners=True).permute(0, 2, 3, 1) # (N, H, W)
    rays_d_cam = F.normalize(rays_d_cam, dim=-1)
    
    # Reshape.
    cloped_mask = cloped_mask.reshape(batch, seq, ddf_H, ddf_H)
    cloped_distance_map = cloped_distance_map.reshape(batch, seq, ddf_H, ddf_H)
    cloped_distance_map[torch.logical_not(cloped_mask)] = 0.
    rays_d_cam = rays_d_cam.reshape(batch, seq, ddf_H, ddf_H, 3)
    return cloped_mask, cloped_distance_map, rays_d_cam



def get_canonical_map(
        H, 
        cam_pos_wrd, 
        rays_d_cam, 
        w2c, 
        input_lat_vec, 
        ddf, 
        with_invdepth_map = False, 
    ):
    # Get rays inputs.
    rays_d = torch.sum(rays_d_cam[..., None, :]*w2c[..., None, None, :, :].permute(0, 1, 2, 4, 3), -1)
    rays_o = cam_pos_wrd[:, None, None, :].expand(-1, H, H, -1)

    # Estimating.
    est_invdepth_map = ddf.forward(rays_o, rays_d, input_lat_vec)
    est_mask = est_invdepth_map > .5
    est_depth_map = torch.zeros_like(est_invdepth_map)
    est_depth_map[est_mask] = 1. / est_invdepth_map[est_mask]

    if with_invdepth_map:
        return est_invdepth_map, est_mask, est_depth_map
    else:
        return est_mask, est_depth_map



def sample_fibonacci_views(n):
    i = arange(0, n, dtype=float)
    phi = arccos(1 - 2*i/n)
    goldenRatio = (1 + 5**0.5)/2
    theta = 2 * pi * i / goldenRatio
    X, Y, Z = cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)
    xyz = np.stack([X, Y, Z], axis=1)
    return xyz



# Sampling view points on a sphere uniformaly by the fibonacci sampling.
from numpy import arange, pi, cos, sin, tan, dot, arccos
def sample_fibonacci_views(n):
    i = arange(0, n, dtype=float)
    phi = arccos(1 - 2*i/n)
    goldenRatio = (1 + 5**0.5)/2
    theta = 2 * pi * i / goldenRatio
    X, Y, Z = cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)
    xyz = np.stack([X, Y, Z], axis=1)
    return xyz



def get_OSMap_obj(distance_map, mask, rays_d_cam, w2c, cam_pos_wrd, o2w, obj_pos_wrd, obj_scale, data_type='shapenet', add_conf='nothing'):
    OSMap_cam = distance_map[..., None] * rays_d_cam
    OSMap_wrd = torch.sum(OSMap_cam[..., None, :]*w2c.permute(0, 2, 1)[..., None, None, :, :], dim=-1) + cam_pos_wrd[..., None, None, :]
    OSMap_obj = torch.sum((OSMap_wrd - obj_pos_wrd[..., None, None, :])[..., None, :]*o2w.permute(0, 2, 1)[..., None, None, :, :], dim=-1) / obj_scale[..., None, None, :]
    OSMap_obj[torch.logical_not(mask)] = 0.
    if data_type == 'scan2cad':
        dist_mask = distance_map < 0.1
        OSMap_obj[dist_mask] = 0.
    if add_conf == 'only_once':
        OSMap_wrd_wMask = torch.cat([OSMap_wrd, mask.to(OSMap_obj.dtype).unsqueeze(-1)], dim=-1)
        return OSMap_wrd_wMask
    else:
        OSMap_obj_wMask = torch.cat([OSMap_obj, mask.to(OSMap_obj.dtype).unsqueeze(-1)], dim=-1)
        return OSMap_obj_wMask



def get_diffOSMap_obj(obs_distance_map, est_distance_map, obs_mask, est_mask, rays_d_cam, w2c, o2w, obj_scale, data_type='shapenet', add_conf='nothing'):
    diff_or_mask = torch.logical_or(obs_mask, est_mask)
    diff_xor_mask = torch.logical_xor(obs_mask, est_mask)

    diff_distance_map = obs_distance_map - est_distance_map
    diffOSMap_cam = diff_distance_map[..., None] * rays_d_cam
    diffOSMap_wrd = torch.sum(diffOSMap_cam[..., None, :]*w2c.permute(0, 2, 1)[..., None, None, :, :], dim=-1)
    diffOSMap_obj = torch.sum(diffOSMap_wrd[..., None, :]*o2w.permute(0, 2, 1)[..., None, None, :, :], dim=-1)
    diffOSMap_obj = diffOSMap_obj / obj_scale[..., None, None, :]

    diffOSMap_obj[torch.logical_not(diff_or_mask)] = 0.
    if data_type == 'scan2cad':
        dist_mask = obs_distance_map <= 0
        diffOSMap_obj[dist_mask] = 0.
    if add_conf == 'only_once':
        diffOSMap_wrd_wMask = torch.cat([diffOSMap_wrd, diff_xor_mask.to(diffOSMap_obj).unsqueeze(-1)], dim=-1)
        return diffOSMap_wrd_wMask
    else:
        diffOSMap_obj_wMask = torch.cat([diffOSMap_obj, diff_xor_mask.to(diffOSMap_obj).unsqueeze(-1)], dim=-1)
        return diffOSMap_obj_wMask



# def get_OSMap_wrd(distance_map, mask, rays_d_cam, w2c, cam_pos_wrd):
#     OSMap_cam = distance_map[..., None] * rays_d_cam
#     OSMap_wrd = torch.sum(OSMap_cam[..., None, :]*w2c.permute(0, 2, 1)[..., None, None, :, :], dim=-1)
#     OSMap_wrd = OSMap_wrd + cam_pos_wrd[..., None, None, :]

#     OSMap_wrd[torch.logical_not(mask)] = 0.
#     OSMap_wrd_wMask = torch.cat([OSMap_wrd, mask.to(OSMap_wrd).unsqueeze(-1)], dim=-1)
#     return OSMap_wrd_wMask



# def get_diffOSMap_wrd(obs_distance_map, est_distance_map, obs_mask, est_mask, rays_d_cam, w2c):
#     diff_or_mask = torch.logical_or(obs_mask, est_mask)
#     diff_xor_mask = torch.logical_xor(obs_mask, est_mask)

#     diff_distance_map = est_distance_map - obs_distance_map
#     diffOSMap_cam = diff_distance_map[..., None] * rays_d_cam
#     diffOSMap_wrd = torch.sum(diffOSMap_cam[..., None, :]*w2c.permute(0, 2, 1)[..., None, None, :, :], dim=-1)

#     diffOSMap_wrd[torch.logical_not(diff_or_mask)] = 0.
#     diffOSMap_wrd_wMask = torch.cat([diffOSMap_wrd, diff_xor_mask.to(diffOSMap_wrd).unsqueeze(-1)], dim=-1)
#     return diffOSMap_wrd_wMask



# def compute_trimesh_chamfer(gt_points, gen_mesh, offset, scale, num_mesh_samples=30000):
def compute_trimesh_chamfer(gt_points_np, gen_points_sampled):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.
    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)
    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)
    """

    # gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]
    # gen_points_sampled = gen_points_sampled / scale - offset
    # only need numpy array of points
    # gt_points_np = gt_points.vertices
    # gt_points_np = gt_points.vertices

    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer



def render_normal_map_from_afar(
        H, 
        o2c, 
        cam_pos_obj, 
        obj_scale, 
        rays_d_cam, 
        input_lat_vec, 
        ddf, ):
    # Get rays direction.
    rays_d_obj = torch.sum(rays_d_cam[..., None, :]*o2c[..., None, None, :, :].permute(0, 1, 2, 4, 3), -1)

    # Get rays origin.
    rays_o_obj = cam_pos_obj[:, None, None, :].expand(-1, H, H, -1)

    # Get rays inputs.
    rays_d = F.normalize(rays_d_obj, dim=-1)
    rays_o = rays_o_obj

    # Estimating.
    est_invdistance_map_obj_scale, negative_D_mask = ddf.forward_from_far(rays_o, rays_d, input_lat_vec)
    est_invdistance_map = est_invdistance_map_obj_scale / obj_scale[:, None, None]

    # Get distance map.
    mask_under_border = 1 / (cam_pos_obj.norm(dim=-1) + .5 * obj_scale * ddf.radius) # 良いのか...？
    est_mask = [map_i > border_i for map_i, border_i in zip(est_invdistance_map, mask_under_border)]
    est_mask = torch.stack(est_mask, dim=0)
    est_distance_map = torch.zeros_like(est_invdistance_map)
    est_distance_map[est_mask] = 1. / est_invdistance_map[est_mask]
    
    # Get dif dir.
    diff_rad = 1e-5 * torch.pi 
    rot_r = Exp(torch.tensor([0., diff_rad, 0.])).to(rays_d_cam)
    rot_u = Exp(torch.tensor([diff_rad ,0., 0.])).to(rays_d_cam)
    
    # Get dif ray.
    rays_d_cam_r = torch.sum(rays_d_cam[:, :, :, None, :] * rot_r[..., None, None, :, :], -1)
    rays_d_obj_r = torch.sum(rays_d_cam_r[:, :, :, None, :] * o2c.permute(0, 2, 1)[:, None, None, :, :], -1)
    rays_d_obj_r = F.normalize(rays_d_obj_r, dim=-1)
    rays_d_cam_u = torch.sum(rays_d_cam[:, :, :, None, :] * rot_u[..., None, None, :, :], -1)
    rays_d_obj_u = torch.sum(rays_d_cam_u[:, :, :, None, :] * o2c.permute(0, 2, 1)[:, None, None, :, :], -1)
    rays_d_obj_u = F.normalize(rays_d_obj_u, dim=-1)
    est_invdistance_r, _ = ddf.forward_from_far(rays_o, rays_d_obj_r, input_lat_vec)
    est_invdistance_u, _ = ddf.forward_from_far(rays_o, rays_d_obj_u, input_lat_vec)
    est_invdistance_r = est_invdistance_r / obj_scale[:, None, None]
    est_invdistance_u = est_invdistance_u / obj_scale[:, None, None]
    
    # Get dif mask.
    est_mask_r = [map_i > border_i for map_i, border_i in zip(est_invdistance_r, mask_under_border)]
    est_mask_u = [map_i > border_i for map_i, border_i in zip(est_invdistance_u, mask_under_border)]
    est_mask_r = torch.stack(est_mask_r, dim=0)
    est_mask_u = torch.stack(est_mask_u, dim=0)
    est_mask *= est_mask_r * est_mask_u

    # Get blur mask.
    fil = torchvision.transforms.GaussianBlur(3, (0.1, 2.0))
    blured_mask = (fil(torch.logical_not(est_mask)[:, None, :, :].expand(-1, 3, -1, -1))<0.5)[:, 0]
    # check_map_torch(torch.logical_xor(est_mask, blured_mask)[0], 'tes_mask.png')
    est_mask = blured_mask

    # Get dif distance.
    est_distance_map = torch.zeros_like(est_invdistance_map)
    est_distance_r = torch.zeros_like(est_invdistance_r)
    est_distance_u = torch.zeros_like(est_invdistance_u)
    est_distance_map[est_mask] = 1. / est_invdistance_map[est_mask]
    est_distance_r[est_mask] = 1 / est_invdistance_r[est_mask]
    est_distance_u[est_mask] = 1 / est_invdistance_u[est_mask]
    # thr = 7e-4
    # dif_mask_cr = torch.abs(est_distance_map - est_distance_r)  < thr
    # dif_mask_cu = torch.abs(est_distance_map - est_distance_u)  < thr
    # dif_mask_ru = torch.abs(est_distance_r - est_distance_u) < thr
    # est_mask *= dif_mask_cr * dif_mask_cu * dif_mask_ru

    # Get 3D points.
    est_point = est_distance_map[..., None] * rays_d_cam
    est_point_r = est_distance_r[..., None] * rays_d_cam_r
    est_point_u = est_distance_u[..., None] * rays_d_cam_u

    # Get normals.
    diff_from_right = est_point - est_point_r
    diff_from_under = est_point - est_point_u
    est_normal = F.normalize(torch.cross(diff_from_right, diff_from_under, dim=-1), dim=-1)
    
    return est_normal, est_distance_map, est_mask



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



def get_pe_target(cam_pos_wrd, w2c, rays_d_cam, obj_pos_wrd, o2w, obj_scale_wrd, bbox_diagonal, H, add_conf):
    pe_target_cam_pos_obj = torch.sum((cam_pos_wrd - obj_pos_wrd)[..., None, :]*o2w.permute(0, 1, 3, 2), dim=-1) / obj_scale_wrd
    center_start = int(-(-H//2)-1)
    center_end = int(H//2+1)
    center_d_cam = rays_d_cam[:, :, center_start:center_end, center_start:center_end, :].mean(dim=(2, 3))
    center_d_wrd = torch.sum(center_d_cam[..., None, :]*w2c.permute(0, 1, 3, 2), dim=-1)
    pe_target_center_d_obj = torch.sum(center_d_wrd[..., None, :]*o2w.permute(0, 1, 3, 2), dim=-1)
    pe_target_bbox = bbox_diagonal
    pe_target_obj = torch.cat([pe_target_cam_pos_obj, pe_target_center_d_obj, pe_target_bbox], dim=-1)
    # return pe_target_obj
    pe_target_cam_pos_wrd = cam_pos_wrd
    pe_target_center_d_wrd = center_d_wrd
    pe_target_wrd = torch.cat([pe_target_cam_pos_wrd, pe_target_center_d_wrd, pe_target_bbox], dim=-1)
    if add_conf == 'only_once':
        return pe_target_wrd
    else:
        return pe_target_obj



def get_riemannian_distance(A, B):
    ATB = torch.bmm(A.permute(0, 2, 1), B).to('cpu').detach().numpy().copy()
    with redirect_stdout(open(os.devnull, 'w')):
        log_ATB = np.stack([logm(R_i) for R_i in ATB], axis=0)
        return torch.from_numpy((np.sqrt((log_ATB**2).sum(axis=(-1, -2))) / math.sqrt(2)).astype(np.float32)).clone()
    


def sample_pose(v, look_at, look_up_randm='normal'):
    pos = v
    look_at = look_at - v
    look_at = look_at / np.linalg.norm(look_at)

    if np.abs(np.dot(look_at, np.array([0, 1, 0]))) < 0.999:
        up = np.array([0, 1, 0])
    else:
        import pdb; pdb.set_trace()
        ang = np.radians(360*np.random.rand())
        up = np.array([np.cos(ang), 0, np.sin(ang)])

    right = np.cross(look_at, up)
    up = np.cross(right, look_at)

    right = right / np.linalg.norm(right)
    up = up / np.linalg.norm(up)

    rot = np.array([right, up, -look_at])

    if look_up_randm != 'normal':
        while True:
            R_up_pi = look_up_randm * np.pi
            R_up = np.array([[np.cos(R_up_pi), -np.sin(R_up_pi), 0.0],
                                [np.sin(R_up_pi), np.cos(R_up_pi), 0.0],
                                [0.0, 0.0, 0.0]])
            up = rot.T@R_up@rot@up

            right = np.cross(look_at, up)
            up = np.cross(right, look_at)

            right = right / np.linalg.norm(right)
            up = up / np.linalg.norm(up)

            if np.abs(np.dot(look_at, up)) > 0.999:
                look_up_randm += 0.1 * np.random.rand() - 0.05 # 一様分布
            else:
                break

        rot = np.array([right, up, -look_at])


    if np.any(rot != rot):
        print('rotation matrix includes nan!!!')
        print('rot = ')
        print(rot)
        print('norm of look_at = ' + str(np.linalg.norm(look_at)))
        print('norm of up      = ' + str(np.linalg.norm(up)))
        print('norm of right   = ' + str(np.linalg.norm(right)))        
        exit(1)

    # print('dot(right, up)      = ' + str(np.dot(right, up)))
    # print('dot(up, look_at)    = ' + str(np.dot(look_at, up)))
    # print('dot(look_at, right) = ' + str(np.dot(right, look_at)))
    # print('|cross(up, right) - look_at| = ' + str(np.linalg.norm(look_at - np.cross(up, right))))
    # print('|cross(look_at, up) - right| = ' + str(np.linalg.norm(right - np.cross(look_at, up))))
    # print('|cross(right, look_at) - up| = ' + str(np.linalg.norm(up - np.cross(right, look_at))))

    # convert to quaternion
    # http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    qw = np.sqrt(np.amax([0, 1 + rot[0,0] + rot[1,1] + rot[2,2]])) / 2
    qx = np.sqrt(np.amax([0, 1 + rot[0,0] - rot[1,1] - rot[2,2]])) / 2
    qy = np.sqrt(np.amax([0, 1 - rot[0,0] + rot[1,1] - rot[2,2]])) / 2
    qz = np.sqrt(np.amax([0, 1 - rot[0,0] - rot[1,1] + rot[2,2]])) / 2

    if qw == np.amax(np.array([qw, qx, qy, qz])):
        # print('|qw| = ' + str(qw))
        qx = (rot[1, 2] - rot[2, 1]) / (4 * qw)
        qy = (rot[2, 0] - rot[0, 2]) / (4 * qw)
        qz = (rot[0, 1] - rot[1, 0]) / (4 * qw)
    elif qx == np.amax(np.array([qx, qy, qz])):
        # print('|qx| = ' + str(qx))
        qw = (rot[1, 2] - rot[2, 1]) / (4 * qx)
        qy = (rot[0, 1] + rot[1, 0]) / (4 * qx)
        qz = (rot[2, 0] + rot[0, 2]) / (4 * qx)
    elif qy == np.amax(np.array([qy, qz])):
        # print('|qy| = ' + str(qy))
        qw = (rot[2, 0] - rot[0, 2]) / (4 * qy)
        qx = (rot[0, 1] + rot[1, 0]) / (4 * qy)
        qz = (rot[1, 2] + rot[2, 1]) / (4 * qy) 
    else:
        # print('|qz| = ' + str(qz))
        qw = (rot[0, 1] - rot[1, 0]) / (4 * qz)
        qx = (rot[2, 0] + rot[0, 2]) / (4 * qz)
        qy = (rot[1, 2] + rot[2, 1]) / (4 * qz)        
    
    if np.any(np.array([qw, qx, qy, qz]) != np.array([qw, qx, qy, qz])):
        print('nan in quarternion!!!')
        print('q = ')
        print(np.array([qw, qx, qy, qz]))
        print('rotation matrix = ')
        print(rot)
        print('norm of look_at = ' + str(np.linalg.norm(look_at)))
        print('norm of up      = ' + str(np.linalg.norm(up)))
        print('norm of right   = ' + str(np.linalg.norm(right)))        
        exit(1)

    return pos, np.array([qw, qx, qy, qz]), up, right, rot



def invdist2pc(invdist, rays_d_cam, w2c, pos, est_pc_wrd_list, b_mask_thr=5, sample_interval=5, inp_is_inv=True):
    rays_d_wrd = torch.sum(rays_d_cam[..., None, :]*w2c.permute(0, 2, 1)[:, None, None, :, :], dim=-1)
    
    if inp_is_inv:
        distance = torch.ones_like(invdist) * 1.e11
        distance[invdist > 0.01] = 1 / invdist[invdist > 0.01]
    else:
        distance = invdist
    mask = distance < 1.25
    
    b_mask = mask.clone()
    b_mask[b_mask_thr:, :] = torch.logical_and(b_mask[b_mask_thr:, :], mask[:-b_mask_thr, :])
    b_mask[:-b_mask_thr, :] = torch.logical_and(b_mask[:-b_mask_thr, :], mask[b_mask_thr:, :])
    b_mask[:, b_mask_thr:] = torch.logical_and(b_mask[:, b_mask_thr:], mask[:, :-b_mask_thr])
    b_mask[:, :-b_mask_thr] = torch.logical_and(b_mask[:, :-b_mask_thr], mask[:, b_mask_thr:])
    pc_wrd = distance[..., None] * rays_d_wrd + pos[:, None, None, :]

    for c_btc_idx, (pt_wrd_i, mask_i) in enumerate(zip(pc_wrd, b_mask)):
        est_pc_wrd_list[c_btc_idx].append(pt_wrd_i[mask_i])
    return 0



from scipy.spatial import ConvexHull
from numpy import *

def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.
   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**
   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def is_clockwise(p):
    x = p[:,0]
    y = p[:,1]
    return np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)) > 0

def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    todo (kent): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
   
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])

    inter_vol = inter_area * max(0.0, ymax-ymin)
    
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d

# ----------------------------------
# Helper functions for evaluation
# ----------------------------------

def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.
    Input:
        box_size: tuple of (length,wide,height)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    R = roty(heading_angle)
    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d



import quaternion
def calc_rotation_diff(q, q00):
    rotation_dot = np.dot(quaternion.as_float_array(q00), quaternion.as_float_array(q))
    rotation_dot_abs = np.abs(rotation_dot)
    try:                                                                                                                                                                                                                                                                                                                      
        error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    except:
        return 0.0
    error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    error_rotation = np.rad2deg(error_rotation_rad)
    return error_rotation
    