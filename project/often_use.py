import os
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
from tqdm import tqdm, trange
import torchvision
from scipy.spatial import cKDTree as KDTree
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



# def batch_vec2skew(v, batch_size):
#     """
#     :param v:  (3, ) torch tensor
#     :return:   (3, 3)
#     """
#     zero = torch.zeros([batch_size, 1], dtype=torch.float32, device=v.device)
#     skew_v0 = torch.cat([ zero,    -v[:, 2:3],   v[:, 1:2]], dim=1)  # (batch, 3, 1)
#     skew_v1 = torch.cat([ v[:, 2:3],   zero,    -v[:, 0:1]], dim=1)
#     skew_v2 = torch.cat([-v[:, 1:2],   v[:, 0:1],   zero], dim=1)
#     skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=1)  # (batch, 3, 3)
#     return skew_v  # (batch, 3, 3)



# def batch_Exp(r):
#     """so(3) vector to SO(3) matrix
#     :param r: (3, ) axis-angle, torch tensor
#     :return:  (3, 3)
#     """
#     batch_size = r.shape[0]
#     skew_r = batch_vec2skew(r, batch_size)  # (batch, 3, 3)
#     norm_r = r.norm(dim=1)[:, None, None] + 1e-15
#     eye = torch.eye(3, dtype=torch.float32, device=r.device).expand(batch_size, -1, -1)
#     R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * torch.bmm(skew_r, skew_r)
#     return R



# def batch_pi2rot_y(pi):
#     # Rotate randomly.
#     if all(pi[:, 0] == 0) and all(pi[:, 2] == 0):
#         zeros = torch.zeros_like(pi[:, 0])
#         ones = torch.ones_like(pi[:, 0])
#         R_0 = torch.stack([torch.cos(pi[:, 1]), zeros, -torch.sin(pi[:, 1])], dim=-1)
#         R_1 = torch.stack([              zeros,  ones,                zeros], dim=-1)
#         R_2 = torch.stack([torch.sin(pi[:, 1]), zeros,  torch.cos(pi[:, 1])], dim=-1)
#         return torch.stack([R_0, R_1, R_2], dim=-1)
#     else:
#         print('only y-axis rotation.')
#         sys.exit()

            
                                    
# def get_weighted_average(target, ratio): # [Batch, Sample, Values]
#     ratio = ratio / torch.sum(ratio, dim=1)[..., None] # [Batch, Sample]
#     return torch.sum(ratio[..., None] * target, dim=1) # [Batch, Sample, Values]



def batch_linspace(start, end, step):
    raw = torch.linspace(0, 1, step)[None, :].to(start)
    return (end.to(start) - start)[:, None] * raw + start[:, None]



# def int_clamp(n, smallest, largest):
#     return int(max(smallest, min(n, largest)))



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



# def clopping_distance_map(mask, distance_map, image_coord, input_H, input_W, ddf_H, bbox_list='not_given'):
#     # Get bbox.
#     if bbox_list == 'not_given':
#         raw_bbox_list = []
#         for i, mask_i in enumerate(mask):
#             masked_image_coord = image_coord[mask_i]
#             if masked_image_coord.shape[0] != 0:
#                 max_y, max_x = masked_image_coord.max(dim=0).values
#                 min_y, min_x = masked_image_coord.min(dim=0).values
#             elif masked_image_coord.shape[0] == 0:
#                 mask[i] = torch.ones_like(mask_i)
#                 max_y, max_x = image_coord.reshape(-1, 2).max(dim=0).values
#                 min_y, min_x = image_coord.reshape(-1, 2).min(dim=0).values
#             raw_bbox_list.append(torch.tensor([[max_x, max_y], [min_x, min_y]]))
#         raw_bbox_list = torch.stack(raw_bbox_list, dim=0)
        
#         # 正方形でClop.
#         bbox_H_xy = torch.stack([raw_bbox_list[:, 0, 0] - raw_bbox_list[:, 1, 0], # H_x
#                                  raw_bbox_list[:, 0, 1] - raw_bbox_list[:, 1, 1]] # H_y
#                                     , dim=-1)
#         bbox_H = bbox_H_xy.max(dim=-1).values # BBoxのxy幅の内、大きい方で揃える
#         diff_bbox_H = (bbox_H[:, None] - bbox_H_xy) / 2
#         bbox_list = raw_bbox_list + torch.stack([diff_bbox_H, -diff_bbox_H], dim=-2) # maxには足りない分を足し、minからは引く

#         # BBoxが画像からはみ出た場合、収まるように戻す
#         border = torch.tensor([[input_W-1, input_H-1], [0, 0]])[None]
#         outside = border - bbox_list
#         outside[:, 0][outside[:, 0] > .0] = 0. # 値が負ならMaxがはみ出た -> ずれを引く
#         outside[:, 1][outside[:, 1] < .0] = 0. # 値が正ならMinがはみ出た -> ずれを足す
#         bbox_list = bbox_list + outside.sum(dim=-2)[:, None, :]
        
#         # 元画像より大きい時
#         for i_, (diff_bbox_H_i, bbox_H_xy_i, bbox_H_i) in enumerate(zip(diff_bbox_H, bbox_H_xy, bbox_H)):
#             if not (bbox_H_i < min(input_H, input_W)):
#                 print('unballance bbox!')
#                 max_y = raw_bbox_list[i_][0, 1]
#                 min_y = raw_bbox_list[i_][1, 1]
#                 upper_limitted = max_y <= input_H
#                 lower_limitted = min_y >= 0
#                 if upper_limitted==lower_limitted:
#                     bbox_list[i_][:, 1] = raw_bbox_list[i_][:, 1] + torch.stack([diff_bbox_H, -diff_bbox_H], dim=-2)[:, 1]
#                 elif upper_limitted: # 上が画像の淵からはみ出ている
#                     bbox_list[i_][1, 1] = raw_bbox_list[i_][1, 1] - 2*diff_bbox_H[1]
#                 elif lower_limitted: # 下が画像の淵からはみ出ている
#                     bbox_list[i_][0, 1] = raw_bbox_list[i_][0, 1] + 2*diff_bbox_H[1]
#                 import pdb; pdb.set_trace()
        
#         # 範囲をそろえる
#         bbox_list[:, :, 0] = bbox_list[:, :, 0] / (0.5*input_W) - 1 # change range [-1, 1]
#         bbox_list[:, :, 1] = bbox_list[:, :, 1] / (0.5*input_H) - 1 # change range [-1, 1]

#     # Clop
#     mask = mask[:, None] # (N, dummy_C, H, W)
#     distance_map = distance_map[:, None] # (N, dummy_C, H, W)

#     coord_x = batch_linspace(bbox_list[:, 1, 0], bbox_list[:, 0, 0], ddf_H)
#     coord_x = coord_x[:, None, :].expand(-1, ddf_H, -1)
#     coord_y = batch_linspace(bbox_list[:, 1, 1], bbox_list[:, 0, 1], ddf_H)
#     coord_y = coord_y[:, :, None].expand(-1, -1, ddf_H)
#     sampling_coord = torch.stack([coord_x, coord_y], dim=-1).to(distance_map.device)
#     cloped_mask = F.grid_sample(
#                         mask.to(sampling_coord.dtype), 
#                         sampling_coord, 
#                         align_corners=True)[:, 0] > 0.99 # (N, H, W)
#     cloped_distance_map = F.grid_sample(
#                         distance_map.to(sampling_coord.dtype), 
#                         sampling_coord, 
#                         align_corners=True)[:, 0] # (N, H, W)
#     return cloped_mask, cloped_distance_map, bbox_list



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
    cloped_mask = F.grid_sample(
                        mask.to(sampling_coord.dtype), 
                        sampling_coord, 
                        align_corners=True)[:, 0] > 0.99 # (N, H, W)
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



# def get_normalized_depth_map(mask, distance_map, rays_d_cam, avg_depth_map='not_given'):
#     # Convert to depth map.
#     depth_map = rays_d_cam[..., -1] * distance_map

#     # Get average.
#     if avg_depth_map=='not_given':
#         avg_depth_map = torch.tensor(
#             [depth_map_i[mask_i].mean() for mask_i, depth_map_i in zip(mask, depth_map)]
#             , device=depth_map.device) # 物体の存在しているピクセルで平均を取る。
#         # top_n = 10 # top_nからtop_n//2までの平均を取る
#         # avg_depth_map = torch.stack([
#         #                     torch.topk(depth_map_i[mask_i], top_n).values[top_n//2:].mean() - torch.topk(-depth_map_i[mask_i], top_n).values[top_n//2:].mean() 
#         #                     for mask_i, depth_map_i in zip(mask, depth_map)], dim=0)
#         # avg_depth_map = avg_depth_map / 2

#     # Normalizing.
#     normalized_depth_map = depth_map - avg_depth_map[:, None, None]
#     normalized_depth_map[torch.logical_not(mask)] = 0. # 物体の存在しているピクセルを正規化
#     return depth_map, normalized_depth_map, avg_depth_map



# def get_clopped_rays_d_cam(size, bbox_list, rays_d_cam):
#     coord_x = batch_linspace(bbox_list[:, 1, 0], bbox_list[:, 0, 0], size)
#     coord_x = coord_x[:, None, :].expand(-1, size, -1)
#     coord_y = batch_linspace(bbox_list[:, 1, 1], bbox_list[:, 0, 1], size)
#     coord_y = coord_y[:, :, None].expand(-1, -1, size)
#     sampling_coord = torch.stack([coord_x, coord_y], dim=-1)
#     rays_d_cam = rays_d_cam.expand(bbox_list.shape[0], -1, -1, -1).permute(0, 3, 1, 2)
#     rays_d_cam = F.grid_sample(rays_d_cam.to(sampling_coord.dtype), sampling_coord, align_corners=True).permute(0, 2, 3, 1)
#     rays_d_cam = F.normalize(rays_d_cam, dim=-1)
#     return rays_d_cam # H, W, 3:xyz



# def diff2estimation(obj_pos_cim, scale_diff, bbox_list, avg_z_map, fov, canonical_bbox_diagonal=1.0, with_cim2cam_info=False):
#     # Get Bbox info.
#     bbox_list = bbox_list.to(obj_pos_cim)
#     fov = torch.deg2rad(torch.tensor(fov, dtype=torch.float))
#     xy_cim = obj_pos_cim[:, :2]
#     bbox_hight = bbox_list[:, 0, 1] - bbox_list[:, 1, 1]
#     bbox_center = bbox_list.mean(1)

#     # clopした深度マップでの予測物体中心(x, y)をカメラ座標系における(x, y)に変換
#     cim2im_scale = (bbox_hight) / 2 # clopしたBBoxの高さ÷画像の高さ２
#     im2cam_scale = avg_z_map * torch.tan(fov/2) # 中心のDepth（z軸の値）×torch.tan(fov/2)
#     xy_im = cim2im_scale[:, None] * xy_cim + bbox_center # 元画像における物体中心
#     xy_cam = im2cam_scale[:, None] * xy_im

#     # 正規化深度画像での物体中心zをカメラ座標系におけるzに変換
#     z_diff = obj_pos_cim[:, 2]
#     z_cam = z_diff + avg_z_map

#     # clopしたBBoxの対角
#     clopping_bbox_diagonal = 2 * math.sqrt(2)

#     # clopしたBBoxの対角とカノニカルBBoxの対角の比を変換
#     scale = scale_diff * im2cam_scale[:, None] * cim2im_scale[:, None] * clopping_bbox_diagonal / canonical_bbox_diagonal
#     if with_cim2cam_info:
#         return torch.cat([xy_cam, z_cam[..., None]], dim=-1), scale, cim2im_scale, im2cam_scale, bbox_center
#     else:
#         return torch.cat([xy_cam, z_cam[..., None]], dim=-1), scale



# def get_clopping_infos(bbox_list, avg_z_map, fov):
#     # Get Bbox info.
#     fov = torch.deg2rad(torch.tensor(fov, dtype=torch.float))
#     bbox_hight = bbox_list[:, 0, 1] - bbox_list[:, 1, 1]
#     bbox_center = bbox_list.mean(1)

#     # clopした深度マップでの予測物体中心(x, y)をカメラ座標系における(x, y)に変換
#     cim2im_scale = (bbox_hight) / 2 # clopしたBBoxの高さ÷画像の高さ２
#     im2cam_scale = avg_z_map * torch.tan(fov/2) # 中心のDepth（z軸の値）×torch.tan(fov/2)
#     return cim2im_scale, im2cam_scale, bbox_center



# def diffcim2diffcam(diff_pos_cim, cim2im_scale, im2cam_scale):
#     # Get Bbox info.
#     diff_xy_cim = diff_pos_cim[:, :-1]
#     diff_z_diff = diff_pos_cim[:, -1]

#     # clopした深度マップでの予測物体中心(x, y)をカメラ座標系における(x, y)に変換
#     diff_xy_im = cim2im_scale[:, None] * diff_xy_cim # 元画像における物体中心
#     diff_xy_cam = im2cam_scale[:, None] * diff_xy_im
#     return torch.cat([diff_xy_cam, diff_z_diff[:, None]], dim=-1)



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



def get_OSMap_obj(distance_map, mask, rays_d_cam, w2c, cam_pos_wrd, o2w, obj_pos_wrd, obj_scale):
    OSMap_cam = distance_map[..., None] * rays_d_cam
    OSMap_wrd = torch.sum(OSMap_cam[..., None, :]*w2c.permute(0, 2, 1)[..., None, None, :, :], dim=-1) + cam_pos_wrd[..., None, None, :]
    OSMap_obj = torch.sum((OSMap_wrd - obj_pos_wrd[..., None, None, :])[..., None, :]*o2w.permute(0, 2, 1)[..., None, None, :, :], dim=-1) / obj_scale[..., None, None, :]
    OSMap_obj[torch.logical_not(mask)] = 0.
    OSMap_obj_wMask = torch.cat([OSMap_obj, mask.to(OSMap_obj.dtype).unsqueeze(-1)], dim=-1)
    return OSMap_obj_wMask
    # OSMap_cam_wMask = torch.cat([OSMap_cam, mask.to(OSMap_obj.dtype).unsqueeze(-1)], dim=-1)
    # return OSMap_cam_wMask



def get_diffOSMap_obj(obs_distance_map, est_distance_map, obs_mask, est_mask, rays_d_cam, w2c, o2w, obj_scale):
    diff_or_mask = torch.logical_or(obs_mask, est_mask)
    diff_xor_mask = torch.logical_xor(obs_mask, est_mask)

    diff_distance_map = obs_distance_map - est_distance_map
    diffOSMap_cam = diff_distance_map[..., None] * rays_d_cam
    diffOSMap_wrd = torch.sum(diffOSMap_cam[..., None, :]*w2c.permute(0, 2, 1)[..., None, None, :, :], dim=-1)
    diffOSMap_obj = torch.sum(diffOSMap_wrd[..., None, :]*o2w.permute(0, 2, 1)[..., None, None, :, :], dim=-1)
    diffOSMap_obj = diffOSMap_obj / obj_scale[..., None, None, :]

    diffOSMap_obj[torch.logical_not(diff_or_mask)] = 0.
    diffOSMap_obj_wMask = torch.cat([diffOSMap_obj, diff_xor_mask.to(diffOSMap_obj).unsqueeze(-1)], dim=-1)
    return diffOSMap_obj_wMask
    # diffOSMap_cam_wMask = torch.cat([diffOSMap_cam, diff_xor_mask.to(diffOSMap_obj).unsqueeze(-1)], dim=-1)
    # return diffOSMap_cam_wMask



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



# class WarmupScheduler(torch.optim.lr_scheduler.LambdaLR):

#     def __init__(self, optimizer, warmup_steps, last_epoch=-1, d_model=512):

#         def lr_lambda(step_num):
#             step_num = step_num + 1
#             return d_model**(-0.5) * min(step_num**(-0.5), step_num*(warmup_steps**(-1.5)))

#         super(WarmupScheduler, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)

#     def get_lr(self):
#         # if not self._get_lr_called_within_step:
#         #     warnings.warn("To get the last learning rate computed by the scheduler, "
#         #                   "please use `get_last_lr()`.")

#         if self.last_epoch == 0: print('warm up !')
#         return [lmbda(self.last_epoch) for lmbda in self.lr_lambdas]



# def get_inits(batch_size, randn_from_pickle, rand_idx_int, rand_P_range, rand_S_range, rand_R_range, 
#     random_axis_list, random_axis_num, rand_Z_center, frame_obj_pos, frame_obj_scale, 
#     frame_gt_obj_axis_green_wrd, frame_gt_obj_axis_red_wrd, gt_shape_code, frame_gt_o2w, rand_idx_max, mode, 
#     rand_P_seed, rand_S_seed, randn_theta_seed, randn_axis_idx):

#     use_log = not(rand_P_seed[0]=='non' or rand_S_seed[0]=='non' or randn_theta_seed[0]=='non' or randn_axis_idx[0]=='non')
#     use_log = use_log and mode!='train'
    
#     if use_log:
#         rand_P_seed = rand_P_seed.to('cpu')
#         rand_S_seed = rand_S_seed.to('cpu')
#         randn_theta_seed = randn_theta_seed.to('cpu')
#         randn_axis_idx = randn_axis_idx.to('cpu')

#     elif randn_from_pickle and (mode!='train'):
#         # Get randn path.
#         rand_idx_str = str(int(rand_idx_int)).zfill(10)
#         randn_path = f'randn/list0randn_batch128_{mode}/{rand_idx_str}.pickle'

#         rand_idx_int = rand_idx_int + 1
#         if rand_idx_int > rand_idx_max[mode]:
#             rand_idx_int = 0

#         # Get randn seeds.
#         randn = pickle_load(randn_path)
#         rand_P_seed = randn[0][:batch_size] # torch.rand(batch_size, 1)
#         rand_S_seed = randn[1][:batch_size] # torch.rand(batch_size, 1)
#         randn_theta_seed = randn[2][:batch_size] # torch.rand(batch_size)
#         randn_axis_idx = randn[3][:batch_size] # np.random.choice(random_axis_num, batch_size)

#     else:
#         rand_P_seed = torch.rand(batch_size, 1)
#         rand_S_seed = torch.rand(batch_size, 1)
#         randn_theta_seed = torch.rand(batch_size)
#         randn_axis_idx = np.random.choice(random_axis_num, batch_size)

#     # Log seeds.
#     if mode in {'val', 'tes'}:
#         rand_seed = {}
#         rand_seed['rand_P_seed'] = rand_P_seed
#         rand_seed['rand_S_seed'] = rand_S_seed
#         rand_seed['randn_theta_seed'] = randn_theta_seed
#         rand_seed['randn_axis_idx'] = randn_axis_idx 
#     else:
#         rand_seed = None


#     # Get initial position.
#     rand_P = 2 * rand_P_range * (rand_P_seed - .5) #  * (torch.rand(batch_size, 1) - .5)
#     ini_obj_pos = frame_obj_pos[:, 0, :] + rand_P.to(frame_obj_pos)

#     # Get initial scale.
#     rand_S = 2 * rand_S_range * (rand_S_seed - .5) + 1. #  * (torch.rand(batch_size, 1) - .5) + 1.
#     ini_obj_scale = frame_obj_scale[:, 0].unsqueeze(-1) * rand_S.to(frame_obj_scale)

#     # Get initial red.
#     randn_theta = rand_R_range * 2 * (randn_theta_seed - .5)
#     randn_axis = random_axis_list[randn_axis_idx]
#     randn_axis = torch.sum(randn_axis[..., None, :]*frame_gt_o2w[:, 0].to(randn_axis), dim=-1)
#     cos_t = torch.cos(randn_theta)
#     sin_t = torch.sin(randn_theta)
#     n_x = randn_axis[:, 0]
#     n_y = randn_axis[:, 1]
#     n_z = randn_axis[:, 2]
#     rand_R = torch.stack([torch.stack([cos_t+n_x*n_x*(1-cos_t), n_x*n_y*(1-cos_t)-n_z*sin_t, n_z*n_x*(1-cos_t)+n_y*sin_t], dim=-1), 
#                             torch.stack([n_x*n_y*(1-cos_t)+n_z*sin_t, cos_t+n_y*n_y*(1-cos_t), n_y*n_z*(1-cos_t)-n_x*sin_t], dim=-1), 
#                             torch.stack([n_z*n_x*(1-cos_t)-n_y*sin_t, n_y*n_z*(1-cos_t)+n_x*sin_t, cos_t+n_z*n_z*(1-cos_t)], dim=-1)], dim=1)
#     obj_red_wrd = frame_gt_obj_axis_red_wrd[:, 0]
#     ini_obj_axis_red_wrd = torch.sum(obj_red_wrd[..., None, :]*rand_R.to(obj_red_wrd), dim=-1)

#     # Get initial green.
#     obj_green_wrd = frame_gt_obj_axis_green_wrd[:, 0]
#     ini_obj_axis_green_wrd = torch.sum(obj_green_wrd[..., None, :]*rand_R.to(obj_green_wrd), dim=-1)

#     # Get initial shape.
#     ini_shape_code = rand_Z_center.unsqueeze(0).expand(batch_size, -1).to(gt_shape_code) # + randn_Z

#     return ini_obj_pos, ini_obj_scale, ini_obj_axis_red_wrd, ini_obj_axis_green_wrd, ini_shape_code, rand_idx_int, rand_seed



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
    rays_o = F.normalize(rays_o_obj, dim=-1)

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
    # est_normal = torch.sum(est_normal[:, :, :, None, :]*o2c[..., None, None, :, :].permute(0, 1, 2, 4, 3), -1)
    # est_normal = F.normalize(est_normal, dim=-1)
    
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



def get_pe_target(cam_pos_wrd, w2c, rays_d_cam, obj_pos_wrd, o2w, obj_scale_wrd, bbox_diagonal, H):
    pe_target_cam_pos_obj = torch.sum((cam_pos_wrd - obj_pos_wrd)[..., None, :]*o2w.permute(0, 1, 3, 2), dim=-1) / obj_scale_wrd
    center_start = int(-(-H//2)-1)
    center_end = int(H//2+1)
    center_d_cam = rays_d_cam[:, :, center_start:center_end, center_start:center_end, :].mean(dim=(2, 3))
    center_d_wrd = torch.sum(center_d_cam[..., None, :]*w2c.permute(0, 1, 3, 2), dim=-1)
    pe_target_center_d_obj = torch.sum(center_d_wrd[..., None, :]*o2w.permute(0, 1, 3, 2), dim=-1)
    pe_target_bbox = bbox_diagonal
    pe_target_obj = torch.cat([pe_target_cam_pos_obj, pe_target_center_d_obj, pe_target_bbox], dim=-1)
    return pe_target_obj
