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





def check_map_torch(image, path = 'sample_images/tes.png', figsize=[10,10]):
    fig = pylab.figure(figsize=figsize)

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title('result')
    ax1.imshow(image.to('cpu').detach().numpy().copy())

    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])
    fig.savefig(path, dpi=300)
    pylab.close()





def check_map_np(image, path = 'sample_images/tes.png', figsize=[10,10]):
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



def batch_vec2skew(v, batch_size):
    """
    :param v:  (3, ) torch tensor
    :return:   (3, 3)
    """
    zero = torch.zeros([batch_size, 1], dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([ zero,    -v[:, 2:3],   v[:, 1:2]], dim=1)  # (batch, 3, 1)
    skew_v1 = torch.cat([ v[:, 2:3],   zero,    -v[:, 0:1]], dim=1)
    skew_v2 = torch.cat([-v[:, 1:2],   v[:, 0:1],   zero], dim=1)
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=1)  # (batch, 3, 3)
    return skew_v  # (batch, 3, 3)



def batch_Exp(r):
    """so(3) vector to SO(3) matrix
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3)
    """
    batch_size = r.shape[0]
    skew_r = batch_vec2skew(r, batch_size)  # (batch, 3, 3)
    norm_r = r.norm(dim=1)[:, None, None] + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device).expand(batch_size, -1, -1)
    R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * torch.bmm(skew_r, skew_r)
    return R



def batch_pi2rot_y(pi):
    # Rotate randomly.
    if all(pi[:, 0] == 0) and all(pi[:, 2] == 0):
        zeros = torch.zeros_like(pi[:, 0])
        ones = torch.ones_like(pi[:, 0])
        R_0 = torch.stack([torch.cos(pi[:, 1]), zeros, -torch.sin(pi[:, 1])], dim=-1)
        R_1 = torch.stack([              zeros,  ones,                zeros], dim=-1)
        R_2 = torch.stack([torch.sin(pi[:, 1]), zeros,  torch.cos(pi[:, 1])], dim=-1)
        return torch.stack([R_0, R_1, R_2], dim=-1)
    else:
        print('only y-axis rotation.')
        sys.exit()

            
                                    
def get_weighted_average(target, ratio): # [Batch, Sample, Values]
    ratio = ratio / torch.sum(ratio, dim=1)[..., None] # [Batch, Sample]
    return torch.sum(ratio[..., None] * target, dim=1) # [Batch, Sample, Values]



def batch_linspace(start, end, step):
    raw = torch.linspace(0, 1, step)[None, :]
    return (end - start)[:, None] * raw + start[:, None]



def int_clamp(n, smallest, largest):
    return int(max(smallest, min(n, largest)))



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
    mask_under_border = 1 / (cam_pos_obj.norm(dim=-1) + 1.0 * obj_scale * ddf.radius) # 良いのか...？
    est_mask = [map_i > border_i for map_i, border_i in zip(est_invdistance_map, mask_under_border)]
    est_mask = torch.stack(est_mask, dim=0)

    if with_invdistance_map:
        return est_invdistance_map, est_mask
    else:
        est_distance_map = torch.zeros_like(est_invdistance_map)
        est_distance_map[est_mask] = 1. / est_invdistance_map[est_mask]
        return est_mask, est_distance_map



def clopping_distance_map(mask, distance_map, image_coord, input_H, input_W, ddf_H, bbox_list='not_given'):
    # Get bbox.
    if bbox_list == 'not_given':
        raw_bbox_list = []
        for i, mask_i in enumerate(mask):
            masked_image_coord = image_coord[mask_i]
            if masked_image_coord.shape[0] != 0:
                max_y, max_x = masked_image_coord.max(dim=0).values
                min_y, min_x = masked_image_coord.min(dim=0).values
            elif masked_image_coord.shape[0] == 0:
                mask[i] = torch.ones_like(mask_i)
                max_y, max_x = image_coord.reshape(-1, 2).max(dim=0).values
                min_y, min_x = image_coord.reshape(-1, 2).min(dim=0).values
            raw_bbox_list.append(torch.tensor([[max_x, max_y], [min_x, min_y]]))
        raw_bbox_list = torch.stack(raw_bbox_list, dim=0)
        
        # 正方形でClop.
        bbox_H_xy = torch.stack([raw_bbox_list[:, 0, 0] - raw_bbox_list[:, 1, 0], # H_x
                                 raw_bbox_list[:, 0, 1] - raw_bbox_list[:, 1, 1]] # H_y
                                    , dim=-1)
        bbox_H = bbox_H_xy.max(dim=-1).values # BBoxのxy幅の内、大きい方で揃える
        diff_bbox_H = (bbox_H[:, None] - bbox_H_xy) / 2
        bbox_list = raw_bbox_list + torch.stack([diff_bbox_H, -diff_bbox_H], dim=-2) # maxには足りない分を足し、minからは引く

        # BBoxが画像からはみ出た場合、収まるように戻す
        border = torch.tensor([[input_W-1, input_H-1], [0, 0]])[None]
        outside = border - bbox_list
        outside[:, 0][outside[:, 0] > .0] = 0. # 値が負ならMaxがはみ出た -> ずれを引く
        outside[:, 1][outside[:, 1] < .0] = 0. # 値が正ならMinがはみ出た -> ずれを足す
        bbox_list = bbox_list + outside.sum(dim=-2)[:, None, :]
        
        # 元画像より大きい時
        for i_, (diff_bbox_H_i, bbox_H_xy_i, bbox_H_i) in enumerate(zip(diff_bbox_H, bbox_H_xy, bbox_H)):
            if not (bbox_H_i < min(input_H, input_W)):
                print('unballance bbox!')
                max_y = raw_bbox_list[i_][0, 1]
                min_y = raw_bbox_list[i_][1, 1]
                upper_limitted = max_y <= input_H
                lower_limitted = min_y >= 0
                if upper_limitted==lower_limitted:
                    bbox_list[i_][:, 1] = raw_bbox_list[i_][:, 1] + torch.stack([diff_bbox_H, -diff_bbox_H], dim=-2)[:, 1]
                elif upper_limitted: # 上が画像の淵からはみ出ている
                    bbox_list[i_][1, 1] = raw_bbox_list[i_][1, 1] - 2*diff_bbox_H[1]
                elif lower_limitted: # 下が画像の淵からはみ出ている
                    bbox_list[i_][0, 1] = raw_bbox_list[i_][0, 1] + 2*diff_bbox_H[1]
                import pdb; pdb.set_trace()
        
        # 範囲をそろえる
        bbox_list[:, :, 0] = bbox_list[:, :, 0] / (0.5*input_W) - 1 # change range [-1, 1]
        bbox_list[:, :, 1] = bbox_list[:, :, 1] / (0.5*input_H) - 1 # change range [-1, 1]

    # Clop
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
    return cloped_mask, cloped_distance_map, bbox_list



def get_normalized_depth_map(mask, distance_map, rays_d_cam, avg_depth_map='not_given'):
    # Convert to depth map.
    depth_map = rays_d_cam[..., -1] * distance_map

    # Get average.
    if avg_depth_map=='not_given':
        avg_depth_map = torch.tensor(
            [depth_map_i[mask_i].mean() for mask_i, depth_map_i in zip(mask, depth_map)]
            , device=depth_map.device) # 物体の存在しているピクセルで平均を取る。
        # top_n = 10 # top_nからtop_n//2までの平均を取る
        # avg_depth_map = torch.stack([
        #                     torch.topk(depth_map_i[mask_i], top_n).values[top_n//2:].mean() - torch.topk(-depth_map_i[mask_i], top_n).values[top_n//2:].mean() 
        #                     for mask_i, depth_map_i in zip(mask, depth_map)], dim=0)
        # avg_depth_map = avg_depth_map / 2

    # Normalizing.
    normalized_depth_map = depth_map - avg_depth_map[:, None, None]
    normalized_depth_map[torch.logical_not(mask)] = 0. # 物体の存在しているピクセルを正規化
    return depth_map, normalized_depth_map, avg_depth_map



def get_clopped_rays_d_cam(size, fov_h, bbox_list, input_H='not_given', input_W='not_given', input_F='not_given'):
    bbox_list_ = 0.5 * bbox_list
    if  input_H == 'not_given':
        fov = torch.deg2rad(torch.tensor(fov_h, dtype=torch.float))
        x_coord = batch_linspace(torch.tan(fov*bbox_list_[:, 1, 0]), torch.tan(fov*bbox_list_[:, 0, 0]), size)
        x_coord = x_coord[:, None, :].expand(-1, size, -1)
        y_coord = batch_linspace(torch.tan(fov*bbox_list_[:, 1, 1]), torch.tan(fov*bbox_list_[:, 0, 1]), size)
        y_coord = y_coord[:, :, None].expand(-1, -1, size)
    else:
        fov_h = 2*torch.arctan(torch.tensor(0.5*input_H/input_F, dtype=torch.float))
        fov_w = 2*torch.arctan(torch.tensor(0.5*input_W/input_F, dtype=torch.float))
        x_coord = batch_linspace(torch.tan(fov_w*bbox_list_[:, 1, 0]), torch.tan(fov_w*bbox_list_[:, 0, 0]), size)
        x_coord = x_coord[:, None, :].expand(-1, size, -1)
        y_coord = batch_linspace(torch.tan(fov_h*bbox_list_[:, 1, 1]), torch.tan(fov_h*bbox_list_[:, 0, 1]), size)
        y_coord = y_coord[:, :, None].expand(-1, -1, size)
    rays_d_cam = torch.stack([x_coord, y_coord, torch.ones_like(x_coord)], dim=-1)
    rays_d_cam = F.normalize(rays_d_cam, dim=-1)
    return rays_d_cam # H, W, 3:xyz



def diff2estimation(obj_pos_cim, scale_diff, bbox_list, avg_z_map, fov, canonical_bbox_diagonal=1.0, with_cim2cam_info=False):
    # Get Bbox info.
    bbox_list = bbox_list.to(obj_pos_cim)
    fov = torch.deg2rad(torch.tensor(fov, dtype=torch.float))
    xy_cim = obj_pos_cim[:, :2]
    bbox_hight = bbox_list[:, 0, 1] - bbox_list[:, 1, 1]
    bbox_center = bbox_list.mean(1)

    # clopした深度マップでの予測物体中心(x, y)をカメラ座標系における(x, y)に変換
    cim2im_scale = (bbox_hight) / 2 # clopしたBBoxの高さ÷画像の高さ２
    im2cam_scale = avg_z_map * torch.tan(fov/2) # 中心のDepth（z軸の値）×torch.tan(fov/2)
    xy_im = cim2im_scale[:, None] * xy_cim + bbox_center # 元画像における物体中心
    xy_cam = im2cam_scale[:, None] * xy_im

    # 正規化深度画像での物体中心zをカメラ座標系におけるzに変換
    z_diff = obj_pos_cim[:, 2]
    z_cam = z_diff + avg_z_map

    # clopしたBBoxの対角
    clopping_bbox_diagonal = 2 * math.sqrt(2)

    # clopしたBBoxの対角とカノニカルBBoxの対角の比を変換
    scale = scale_diff * im2cam_scale[:, None] * cim2im_scale[:, None] * clopping_bbox_diagonal / canonical_bbox_diagonal
    if with_cim2cam_info:
        return torch.cat([xy_cam, z_cam[..., None]], dim=-1), scale, cim2im_scale, im2cam_scale, bbox_center
    else:
        return torch.cat([xy_cam, z_cam[..., None]], dim=-1), scale




def diffcim2diffcam(diff_pos_cim, cim2im_scale, im2cam_scale):
    # Get Bbox info.
    diff_xy_cim = diff_pos_cim[:, :-1]
    diff_z_diff = diff_pos_cim[:, -1]

    # clopした深度マップでの予測物体中心(x, y)をカメラ座標系における(x, y)に変換
    diff_xy_im = cim2im_scale[:, None] * diff_xy_cim # 元画像における物体中心
    diff_xy_cam = im2cam_scale[:, None] * diff_xy_im
    return torch.cat([diff_xy_cam, diff_z_diff[:, None]], dim=-1)



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