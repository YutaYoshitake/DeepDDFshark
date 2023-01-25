# python check_ddf.py --config=../configs/paper_exp/chair/view5/txt.txt

import numpy as np
import os
import sys
import glob
import tqdm
import random
from often_use import *
sys.path.append("../")
from parser_get_arg import *
from DDF.train_pl import DDF



# 書き換える
out_dir = 'results'
data_dir_path = '/home/yyoshitake/works/DeepSDF/project/scannet/results/03001627/scene0314_00/c585ee093bfd52af6512b7b24f3d84_002/data_dict'
cat_id = data_dir_path.split('/')[8]

category_incetance_id = data_dir_path.split('/')[-2]
instance_id = category_incetance_id.split('_')[0]
obj_id = category_incetance_id.split('_')[1]
loc_dict = pickle_load(os.path.join(out_dir, 'loc_dict.pickle')) # make_offet_pickle.pyで作る
loc = loc_dict[f'{cat_id}_{instance_id}']

# Get args
args = get_args()
args.ddf_model_path = '/home/yyoshitake/works/DeepSDF/project/DDF/lightning_logs/chair/cat_depth_mae_normal_mae_seed0_normal001_lr00001/checkpoints/0000010000.ckpt'

# Set ddf.
ddf = DDF(args)
ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args)
ddf.eval()
ddf_instance_list = txt2list(args.ddf_instance_list_txt)

def render_distance_map_from_axis_for_scannet(
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
    rays_d_obj = rays_d_obj / obj_scale[:, None, None, :] # あってる？
    rays_d_obj = rays_d_obj / torch.norm(rays_d_obj, dim=-1)[..., None]

    # Get rays origin.
    if obj_pos_cam == 'not_given':
        obj_pos_cam = torch.sum((obj_pos_wrd - cam_pos_wrd)[..., None, :]*w2c, dim=-1)
    cam_pos_obj = - torch.sum(obj_pos_cam[..., None, :]*o2c.permute(0, 2, 1), dim=-1) / obj_scale # あってる？
    rays_o_obj = cam_pos_obj[:, None, None, :].expand(-1, H, H, -1)

    # Get rays inputs.
    rays_d = rays_d_obj
    rays_o = rays_o_obj

    # Estimating.
    est_invdistance_map_obj_scale, _ = ddf.forward_from_far(rays_o, rays_d, input_lat_vec)
    est_invdistance_map = torch.zeros_like(est_invdistance_map_obj_scale)
    noinf_dist_mask = est_invdistance_map_obj_scale > 1.e-5
    est_invdistance_map[noinf_dist_mask] = 1 / torch.norm(
        1 / est_invdistance_map_obj_scale[noinf_dist_mask][:, None] * rays_d_obj[noinf_dist_mask] * obj_scale[:, None, None, :].expand(-1, H, H, -1)[noinf_dist_mask], 
        dim=-1)
    est_invdistance_map[~noinf_dist_mask] = est_invdistance_map_obj_scale[~noinf_dist_mask]

    # Get distance map.
    mask_under_border = 1 / (cam_pos_obj.norm(dim=-1) + .5 * obj_scale.max(dim=-1).values * ddf.radius) # 良いのか...？
    est_mask = [map_i > border_i for map_i, border_i in zip(est_invdistance_map, mask_under_border)]
    est_mask = torch.stack(est_mask, dim=0)
    est_distance_map = torch.zeros_like(est_invdistance_map)
    est_distance_map[est_mask] = 1. / est_invdistance_map[est_mask]

    if with_invdistance_map:
        return est_invdistance_map, est_mask, est_distance_map
    else:
        return est_mask, est_distance_map



#########################
# データの確認
#########################
# データの取得
data_path_list = glob.glob(os.path.join(data_dir_path, '*.pickle'))
est_pt_list = []
gt_pt_list = []
gt_deppt_list = []
samle_num = 5
if len(data_path_list) > samle_num:
    random.shuffle(data_path_list)
check_length = min(len(data_path_list), samle_num)

# Get shape code.
instance_name = data_dir_path.split('/')[-2]
instance_id = instance_name.split('_')[0]
instance_idx = [ddf_instance_list.index(instance_id)] # _i) for instance_id_i in instance_id]
gt_shape_code = ddf.lat_vecs(torch.tensor(instance_idx, device=ddf.device)).detach()

# レンダリング開始
data_path_list = ['/home/yyoshitake/works/DeepSDF/project/scannet/results/03001627/scene0314_00/c585ee093bfd52af6512b7b24f3d84_002/data_dict/00017_00000_00035.pickle', ]
#                   '/home/yyoshitake/works/DeepSDF/project/scannet/results/03001627/scene0000_01/3289bcc9bf8f5dab48d8ff57878739ca_006/data_dict/00035_00003_00380.pickle', 
#                   '/home/yyoshitake/works/DeepSDF/project/scannet/results/03001627/scene0000_01/3289bcc9bf8f5dab48d8ff57878739ca_006/data_dict/00040_00003_00385.pickle', 
#                   '/home/yyoshitake/works/DeepSDF/project/scannet/results/03001627/scene0000_01/3289bcc9bf8f5dab48d8ff57878739ca_006/data_dict/00045_00003_00390.pickle', 
#                   '/home/yyoshitake/works/DeepSDF/project/scannet/results/03001627/scene0000_01/3289bcc9bf8f5dab48d8ff57878739ca_006/data_dict/00050_00004_00412.pickle', ]
vis_map = []
for data_path in data_path_list[:check_length]:
    # データの読み込み
    data_dict = pickle_load(data_path)
    clopped_mask       = torch.from_numpy(data_dict['clopped_mask'].astype(np.float32)).clone()
    clopped_fore_mask  = torch.from_numpy(data_dict['clopped_fore_mask']).clone()
    clopped_distance   = torch.from_numpy(data_dict['clopped_distance'].astype(np.float32)).clone()
    clopped_depth      = torch.from_numpy(data_dict['clopped_depth'].astype(np.float32)).clone()
    clopped_rays_d_cam = torch.from_numpy(data_dict['clopped_rays_d_cam'].astype(np.float32)).clone()
    cam_pos_wrd        = torch.from_numpy(data_dict['cam_pos_wrd'].astype(np.float32)).clone()
    w2c                = torch.from_numpy(data_dict['w2c'].astype(np.float32)).clone()
    o2w                = torch.from_numpy(data_dict['o2w'].astype(np.float32)).clone()
    obj_scale_wrd      = torch.from_numpy(data_dict['obj_scale_wrd'].astype(np.float32)).clone()
    obj_pos_wrd        = torch.from_numpy(data_dict['obj_pos_wrd'].astype(np.float32)).clone() # - o2w @ loc * obj_scale_wrd

    # データの前処理
    H = clopped_distance.shape[0]
    o2c = w2c @ o2w
    obj_green_cam = o2c[:, 1] # Y
    obj_red_cam = o2c[:, 0] # X

    # ダミーのバッチ
    clopped_mask = clopped_mask[None]
    clopped_depth = clopped_depth[None]
    clopped_distance = clopped_distance[None]
    inp_obj_pos_wrd = obj_pos_wrd[None]
    inp_axis_green_cam = obj_green_cam[None]
    inp_axis_red_cam = obj_red_cam[None]
    inp_obj_scale = obj_scale_wrd[None]
    inp_shape_code = gt_shape_code
    inp_cam_pos_wrd = cam_pos_wrd[None]
    inp_rays_d_cam = clopped_rays_d_cam[None]
    inp_w2c = w2c[None]
    inp_o2w = o2w[None]

    est_mask, est_distance_map = render_distance_map_from_axis_for_scannet(H = H, 
                                                                axis_green = inp_axis_green_cam,
                                                                axis_red = inp_axis_red_cam,
                                                                obj_scale = inp_obj_scale, 
                                                                obj_pos_wrd = inp_obj_pos_wrd, 
                                                                rays_d_cam = inp_rays_d_cam, 
                                                                input_lat_vec = inp_shape_code, 
                                                                ddf = ddf, 
                                                                cam_pos_wrd = inp_cam_pos_wrd, 
                                                                w2c = inp_w2c, )
    vis_map_i = torch.cat([clopped_distance[0,:, :], 
                            est_distance_map[0,:, :], 
                            torch.abs(clopped_distance[0,:, :] - est_distance_map[0,:, :])], dim=0)
    vis_map.append(vis_map_i)
    check_map_torch(vis_map_i, f'tes.png')

    # 点群の取得
    OSMap_obj = get_OSMap_obj(est_distance_map, est_mask, inp_rays_d_cam, inp_w2c, inp_cam_pos_wrd, inp_o2w, inp_obj_pos_wrd, inp_obj_scale)[..., :3]
    est_pt_list.append(OSMap_obj[est_mask].to('cpu').detach().numpy().copy())
    gt_mask = torch.logical_and(clopped_mask, clopped_distance > 0)
    OSMap_obj = get_OSMap_obj(clopped_distance, gt_mask, inp_rays_d_cam, inp_w2c, inp_cam_pos_wrd, inp_o2w, inp_obj_pos_wrd, inp_obj_scale)[..., :3]
    gt_pt_list.append(OSMap_obj[gt_mask].to('cpu').detach().numpy().copy())
    OSMap_obj = get_OSMap_obj(clopped_depth, gt_mask, inp_rays_d_cam, inp_w2c, inp_cam_pos_wrd, inp_o2w, inp_obj_pos_wrd, inp_obj_scale)[..., :3]
    gt_deppt_list.append(OSMap_obj[gt_mask].to('cpu').detach().numpy().copy())

# 深度画像の可視化
check_map_torch(torch.cat(vis_map, dim=1), f'{instance_name}_dst.png')

fig = pylab.figure(figsize = [10, 10])
ax = fig.add_subplot(projection='3d')
for est_pt, gt_pt in zip(est_pt_list, gt_pt_list):
    point = est_pt
    ax.scatter3D(point[:, 0], point[:, 1], point[:, 2], c='m', s=0.5)
    point = gt_pt
    ax.scatter3D(point[:, 0], point[:, 1], point[:, 2], c='g', s=0.5)
ax.set_xlabel('x_value')
ax.set_ylabel('y_value')
ax.set_zlabel('z_value')
fig.savefig(f'{instance_name}_pt.png', dpi=300)
pylab.close()
import pdb; pdb.set_trace()

# 点群の一貫性の確認
# fig = pylab.figure(figsize = [10, 10])
# ax = fig.add_subplot(projection='3d')
# c_list = ['g', 'c', 'm', 'r', 'b']
# for pt_idx, est_pt in enumerate(est_pt_list):
#     point = est_pt
#     ax.scatter3D(point[:, 0], point[:, 1], point[:, 2], c=c_list[pt_idx], s=0.5)
# ax.set_xlabel('x_value')
# ax.set_ylabel('y_value')
# ax.set_zlabel('z_value')
# # ax.view_init(elev=0, azim=90)
# fig.savefig('est_tes.png', dpi=300)
# pylab.close()

# fig = pylab.figure(figsize = [10, 10])
# ax = fig.add_subplot(projection='3d')
# c_list = ['g', 'c', 'm', 'r', 'b']
# for pt_idx, gt_pt in enumerate(gt_pt_list):
#     point = gt_pt
#     ax.scatter3D(point[:, 0], point[:, 1], point[:, 2], c=c_list[pt_idx], s=0.5)
# ax.set_xlabel('x_value')
# ax.set_ylabel('y_value')
# ax.set_zlabel('z_value')
# ax.view_init(elev=0, azim=90)
# fig.savefig('_gt_tes_0.png', dpi=300)
# ax.view_init(elev=0, azim=0)
# fig.savefig('_gt_tes_1.png', dpi=300)
# ax.view_init(elev=90, azim=90)
# fig.savefig('_gt_tes_2.png', dpi=300)
# pylab.close()


# # ScanNet Scene_id
# scene_id = 'scene0000_01'
# target_shapenet_cat = {'03001627': 'chair', }
# # Full scan2cad annotationsの読み込み
# import json
# annotations_json_open = open('data2/scan2cad/full_annotations.json', 'r')
# annotations_json_load = json.load(annotations_json_open)
# # Prepare Annotation dictを作成
# annotations_dict = {}
# for annotation_i in annotations_json_load:
#     scan_id_i = annotation_i['id_scan']
#     annotations_dict[scan_id_i] = {}
#     annotations_dict[scan_id_i]['trs'] = annotation_i['trs'] # <-- transformation from scan space to world space
#     annotations_dict[scan_id_i]['cad'] = {}
#     # annotation_i['aligned_models']にはリスト形式でCADモデルが格納されている
#     if scan_id_i == scene_id:
#         for cad_idx, cad_i in enumerate(annotation_i['aligned_models']):
#             catid_cad = cad_i['catid_cad']
#             id_cad = cad_i['id_cad']
#             if catid_cad in target_shapenet_cat.keys():
#                 total_id = f'{catid_cad}_{id_cad}_{str(cad_idx).zfill(3)}'
#                 annotations_dict[scan_id_i]['cad'][total_id] = {}
#                 annotations_dict[scan_id_i]['cad'][total_id]['trs'] = cad_i['trs'] # <-- transformation from CAD space to world space 
#                 annotations_dict[scan_id_i]['cad'][total_id]['sym'] = cad_i['sym']

# scene_trs = annotations_dict[scene_id]['trs']
# Mscan = make_M_from_tqs(scene_trs["translation"], scene_trs["rotation"], scene_trs["scale"])

# DATA_PATH = 'data2/scans/scene0000_00/intrinsic/intrinsic_depth.txt'

# # Ray方向の取得
# intrinsic_depth = load_matrix_from_txt(DATA_PATH)
# d_H, d_W = 480, 640
# u_coord = np.tile(np.arange(0, d_W)[None, :], (d_H, 1))
# v_coord = np.tile(np.arange(0, d_H)[:, None], (1, d_W))
# fx = intrinsic_depth[0, 0]
# fy = intrinsic_depth[1, 1]
# cx = intrinsic_depth[0, 2]
# cy = intrinsic_depth[1, 2]
# total_rays_d_cam_z1 = np.stack([(u_coord - cx) / fx, (v_coord - cy) / fy, np.ones((d_H, d_W))], axis=-1)

# pt_list = []
# for frame_idx in [5100, 5125, 5150]:
#     frame_idx_str = str(frame_idx)
#     depth = np.load(f'/home/yyoshitake/works/DeepSDF/project/scannet/data2/scans/{scene_id}/depth/{frame_idx_str}.npy') / 1000
#     total_distance = depth * np.linalg.norm(total_rays_d_cam_z1, axis=-1)
#     total_rays_d_cam = total_rays_d_cam_z1 / np.linalg.norm(total_rays_d_cam_z1, axis=-1)[:, :, None]

#     cam_pt = (total_distance[:, :, None] * total_rays_d_cam).reshape(-1, 3)
#     pose = load_matrix_from_txt(f'/home/yyoshitake/works/DeepSDF/project/scannet/data2/scans/{scene_id}/pose/{frame_idx_str}.txt')
#     pose = Mscan @ pose # M_c2w = Mscan @ pose
#     cam_pt = np.concatenate([cam_pt, np.ones((cam_pt.shape[0], 1))], axis=-1)
#     wrd_pt = np.sum(cam_pt[:, None, :] * pose[None, :, :], axis=-1)
#     wrd_pt = wrd_pt[:, :3] / wrd_pt[:, 3:]
#     import pdb; pdb.set_trace()

#     pt_list.append(wrd_pt)

# fig = pylab.figure(figsize = [10, 10])
# ax = fig.add_subplot(projection='3d')
# c_list = ['g', 'c', 'm', 'r', 'b']
# for pt_idx, est_pt in enumerate(pt_list):
#     point = est_pt
#     ax.scatter3D(point[:, 0], point[:, 1], point[:, 2], c=c_list[pt_idx], s=0.5)
# ax.set_xlabel('x_value')
# ax.set_ylabel('y_value')
# ax.set_zlabel('z_value')
# ax.view_init(elev=0, azim=-90)
# fig.savefig('tes.png', dpi=300)
# pylab.close()
