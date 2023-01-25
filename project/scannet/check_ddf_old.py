# python check_ddf.py --config=../configs/paper_exp/chair/view5/txt.txt

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import glob
import tqdm
import pylab
# from segmentation_color import get_colored_segmap
from often_use import *
from load_scannet_data import export
import sys
sys.path.append("../")
from parser_get_arg import *
from DDF.train_pl import DDF

SCANNET_DIR = '/home/yyoshitake/works/DeepSDF/project/scannet/ODAM/data2/scans'

# ScanNet Scene_id
scene_id = 'scene0000_01'
target_shapenet_cat = {'03001627': 'chair', }

# Full scan2cad annotationsの読み込み
annotations_json_open = open('data2/scan2cad/full_annotations.json', 'r')
annotations_json_load = json.load(annotations_json_open)
# Prepare Annotation dictを作成
annotations_dict = {}
for annotation_i in annotations_json_load:
    scan_id_i = annotation_i['id_scan']
    annotations_dict[scan_id_i] = {}
    annotations_dict[scan_id_i]['trs'] = annotation_i['trs'] # <-- transformation from scan space to world space
    annotations_dict[scan_id_i]['cad'] = {}
    # annotation_i['aligned_models']にはリスト形式でCADモデルが格納されている
    if scan_id_i == scene_id:
        for cad_idx, cad_i in enumerate(annotation_i['aligned_models']):
            catid_cad = cad_i['catid_cad']
            id_cad = cad_i['id_cad']
            if catid_cad in target_shapenet_cat.keys():
                total_id = f'{catid_cad}_{id_cad}_{str(cad_idx).zfill(3)}'
                annotations_dict[scan_id_i]['cad'][total_id] = {}
                annotations_dict[scan_id_i]['cad'][total_id]['trs'] = cad_i['trs'] # <-- transformation from CAD space to world space 
                annotations_dict[scan_id_i]['cad'][total_id]['sym'] = cad_i['sym']

# Instance info.
category_incetance_id = '03001627_3289bcc9bf8f5dab48d8ff57878739ca_007' # '03001627_3289bcc9bf8f5dab48d8ff57878739ca_016'
category_id = category_incetance_id.split('_')[0]
instance_id = category_incetance_id.split('_')[1]
obj_id = category_incetance_id.split('_')[2]
instance_infos = annotations_dict[scene_id]['cad'][category_incetance_id]
# Img info.
img_id = '00026_00002_00515' # '00045_00045_00390' # '00035_00035_00380'

# Img info.
out_dir = 'results'
data_dict = pickle_load(os.path.join(out_dir, f'{category_id}/{scene_id}/{instance_id}_{obj_id}/data_dict/{img_id}.pickle'))

# # 並進のズレの取得
loc_dict = pickle_load(os.path.join(out_dir, 'loc_dict.pickle'))
loc = loc_dict[f'{category_id}_{instance_id}']
pose = data_dict['pose']
rays_d_cam = data_dict['clopped_rays_d_cam']
distance = data_dict['clopped_distance']


##############################
T_o2w = np.array(instance_infos['trs']["translation"])
Q_o2w = np.array(instance_infos['trs']["rotation"])
S_o2w = np.array(instance_infos['trs']["scale"])
M_o2w = make_M_from_tqs(T_o2w, Q_o2w, S_o2w)
R_o2w = quaternion2rotation(Q_o2w)
T_o2w = np.array(instance_infos['trs']["translation"]) - R_o2w @ loc * S_o2w
##############################
T_c2w = pose[:3, 3]
R_c2w = pose[:3, :3]
##############################
T_o2w = torch.from_numpy(T_o2w.astype(np.float32)).clone()
S_o2w = torch.from_numpy(S_o2w.astype(np.float32)).clone()
R_o2w = torch.from_numpy(R_o2w.astype(np.float32)).clone()
T_c2w = torch.from_numpy(T_c2w.astype(np.float32)).clone()
R_c2w = torch.from_numpy(R_c2w.astype(np.float32)).clone()
rays_d_cam = torch.from_numpy(rays_d_cam.astype(np.float32)).clone()
distance = torch.from_numpy(distance.astype(np.float32)).clone()
##############################


# Get args
args = get_args()
args.ddf_model_path = '/home/yyoshitake/works/DeepSDF/project/DDF/lightning_logs/chair/cat_depth_mae_normal_mae_seed0_normal001_lr00001/checkpoints/0000010000.ckpt'

# Set ddf.
ddf = DDF(args)
ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args)
ddf.eval()
ddf_instance_list = txt2list(args.ddf_instance_list_txt)

# Get shape code.
instance_idx = [ddf_instance_list.index(instance_id)] # _i) for instance_id_i in instance_id]
gt_shape_code = ddf.lat_vecs(torch.tensor(instance_idx, device=ddf.device)).detach()



# まずはスケール１でやる
cam_pos_cam = torch.zeros_like(T_c2w)
cam_pos_wrd = torch.sum((cam_pos_cam)[..., None, :]*R_c2w.permute(1, 0), dim=-1) + T_c2w
cam_pos_obj = torch.sum((cam_pos_wrd-T_o2w)[..., None, :]*R_o2w.permute(1, 0), dim=-1) / S_o2w # あってる？
rays_d_wrd = torch.sum(rays_d_cam[..., None, :]*R_c2w[None, None, :, :], dim=-1)
rays_d_obj = torch.sum(rays_d_wrd[..., None, :]*R_o2w[None, None, :, :].permute(0, 1, 3, 2), dim=-1)
rays_d_obj = rays_d_obj / S_o2w[None, None, :] # あってる？
rays_d_obj = rays_d_obj / torch.norm(rays_d_obj, dim=-1)[..., None]

# Get rays inputs.
rays_d = rays_d_obj[None, :, :, :]
rays_o = cam_pos_obj[None, None, None, :].expand(-1, rays_d_cam.shape[0], rays_d_cam.shape[1], -1)
input_lat_vec = gt_shape_code

# Estimating.
est_invdistance_map_obj_scale, negative_D_mask = ddf.forward_from_far(rays_o, rays_d, input_lat_vec)
check_map_torch(torch.cat([distance, est_invdistance_map_obj_scale[0,:, :], torch.abs(distance-est_invdistance_map_obj_scale[0,:, :])], dim=0), f'tes_{img_id}.png')
import pdb; pdb.set_trace()