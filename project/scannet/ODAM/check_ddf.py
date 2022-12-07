import json
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import glob
import tqdm
import pylab
from segmentation_color import get_colored_segmap
from often_use import *
from load_scannet_data import export
sys.path.append("../../")
from parser_get_arg import *
from DDF.train_pl import DDF

SCANNET_DIR = '/home/yyoshitake/works/DeepSDF/project/scannet/ODAM/data2/scans'
annotations_dict = pickle_load('/home/yyoshitake/works/DeepSDF/project/scannet/ODAM/annotations_dict.pickle')


# ScanNet Scene_id
scan_id = 0
sub_scene_id = 0
scene_id = 'scene{}_{}'.format(str(0).zfill(4), str(0).zfill(2))
# Instance info.
category_incetance_id = '03001627_3289bcc9bf8f5dab48d8ff57878739ca' # '03001627_95d082c78ea0b1befe52ffd0e748a1ab' 
category_id = category_incetance_id.split('_')[0]
instance_id = category_incetance_id.split('_')[-1]
instance_infos = annotations_dict[scene_id]['cad'][category_incetance_id]
# Img info.
img_id = 0
pose = np.load(os.path.join(SCANNET_DIR, scene_id, category_incetance_id, 'pose', '{}.npy'.format(img_id)))
rays_d_cam = np.load(os.path.join(SCANNET_DIR, scene_id, category_incetance_id, 'rays_d_cam', '{}.npy'.format(img_id)))


##############################
T_o2w = np.array(instance_infos['trs']["translation"])
Q_o2w = np.array(instance_infos['trs']["rotation"])
S_o2w = np.array(instance_infos['trs']["scale"])
M_o2w = make_M_from_tqs(T_o2w, Q_o2w, S_o2w)
R_o2w = quaternion2rotation(Q_o2w)
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
check_map_torch(est_invdistance_map_obj_scale[0,:, :], f'tes_{img_id}.png')
import pdb; pdb.set_trace()



cad_file = f'/d/workspace/yyoshitake/ShapeNet/ShapeNetCore.v2/{category_id}/{instance_id}/models/model_normalized.obj'
cad_vertices = loadOBJ(cad_file) 
vertices = cad_vertices[..., :3]

fig = pylab.figure(figsize = [10, 10])
ax = fig.add_subplot(projection='3d')
point = vertices[::3, :]
ax.scatter(point[:, 0], point[:, 1], point[:, 2], c='c', s=0.5)
ax.scatter3D(point[:, 0], point[:, 1], point[:, 2], c='m', s=0.5)
ax.set_xlabel('x_value')
ax.set_ylabel('y_value')
ax.set_zlabel('z_value')
ax.view_init(elev=0, azim=90)
fig.savefig('tes.png', dpi=300)
pylab.close()