import os
import sys
import tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.append("../")
from parser_get_arg import *
from DDF.train_pl import DDF
from often_use import check_map_torch, get_OSMap_obj
from scannet_dataset import scannet_dataset, render_distance_map_from_axis_for_scannet

args = get_args()





#########################
# 書き換える
#########################
# CUDA_VISIBLE_DEVICES=0 python check_annotation.py --config=../configs/paper_exp/chair_re/view5/rdn.txt
args.cat_id = '03001627'
args.total_obs_num = 5
args.scannet_data_dir = 'results_chair_tes'
out_dir_name = 'train' # 'ValTest'
# check_txt_list = [
#                     f'/home/yyoshitake/works/DeepSDF/project/scannet/{args.scannet_data_dir}/match_another_shapecat.txt', 
#                     f'/home/yyoshitake/works/DeepSDF/project/scannet/{args.scannet_data_dir}/match_one_not_in_shapecat.txt', 
#                     f'/home/yyoshitake/works/DeepSDF/project/scannet/{args.scannet_data_dir}/unmatched_init_one.txt', 
#                     ]
check_txt_list = [f'/home/yyoshitake/works/DeepSDF/project/scannet/{args.scannet_data_dir}/match_removed.txt']
args.shape_code_dir = '/home/yyoshitake/works/DeepSDF/project/scannet/gt_latent_code/03001627'



# tgt_ins_list = []
# for check_txt in check_txt_list:
#     ins_list = txt2list(check_txt)
#     ins_path_list = [
#             os.path.join(args.scannet_data_dir, ins_list_i.split(':')[1][:8], ins_list_i.split(':')[0], ins_list_i.split(':')[1][9:]) 
#             for ins_list_i in ins_list]
#     tgt_ins_list += ins_path_list
tgt_ins_list = [
    '/home/yyoshitake/works/DeepSDF/project/scannet/results_chair_tes/03001627/scene0500_01/d2fe67741e0f3cc845613f5c2df1029a_032', 
    ]
check_tgt_sampled_image_names = ['00003_00000_00003']
# import pdb; pdb.set_trace()

# 保存用
out_dir_path = f'check_annotations/{args.cat_id}/{out_dir_name}'
if not os.path.exists(out_dir_path):
    os.mkdir(out_dir_path)

# Set ddf.
ddf = DDF(args)
ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args)
ddf.eval()
ddf_instance_list = txt2list(args.ddf_instance_list_txt)


# Make dummy data loader
args.scannet_view_selection = 'use_top_mask'
args.use_top_mask = True
dummy_dataset = scannet_dataset(args, 'val', tgt_ins_list)
dummy_dataloader = DataLoader(dummy_dataset, batch_size=1, num_workers=0)
# dummy_dataset.data_list = tgt_ins_list
dummy_dataset.total_obs_num = 1
dummy_dataset.check_tgt_sampled_image_names = check_tgt_sampled_image_names


for batch in tqdm.tqdm(dummy_dataloader):
    frame_idx = 0
    mask, distance_map, instance_id, cam_pos_wrd, w2c, bbox_diagonal, bbox_list, rays_d_cam, obj_pos_wrd, o2w, obj_green_wrd, obj_red_wrd, o2c, obj_green_cam, obj_red_cam, obj_scale_wrd, model_bbox_obj, \
    _, _, _, _, _, _, scene_path, _, gt_shape_code, _, _ = batch
    batch_size = mask.shape[0]

    # Rendering
    est_mask, est_distance_map = render_distance_map_from_axis_for_scannet(H = args.input_H, 
                                                                axis_green = obj_green_cam[:, frame_idx],
                                                                axis_red = obj_red_cam[:, frame_idx],
                                                                obj_scale = obj_scale_wrd[:, frame_idx], 
                                                                obj_pos_wrd = obj_pos_wrd[:, frame_idx], 
                                                                rays_d_cam = rays_d_cam[:, frame_idx], 
                                                                input_lat_vec = gt_shape_code, 
                                                                ddf = ddf, 
                                                                cam_pos_wrd = cam_pos_wrd[:, frame_idx], 
                                                                w2c = w2c[:, frame_idx], )
    
    # vis_map_i = torch.cat([distance_map[0, 0,:, :], 
    #                         est_distance_map[0,:, :], 
    #                         torch.abs(distance_map[0, 0, :, :] - est_distance_map[0,:, :])], dim=0)
    # check_map_torch(vis_map_i, f'tes.png')

    scene_path = scene_path[0][frame_idx]
    img_path = '/'.join(scene_path.split('/')[:-2] + ['video'] + [scene_path.split('/')[-1].split('.')[0] + '.png'])
    video_frame = cv2.imread(img_path)

    est_distance_map = est_distance_map[0].to('cpu').detach().numpy().copy()
    if (est_distance_map > 0).sum() > 0:
        d_max = est_distance_map[est_distance_map>0].max()
        d_min = est_distance_map[est_distance_map>0].min()
    else:
        d_max = 0
        d_min = 1
    vis_d = (est_distance_map - d_min) / (d_max - d_min)
    vis_d = (255 * np.clip(vis_d, 0.0, 1.0)).astype(np.uint8)
    vis_d = cv2.applyColorMap(vis_d, cv2.COLORMAP_JET)

    # 画像を保存
    scene_id = img_path.split('/')[2]
    obj_id = img_path.split('/')[3]
    img_path = f'{out_dir_path}/{scene_id}_{obj_id}.png'
    # cv2.imwrite(img_path, np.concatenate([video_frame, vis_d], axis=1))
    cv2.imwrite('tes.png', np.concatenate([video_frame, vis_d], axis=1))
