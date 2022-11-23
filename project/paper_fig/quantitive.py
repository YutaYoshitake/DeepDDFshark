import matplotlib.pyplot as plt
import tqdm
import sys
import os
import numpy as np
import glob
import datetime
import cv2
sys.path.append("../")
from often_use import *
from parser_get_arg import *
from DDF.train_pl import DDF
from renderer import get_rot_views



# python renderer.py --config=../configs/paper_exp/chair/view5/tes.txt
if __name__=='__main__':
    # Get args
    args = get_args()
    args.ddf_model_path = '/home/yyoshitake/works/DeepSDF/project/DDF/lightning_logs/chair/cat_depth_mae_normal_mae_seed0_normal001_lr00001/checkpoints/0000010000.ckpt'
    H = 256
    Fov = 53
    rays_d_cam = get_ray_direction(H, Fov)
    
    # Set ddf.
    ddf = DDF(args)
    ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args)
    ddf.eval()


    # Configs.
    result_dir = '/home/yyoshitake/works/DeepSDF/project/paper_fig/depth_map/result'
    dt_now = datetime.datetime.now()
    result_dir = '/home/yyoshitake/works/DeepSDF/project/paper_fig/normal_map/results'
    time_log = dt_now.strftime('%Y_%m_%d_%H_%M_%S')
    os.mkdir(os.path.join(result_dir, time_log))

    gray_scale = 0.85
    mask_gray_color = 70
    interpolate_freq = 3
    origin_ins_num = 3
    freq = 1
    lat_deg = 30
    cam_pos, w2c = get_rot_views(lat_deg, 10, ddf)
    cam_pos_gtobj = cam_pos[2][None, :].expand(freq, -1)
    w2c = w2c[2][None, :, :].expand(freq, -1, -1)
    rays_d_cam = rays_d_cam.expand(1, -1, -1, -1)
    
    total_map = []
    gt_pickle_list = glob.glob('normal_map/gt/*.pickle')
    gt_id_list = [gt_pickle_path.split('/')[-1].split('.')[0] for gt_pickle_path in gt_pickle_list]
    est_list = glob.glob('normal_map/src/autoreg/*.pickle')
    for map_i, est_path in enumerate(tqdm.tqdm(est_list)):
        est_dict = pickle_load(est_path)
        est_instance_id = est_path.split('/')[-1].split('.')[0]
        gt_dict = pickle_load(gt_pickle_list[gt_id_list.index(est_instance_id.split('_')[-1])])

        gtobj_pos_wrd = torch.from_numpy(est_dict['gt_pos'].astype(np.float32)).clone()[None, ...].to(w2c)
        gt_o2w = torch.from_numpy(est_dict['gt_o2w'].astype(np.float32)).clone()[None, ...].to(w2c)
        gt_scale = torch.from_numpy(est_dict['gt_scale'].astype(np.float32)).clone()[None, ...].to(w2c)
        
        estobj_pos_wrd = torch.from_numpy(est_dict['est_pos'].astype(np.float32)).clone()[None, ...].to(w2c)
        est_o2w = torch.from_numpy(est_dict['est_o2w'].astype(np.float32)).clone()[None, ...].to(w2c)
        est_scale = torch.from_numpy(est_dict['est_scale'].astype(np.float32)).clone()[None, ...].to(w2c)
        input_lat_vec = torch.from_numpy(est_dict['est_shape'].astype(np.float32)).clone()[None, ...].to(w2c)

        o2c = w2c
        o2c = torch.bmm(est_o2w.permute(0, 2, 1), torch.bmm(gt_o2w, o2c))
        obj_scale = gt_scale / est_scale
        cam_pos_wrd = torch.sum((cam_pos_gtobj - gtobj_pos_wrd)[..., None, :] * gt_o2w.permute(0, 2, 1), dim=-1)
        cam_pos_obj = (torch.sum(cam_pos_wrd[..., None, :] * est_o2w, dim=-1) + estobj_pos_wrd) / obj_scale # cam_pos_estobj

        est_normal_map, est_distance_map, est_mask = \
            render_normal_map_from_afar(H, o2c, cam_pos_obj, obj_scale[:, 0], rays_d_cam, input_lat_vec, ddf)
        est_normal_map = est_normal_map.to('cpu').detach().numpy().copy()
        est_mask = est_mask.to('cpu').detach().numpy().copy()
        normal_img = np.tile((255*gray_scale*np.clip(-est_normal_map[0, ..., -1], 0, 1)**(1/2.2)).astype(np.uint8)[..., None], (1, 1, 3))

        #########################
        #     Ground truth      #
        #########################
        est_batch_idx = est_instance_id.split('_')[0]
        obs_map_path = sorted(glob.glob(f'depth_map/result/2022_11_01_13_05_15/gt_{est_batch_idx}_*_00.png'))
        obs_map_list = [cv2.imread(map_path) for map_path in obs_map_path]

        side_buffer = 40
        total_map = 255 * np.ones((H, H + side_buffer, 3), dtype=np.uint8)

        obs_mapid_list = [0, 1, -1]
        obs_map_size = 60
        obs_map_gap = 15
        obs_pos_list = [[0, obs_map_size], 
                        [obs_map_size+obs_map_gap, 2*obs_map_size+obs_map_gap], 
                        [-obs_map_size, None]]
        for map_idx, obs_map_idx in enumerate(obs_mapid_list):
            total_map[obs_pos_list[map_idx][0]:obs_pos_list[map_idx][-1], :obs_map_size] \
                 = cv2.resize(obs_map_list[obs_map_idx], (obs_map_size, obs_map_size))

        three_points_img = cv2.imread('a/three_points.png')
        obs_start = obs_pos_list[-2][-1]
        obs_end = H + obs_pos_list[-1][0]
        three_points_size = max(obs_map_size, obs_start - obs_end)
        three_points_Hstart = obs_start + (obs_end - obs_start - three_points_size) // 2
        three_points_Wstart = (obs_map_size - three_points_size) // 2
        total_map[three_points_Hstart:three_points_Hstart+three_points_size, three_points_Wstart:three_points_Wstart+three_points_size] \
            = cv2.resize(three_points_img, (three_points_size, three_points_size))

        gt_normal_cam = np.sum(gt_dict['normal_map'][..., None, :] * w2c[:, None, :, :].to('cpu').detach().numpy().copy(), axis=-1)
        gt_nz = gt_normal_cam[..., -1]
        gt_normal_img = np.tile((255*gray_scale*np.clip(-gt_nz,0,1)**(1/2.2)).astype(np.uint8)[..., None], (1, 1, 3))
        total_map[:, -H:, :][gt_dict['mask']] = gt_normal_img[gt_dict['mask']]
        cv2.imwrite(os.path.join(result_dir, time_log, est_instance_id + '_gt.png'), total_map)

        #########################
        #          Est          #
        #########################
        total_map = 255 * np.ones((H, H + side_buffer, 3), dtype=np.uint8)
        
        dif_map_path = sorted(glob.glob(f'depth_map/result/2022_11_01_13_05_15/dif_{est_batch_idx}_*_04.png'))
        dif_map_list = [cv2.imread(map_path) for map_path in dif_map_path]
        # for map_idx, dif_map_idx in enumerate(obs_mapid_list):
        #     total_map[obs_pos_list[map_idx][0]:obs_pos_list[map_idx][-1], :obs_map_size] \
        #          = cv2.resize(dif_map_list[dif_map_idx], (obs_map_size, obs_map_size))
        # total_map[three_points_Hstart:three_points_Hstart+three_points_size, three_points_Wstart:three_points_Wstart+three_points_size] \
        #     = cv2.resize(three_points_img, (three_points_size, three_points_size))
        total_map_wmask = total_map.copy()

        total_map_wmask[:, -H:, :][gt_dict['mask']] = np.array([mask_gray_color, mask_gray_color, mask_gray_color])
        total_map_wmask[:, -H:, :][est_mask[0]] = normal_img[est_mask[0]]
        total_map[:, -H:, :][est_mask[0]] = normal_img[est_mask[0]]
        
        # cv2.imwrite(os.path.join(result_dir, time_log, est_instance_id + '.png'), total_map)
        # cv2.imwrite(os.path.join(result_dir, time_log, est_instance_id + 'wmask.png'), total_map_wmask)

        # total_map[-87:, :87, :] = cv2.resize(total_map_wmask[:, -H:, :], (87, 87))
        cv2.imwrite(os.path.join(result_dir, time_log, est_instance_id + '.png'), total_map)
