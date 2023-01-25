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

if __name__=='__main__':
    # 椅子の場合
    # CUDA_VISIBLE_DEVICES=0 python quantitive.py --config=../configs/paper_exp/chair_re/view5/rdn.txt
    est_pickle_dir = '/home/yyoshitake/works/DeepSDF/project/paper_fig/normal_map/src/chair_re/2023_01_18_13_20_40_rdn_autoreg_epo_0000000128'
    Fov = 53
    
    # Display
    # CUDA_VISIBLE_DEVICES=0 python quantitive.py --config=../configs/paper_exp/display/view5/rdn.txt
    est_pickle_dir = '/home/yyoshitake/works/DeepSDF/project/paper_fig/normal_map/src/display/2023_01_19_00_16_47_rdn_autoreg_prg_epo_0000000512'
    Fov = 57
    
    # Cabinet
    Fov = 57
    
    # Table
    Fov = 60

    # Get args
    args = get_args()
    args.ddf_model_path = args.ddf_model_path
    H = 256
    rays_d_cam = get_ray_direction(H, Fov)
    
    # Set ddf.
    ddf = DDF(args)
    ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args) #.to(device)
    ddf.eval()

    # 保存
    exp_name = est_pickle_dir.split('/')[-1]
    cat_name = est_pickle_dir.split('/')[-2]
    dt_now = datetime.datetime.now()
    result_dir = '/home/yyoshitake/works/DeepSDF/project/paper_fig/normal_map/results'
    time_log = '_'.join([dt_now.strftime('%Y_%m_%d_%H_%M_%S')] + exp_name.split('_')[-4:])
    os.makedirs(os.path.join(result_dir, cat_name, time_log), exist_ok=True)

    # Configs.
    gray_scale = 0.85
    mask_gray_color = 70
    interpolate_freq = 3
    origin_ins_num = 3
    freq = 1
    lat_deg = 30
    cam_pos, gto2c = get_rot_views(lat_deg, 10, ddf)
    cam_pos_gtobj = cam_pos[8][None, :].expand(freq, -1) #.to(device)
    gto2c = gto2c[8][None, :, :].expand(freq, -1, -1) #.to(device)
    rays_d_cam = rays_d_cam.expand(1, -1, -1, -1) #.to(device)

    # 生成開始
    total_map = []
    gt_pickle_list = glob.glob(f'normal_map/gt/{cat_name}/*.pickle')
    gt_id_list = [gt_pickle_path.split('/')[-1].split('.')[0] for gt_pickle_path in gt_pickle_list]
    est_list = glob.glob(est_pickle_dir + '/*')
    for map_i, est_path in enumerate(tqdm.tqdm(est_list)):
    
        est_dict = pickle_load(est_path)
        est_instance_id = est_path.split('/')[-1].split('.')[0]
        gt_dict = pickle_load(gt_pickle_list[gt_id_list.index(est_instance_id.split('_')[-1])])

        gtobj_pos_wrd = torch.from_numpy(est_dict['gt_pos'].astype(np.float32)).clone()[None, ...].to(gto2c)
        gt_o2w = torch.from_numpy(est_dict['gt_o2w'].astype(np.float32)).clone()[None, ...].to(gto2c)
        gt_scale = torch.from_numpy(est_dict['gt_scale'].astype(np.float32)).clone()[None, ...].to(gto2c)
        
        estobj_pos_wrd = torch.from_numpy(est_dict['est_pos'].astype(np.float32)).clone()[None, ...].to(gto2c)
        est_o2w = torch.from_numpy(est_dict['est_o2w'].astype(np.float32)).clone()[None, ...].to(gto2c)
        est_scale = torch.from_numpy(est_dict['est_scale'].astype(np.float32)).clone()[None, ...].to(gto2c)
        input_lat_vec = torch.from_numpy(est_dict['est_shape'].astype(np.float32)).clone()[None, ...].to(gto2c)

        w2c = torch.bmm(gto2c, gt_o2w.permute(0, 2, 1))
        gt_o2c = torch.bmm(w2c, gt_o2w)
        cam_pos_wrd = torch.sum(cam_pos_gtobj[..., None, :] * gt_o2w, dim=-1) * gt_scale + gtobj_pos_wrd

        est_o2c = torch.bmm(w2c, est_o2w)
        obj_pos_cam = torch.sum((estobj_pos_wrd - cam_pos_wrd)[..., None, :] * w2c, dim=-1)
        cam_pos_obj = - torch.sum(obj_pos_cam[..., None, :]*est_o2c.permute(0, 2, 1), dim=-1) / est_scale # cam_pos_gtobj # 
        est_normal_map, est_distance_map, est_mask = \
            render_normal_map_from_afar(H, est_o2c, cam_pos_obj, est_scale[:, 0], rays_d_cam, input_lat_vec, ddf)
        # est_normal_map, est_distance_map, est_mask = \
        #   render_normal_map_from_afar(H, o2c, cam_pos_obj, obj_scale[:, 0], rays_d_cam, input_lat_vec, ddf)
        est_normal_map = est_normal_map.to('cpu').detach().numpy().copy()
        est_mask = est_mask.to('cpu').detach().numpy().copy()
        normal_img = np.tile((255*gray_scale*np.clip(-est_normal_map[0, ..., -1], 0, 1)**(1/2.2)).astype(np.uint8)[..., None], (1, 1, 3))

        #########################
        #         観測          #
        #########################
        est_batch_idx = est_instance_id.split('_')[0]
        obs_mapid_list = [0, 1, -1]
        top_bottom_buffer = 8
        obs_map_size = 65
        obs_map_gap = 12
        total_map = 255 * np.ones((H, obs_map_size, 3), dtype=np.uint8)
        obs_pos_list = [[top_bottom_buffer, top_bottom_buffer+obs_map_size], [top_bottom_buffer+obs_map_size+obs_map_gap, top_bottom_buffer+2*obs_map_size+obs_map_gap], [-top_bottom_buffer-obs_map_size, -top_bottom_buffer]]
        for map_idx, obs_map_idx in enumerate(obs_mapid_list):
            distance_map_i = est_dict['obs_dist_map'][obs_map_idx]
            d_max = distance_map_i.max()
            d_min = distance_map_i.min()
            vis_d = (distance_map_i - d_min) / (d_max - d_min)
            vis_d = (255 * np.clip(vis_d, 0.0, 1.0)).astype(np.uint8)
            vis_d = cv2.applyColorMap(vis_d, cv2.COLORMAP_VIRIDIS)
            total_map[obs_pos_list[map_idx][0]:obs_pos_list[map_idx][-1], :obs_map_size] \
                 = cv2.resize(vis_d, (obs_map_size, obs_map_size))

        three_points_img = cv2.imread('a/three_points.png')
        obs_start = obs_pos_list[-2][-1]
        obs_end = H + obs_pos_list[-1][0]
        three_points_size = min(obs_map_size, obs_end - obs_start)
        three_points_Hstart = obs_start + (obs_end - obs_start - three_points_size) // 2
        three_points_Wstart = (obs_map_size - three_points_size) // 2
        total_map[three_points_Hstart:three_points_Hstart+three_points_size, three_points_Wstart:three_points_Wstart+three_points_size] \
            = cv2.resize(three_points_img, (three_points_size, three_points_size))
        # cv2.imwrite(os.path.join(result_dir, cat_name, time_log, est_instance_id + '_obs.png'), total_map)
        obs_maps = total_map.copy()

        #########################
        #         真値          #
        #########################
        gt_normal_cam = np.sum(gt_dict['normal_map'][..., None, :] * gto2c[:, None, :, :].to('cpu').detach().numpy().copy(), axis=-1)
        gt_nz = gt_normal_cam[..., -1]
        gt_normal_img = np.tile((255*gray_scale*np.clip(-gt_nz,0,1)**(1/2.2)).astype(np.uint8)[..., None], (1, 1, 3))
        # total_map[:, -H:, :][gt_dict['mask']] = gt_normal_img[gt_dict['mask']]
        # cv2.imwrite(os.path.join(result_dir, cat_name, time_log, est_instance_id + '_gt.png'), total_map)
        gt_normal_img[~gt_dict['mask']] = 255
        result = np.concatenate([obs_maps, gt_normal_img[:, 10:]], axis=1)
        cv2.imwrite(os.path.join(result_dir, cat_name, time_log, est_instance_id + '_gt.png'), result)

        # #########################
        # #          Est          #
        # #########################
        side_buffer = 55
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
        
        # cv2.imwrite(os.path.join(result_dir, cat_name, time_log, est_instance_id + '.png'), total_map)
        # cv2.imwrite(os.path.join(result_dir, cat_name, time_log, est_instance_id + 'wmask.png'), total_map_wmask)
        dif_mask = 110
        map_wmask_mask = cv2.resize(255 * np.logical_or(est_mask[0], gt_dict['mask']).astype(np.uint8), (dif_mask, dif_mask))
        map_wmask_mask = map_wmask_mask > 100
        total_map[-dif_mask:, :dif_mask, :][map_wmask_mask] = cv2.resize(total_map_wmask[:, -H:, :], (dif_mask, dif_mask))[map_wmask_mask]
        cv2.imwrite(os.path.join(result_dir, cat_name, time_log, est_instance_id + '.png'), total_map)
        import pdb; pdb.set_trace()
