import os
import sys
import numpy as np
import glob
import torch.utils.data as data
import torch
import tqdm
from often_use import pickle_load
from often_use import txt2list
import linecache
import torch.nn.functional as F



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
    mask_under_border = 1 / (cam_pos_obj.norm(dim=-1) + 1.0 * obj_scale.max(dim=-1).values * ddf.radius) # 良いのか...？
    est_mask = [map_i > border_i for map_i, border_i in zip(est_invdistance_map, mask_under_border)]
    est_mask = torch.stack(est_mask, dim=0)
    est_distance_map = torch.zeros_like(est_invdistance_map)
    est_distance_map[est_mask] = 1. / est_invdistance_map[est_mask]

    if with_invdistance_map:
        return est_invdistance_map, est_mask, est_distance_map
    else:
        return est_mask, est_distance_map



class scannet_dataset(data.Dataset):
    def __init__(self, args, mode, raw_data_list):
        self.mode = mode
        self.total_obs_num = args.total_obs_num
        self.total_obs_range = self.total_obs_num * 3
        self.check_tgt_obj_data_dir = None
        self.check_tgt_sampled_image_names = None
        self.canonical_path = args.canonical_data_path
        self.point_cloud_path = args.pt_path
        self.randn_from_log = False
        self.shape_code_dir = args.shape_code_dir
        self.scannet_view_selection = args.scannet_view_selection
        self.rs0 = np.random.RandomState(888)

        self.box_corner_vertices = np.array([
                                        [-1, -1, -1],
                                        [ 1, -1, -1],
                                        [ 1,  1, -1],
                                        [-1,  1, -1],
                                        [-1, -1,  1],
                                        [ 1, -1,  1],
                                        [ 1,  1,  1],
                                        [-1,  1,  1],
                                    ], dtype=np.float64)
        
        # DataListの確認
        self.data_list = []
        invalid_data_list = []
        self.mask_ratio_threshold = 0.02
        print('check each frames!!!')
        for data_i in tqdm.tqdm(raw_data_list):
            mask_ratio = pickle_load(os.path.join(data_i, 'total_infos.pickle'))['frame']['mask_ratio_list'] / 10000
            valid_frame_num = (mask_ratio > self.mask_ratio_threshold).sum()
            if valid_frame_num > 5:
                self.data_list.append(data_i)
            else:
                invalid_data_list.append(data_i)
        
        invalid_data_num = len(invalid_data_list)
        if invalid_data_num > 10:
            print(f"{invalid_data_num} data was invalid !!!")
        else:
            for data_i in invalid_data_list:
                print(data_i)

        # self.data_dir = os.path.join(args.scannet_data_dir, args.cat_id)
        # self.data_list = []
        # for scene_id in scene_list:
        #     scene_total_infos = pickle_load(os.path.join(self.data_dir, scene_id, 'scene_total_infos.pickle'))
        #     data_path_list = [os.path.join(self.data_dir, dir_name_list) for dir_name_list in scene_total_infos['dir_list']]
        #     self.data_list += data_path_list
        
    def __getitem__(self, index):
        # 物体データのディレクトリパス取得
        obj_data_dir = self.data_list[index]
        instance_id, obj_idx = obj_data_dir.split('/')[-1].split('_')

        # Total Infoの取得
        total_infos = pickle_load(os.path.join(obj_data_dir, 'total_infos.pickle'))

        # 全視点情報の取得
        total_infos_frame = total_infos['frame']
        cam_pos_wrd_list = total_infos_frame['cam_pos_wrd_list']
        w2c_list = total_infos_frame['w2c_list']
        img_name_list = total_infos_frame['img_name_list']
        mask_ratio_list = total_infos_frame['mask_ratio_list'] / 10000
        dist_med_list = total_infos_frame['dist_med_list']

        # 物体ポーズの取得
        total_infos_obj = total_infos['obj']
        obj_pos_wrd = total_infos_obj['obj_pos_wrd']
        o2w = total_infos_obj['o2w']
        obj_scale_wrd = total_infos_obj['obj_scale_wrd']
        sym_label = total_infos_obj['sym_label']
        model_bbox_shapenet = total_infos_obj['bbox']

        # 物体情報をReshapeしTorchに変換
        obj_pos_wrd = torch.from_numpy(obj_pos_wrd.astype(np.float32)).clone()[None, :].expand(self.total_obs_num, -1)
        o2w = torch.from_numpy(o2w.astype(np.float32)).clone()[None, :, :].expand(self.total_obs_num, -1, -1)
        obj_scale_wrd = torch.from_numpy(obj_scale_wrd.astype(np.float32)).clone()[None, :].expand(self.total_obs_num, -1)

        # 視点を選ぶ
        if self.scannet_view_selection == 'use_top_mask':
            # マスク領域のTop使用
            sampled_frame_idxes = np.argsort(-mask_ratio_list)[:self.total_obs_num]
        elif self.scannet_view_selection == 'rdn_sample_with_mask_latio':
            # 一定閾値以上からランダムに選択
            total_frame_num = mask_ratio_list.shape[0]
            frame_idx_list = np.arange(total_frame_num)
            valid_mask = mask_ratio_list > self.mask_ratio_threshold
            valid_index_list = frame_idx_list[valid_mask]
            sampled_frame_idxes = self.rs0.choice(valid_index_list, self.total_obs_num, replace=True) # 重複を許す
        else:
            # マスクの領域が広いTopK個のインデックス取得
            total_frame_num = mask_ratio_list.shape[0]
            if total_frame_num < self.total_obs_num:
                import pdb; pdb.set_trace()
            total_obs_range = min(total_frame_num, self.total_obs_range)
            topK_idx = np.argpartition(-mask_ratio_list, total_obs_range)[:total_obs_range]

            # TopK個のインデックスを取得し、
            sample_step = total_obs_range // self.total_obs_num
            sampled_frame_idxes = np.sort(topK_idx)[::sample_step]

        # 観測情報Dictの取得
        if self.check_tgt_sampled_image_names is None:
            sampled_image_names = img_name_list[sampled_frame_idxes]
        else:
            sampled_image_names = self.check_tgt_sampled_image_names
        frame_path_list = [os.path.join(obj_data_dir, 'data_dict', sampled_image_name + '.pickle') for sampled_image_name in sampled_image_names]
        frame_dict_list = [pickle_load(frame_path) for frame_path in frame_path_list]

        # 観測情報を１つのテンソルにまとめる
        mask = np.stack([frame_dict['clopped_mask'] for frame_dict in frame_dict_list], axis=0)
        fore_mask = np.stack([frame_dict['clopped_fore_mask'] for frame_dict in frame_dict_list], axis=0)
        distance_map = np.stack([frame_dict['clopped_distance'] for frame_dict in frame_dict_list], axis=0)
        rays_d_cam = np.stack([frame_dict['clopped_rays_d_cam'] for frame_dict in frame_dict_list], axis=0)
        cam_pos_wrd = np.stack([frame_dict['cam_pos_wrd'] for frame_dict in frame_dict_list], axis=0)
        w2c = np.stack([frame_dict['w2c'] for frame_dict in frame_dict_list], axis=0)
        bbox_list = np.stack([frame_dict['bbox_list'] for frame_dict in frame_dict_list], axis=0)
        bbox_diagonal = np.array([frame_dict['bbox_diagonal'] for frame_dict in frame_dict_list])

        # 観測情報をTorchに変換
        mask = torch.from_numpy(mask).clone()
        fore_mask = torch.from_numpy(fore_mask).clone()
        distance_map = torch.from_numpy(distance_map.astype(np.float32)).clone()
        rays_d_cam = torch.from_numpy(rays_d_cam.astype(np.float32)).clone()
        cam_pos_wrd = torch.from_numpy(cam_pos_wrd.astype(np.float32)).clone()
        w2c = torch.from_numpy(w2c.astype(np.float32)).clone()
        bbox_list = torch.from_numpy(bbox_list.astype(np.float32)).clone()
        bbox_diagonal = torch.from_numpy(bbox_diagonal.astype(np.float32)).clone()

        # 回転行列の取得
        o2c = w2c @ o2w
        obj_pos_cam = torch.sum((obj_pos_wrd - cam_pos_wrd)[..., None, :]*w2c, dim=-1)
        cam_pos_obj = - torch.sum(obj_pos_cam[..., None, :]*o2c.permute(0, 2, 1), dim=-1)

        # 1つめのカメラを世界座標に
        o2w = o2c[0, :, :][None, :, :].expand(self.total_obs_num, -1, -1) # １つ目のカメラ座標が世界座標に置き換わる
        obj_pos_wrd = o2w[0, :, :] @ (torch.zeros_like(cam_pos_obj[0, :]) - cam_pos_obj[0, :])
        obj_pos_wrd = obj_pos_wrd[None, :].expand(self.total_obs_num, -1)
        # cam_pos_wrd_ = torch.sum((cam_pos_wrd - cam_pos_wrd[0, :][None, :])[:, None, :] * w2c[0, :, :][None, :, :], axis=-1) # チェック用
        w2c = o2c @ o2w[0, :, :].T
        cam_pos_wrd = torch.sum(cam_pos_obj[:, None, :]*o2w[0, :, :][None, :, :], axis=-1) + obj_pos_wrd

        # 軸の取得
        obj_green_cam = o2c[:, :, 1] # Y
        obj_red_cam = o2c[:, :, 0] # X
        obj_green_wrd = o2w[:, :, 1] # Y
        obj_red_wrd = o2w[:, :, 0] # X

        # Get canonical maps.
        # if self.mode in {'val', 'tes'}:
        #     # scene_path_list = False
        #     canonical_path = os.path.join(self.canonical_path, instance_id + '.pickle')
        #     canonical_data_dict = pickle_load(canonical_path)
        #     canonical_distance_map = canonical_data_dict['depth_map'].astype(np.float32)
        #     canonical_camera_pos = canonical_data_dict['camera_pos'].astype(np.float32)
        #     canonical_camera_rot = canonical_data_dict['camera_rot'].astype(np.float32)
        # else:
        canonical_distance_map = canonical_camera_pos = canonical_camera_rot = False

        # BBoxの取得
        model_bbox_obj = torch.from_numpy(model_bbox_shapenet) # + loc

        # Get randn seeds.
        if self.randn_from_log:
            rand_seed = {}
            rand_seed['gt_S_seed'] = self.gt_S_seed[index]
            rand_seed['rand_P_seed'] = self.rand_P_seed[index]
            rand_seed['rand_S_seed'] = self.rand_S_seed[index]
            rand_seed['randn_theta_seed'] = self.randn_theta_seed[index]
            rand_seed['randn_axis_idx'] = self.randn_axis_idx[index]
        else:
            rand_seed = {}
            rand_seed['rand_P_seed'] = 'not_given'

        # Eval point clouds.
        # if self.mode in {'tes'}: # {'tes', 'val'}: # 
        #     gt_pc_obj = np.load(os.path.join(self.point_cloud_path, instance_id+'.npy')).astype(np.float32)
        # else:
        gt_pc_obj = False

        # 深度観測にマスク＋スケーリング
        distance_map[~mask] = 0.

        # GTの3DBboxの内側のみ使う
        OSMap_cam = distance_map[..., None] * rays_d_cam
        OSMap_wrd = torch.sum(OSMap_cam[..., None, :]*w2c.permute(0, 2, 1)[..., None, None, :, :], dim=-1) + cam_pos_wrd[..., None, None, :]
        OSMap_obj = torch.sum((OSMap_wrd - obj_pos_wrd[..., None, None, :])[..., None, :]*o2w.permute(0, 2, 1)[..., None, None, :, :], dim=-1) / obj_scale_wrd[..., None, None, :]
        bbox_mask_max = (OSMap_obj > 1.1 * model_bbox_obj.max(dim=0).values).any(dim=-1) # BBoxのMaxよりも大きい
        bbox_mask_min = (OSMap_obj < 1.1 * model_bbox_obj.min(dim=0).values).any(dim=-1) # BBoxのMinよりも小さい
        outside_bbox_mask = torch.logical_or(bbox_mask_max, bbox_mask_min)
        # check_map_torch(distance_map.reshape(-1, 128), 'tes_a.png')
        distance_map[outside_bbox_mask] = 0.
        # check_map_torch(distance_map.reshape(-1, 128), 'tes_r.png')

        # # BBoｘによるマスクの確認
        # OSMap_cam = distance_map[..., None] * rays_d_cam
        # OSMap_wrd = torch.sum(OSMap_cam[..., None, :]*w2c.permute(0, 2, 1)[..., None, None, :, :], dim=-1) + cam_pos_wrd[..., None, None, :]
        # OSMap_obj = torch.sum((OSMap_wrd - obj_pos_wrd[..., None, None, :])[..., None, :]*o2w.permute(0, 2, 1)[..., None, None, :, :], dim=-1) / obj_scale_wrd[..., None, None, :]
        # import pylab
        # fig = pylab.figure(figsize = [10, 10])
        # ax = fig.add_subplot(projection='3d')
        # c_list = ['g', 'c', 'm', 'r', 'b']
        # for pt_idx, (gt_pt_i, gt_mask_i) in enumerate(zip(OSMap_obj, mask)):
        #     point = gt_pt_i[gt_mask_i].to('cpu').detach().numpy().copy()
        #     ax.scatter3D(point[:, 0], point[:, 1], point[:, 2], c=c_list[pt_idx], s=0.5)
        # point = model_bbox_obj.to('cpu').detach().numpy().copy()
        # ax.scatter3D(point[:, 0], point[:, 1], point[:, 2], c='r', s=5)
        # point = 1.05 * model_bbox_obj.to('cpu').detach().numpy().copy()
        # ax.scatter3D(point[:, 0], point[:, 1], point[:, 2], c='g', s=5)
        # ax.set_xlim(-0.5, 0.5)
        # ax.set_ylim(-0.5, 0.5)
        # ax.set_zlim(-0.5, 0.5)
        # ax.set_xlabel('x_value')
        # ax.set_ylabel('y_value')
        # ax.set_zlabel('z_value')
        # ax.view_init(elev=0, azim=0)
        # fig.savefig(f'ptgtr_00_00.png', dpi=300)
        # ax.view_init(elev=0, azim=90)
        # fig.savefig(f'ptgtr_00_90.png', dpi=300)
        # ax.view_init(elev=90, azim=0)
        # fig.savefig(f'ptgtr_90_00.png', dpi=300)
        # pylab.close()

        # 形状コード
        gt_shape_code = np.load(os.path.join(self.shape_code_dir, instance_id + '.npy'))
        gt_shape_code = torch.from_numpy(gt_shape_code.astype(np.float32)).clone().reshape(-1)

        return mask, distance_map, instance_id, cam_pos_wrd, w2c, bbox_diagonal, bbox_list, rays_d_cam, obj_pos_wrd, o2w, \
            obj_green_wrd, obj_red_wrd, o2c, obj_green_cam, obj_red_cam, obj_scale_wrd, model_bbox_obj, canonical_distance_map, \
            canonical_camera_pos, canonical_camera_rot, gt_pc_obj, rand_seed, sym_label, frame_path_list, [path[-24:-7] for path in frame_path_list], gt_shape_code


    def __len__(self):
        return len(self.data_list)



# python scannet_dataset.py --config=./configs/paper_exp/chair/view5/txt.txt
if __name__=='__main__':
    from DDF.train_pl import DDF
    from often_use import check_map_torch, get_OSMap_obj

    from parser_get_arg import get_args
    args = get_args()

    #########################
    # 書き換える
    #########################
    args.cat_id = '03001627'
    args.ddf_model_path = 'DDF/lightning_logs/chair/cat_depth_mae_normal_mae_seed0_normal001_lr00001/checkpoints/0000010000.ckpt'
    args.total_obs_num = 5
    args.val_scene_list = ['scene0314_00']
    args.scannet_data_dir = 'scannet/results'
    tgt_obj_id = 'c585ee093bfd52af6512b7b24f3d84_002' # 'c585ee093bfd52af6512b7b24f3d84_002'
    check_tgt_sampled_image_names = ['00010_00000_00028', '00028_00000_00046', '00032_00000_00050', '00044_00000_00062', '00052_00000_00070'] # ['00003_00002_00488', '00010_00002_00495', '00021_00002_00506', '00027_00002_00512', '00030_00002_00515']
    check_tgt_obj_data_list = [os.path.join(args.scannet_data_dir, args.cat_id, args.val_scene_list[0], tgt_obj_id)]

    # scene0011_01, d6da5457b0682e24696b74614952b2d0_004, ['00098_00006_00238', '00068_00003_00086', '00055_00002_00072', '00038_00001_00054', '00043_00001_00059']
    # 'scene0314_00', 'c585ee093bfd52af6512b7b24f3d84_002', ['00005_00000_00023', '00012_00000_00030', '00022_00000_00040', '00032_00000_00050', '00042_00000_00060']

    # Set ddf.
    ddf = DDF(args)
    ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args)
    ddf.eval()
    ddf_instance_list = txt2list(args.ddf_instance_list_txt)


    # Make dummy data loader
    from torch.utils.data import DataLoader
    dummy_dataset = scannet_dataset(args, 'val', args.val_scene_list)
    dummy_dataloader = DataLoader(dummy_dataset, batch_size=1, num_workers=0)

    # 書き換える
    dummy_dataset.check_tgt_sampled_image_names = check_tgt_sampled_image_names
    dummy_dataset.data_list = check_tgt_obj_data_list

    for batch in dummy_dataloader:
        batch_idx = 0
        mask, distance_map, instance_id, cam_pos_wrd, w2c, bbox_diagonal, bbox_list, rays_d_cam, obj_pos_wrd, o2w, obj_green_wrd, obj_red_wrd, o2c, obj_green_cam, obj_red_cam, obj_scale_wrd, model_bbox_obj, \
        _, _, _, _, _, _, _, _ = batch
        batch_size = mask.shape[0]

        # Get shape code.
        instance_idx = [ddf_instance_list.index(instance_id_i) for instance_id_i in instance_id]
        gt_shape_code = ddf.lat_vecs(torch.tensor(instance_idx, device=ddf.device)).detach()
        gt_shape_code = gt_shape_code.expand(args.total_obs_num, -1)

        # Rendering
        est_mask, est_distance_map = render_distance_map_from_axis_for_scannet(H = args.input_H, 
                                                                    axis_green = obj_green_cam[batch_idx],
                                                                    axis_red = obj_red_cam[batch_idx],
                                                                    obj_scale = obj_scale_wrd[batch_idx], 
                                                                    obj_pos_wrd = obj_pos_wrd[batch_idx], 
                                                                    rays_d_cam = rays_d_cam[batch_idx], 
                                                                    input_lat_vec = gt_shape_code, 
                                                                    ddf = ddf, 
                                                                    cam_pos_wrd = cam_pos_wrd[batch_idx], 
                                                                    w2c = w2c[batch_idx], )
        
        # vis_map_i = torch.cat([distance_map[0, 0,:, :], 
        #                         est_distance_map[0,:, :], 
        #                         torch.abs(distance_map[0, 0, :, :] - est_distance_map[0,:, :])], dim=0)
        # check_map_torch(vis_map_i, f'tes.png')
        # vis_map_i = torch.cat([distance_map[0, 1,:, :], 
        #                         est_distance_map[1,:, :], 
        #                         torch.abs(distance_map[0, 1, :, :] - est_distance_map[1,:, :])], dim=0)
        # check_map_torch(vis_map_i, f'tes_.png')

        vis_map = []
        for map_idx in range(args.total_obs_num):
            vis_map_i = torch.cat([distance_map[0, map_idx,:, :], 
                                    est_distance_map[map_idx,:, :], 
                                    torch.abs(distance_map[0, map_idx, :, :] - est_distance_map[map_idx,:, :])], dim=0)
            vis_map.append(vis_map_i)

        # 点群の取得
        est_OSMap_obj = get_OSMap_obj(est_distance_map, est_mask, rays_d_cam[batch_idx], w2c[batch_idx], cam_pos_wrd[batch_idx], o2w[batch_idx], obj_pos_wrd[batch_idx], obj_scale_wrd[batch_idx])[..., :3]
        gt_mask = torch.logical_and(mask[batch_idx], distance_map[batch_idx] > 0)
        gt_OSMap_obj = get_OSMap_obj(distance_map[batch_idx], gt_mask, rays_d_cam[batch_idx], w2c[batch_idx], cam_pos_wrd[batch_idx], o2w[batch_idx], obj_pos_wrd[batch_idx], obj_scale_wrd[batch_idx])[..., :3]

        # 深度画像の可視化
        check_map_torch(torch.cat(vis_map, dim=1), f'dst.png')

        # 観測と予測点群の一致を確認
        import pylab
        fig = pylab.figure(figsize = [10, 10])
        ax = fig.add_subplot(projection='3d')
        for est_pt_i, est_mask_i, gt_pt_i, gt_mask_i in zip(est_OSMap_obj, est_mask, gt_OSMap_obj, gt_mask):
            point = est_pt_i[est_mask_i].to('cpu').detach().numpy().copy()
            ax.scatter3D(point[:, 0], point[:, 1], point[:, 2], c='m', s=0.5)
            point = gt_pt_i[gt_mask_i].to('cpu').detach().numpy().copy()
            ax.scatter3D(point[:, 0], point[:, 1], point[:, 2], c='g', s=0.5)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
        ax.set_xlabel('x_value')
        ax.set_ylabel('y_value')
        ax.set_zlabel('z_value')
        ax.view_init(elev=0, azim=0)
        fig.savefig(f'pt_00_00.png', dpi=300)
        ax.view_init(elev=0, azim=90)
        fig.savefig(f'pt_00_90.png', dpi=300)
        ax.view_init(elev=90, azim=0)
        fig.savefig(f'pt_90_00.png', dpi=300)
        pylab.close()

        # 点群の一貫性の確認
        import pylab
        fig = pylab.figure(figsize = [10, 10])
        ax = fig.add_subplot(projection='3d')
        c_list = ['g', 'c', 'm', 'r', 'b']
        for pt_idx, (est_pt_i, est_mask_i) in enumerate(zip(est_OSMap_obj, est_mask)):
            point = est_pt_i[est_mask_i].to('cpu').detach().numpy().copy()
            ax.scatter3D(point[:, 0], point[:, 1], point[:, 2], c=c_list[pt_idx], s=0.5)
        point = model_bbox_obj[0].to('cpu').detach().numpy().copy()
        ax.scatter3D(point[:, 0], point[:, 1], point[:, 2], c='r', s=5)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
        ax.set_xlabel('x_value')
        ax.set_ylabel('y_value')
        ax.set_zlabel('z_value')
        ax.view_init(elev=0, azim=0)
        fig.savefig(f'ptest_00_00.png', dpi=300)
        ax.view_init(elev=0, azim=90)
        fig.savefig(f'ptest_00_90.png', dpi=300)
        ax.view_init(elev=90, azim=0)
        fig.savefig(f'ptest_90_00.png', dpi=300)
        pylab.close()

        import pylab
        fig = pylab.figure(figsize = [10, 10])
        ax = fig.add_subplot(projection='3d')
        c_list = ['g', 'c', 'm', 'r', 'b']
        for pt_idx, (gt_pt_i, gt_mask_i) in enumerate(zip(gt_OSMap_obj, gt_mask)):
            point = gt_pt_i[gt_mask_i].to('cpu').detach().numpy().copy()
            ax.scatter3D(point[:, 0], point[:, 1], point[:, 2], c=c_list[pt_idx], s=0.5)
        point = model_bbox_obj[0].to('cpu').detach().numpy().copy()
        ax.scatter3D(point[:, 0], point[:, 1], point[:, 2], c='r', s=5)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
        ax.set_xlabel('x_value')
        ax.set_ylabel('y_value')
        ax.set_zlabel('z_value')
        ax.view_init(elev=0, azim=0)
        fig.savefig(f'ptgt_00_00.png', dpi=300)
        ax.view_init(elev=0, azim=90)
        fig.savefig(f'ptgt_00_90.png', dpi=300)
        ax.view_init(elev=90, azim=0)
        fig.savefig(f'ptgt_90_00.png', dpi=300)
        pylab.close()
        import pdb; pdb.set_trace()

        # # ポーズを世界座標系で評価：大丈夫なはず
        # def calc_rotation_diff(q, q00):
        #     rotation_dot = np.dot(quaternion.as_float_array(q00), quaternion.as_float_array(q))
        #     rotation_dot_abs = np.abs(rotation_dot)
        #     try:                                                                                                                                                                                                                                                                                                                      
        #         error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
        #     except:
        #         return 0.0
        #     error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
        #     error_rotation = np.rad2deg(error_rotation_rad)
        #     return error_rotation
        
        # valid_transformation_list = []
        # for batch_idx in range(batch_size):

        #     error_translation = np.linalg.norm(t - t_gt, ord=2)
        #     error_scale = 100.0*np.abs(np.mean(s/s_gt) - 1)

        #     # --> resolve symmetry
        #     if sym == "__SYM_ROTATE_UP_2":
        #         m = 2
        #         tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
        #         error_rotation = np.min(tmp)
        #     elif sym == "__SYM_ROTATE_UP_4":
        #         m = 4
        #         tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
        #         error_rotation = np.min(tmp)
        #     elif sym == "__SYM_ROTATE_UP_INF":
        #         m = 36
        #         tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
        #         error_rotation = np.min(tmp)
        #     else:
        #         error_rotation = calc_rotation_diff(q, q_gt)


        #     threshold_translation = 0.2 # <-- in meter
        #     threshold_rotation = 20 # <-- in deg
        #     threshold_scale = 20 # <-- in %

        #     is_valid_transformation = error_translation <= threshold_translation and error_rotation <= threshold_rotation and error_scale <= threshold_scale
        #     valid_transformation_list.append(is_valid_transformation)


        # # IoUの評価
        # # 回転は赤軸のみにする：FroDOとMOLTORの実装要確認
        # # GTのBBoxをどうやって作るか？：Scan2CAD見る

        # # https://pytorch3d.org/docs/iou3d
        # # https://github.com/AlienCat-K/3D-IoU-Python/blob/c07df684a31171fa4cbcb8ff0d50caddc9e99a13/3D-IoU-Python.py#L119
        # from pytorch3d.ops import box3d_overlap
        # # Assume inputs: boxes1 (M, 8, 3) and boxes2 (N, 8, 3)
        # intersection_vol, iou_3d = box3d_overlap(boxes1, boxes2)
        

        # # # 形状の評価
        # # まずはGTのスケールで評価してみる
