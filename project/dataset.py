import os
import sys
import numpy as np
import glob
import torch.utils.data as data
import torch
import tqdm
from often_use import pickle_load





class TaR_dataset(data.Dataset):
    def __init__(self, args, mode, instance_list_txt, data_dir, scene_list_path):

        scenenum_log_txt = os.path.join(('/').join(data_dir.split('/')[:-1]), 'rendered_instances.txt')
        scenenum_log = {}
        if os.path.isfile(scenenum_log_txt):
            with open(scenenum_log_txt, 'r') as f:
                instance_scene_nums = f.read().splitlines()
                for max_scene_info in instance_scene_nums:
                    instance_id, scene_num = max_scene_info.split(' : ')
                    scenenum_log[instance_id] = int(scene_num)

        self.mode = mode
        self.view_position = args.view_position
        self.total_obs_num = args.total_obs_num
        self.scene_pos_array_path ='/d/workspace/yyoshitake/moving_camera/volumetric/instance_infos/revised/'
        self.canonical_path = args.canonical_data_path
        
        self.data_list = []
        if  self.mode in {'train', 'make_val_list', 'make_tes_list'}:
            with open(instance_list_txt, 'r') as f:
                instances = f.read().splitlines()
                for instance_id in instances:
                    instance_path = os.path.join(data_dir, instance_id.rstrip('\n'))
                    # Get max scene num for each instance.
                    if instance_id in scenenum_log.keys():
                        scene_max_num = scenenum_log[instance_id]
                    else:
                        scene_max_num = len(glob.glob(os.path.join(instance_path, '*.pickle')))
                    # Append to datalist.
                    if self.view_position == 'continuous':
                        scene_pos_array = np.load(self.scene_pos_array_path + instance_id + '.npy')
                        self.data_list.append([instance_path, scene_max_num, scene_pos_array])
                    elif self.view_position == 'randn':
                        self.data_list.append([instance_path, scene_max_num])
        
        del scenenum_log_txt, scenenum_log

        if self.mode in {'val', 'tes'}:
            self.data_list = pickle_load(scene_list_path)
        # self.data_list = self.data_list[:5000] # self.data_list[:15000]

        self.input_H = args.input_H
        self.input_W = args.input_W
        if self.input_H != self.input_W:
            print('Non-supported image shapes!')
            sys.exit()
        x_coord = np.tile(np.arange(0, self.input_H)[None, :], (self.input_W, 1)).astype(np.float32)
        y_coord = np.tile(np.arange(0, self.input_W)[:, None], (1, self.input_H)).astype(np.float32)
        self.image_coord = np.stack([y_coord, x_coord], axis=-1)

        self.randn_from_log = not args.pickle_to_check_qantitive_results=='not_given'
        if self.randn_from_log:
            print('pickle_to_check_qantitive_results')
            pickle_path = args.pickle_to_check_qantitive_results
            targets = pickle_load(pickle_path)
            self.data_list = [data_path.tolist() for data_path in targets['path']]
            self.gt_S_seed = targets['gt_S_seed']
            self.rand_P_seed = targets['rand_P_seed']
            self.rand_S_seed = targets['rand_S_seed']
            self.randn_theta_seed = targets['randn_theta_seed']
            self.randn_axis_idx = targets['randn_axis_idx']
        
        # Eval point clouds.
        self.point_cloud_path = '/home/yyoshitake/works/make_depth_image/project/tmp_point_clouds/000/'


    def __getitem__(self, index):
        if self.mode in {'train', 'make_val_list', 'make_tes_list'}:
            if self.view_position == 'continuous':
                instance_path, scene_max_num, scene_pos_array = self.data_list[index]
                ini_scene_id = np.random.randint(1, scene_max_num+1, 1).item()
                ini_scene_path = glob.glob(os.path.join(instance_path, str(ini_scene_id).zfill(10) + '*.pickle'))[0]
                if self.mode in {'train', 'make_val_list'}:
                    str_pos_id = ini_scene_path.split('_')[-2]
                    ini_pos_id = int(str_pos_id)
                    previous_grid_id = scene_pos_array[ini_pos_id]
                elif self.mode in {'make_tes_list'}:
                    str_pos_id = ini_scene_path.split('_')[-2] # = str_grid_id
                    previous_grid_id = np.array([[str_pos_id[:4], str_pos_id[4:7], str_pos_id[7:]]], dtype=np.int16)
                    # ini_pos_id = np.where(np.all(scene_pos_array==previous_grid_id, axis=-1))[0][0]
                    # previous_grid_id_ = scene_pos_array[ini_pos_id]
                pos_str_list = [str_pos_id]
                for sample_idx in range(self.total_obs_num-1):
                    pos_maxdist_list = np.abs(scene_pos_array - previous_grid_id).max(axis=-1)
                    near_pos_mask = (pos_maxdist_list<=2) * np.any(scene_pos_array!=previous_grid_id, axis=-1) * (scene_pos_array[:, 0]>=0) # all_pos
                    if near_pos_mask.sum() > 0:
                        near_pos_array = scene_pos_array[near_pos_mask]
                        previous_grid_id = near_pos_array[np.random.randint(0, near_pos_array.shape[0], 1)]
                    else:
                        previous_grid_id = previous_grid_id
                    if self.mode in {'train', 'make_val_list'}:
                        pos_str_list.append(str(np.where(np.all(scene_pos_array==previous_grid_id, axis=-1))[0].item()).zfill(10))
                    elif self.mode in {'make_tes_list'}:
                        pos_str_list.append(str(previous_grid_id[0, 0]).zfill(4) + str(previous_grid_id[0, 1]).zfill(3) + str(previous_grid_id[0, 2]).zfill(3))
                    # For check !
                    # np.linspace(-1.5, 1.5, 11)
                    # pos_str_list[-1]
                    # pickle_load(np.random.choice(glob.glob(os.path.join(instance_path, '*_'+pos_str_list[-1]+'_*.pickle'))))['pos']
                    # import pdb; pdb.set_trace()
                scene_path_list = []
                for pos_str in pos_str_list:
                    tgt_path_list = glob.glob(os.path.join(instance_path, '*_'+pos_str+'_*.pickle'))
                    if len(tgt_path_list) > 0:
                        scene_path_list.append(np.random.choice(tgt_path_list))
                    else:
                        import pdb; pdb.set_trace()
                # scene_path_list = [np.random.choice(glob.glob(os.path.join(instance_path, '*_'+pos_str+'_*.pickle'))) for pos_str in pos_str_list]
            elif self.view_position == 'randn':
                instance_path, scene_max_num = self.data_list[index]
                scene_id_list = list(range(1, scene_max_num+1))
                sampled_id = np.random.choice(scene_id_list, self.total_obs_num, replace=False)
                sampled_scene_list = [str(scene_id).zfill(10) for scene_id in sampled_id]
                scene_path_list = [glob.glob(os.path.join(instance_path, scene_id+'*.pickle'))[0] for scene_id in sampled_scene_list]
            if self.mode in {'make_val_list', 'make_tes_list'}:
                scene_dict_list = [pickle_load(scene_path) for scene_path in scene_path_list]
                return scene_path_list
        elif self.mode in {'val', 'tes'}:
            scene_path_list = self.data_list[index]

        # Get data.
        instance_id = scene_path_list[0].split('/')[-2]
        scene_dict_list = [pickle_load(scene_path) for scene_path in scene_path_list]
        camera_pos_obj = np.stack([scene_dict['pos'] for scene_dict in scene_dict_list], axis=0).astype(np.float32)
        camera_o2c = np.stack([scene_dict['w2c'] for scene_dict in scene_dict_list], axis=0).astype(np.float32)
        distance_map = np.stack([scene_dict['distance'] for scene_dict in scene_dict_list], axis=0).astype(np.float32)

        # Get bbox.
        mask = distance_map > 0
        bbox_list = []
        for i, mask_i in enumerate(mask):
            if mask[0].sum() > 0:
                masked_image_coord = self.image_coord[mask_i]
                max_y, max_x = masked_image_coord.max(axis=0)
                min_y, min_x = masked_image_coord.min(axis=0)
            else:
                mask[i] = np.full_like(mask_i, True)
                max_y, max_x = self.image_coord.reshape(-1, 2).max(axis=0)
                min_y, min_x = self.image_coord.reshape(-1, 2).min(axis=0)
            bbox_list.append(np.array([[max_x, max_y], [min_x, min_y]]))
        bbox_list = np.stack(bbox_list, axis=0)

        # 正方形でClop.
        bbox_H_xy = np.stack([bbox_list[:, 0, 0] - bbox_list[:, 1, 0],             # H_x
                              bbox_list[:, 0, 1] - bbox_list[:, 1, 1]], axis = -1) # H_y
        bbox_H = bbox_H_xy.max(axis = -1) # BBoxのxy幅の内、大きい方で揃える
        diff_bbox_H = (bbox_H[:, None] - bbox_H_xy) / 2
        bbox_list = bbox_list + np.stack([diff_bbox_H, -diff_bbox_H], axis=-2) # maxには足りない分を足し、minからは引く

        # BBoxが画像からはみ出た場合、収まるように戻す
        border = np.array([[self.input_W-1, self.input_H-1], [0., 0.]]).astype(np.float32)
        outside = border[None, :, :] - bbox_list
        outside[:, 0][outside[:, 0] > .0] = 0. # 値が負ならMaxがはみ出た -> ずれを引く
        outside[:, 1][outside[:, 1] < .0] = 0. # 値が正ならMinがはみ出た -> ずれを足す
        bbox_list = bbox_list + outside.sum(axis = -2)[:, None, :]
        bbox_list[:, :, 0] = bbox_list[:, :, 0] / (0.5*self.input_W) - 1 # change range [-1, 1]
        bbox_list[:, :, 1] = bbox_list[:, :, 1] / (0.5*self.input_H) - 1 # change range [-1, 1]
        bbox_diagonal = (bbox_list[:, 0, 0] - bbox_list[:, 1, 0]) / 2

        # Get object and camera poses against first frame.
        obj_scale = np.ones(1, dtype=np.float32) # dummy scale.
        obj_scale = np.tile(obj_scale[None, :], (self.total_obs_num, 1))
        # distance_map *= obj_scale[0, :]
        # camera_pos_obj *= obj_scale[0, :]
        o2w = np.tile(camera_o2c[0, :, :][None, :, :], (self.total_obs_num, 1, 1))
        obj_pos_wrd = o2w[0, :, :] @ (np.zeros(3, dtype=np.float32) - camera_pos_obj[0, :])
        obj_pos_wrd = np.tile(obj_pos_wrd[None, :], (self.total_obs_num, 1))
        w2c = camera_o2c @ o2w[0, :, :].T
        camera_pos_wrd = np.sum(camera_pos_obj[:, None, :]*o2w[0, :, :][None, :, :], axis=-1) + obj_pos_wrd
        obj_green_cam = camera_o2c[:, :, 1] # Y
        obj_red_cam = camera_o2c[:, :, 0] # X
        obj_green_wrd = o2w[:, :, 1] # Y
        obj_red_wrd = o2w[:, :, 0] # X

        # Get canonical maps.
        if self.mode in {'val', 'tes'}:
            # scene_path_list = False
            canonical_path = os.path.join(self.canonical_path, instance_id + '.pickle')
            canonical_data_dict = pickle_load(canonical_path)
            canonical_distance_map = canonical_data_dict['depth_map'].astype(np.float32)
            canonical_camera_pos = canonical_data_dict['camera_pos'].astype(np.float32)
            canonical_camera_rot = canonical_data_dict['camera_rot'].astype(np.float32)
        else:
            canonical_distance_map = canonical_camera_pos = canonical_camera_rot = False

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
        # if self.mode in {'tes'}:
        gt_pc_obj = np.load(os.path.join(self.point_cloud_path, instance_id+'.npy')).astype(np.float32)
        # else:
        #     gt_pc_obj = False
        
        return mask, distance_map, instance_id, camera_pos_wrd, w2c, bbox_diagonal, bbox_list, obj_pos_wrd, o2w, obj_green_wrd, \
            obj_red_wrd, camera_o2c, obj_green_cam, obj_red_cam, obj_scale, canonical_distance_map, canonical_camera_pos, \
            canonical_camera_rot, scene_path_list, rand_seed, [path[-39:-29] for path in scene_path_list], gt_pc_obj


    def __len__(self):
        return len(self.data_list)





if __name__=='__main__':
    from parser_get_arg import get_args
    args = get_args()
    eval_mode = 'tes' # 'val'
    args.dummy_instance_list_txt = '/home/yyoshitake/works/DeepSDF/project/instance_lists/paper_exp/tes_unknown.txt' # val.txt'
    args.dummy_data_dir = '/d/workspace/yyoshitake/moving_camera/volumetric/tmp_1/results' # tmp_2/results'
    args.dummy_N_scenes = 32 # 16
    args.view_position = 'continuous' # 'randn'
    pickle_path = f'/home/yyoshitake/works/DeepSDF/project/randn/{eval_mode}/{args.view_position}_{args.dummy_N_scenes}.pickle'

    # Make dummy data loader
    from torch.utils.data import DataLoader
    def seed_worker(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
    dummy_generator = torch.Generator().manual_seed(5)
    dummy_dataset = TaR_dataset(args, f'make_{eval_mode}_list', args.dummy_instance_list_txt, args.dummy_data_dir, args.dummy_N_scenes)
    dummy_dataloader = DataLoader(dummy_dataset, batch_size=1, num_workers=1, shuffle=False, worker_init_fn=seed_worker, generator=dummy_generator)

    from often_use import pickle_dump
    data_list = []
    out_cnt = 0
    for epoch in tqdm.tqdm(range(args.dummy_N_scenes)):
        np.random.seed(100 + epoch)
        for batch in tqdm.tqdm(dummy_dataloader):
            scene_path_list = [batch_i[0] for batch_i in batch]
            # print([scene_path.split('/')[-1].split('_')[1] for scene_path in scene_path_list])
            data_list.append(scene_path_list)
            pos_list = [pickle_load(path)['pos'] for path in scene_path_list]
            if 1.04 < max([np.linalg.norm(pos_list[i]-pos_list[i+1]) for i in range(len(pos_list)-1)]):
                out_cnt += 1
    print(out_cnt)
    pickle_dump(data_list, pickle_path)
    