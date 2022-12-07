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
        # self.data_list = self.data_list[:116] # import random; random.shuffle(self.data_list) # [self.data_list[5]]

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
        if self.mode in {'tes'}:
            gt_pc_obj = np.load(os.path.join(self.point_cloud_path, instance_id+'.npy')).astype(np.float32)
        else:
            gt_pc_obj = False
        
        return mask, distance_map, instance_id, camera_pos_wrd, w2c, bbox_diagonal, bbox_list, obj_pos_wrd, o2w, obj_green_wrd, \
            obj_red_wrd, camera_o2c, obj_green_cam, obj_red_cam, obj_scale, canonical_distance_map, canonical_camera_pos, \
            canonical_camera_rot, scene_path_list, rand_seed, [path[-39:-29] for path in scene_path_list], gt_pc_obj


    def __len__(self):
        return len(self.data_list)



class txt2dataset(data.Dataset):
    def __init__(self, args, mode, instance_list_txt, current_epo, scene_list_path):

        self.mode = mode
        self.total_obs_num = args.total_obs_num

        if self.mode in {'train'}:
            self.data_list = txt2list(instance_list_txt) # instance_name のリスト
            self.data_list_files = {}
            for instance_name in self.data_list:
                self.data_list_files[instance_name] = os.path.join(scene_list_path, instance_name + '.txt') # open(os.path.join(scene_list_path, instance_name + '.txt'), 'r')
                # for epoch in range(current_epo):
                #     _ = self.data_list_files[instance_name].readline()
        elif self.mode in {'val', 'tes'}:
            self.data_list = pickle_load(scene_list_path)
        # self.data_list = self.data_list[:116] # import random; random.shuffle(self.data_list) # [self.data_list[5]]

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

        self.canonical_path = args.canonical_data_path
        self.point_cloud_path = '/home/yyoshitake/works/make_depth_image/project/tmp_point_clouds/000/'


    def __getitem__(self, index):
        if self.mode in {'val', 'tes'}:
            scene_path_list = self.data_list[index]
        elif self.mode in {'train'}:
            scene_path_txt = linecache.getline(self.data_list_files[self.data_list[index]], self.dataset_current_epoch) # self.data_list_files[self.data_list[index]].readline()
            scene_path_list = scene_path_txt.replace('\n','').split(' ')

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
        if self.mode in {'tes'}:
            gt_pc_obj = np.load(os.path.join(self.point_cloud_path, instance_id+'.npy')).astype(np.float32)
        else:
            gt_pc_obj = False
        
        return mask, distance_map, instance_id, camera_pos_wrd, w2c, bbox_diagonal, bbox_list, obj_pos_wrd, o2w, obj_green_wrd, \
            obj_red_wrd, camera_o2c, obj_green_cam, obj_red_cam, obj_scale, canonical_distance_map, canonical_camera_pos, \
            canonical_camera_rot, scene_path_list, rand_seed, [path[-39:-29] for path in scene_path_list], gt_pc_obj


    def __len__(self):
        return len(self.data_list)



class make_dataset(data.Dataset):
    def __init__(self, args, mode, instance_list_txt, data_dir, scene_list_path):

        self.mode = mode
        self.view_position = args.view_position
        self.total_obs_num = args.total_obs_num
        self.scene_pos_array_path = args.scene_pos_array_path
        self.canonical_path = args.canonical_data_path
        self.dummy_epoch = args.dummy_epoch
        self.cnt_max_length = args.cnt_max_length
        # 視点位置を一定範囲に制限する際、ファイル名から視点位置を計算
        self.pos_bound_norm = args.pos_bound_norm
        self.grid_num = args.grid_num
        self.grid_med  = np.array([10/2, 0, 10/2]) # xyzの中心に相当するGrid_id
        self.cmera_pos_range = 1.5
        self.grig2cad_scale = self.cmera_pos_range / ((self.grid_num - 1) / 2)
        # テストや訓練データの位置の書かれ方。
        self.file_name_pos_id_mode = 'order' 
        if data_dir.split('/')[-2] == 'tmp_1':
            self.file_name_pos_id_mode = 'grids'

        self.data_list = []
        with open(instance_list_txt, 'r') as f:
            instances = f.read().splitlines()
            for instance_id in instances:
                instance_path = os.path.join(data_dir, instance_id.rstrip('\n'))
                # Append to datalist.
                scene_pos_array = np.load(self.scene_pos_array_path + instance_id + '.npy')
                self.data_list.append([instance_path, scene_pos_array])


    def __getitem__(self, index):
        instance_path, scene_pos_array = self.data_list[index]
        instance_name = instance_path.split('/')[-1]
    
        # ある範囲内の視点のみ取ってくる。
        scene_pos_cad_coordinate_array = self.grig2cad_scale * (scene_pos_array - self.grid_med)
        inside_pos_mask = np.linalg.norm(scene_pos_cad_coordinate_array, axis=-1) < self.pos_bound_norm
        inside_pos_id = np.arange(scene_pos_cad_coordinate_array.shape[0])[inside_pos_mask] # pos_idは0始まり。
        inside_pos_num = inside_pos_id.shape[0]

        # Trainの場合はTxtに書き込んでしまう。
        total_data_list = []
        if self.mode in {'make_train_list'}:
            insdata_txt_file = open(os.path.join(self.txt_dir_path, f'{instance_name}.txt'), 'wt')

        if self.view_position == 'randn':
            # エポックの数だけサンプル
            for epoch in range(self.dummy_epoch): # for epoch in tqdm.tqdm(range(self.dummy_epoch)):
                pos_id_list = np.random.choice(inside_pos_id, self.total_obs_num)
                sampled_img_paths = []
                for pos_id in pos_id_list:
                    if self.file_name_pos_id_mode == 'order':
                        str_pos_id = str(pos_id).zfill(10)
                    elif self.file_name_pos_id_mode == 'grids':
                        grid_id = scene_pos_array[pos_id]
                        str_pos_id = str(grid_id[0]).zfill(4) + str(grid_id[1]).zfill(3) + str(grid_id[2]).zfill(3)
                    sampled_img_path = np.random.choice(np.array(glob.glob(os.path.join(instance_path, '*_' + str_pos_id + '_*.pickle'))))
                    sampled_img_paths.append(sampled_img_path)
                
                # 結果を保存
                if self.mode in {'make_train_list'}:
                    sampled_txt_paths = (' ').join(sampled_img_paths) + '\n'
                    insdata_txt_file.write(sampled_txt_paths)
                total_data_list.append(sampled_img_paths)

        elif self.view_position == 'continuous':
            # エポックの数だけサンプル
            for epoch in range(self.dummy_epoch): # for epoch in tqdm.tqdm(range(self.dummy_epoch)):
                sampled_img_paths = []
                # 開始地点
                ini_pos_id = np.random.choice(inside_pos_id, 1)[0]
                previous_grid_id = scene_pos_array[ini_pos_id]
                if self.file_name_pos_id_mode == 'order':
                    str_pos_id = str(np.where(np.all(scene_pos_array==previous_grid_id, axis=-1))[0].item()).zfill(10)
                elif self.file_name_pos_id_mode == 'grids':
                    str_pos_id = str(previous_grid_id[0]).zfill(4) + str(previous_grid_id[1]).zfill(3) + str(previous_grid_id[2]).zfill(3)
                ini_scene_path = np.random.choice(np.array(glob.glob(os.path.join(instance_path, '*_' + str_pos_id + '_*.pickle'))))
                sampled_img_paths.append(ini_scene_path)

                for sample_idx in range(self.total_obs_num-1):
                    pos_maxdist_list = np.abs(scene_pos_array - previous_grid_id).max(axis=-1)
                    near_pos_mask = (pos_maxdist_list<=self.cnt_max_length) * np.any(scene_pos_array!=previous_grid_id, axis=-1) * (scene_pos_array[:, 0]>=0) # all_pos
                    near_pos_mask = near_pos_mask * inside_pos_mask

                    if near_pos_mask.sum() > 0:
                        near_pos_array = scene_pos_array[near_pos_mask]
                        previous_grid_id = near_pos_array[np.random.randint(0, near_pos_array.shape[0], 1)[0]]
                    else:
                        previous_grid_id = previous_grid_id # なければとどまる。
                    
                    if self.file_name_pos_id_mode == 'order':
                        str_pos_id = str(np.where(np.all(scene_pos_array==previous_grid_id, axis=-1))[0].item()).zfill(10)
                    elif self.file_name_pos_id_mode == 'grids':
                        str_pos_id = str(previous_grid_id[0]).zfill(4) + str(previous_grid_id[1]).zfill(3) + str(previous_grid_id[2]).zfill(3)
                    sampled_img_path = np.random.choice(np.array(glob.glob(os.path.join(instance_path, '*_' + str_pos_id + '_*.pickle'))))
                    sampled_img_paths.append(sampled_img_path)

                # 結果を保存
                if self.mode in {'make_train_list'}:
                    sampled_txt_paths = (' ').join(sampled_img_paths) + '\n'
                    insdata_txt_file.write(sampled_txt_paths)
                total_data_list.append(sampled_img_paths)
                
        if self.mode in {'make_train_list'}:
            insdata_txt_file.close()
        return total_data_list, inside_pos_num

    def __len__(self):
        return len(self.data_list)





if __name__=='__main__':
    import datetime
    from parser_get_arg import get_args
    eval_mode = 'train' # 'train' 'tes' 'val' 
    args = get_args()
    args.dummy_instance_list_txt = '/home/yyoshitake/works/DeepSDF/project/instance_lists/paper_exp/fultrain.txt' # tes_unknown.txt # val.txt # fultrain.txt
    if eval_mode in {'train', 'val'}:
        args.dummy_data_dir = '/d/workspace/yyoshitake/moving_camera/volumetric/tmp_2/results'
    elif eval_mode in {'tes'}:
        args.dummy_data_dir = '/d/workspace/yyoshitake/moving_camera/volumetric/tmp_1/results'
    args.scene_pos_array_path ='/d/workspace/yyoshitake/moving_camera/volumetric/instance_infos/revised/'
    args.dummy_epoch = 10 # 5 # 16
    args.view_position = 'randn' # 'continuous' 'randn'
    args.total_obs_num = 5
    args.pos_bound_norm = 0.70 # ノルムの最大値
    args.grid_num = 11 # データを作った際のグリッド数
    args.cnt_max_length = 1 # 連続的の最大移動距離



    dt_now = datetime.datetime.now()
    time_log = dt_now.strftime('%Y_%m_%d_%H_%M_%S').replace('_', '')
    str_bound = str(args.pos_bound_norm).replace('.', 'd')
    instance_txt_name = args.dummy_instance_list_txt.split('/')[-1].split('.')[0]
    if eval_mode in {'tes', 'val'}:
        pickle_path = f'dataset/sampled_path/chair/{eval_mode}/{time_log}_{instance_txt_name}_{args.view_position}_{args.dummy_epoch}_{str_bound}.pickle'
    if eval_mode in {'train'}:
        txt_dir_path = f'dataset/sampled_path/chair/{eval_mode}/{time_log}_{instance_txt_name}_{args.view_position}_{args.dummy_epoch}_{str_bound}'
        os.mkdir(txt_dir_path)

    # Make dummy data loader
    from torch.utils.data import DataLoader
    def seed_worker(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
    dummy_generator = torch.Generator().manual_seed(8)
    dummy_dataset = make_dataset(args, f'make_{eval_mode}_list', args.dummy_instance_list_txt, args.dummy_data_dir, False)
    if eval_mode in {'train'}:
        dummy_dataloader = DataLoader(dummy_dataset, batch_size=32, num_workers=32, shuffle=False, worker_init_fn=seed_worker, generator=dummy_generator)
    elif eval_mode in {'val', 'tes'}:
        dummy_dataloader = DataLoader(dummy_dataset, batch_size=1, num_workers=1, shuffle=False, worker_init_fn=seed_worker, generator=dummy_generator)
    if eval_mode in {'train'}:
        dummy_dataset.txt_dir_path = txt_dir_path

    # #########################
    # #                       #
    # #########################
    # from often_use import pickle_dump

    # data_list = []
    # pos_num = []
    # out_cnt = 0
    # np.random.seed(8)
    # for batch in tqdm.tqdm(dummy_dataloader):
    #     batch_data_list = batch[0]
    #     pos_num.append(batch[1])

    #     current_data_list = []
    #     for batch_i in batch_data_list:
    #         current_data_list.append([path_i[0] for path_i in batch_i]) # Dataloaderの返し方的にこうまとめる必要
    #     data_list += current_data_list

    #     # 連続的になっているかの確認
    #     for scene_path_list in current_data_list:
    #         pos_list = [pickle_load(path)['pos'] for path in scene_path_list]
    #         buffer = 1.01
    #         if np.sqrt(3*(args.cnt_max_length*0.3)**2) * buffer < max([np.linalg.norm(pos_list[i]-pos_list[i+1]) for i in range(len(pos_list)-1)]):
    #             out_cnt += 1

    # if eval_mode in {'tes', 'val'}:
    #     pickle_dump(data_list, pickle_path)
        
    # print(f'{out_cnt} samples are out side cont')

    # valid_pos_nums = torch.cat(pos_num).to(torch.float)
    # print(f'average {valid_pos_nums.mean()} are inside bound')
    # print(f'max {valid_pos_nums.min()} are inside bound')



    # debug
    from often_use import check_map_torch
    args.input_H = 128
    args.input_W = 128
    finished_epo = 0 # 5エポックまで終わってる
    from torch.utils.data import DataLoader
    dummy_generator = torch.Generator().manual_seed(15)
    def seed_worker(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
    dummy_dataset = txt2dataset(args, f'train', args.dummy_instance_list_txt, finished_epo, 'dataset/sampled_path/chair/train/20221206152812_fultrain_randn_1000_0d7')
    dummy_dataloader = DataLoader(dummy_dataset, batch_size=2, num_workers=0, shuffle=False, worker_init_fn=seed_worker, generator=dummy_generator)

    for epo in range(args.dummy_epoch):
        dummy_dataset.dataset_current_epoch = epo + 1
        for batch_idx, batch in enumerate(tqdm.tqdm(dummy_dataloader)):
            import pdb; pdb.set_trace()
            # dstmap = batch[1]
            # dstmap_img = torch.cat([dstmap_i.reshape(-1, 128) for dstmap_i in dstmap], dim=-1)
            # check_map_torch(dstmap_img, f'tes_{str(epo).zfill(5)}_{str(batch_idx).zfill(5)}.png')
            # if batch_idx > 30:
