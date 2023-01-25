import cv2
import os
import numpy as np
import linecache
import pickle


def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data


data_path = '/home/yyoshitake/works/DeepSDF/disks/old/chair/tmp_2/results'.split('/')
sampled_txt_path = '/home/yyoshitake/works/DeepSDF/project/dataset/sampled_path/chair/train/20221211035003_fultrain_continuous_1000_10_2_all'
ins_name = '1e7bc7fd20c61944f51f77a6d7299806' # '1842797b090ce2ecebc1a7ae7c4c250d'

for line_i in range(1000):
    ins_sampled_txt_file = os.path.join(sampled_txt_path, ins_name + '.txt')
    scene_path_txt = linecache.getline(ins_sampled_txt_file, line_i+1)
    scene_path_list = ['/'.join(data_path + path_i.split('/')[-2:]) for path_i in scene_path_txt.replace('\n','').split(' ')]

    scene_dict_list = [pickle_load(scene_path) for scene_path in scene_path_list]
    distance_map = np.stack([scene_dict['distance'] for scene_dict in scene_dict_list], axis=0).astype(np.float32)

    vis_d_list = []
    for map_idx, distance_map_i in enumerate(distance_map):
        d_max = distance_map_i.max()
        d_min = distance_map_i.min()
        vis_d = (distance_map_i - d_min) / (d_max - d_min)
        vis_d = (255 * np.clip(vis_d, 0.0, 1.0)).astype(np.uint8)
        vis_d = cv2.applyColorMap(vis_d, cv2.COLORMAP_VIRIDIS)
        vis_d_list.append(vis_d)
    
        buffer_ = 255 * np.ones((128, 10, 3)).astype(np.uint8)
        if 4 > map_idx:
            vis_d_list.append(buffer_)
    result = np.concatenate(vis_d_list, axis=1)
    cv2.imwrite(f'seq_1/{line_i}.png', result)

    # import pdb; pdb.set_trace()
