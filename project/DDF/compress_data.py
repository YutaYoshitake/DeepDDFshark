import os
import pickle
import numpy as np


def txt2list(txt_file):
    result_list = []
    with open(txt_file, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            result_list.append(line.rstrip('\n'))
    return result_list

def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)

def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data


###
base_dir = '/home/yyoshitake/works/DeepSDF/project/DDF/dataset/table/results'
out_dir = '/home/yyoshitake/works/DeepSDF/project/DDF/dataset/table/results_cmp'
ins_list_all = txt2list('/home/yyoshitake/works/make_depth_image/project/instance_list/paper/table/total_havibg_ddf_data.txt') # 直す！！！
ins_list_not = txt2list('/home/yyoshitake/works/make_depth_image/project/instance_list/paper/table/remake.txt') # 直す！！！
ins_list = ins_list_all # list(set(ins_list_all) - set(ins_list_not))
###
cnt = 0
for ins in ins_list:
    if not os.path.exists(os.path.join(out_dir, ins)):
        os.mkdir(os.path.join(out_dir, ins))
    cnt += 1
    print(cnt)
    for view_i in range(200):

        # データの読み込み
        map_dict = pickle_load(f'{base_dir}/{ins}/{str(view_i).zfill(5)}_mask.pickle')
        inverced_depth = map_dict['inverced_depth']
        normal_map = map_dict['normal_map']
        blur_mask = map_dict['blur_mask']
        pose_dict = pickle_load(f'{base_dir}/{ins}/{str(view_i).zfill(5)}_pose.pickle')
        pos = pose_dict['pos'].astype(np.float32)
        rot = pose_dict['rot'].astype(np.float32)

        # データの圧縮？
        masked_invdepth_map = inverced_depth[blur_mask].astype(np.float32)
        masked_normal_map = normal_map[blur_mask].astype(np.float32)

        # データの保存
        map_path = f'{out_dir}/{ins}/{str(view_i).zfill(5)}_mask.pickle'
        pose_path = f'{out_dir}/{ins}/{str(view_i).zfill(5)}_pose.pickle'
        map_dict = {'masked_invdepth_map': masked_invdepth_map, 'masked_normal_map': masked_normal_map, 'blur_mask': blur_mask} # 距離場画像か深度画像か注意
        camera_info_dict = {'pos':pos, 'rot':rot}
        pickle_dump(map_dict, map_path)
        pickle_dump(pose_dict, pose_path)
