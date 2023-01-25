# 
import os
import tqdm
import glob
from often_use import *

# 書き換える必要がある
ori_ins_list = txt2list('./DDF/instance_list/table/total_havibg_ddf_data.txt') + txt2list('/home/yyoshitake/works/make_depth_image/project/instance_list/paper/table/tes.txt')
data_dir = '/disks/local/yyoshitake/moving_camera/volumetric/table'

# 各ディレクトリのパス
pos_list_dir = os.path.join(data_dir, 'instance_infos', 'revised')
results_dir = os.path.join(data_dir, 'results')

# Logされたシーン数の取得
rendered_instances_list = txt2list(os.path.join(data_dir, 'rendered_instances.txt'))
rendered_instances_dict = {}
for rendered_instances_info in rendered_instances_list:
    instance_name, scene_num, lack_pos_num = rendered_instances_info.split(' : ')
    rendered_instances_dict[instance_name] = [int(scene_num), int(lack_pos_num)]

# レンダリングされたインスタンスのパス
lack_list = []
rendered_ins_paths = glob.glob(os.path.join(results_dir, '*'))
rendered_ins_names = [rendered_ins_path.split('/')[-1] for rendered_ins_path in rendered_ins_paths]
for ins_name in ori_ins_list:
    if not ins_name in rendered_ins_names:
        lack_list.append(ins_name)
import pdb; pdb.set_trace()

# 確認開始
lack_pos_list_ins_list = [] # PosListがないインスタンス
lack_rendered_log_ins_list = [] # LogTxtの中にインスタンスがあるか？
incoreect_log_ins = [] # 
incorrect_pos_list = [] # 
incorrect_pos_minus = [] # 
few_scene_list = [] # シーンが足りないインスタンス
small_size_list = [] # サイズの小さすぎるファイル
for instance_path in tqdm.tqdm(rendered_ins_paths):
    instance_name = instance_path.split('/')[-1]
    if instance_name in ori_ins_list:
        scene_paths = glob.glob(os.path.join(instance_path, '*'))
        correct_file_num = sum([os.path.getsize(scene_path_i) > 1e5 for scene_path_i in scene_paths])
        if correct_file_num != len(scene_paths):
            small_size_list.append(instance_name)
            continue

        # PosListが存在するか？
        pos_list_path = os.path.join(pos_list_dir, instance_name + '.npy')
        if os.path.isfile(pos_list_path):
            pos_list = np.load(pos_list_path)
        else:
            lack_pos_list_ins_list.append(instance_name)
            continue

        # LogTxtの中にあるか？
        exist_in_log = instance_name in rendered_instances_dict.keys()
        if exist_in_log:
            scene_num, lack_pos_num = rendered_instances_dict[instance_name]
        else:
            lack_rendered_log_ins_list.append(instance_name)
            continue

        # Logの数が合っているか？
        if not len(scene_paths) == scene_num:
            if len(scene_paths) > 0:
                import pdb; pdb.set_trace()
            incoreect_log_ins.append(instance_name)

        # PosListが合っているか？
        pos_in_paths = set([scene_path.split('_')[-2] for scene_path in scene_paths])
        if not len(pos_list) == (len(pos_in_paths) + lack_pos_num): # PosListのTotal合っているか？
            incorrect_pos_list.append(instance_name)
            continue
        if not (pos_list[:, 0] < 0).sum() == lack_pos_num: # 欠けた部分はPosマイナスに成る
            incorrect_pos_minus.append(instance_name)
            continue

        # Sceneの数とPosの数
        if scene_num < 2000 and len(pos_in_paths) < 200:
            few_scene_list.append(instance_name)

# 結果の保存
list2txt(lack_pos_list_ins_list, 'lack_pos_list_ins_list.txt')
list2txt(lack_rendered_log_ins_list, 'lack_rendered_log_ins_list.txt')
list2txt(incoreect_log_ins, 'incoreect_log_ins.txt')
list2txt(incorrect_pos_list, 'incorrect_pos_list.txt')
list2txt(incorrect_pos_minus, 'incorrect_pos_minus.txt')
list2txt(few_scene_list, 'few_scene_list.txt')
list2txt(small_size_list, 'small_size_list.txt')

import pdb; pdb.set_trace()
# # Renderingできたインスタンス数
# failed_ins = set(lack_pos_list_ins_list + lack_rendered_log_ins_list + incoreect_log_ins + incorrect_pos_list + incorrect_pos_minus + few_scene_list)
# correctly_rendered_ins = set(rendered_ins_paths) - failed_ins
# miss = set(ori_ins_list) - correctly_rendered_ins