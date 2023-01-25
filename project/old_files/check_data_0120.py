import os
import glob
import linecache
import numpy as np
import pickle


def txt2list(txt_file):
    result_list = []
    with open(txt_file, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            result_list.append(line.rstrip('\n'))
    return result_list


def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data



train_ins_list = txt2list('/home/yyoshitake/works/DeepSDF/project/instance_lists/paper_exp/fultrain.txt')
val_ins_list   = txt2list('/home/yyoshitake/works/DeepSDF/project/instance_lists/paper_exp/val.txt')
tes_ins_list   = txt2list('/home/yyoshitake/works/DeepSDF/project/instance_lists/paper_exp/tes_unknown.txt')

train_data_dir = '/home/yyoshitake/works/DeepSDF/disks/old/chair/tmp_2/results'
tes_data_dir = '/home/yyoshitake/works/DeepSDF/disks/old/chair/tmp_1/squashfs-root'
scene_pos_array_path ='/home/yyoshitake/works/DeepSDF/disks/old/chair/instance_infos/revised/'



# サンプルされたパスのDir
# sampled_path_dir = '/home/yyoshitake/works/DeepSDF/project/dataset/sampled_path/chair/train/20221211035003_fultrain_continuous_1000_10_2_all'
sampled_path_dir = '/home/yyoshitake/works/DeepSDF/project/dataset/sampled_path/chair/train/20221211065550_fultrain_randn_1000_10_2_all'

ins_list = train_ins_list
data_dir = train_data_dir.split('/')

sampled_ins_list = glob.glob(os.path.join(sampled_path_dir, '*'))
out_cnt = 0
cnt = 0
for sampled_ins in sampled_ins_list:
    for epoch in range(1000):
        print(cnt)
        scene_path_txt = linecache.getline(sampled_ins, epoch + 1) # self.data_list_files[self.data_list[index]].readline()
        scene_path_list = ['/'.join(data_dir + path_i.split('/')[-2:]) for path_i in scene_path_txt.replace('\n','').split(' ')]
        scene_dict_list = [pickle_load(scene_path) for scene_path in scene_path_list]
        camera_pos_obj = np.stack([scene_dict['pos'] for scene_dict in scene_dict_list], axis=0).astype(np.float32)

        cnt += 1
        buffer = 1.01
        if np.sqrt(3*(2*0.3)**2) * buffer < max([np.linalg.norm(camera_pos_obj[i]-camera_pos_obj[i+1]) for i in range(len(camera_pos_obj)-1)]):
            out_cnt += 1
print('###############')
print(cnt, out_cnt)



# check_path_list = [
#     # '/home/yyoshitake/works/DeepSDF/project/dataset/sampled_path/chair/tes/20221214021558_tes_unknown_continuous_8_10_2_all.pickle', 
#     # '/home/yyoshitake/works/DeepSDF/project/dataset/sampled_path/chair/tes/20221210160130_tes_unknown_randn_8_10_1_all.pickle', 
#     # '/home/yyoshitake/works/DeepSDF/project/dataset/sampled_path/chair/val/20221211065755_val_randn_7_10_2_all.pickle', 
#     '/home/yyoshitake/works/DeepSDF/project/dataset/sampled_path/chair/val/20221211061032_val_continuous_7_10_2_all.pickle', 
# ]

# cnt_list = []
# for check_path in check_path_list:

#     ins_list = train_ins_list
#     data_dir = train_data_dir.split('/')

#     if check_path.split('/')[-2] == 'tes':
#         data_dir = tes_data_dir.split('/')
#     if check_path.split('/')[-2] == 'val':
#         data_dir = train_data_dir.split('/')

#     sampled_path_list = pickle_load(check_path)
#     out_cnt = 0
#     cnt = 0
#     for sampled_ins in sampled_path_list:
#         print(cnt)
#         scene_path_list = ['/'.join(data_dir + path_i.split('/')[-2:]) for path_i in sampled_ins]
#         scene_dict_list = [pickle_load(scene_path) for scene_path in scene_path_list]
#         camera_pos_obj = np.stack([scene_dict['pos'] for scene_dict in scene_dict_list], axis=0).astype(np.float32)
#         cnt += 1
#         buffer = 1.01
#         if np.sqrt(3*(2*0.3)**2) * buffer < max([np.linalg.norm(camera_pos_obj[i]-camera_pos_obj[i+1]) for i in range(len(camera_pos_obj)-1)]):
#             out_cnt += 1
    
#     cnt_list.append([out_cnt, cnt])
# import pdb; pdb.set_trace()