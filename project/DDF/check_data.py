import os
import pickle
import glob



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



# ins_list = txt2list('/home/yyoshitake/works/DeepSDF/project/DDF/instance_list/miniddf_train_list1000.txt')
# base_dir = '/disks/local/yyoshitake/ddf/cabinet/results'
# base_dir = '/d/workspace/yyoshitake/ShapeNet/ddf/table/results'
# ins_list = [ins_path.split('/')[-1] for ins_path in glob.glob(os.path.join(base_dir, '*'))]
# total_ins = '/home/yyoshitake/works/make_depth_image/project/instance_list/paper/table/total.txt'
# total_ins = '/home/yyoshitake/works/make_depth_image/project/instance_list/paper/cabinet/total.txt'
# renderesd_ins = '/home/yyoshitake/works/make_depth_image/project/instance_list/paper/cabinet/rendered_instances.txt'

# checked_ins = txt2list('total_havibg_ddf_data.txt') # []
# import pdb; pdb.set_trace()
# ins_list = list(set(ins_list) - set(checked_ins))
# import pdb; pdb.set_trace()

###
base_dir = '/home/yyoshitake/works/DeepSDF/project/DDF/dataset/table/results'
ins_list_seal = txt2list('/home/yyoshitake/works/make_depth_image/project/tmp_table/rendered_instances.txt')
ins_list = txt2list('/home/yyoshitake/works/make_depth_image/project/tmp_table/rendered_instances.txt') # list(set(txt2list('/home/yyoshitake/works/make_depth_image/project/instance_list/paper/table/total.txt')) - set(ins_list_seal))
###

lack_map = []
lack_views = []
ins_list = ins_list
# for ins in tqdm.tqdm(ins_list):
cnt = 0
for ins in ins_list:
    cnt += 1
    print(cnt)
    for view_i in range(200): # tqdm.tqdm(range(200)):
        map_exist = os.path.isfile(f'{base_dir}/{ins}/{str(view_i).zfill(5)}_mask.pickle') and os.path.getsize(f'{base_dir}/{ins}/{str(view_i).zfill(5)}_mask.pickle') != 0
        pose_exist = os.path.isfile(f'{base_dir}/{ins}/{str(view_i).zfill(5)}_pose.pickle') and os.path.getsize(f'{base_dir}/{ins}/{str(view_i).zfill(5)}_pose.pickle') != 0
        if map_exist and pose_exist:
            data_dict = pickle_load(f'{base_dir}/{ins}/{str(view_i).zfill(5)}_mask.pickle')
            if not {'normal_map', 'inverced_depth', 'blur_mask'} == data_dict.keys():
                # print(f'{ins}_{str(view_i).zfill(5)}')
                # lack_map.append(ins)
                with open('lack_map_clione_tmp_1228.txt', 'a') as f:
                    print(ins, file=f)
            # else:
            #     with open('correct_1228.txt', 'a') as f:
            #         print(ins, file=f)
        else:
            # lack_views.append(f'{str(view_i).zfill(3)} : {ins}')
            with open('lack_views_clione_tmp_1228.txt', 'a') as f:
                print(ins, file=f)
            break

# list2txt(lack_map, 'lack_map_clione.txt')
# list2txt(lack_views, 'lack_views_clione.txt')

import pdb; pdb.set_trace()
# list2txt(ins_list - lack_views, 'total_havibg_ddf_data.txt')

# 誤ったもの
miss_ins = list(set(lack_map + lack_views))
# Totalの内、レンダリングできていないもの
miss_in_total = list(set(total_ins - miss_ins))
list2txt(miss_in_total, 'miss_in_total.txt')
# Renderedの内、レンダリングできていないもの
miss_in_renderesd = list(set(renderesd_ins - miss_ins))
list2txt(miss_in_renderesd, 'miss_in_renderesd.txt')
