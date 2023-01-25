import json
import pickle



def txt2list(txt_file):
    result_list = []
    with open(txt_file, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            result_list.append(line.rstrip('\n'))
    return result_list


def list2txt(result_list, txt_file):
    with open(txt_file, 'a') as f:
        for result in result_list:
            f.write(result + '\n')


def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data



# 書き換える
target_cat_id = '03001627'
# scene_list = txt2list('/home/yyoshitake/works/DeepSDF/project/scannet/data2/scan2cad/split/train.txt')
# result_dir = 'results_chair_train'
# scene_list = txt2list('/home/yyoshitake/works/DeepSDF/project/scannet/data2/scan2cad/split/test.txt')
# result_dir = 'results_chair_tes'
scene_list = txt2list('/home/yyoshitake/works/DeepSDF/project/scannet/data2/scan2cad/split/val.txt')
result_dir = 'results_chair_val'
match_removed_txt = f'/home/yyoshitake/works/DeepSDF/project/scannet/{result_dir}/match_removed.txt'


# Scan2CADのデータにあるインスタンス
annotations_json_open = open('data2/scan2cad/full_annotations.json', 'r')
annotations_json_load = json.load(annotations_json_open)
# Prepare Annotation dictを作成
data_path_list = {}
for annotation_i in annotations_json_load:
    scan_id_i = annotation_i['id_scan']
    if scan_id_i in scene_list:
        shapenet_idx_cnt = {}
        for cad_idx, cad_i in enumerate(annotation_i['aligned_models']):
            catid_cad = cad_i['catid_cad']
            id_cad = cad_i['id_cad']
            if not id_cad in shapenet_idx_cnt.keys():
                shapenet_idx_cnt[id_cad] = 1
            else:
                shapenet_idx_cnt[id_cad] = shapenet_idx_cnt[id_cad] + 1
            if catid_cad == target_cat_id:
                if not scan_id_i in data_path_list.keys():
                    data_path_list[scan_id_i] = []
                tgt_cad_idx = str(shapenet_idx_cnt[id_cad])
                data_path_list[scan_id_i].append(f'{id_cad}_{str(tgt_cad_idx).zfill(3)}')


##################################################
# 出来上がったmatchedを試す
##################################################
sampled_data_list = txt2list(match_removed_txt)
sampled_path_list = []
for sampled_data in sampled_data_list:
    scene_id = sampled_data.split(':')[0]
    obj_id = sampled_data.split(':')[1]
    obj_id = obj_id[9:]
    sampled_path_list.append(f'scannet/{result_dir}/{target_cat_id}/{scene_id}/{obj_id}')

# 確認開始
for i, scene_id in enumerate(data_path_list.keys()):
    for obj_id in data_path_list[scene_id]:
        if not f'scannet/{result_dir}/{target_cat_id}/{scene_id}/{obj_id}' in sampled_path_list:
            with open(f'{result_dir}/lacks_data_list.txt', 'a') as f:
                print(f'{scene_id}_{obj_id}', file=f)

list2txt(sampled_path_list, f'{result_dir}/data_list.txt')