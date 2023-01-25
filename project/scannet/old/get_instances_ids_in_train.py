import json
from often_use import *
import tqdm



# Full scan2cad annotations
annotations_json_open = open('data2/scan2cad/full_annotations.json', 'r')
annotations_json_load = json.load(annotations_json_open)

# simple scan2cad annotations
appearances_json_open = open('data2/scan2cad/cad_appearances.json')
appearances_json_load = json.load(appearances_json_open)

# 訓練と評価のシーンID
train_scene_ids = txt2list('/home/yyoshitake/works/DeepSDF/project/scannet/data2/scan2cad/split/train.txt')
val_sccene_ids = txt2list('/home/yyoshitake/works/DeepSDF/project/scannet/data2/scan2cad/split/val.txt')
train_scene_ids = set(train_scene_ids + val_sccene_ids)

# Prepare Annotation dict.
target_cat_ids = ['02933112', '03001627', '03211117', '04379243']
isinstance_id_dict = {}
for cat_id in target_cat_ids:
    isinstance_id_dict[cat_id] = []
target_cat_ids = set(target_cat_ids)

# インスタンスIDの抽出
annotations_dict = {}
unused = []
for annotation_i in tqdm.tqdm(annotations_json_load):
    scan_id = annotation_i['id_scan']
    if scan_id in train_scene_ids:
        for cad_i in annotation_i['aligned_models']:
            catid_cad = cad_i['catid_cad']
            if catid_cad in target_cat_ids:
                isinstance_id_dict[catid_cad].append(cad_i['id_cad'])
    else:
        unused.append(scan_id)

# 使われなかったシーン
print(len(unused))

# 使われているシーンのインスタンスID
for cat_id in target_cat_ids:
    ins_ids = set(isinstance_id_dict[cat_id])
    isinstance_id_dict[cat_id] = list(ins_ids)
pickle_dump(isinstance_id_dict, 'used_instance_id.pickle')
import pdb; pdb.set_trace()
