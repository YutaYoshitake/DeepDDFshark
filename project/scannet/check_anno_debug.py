import json
# from often_use import *

# def txt2list(txt_file):
#     result_list = []
#     with open(txt_file, 'r') as f:
#         lines = f.read().splitlines()
#         for line in lines:
#             result_list.append(line.rstrip('\n'))
#     return result_list

# target_shapenet_cat = {'04379243': 'table', 
#                        '03001627': 'chair', 
#                        '03211117': 'display', 
#                        '02933112': 'cabinet', }
# shapenet_cat_counter = {}
# scene_id_list = txt2list('/home/yyoshitake/works/DeepSDF/project/scannet/data2/scan2cad/split/test.txt')

# # Full scan2cad annotationsの読み込み
# annotations_json_open = open('/home/yyoshitake/works/DeepSDF/project/scannet/data2/scan2cad/full_annotations.json', 'r')
# annotations_json_load = json.load(annotations_json_open)
# # Prepare Annotation dictを作成
# for annotation_i in annotations_json_load:
#     scan_id_i = annotation_i['id_scan']
#     if scan_id_i in scene_id_list:
#         for cad_idx, cad_i in enumerate(annotation_i['aligned_models']):
#             catid_cad = cad_i['catid_cad']
#             id_cad = cad_i['id_cad']

#             if not catid_cad in shapenet_cat_counter.keys():
#                 shapenet_cat_counter[catid_cad] = 0
#             shapenet_cat_counter[catid_cad] += 1

# total_cnt = 0
# for catid_cad in shapenet_cat_counter.keys():
#     total_cnt += shapenet_cat_counter[catid_cad]
# print(total_cnt)
# print('chair', shapenet_cat_counter['03001627'])

# shapenet_cat_counter_by_app = {}
# annotations_json_open = open('data2/scan2cad/cad_appearances.json', 'r')
# annotations_json_load = json.load(annotations_json_open)
# for scene_id_i in annotations_json_load.keys():
#     if scene_id_i in scene_id_list:
#         for cad_id in annotations_json_load[scene_id_i].keys():
#             cat_id, shapenet_id = cad_id.split('_')
#             if not cat_id in shapenet_cat_counter_by_app.keys():
#                 shapenet_cat_counter_by_app[cat_id] = 0
#             shapenet_cat_counter_by_app[cat_id] += annotations_json_load[scene_id_i][cad_id]

# total_cnt = 0
# for catid_cad in shapenet_cat_counter_by_app.keys():
#     total_cnt += shapenet_cat_counter_by_app[catid_cad]
# print(total_cnt)



# Full scan2cad annotationsの読み込み
annotations_json_open = open('data2/scan2cad/full_annotations.json', 'r')
annotations_json_load = json.load(annotations_json_open)
# Prepare Annotation dictを作成
annotations_dict = {}
for annotation_i in annotations_json_load:
    scan_id_i = annotation_i['id_scan']
    annotations_dict[scan_id_i] = {}
    annotations_dict[scan_id_i]['trs'] = annotation_i['trs'] # <-- transformation from scan space to world space
    annotations_dict[scan_id_i]['cad'] = {}
    # annotation_i['aligned_models']にはリスト形式でCADモデルが格納されている
    shapenet_idx_cnt = {}
    for cad_idx, cad_i in enumerate(annotation_i['aligned_models']):
        catid_cad = cad_i['catid_cad']
        id_cad = cad_i['id_cad']
        if not id_cad in shapenet_idx_cnt.keys():
            shapenet_idx_cnt[id_cad] = 1
        else:
            shapenet_idx_cnt[id_cad] = shapenet_idx_cnt[id_cad] + 1
        tgt_cad_idx = str(shapenet_idx_cnt[id_cad])
        total_id = f'{catid_cad}_{id_cad}_{str(tgt_cad_idx).zfill(3)}'
        annotations_dict[scan_id_i]['cad'][total_id] = {}
        annotations_dict[scan_id_i]['cad'][total_id]['trs'] = cad_i['trs'] # <-- transformation from CAD space to world space 
        annotations_dict[scan_id_i]['cad'][total_id]['sym'] = cad_i['sym']
        annotations_dict[scan_id_i]['cad'][total_id]["bbox"] = cad_i["bbox"]
        annotations_dict[scan_id_i]['cad'][total_id]["center"] = cad_i["center"]

import pdb; pdb.set_trace()