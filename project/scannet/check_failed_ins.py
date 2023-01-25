import os
import glob
import json
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



# 書き換える
target_cat_id = '03001627'
scene_list = txt2list('/home/yyoshitake/works/DeepSDF/project/scannet/data2/scan2cad/split/val.txt')
result_dir = 'results_chair_val'


# Scan2CADのデータにあるインスタンス
annotations_json_open = open('data2/scan2cad/full_annotations.json', 'r')
annotations_json_load = json.load(annotations_json_open)
# Prepare Annotation dictを作成
data_path_list = {}
for annotation_i in annotations_json_load:
    scan_id_i = annotation_i['id_scan']
    # if scan_id_i in scene_list:
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
import pdb; pdb.set_trace()
##################################################
# Target Sceneでループ
##################################################
for i, scene_id in enumerate(data_path_list.keys()):
    print(f'{i} / {len(data_path_list.keys())}')
    # sceneのDirが存在するか？
    scene_dir = os.path.join(result_dir, target_cat_id, scene_id)
    if not os.path.exists(scene_dir):
        with open(f'{target_cat_id}_lack_scene.txt', 'a') as f:
            print(scene_id, file=f)
    else:
         # PosListが存在するか？
        scene_total_info_path = os.path.join(result_dir, target_cat_id, scene_id, 'scene_total_infos.pickle')
        if not os.path.exists(scene_total_info_path):
            with open(f'{target_cat_id}_lack_sceneinfo.txt', 'a') as f:
                print(scene_id, file=f)
        else:
            ##################################################
            # Scan2CADのアノテーションでループ回す
            ##################################################
            scene_total_info = pickle_load(scene_total_info_path)
            for obj_id in data_path_list[scene_id]:
                obj_data_path = os.path.join(scene_id, obj_id)
                
                # PosListにあるか？
                if not obj_data_path in scene_total_info['dir_list']:
                    with open(f'{target_cat_id}_uncorrect_sceneinfo.txt', 'a') as f:
                        print(f'{scene_id}_{obj_id}', file=f)
                
                # DataのDirがあるか？
                obj_data_path = os.path.join(result_dir, target_cat_id, obj_data_path)
                if not os.path.exists(obj_data_path):
                    with open(f'{target_cat_id}_lack_datadir.txt', 'a') as f:
                        print(f'{scene_id}_{obj_id}', file=f)
                else:
                    # ObjInfoがあるか?
                    obj_total_info_path = os.path.join(obj_data_path, 'total_infos.pickle')
                    if not os.path.exists(obj_total_info_path):
                        with open(f'{target_cat_id}_lack_obj_total_info.txt', 'a') as f:
                            print(f'{scene_id}_{obj_id}', file=f)
                    else:
                        if not os.path.exists(scene_total_info_path):
                            with open(f'{target_cat_id}_lack_objinfo.txt', 'a') as f:
                                print(f'{scene_id}_{obj_id}', file=f)
                        else:
                            # FrameKeyがあるか？
                            obj_total_info = pickle_load(obj_total_info_path)
                            if not('frame' in obj_total_info.keys() and 'obj' in obj_total_info.keys()):
                                with open(f'{target_cat_id}_objinfo_has_uncorrect_key.txt', 'a') as f:
                                    print(f'{scene_id}_{obj_id}', file=f)
                            else:
                                # 視点の数がObjInfoと合っているか？
                                mask_ratio_list = obj_total_info['frame']['mask_ratio_list'] / 10000
                                frame_path_list = glob.glob(os.path.join(obj_data_path, 'data_dict/*'))
                                if len(frame_path_list) != len(mask_ratio_list):
                                    with open(f'{target_cat_id}_objinfo_has_uncorrect_viewnum.txt', 'a') as f:
                                        print(f'{scene_id}_{obj_id}', file=f)
                                # else:
                                #     with open(f'{target_cat_id}_data_list.txt', 'a') as f:
                                #         print(obj_data_path, file=f)
