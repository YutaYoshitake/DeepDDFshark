import pickle
import glob
import os
import json
import numpy as np



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



def quaternion2rotation(q):
    qw, qx, qy, qz = q
    R = np.array([[1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw], 
                  [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw], 
                  [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]])
    return R



def make_M_from_tqs(t, q, s):
    # q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion2rotation(q) # quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)
    M = T.dot(R).dot(S)
    return M

target_dir = '/home/yyoshitake/works/DeepSDF/project/scannet/results_chair_val/03001627'
scene_path_list = glob.glob(os.path.join(target_dir, '*/'))
loc_dict = pickle_load('/home/yyoshitake/works/DeepSDF/project/scannet/loc_dict.pickle')

for scene_path_i in scene_path_list:
    ins_path_list = glob.glob(os.path.join(scene_path_i, '*/'))
    for i, ins_path_i in enumerate(ins_path_list):
        print(i)
        _, _, _, _, _, _, _, _, cat_id, scene_id, obj_id, _ = ins_path_i.split('/')

        annotations_dict[scene_id]['cad'][f'{cat_id}_{obj_id}']
        ins_id, _ = obj_id.split('_')


        # 物体のポーズ
        loc = loc_dict[f'{cat_id}_{ins_id}']
        T_o2w = np.array(annotations_dict[scene_id]['cad'][f'{cat_id}_{obj_id}']['trs']["translation"]) # + loc # 並進を足す
        Q_o2w = np.array(annotations_dict[scene_id]['cad'][f'{cat_id}_{obj_id}']['trs']["rotation"])
        S_o2w = np.array(annotations_dict[scene_id]['cad'][f'{cat_id}_{obj_id}']['trs']["scale"])
        M_o2w = make_M_from_tqs(T_o2w, Q_o2w, S_o2w)
        o2w = quaternion2rotation(Q_o2w)
        obj_scale_wrd = S_o2w
        obj_pos_wrd = T_o2w - o2w @ (loc * obj_scale_wrd)

        # 修正
        if os.path.exists(os.path.join(ins_path_i, 'total_infos.pickle')):
            total_infos = pickle_load(os.path.join(ins_path_i, 'total_infos.pickle'))
            total_infos['obj']['obj_pos_wrd'] = obj_pos_wrd
            pickle_dump(total_infos, os.path.join(ins_path_i, 'total_infos.pickle'))
        else:
            import pdb; pdb.set_trace()
