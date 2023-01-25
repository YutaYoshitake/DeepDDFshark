import numpy as np
import os
import sys
import glob
import tqdm
import random
import json
from often_use import *
sys.path.append("../")
from parser_get_arg import *
from DDF.train_pl import DDF





# 書き換える
# python check_ddf.py --config=../configs/paper_exp/chair/view5/txt.txt
target_cat_id = '03001627'

# Get args
args = get_args()
# args.ddf_model_path = '/home/yyoshitake/works/DeepSDF/project/DDF/lightning_logs/chair/cat_depth_mae_normal_mae_seed0_normal001_lr00001/checkpoints/0000010000.ckpt'

# Set ddf.
ddf = DDF(args)
ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args)
ddf.eval()
ddf_instance_list = txt2list(args.ddf_instance_list_txt)

# Scan2CADのデータにあるインスタンス
annotations_json_open = open('data2/scan2cad/full_annotations.json', 'r')
annotations_json_load = json.load(annotations_json_open)
# Prepare Annotation dictを作成
ins_list = []
for annotation_i in annotations_json_load:
    scan_id_i = annotation_i['id_scan']
    shapenet_idx_cnt = {}
    for cad_idx, cad_i in enumerate(annotation_i['aligned_models']):
        catid_cad = cad_i['catid_cad']
        id_cad = cad_i['id_cad']
        if catid_cad == target_cat_id:
            ins_list.append(id_cad)
        # if catid_cad in {'03001627', '02933112', '03211117', '04379243', }:
        #     ins_list.append(f'{catid_cad}_{id_cad}')



scan2cad_ins_list = list(set(ins_list))
for ins_id in tqdm.tqdm(scan2cad_ins_list):
    # 訓練データにあるものを保存
    if ins_id in ddf_instance_list:
        instance_idx = [ddf_instance_list.index(ins_id)]
        gt_shape_code = ddf.lat_vecs(torch.tensor(instance_idx))
        gt_shape_code = gt_shape_code.to('cpu').detach().numpy().copy()
        save_path = os.path.join('gt_latent_code', target_cat_id, ins_id)
        np.save(save_path, gt_shape_code)
        ...
    else:
        txt_log_path = os.path.join('gt_latent_code', target_cat_id + '.txt')
        with open(txt_log_path, 'a') as f:
            print(ins_id, file=f)




# scan2cad_ins_list = list(set(ins_list))
# for ins_id in tqdm.tqdm(scan2cad_ins_list):
#     txt_log_path = os.path.join(f'in_scannet.txt')
#     with open(txt_log_path, 'a') as f:
#         print(ins_id, file=f)
