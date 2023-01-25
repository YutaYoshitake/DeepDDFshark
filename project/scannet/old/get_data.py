import json
import numpy as np
import sys
import os
import cv2
import glob
import tqdm
import shutil
from often_use import *
from load_scannet_data import export
import scannet_utils
import quaternion


total_cat_in_shapenet_list = ['table', 'jar', 'skateboard', 'car', 'bottle', 'tower', 'chair', 'bookshelf', 'camera', 'airplane', 
'laptop', 'basket', 'sofa', 'knife', 'can', 'rifle', 'train', 'pillow', 'lamp', 'trash bin', 'mailbox', 
'watercraft', 'motorbike', 'dishwasher', 'bench', 'pistol', 'rocket', 'loudspeaker', 'file cabinet', 
'bag', 'cabinet', 'bed', 'birdhouse', 'display', 'piano', 'earphone', 'telephone', 'stove', 'microphone', 
'bus', 'mug', 'remote', 'bathtub', 'bowl', 'keyboard', 'guitar', 'washer', 'bicycle', 'faucet', 'printer', 'cap', 'trash_bin', 'tv or monitor']

error_cat_list = {
    'total': [
        'zero_pt', 'wall', 'keyboard', 'cup or mug', 'crate', 'floor', 'computer tower', 'tissue box', 
        'paper', 'book', 'door', 'window', 'pillow', 'picture', 'ceiling', 'plant', 'toiletry', 'chain', 'backpack', 'salt', 'tray'
    ]
}

ok_cat_list = {
    'table': ['nightstand'], 
    'chair': ['toilet'], 
    'display': [], 
    'cabinet': ['kitchen cabinet', 'refrigerator', 'dresser'], 
}

#########################
# アノテーションの作成
#########################
SCANNET_DIR = '/home/yyoshitake/works/DeepSDF/project/scannet/data2/scans/raw' # 'data2/scans' # 
SCANNET_DIR_unzipped = 'data2/scans'
LABEL_MAP_FILE = 'data2/meta_data/scannetv2-labels.combined.tsv'
target_shapenet_cat = {'04379243': 'table', 
                       '03001627': 'chair', 
                       '03211117': 'display', 
                       '02933112': 'cabinet', }
# target_shapenet_cat = {'03001627': 'chair', } # {'04379243': 'table'} # 
out_dir = 'results'
inv_target_shapenet_cat = inverse_dict(target_shapenet_cat)

# ScanNet target_scan_id
target_scan_list =  [sys.argv[-1]] # [sys.argv[-1]] # ['scene0314_00'] # ['scene0030_01']

# target_scan_id = 'scene0320_00'
for target_scan_id in target_scan_list:
    if not os.path.exists(os.path.join(SCANNET_DIR, target_scan_id)):
        print('not exist!!!')
        sys.exit()
    print('###################################')
    print(f'start making {target_scan_id}')
    print('###################################')

    # 出力用のDir作成
    loc_dict = pickle_load(os.path.join(out_dir, 'loc_dict.pickle')) # make_offet_pickle.pyで作る
    for cat_id in target_shapenet_cat.keys():
        cat_dir_path = os.path.join(out_dir, cat_id)
        # if not os.path.exists(cat_dir_path):
        #     os.mkdir(cat_dir_path)
    for cat_id in target_shapenet_cat.keys():
        scene_dir_path = os.path.join(out_dir, cat_id, target_scan_id)
        # if not os.path.exists(scene_dir_path):
        #     os.mkdir(scene_dir_path)

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
        if scan_id_i == target_scan_id:
            shapenet_idx_cnt = {}
            for cad_idx, cad_i in enumerate(annotation_i['aligned_models']):
                catid_cad = cad_i['catid_cad']
                id_cad = cad_i['id_cad']
                if not id_cad in shapenet_idx_cnt.keys():
                    shapenet_idx_cnt[id_cad] = 1
                else:
                    shapenet_idx_cnt[id_cad] = shapenet_idx_cnt[id_cad] + 1
                if catid_cad in target_shapenet_cat.keys():
                    tgt_cad_idx = str(shapenet_idx_cnt[id_cad])
                    total_id = f'{catid_cad}_{id_cad}_{str(tgt_cad_idx).zfill(3)}'
                    annotations_dict[scan_id_i]['cad'][total_id] = {}
                    annotations_dict[scan_id_i]['cad'][total_id]['trs'] = cad_i['trs'] # <-- transformation from CAD space to world space 
                    annotations_dict[scan_id_i]['cad'][total_id]['sym'] = cad_i['sym']
                    annotations_dict[scan_id_i]['cad'][total_id]["bbox"] = cad_i["bbox"]
                    annotations_dict[scan_id_i]['cad'][total_id]["center"] = cad_i["center"]
    if len(annotations_dict[target_scan_id]['cad']) == 0:
        print('No instance ...')
        sys.exit()
    else:
        for cat_id in target_shapenet_cat.keys():
            cat_dir_path = os.path.join(out_dir, cat_id)
            # if not os.path.exists(cat_dir_path):
            #     os.mkdir(cat_dir_path)

    # Vis Seg Mesh
    mesh_file = os.path.join(SCANNET_DIR, target_scan_id, target_scan_id + '_vh_clean_2.ply')
    agg_file = os.path.join(SCANNET_DIR, target_scan_id, target_scan_id + '.aggregation.json')
    seg_file = os.path.join(SCANNET_DIR, target_scan_id, target_scan_id + '_vh_clean_2.0.010000.segs.json')
    meta_file = os.path.join(SCANNET_DIR, target_scan_id, target_scan_id + '.txt') # includes axisAlignment info for the train set scans. 
    scene_vertices, semantic_labels, vertices_instance_labels, shapenet_instance_bboxes, instance2semantic, shapenet_cat_list, scannet_cat_list = \
        export(mesh_file, agg_file, seg_file, meta_file, LABEL_MAP_FILE, None)
    scene_trs = annotations_dict[target_scan_id]['trs']
    Mscan = make_M_from_tqs(scene_trs["translation"], scene_trs["rotation"], scene_trs["scale"]) # <-- transformation from scan space to world space
    # 世界座標系のScan頂点
    vertices = scene_vertices[..., :3]
    vertices = np.sum(np.concatenate([vertices, np.ones_like(vertices)[..., :1]], axis=-1)[..., None, :]*Mscan[None, :, :], axis=-1)[..., :3]
    scene_vertices[..., :3] = vertices.copy() 
    # 世界座標系のBBox
    vertices = shapenet_instance_bboxes[..., :3] # shapenet_instance_bboxes is (cx,cy,cz) and (dx,dy,dz) and label id
    vertices = np.sum(np.concatenate([vertices, np.ones_like(vertices)[..., :1]], axis=-1)[..., None, :]*Mscan[None, :, :], axis=-1)[..., :3]
    scannet_centers_wrd = vertices.copy() # BBoxの世界座標系での中心位置

    # Scan2CADのアノテーションからターゲットではないカテゴリを削除
    # また、各CADモデルの世界座標系における中心位置を取得
    scan2cad_centers_wrd = [] # Scan2Cadの中心位置を取得
    for _, instance_infos in annotations_dict[target_scan_id]['cad'].items():
        Mcad = make_M_from_tqs(instance_infos['trs']["translation"], instance_infos['trs']["rotation"], instance_infos['trs']["scale"])
        scan2cad_centers_wrd.append(np.dot(Mcad, np.array([0., 0., 0., 1.]))[0:3])
    scan2cad_centers_wrd = np.array(scan2cad_centers_wrd) # Scan2Cadの中心位置を取得

    # 時間のログ
    import datetime
    dt_now = datetime.datetime.now()
    time_log = dt_now.strftime('%Y%m%d%H%M%S')
    # Scan2CADの中心位置と、ScanNetの中心位置で最も距離が近い者同士を結びつける。
    center_distances = np.linalg.norm(scannet_centers_wrd[:, None, :] - scan2cad_centers_wrd[None, :, :], axis=-1) # 
    scan2cad2scannet = {}
    label_map = scannet_utils.read_label_mapping(LABEL_MAP_FILE, label_from='raw_category', label_to='nyu40id') 
    for idx, (instance_id, _) in enumerate(annotations_dict[target_scan_id]['cad'].items()):
        print(f'{instance_id}')
        catid = instance_id.split('_')[0]
        # scannet_instance_idx = np.argmin(center_distances[:, idx]) # 最も中心距離が近いモデルを取得
        # annotations_dict[target_scan_id]['cad'][instance_id]['shapened_semantic_id'] = int(shapenet_instance_bboxes[scannet_instance_idx, -1])
        # annotations_dict[target_scan_id]['cad'][instance_id]['shapened_instance_id'] = scannet_instance_idx + 1
        # # 正しくマッチしたかの確認
        # #    Scan2CADのカテゴリとScanNetのカテゴリを比較
        # #    -> Scan2CADのカテゴリ:ScanNetのカテゴリ:::Scannetのカテゴリ:ScannetのマッチしたShapeNetCoreのカテゴリ
        # bbox_shapenet_cat = shapenet_cat_list[scannet_instance_idx]
        # bbox_scannet_cat = scannet_cat_list[scannet_instance_idx]
        # if bbox_shapenet_cat == '':
        #     with open(os.path.join(out_dir, f'unmatch_blank.txt'), 'a') as f: # カテゴリが空白
        #         print(f'{instance_id}:{target_shapenet_cat[catid]}:::{bbox_scannet_cat}:{bbox_shapenet_cat}', file=f)
        # else:
        #     matched = False
        #     if bbox_shapenet_cat in inv_target_shapenet_cat.keys(): # カテゴリがターゲットのものでない
        #         bbox_shapenet_catid = inv_target_shapenet_cat[bbox_shapenet_cat]
        #         matched = catid == bbox_shapenet_catid # カテゴリがターゲットのもの
        #     else:
        #         with open(os.path.join(out_dir, f'unmatch.txt'), 'a') as f:
        #             print(f'{instance_id}:{target_shapenet_cat[catid]}:::{bbox_scannet_cat}:{bbox_shapenet_cat}', file=f)

        # 変えてみた
        sorted_near_instance_idx_list = np.argsort(center_distances[:, idx])
        past_past_matched_idx_list = []
        for list_idx, scannet_instance_idx in enumerate(sorted_near_instance_idx_list):
            past_past_matched_idx_list.append(scannet_instance_idx)
            bbox_shapenet_cat = shapenet_cat_list[scannet_instance_idx]
            bbox_scannet_cat = scannet_cat_list[scannet_instance_idx]

            print(f'unmatch? {list_idx}')
            if center_distances[scannet_instance_idx, idx] > 2.0: # BBox同士の距離が遠すぎる
                with open(os.path.join(out_dir, f'too_far.txt'), 'a') as f:
                    print(f'{instance_id}:{target_shapenet_cat[catid]}:::{bbox_scannet_cat}:{bbox_shapenet_cat}', file=f)
                    for past_matched_idx in range(list_idx):
                        print(f'     {shapenet_cat_list[past_matched_idx]}:{shapenet_cat_list[past_matched_idx]}', file=f)
                break

            # 正しくマッチしたかの確認
            #    Scan2CADのカテゴリとScanNetのカテゴリを比較
            #    -> Scan2CADのカテゴリ:ScanNetのカテゴリ:::Scannetのカテゴリ:ScannetのマッチしたShapeNetCoreのカテゴリ
            matched = False
            if bbox_shapenet_cat in total_cat_in_shapenet_list: # カテゴリがShapeNetに存在するか？
                matched = target_shapenet_cat[catid] == bbox_shapenet_cat # カテゴリがターゲットのものならOK
                if not(matched) and target_shapenet_cat[catid] == 'display':
                    matched = 'tv or monitor' == bbox_shapenet_cat
            else:
                if not bbox_scannet_cat in error_cat_list['total']: # 小さなものと誤って結びついたわけではないか？
                    if bbox_scannet_cat in ok_cat_list[target_shapenet_cat[catid]]:
                        matched = True
                    else:
                        with open(os.path.join(out_dir, f'match_one_not_in_shapecat.txt'), 'a') as f:
                            print(f'{target_scan_id}:{instance_id}:{target_shapenet_cat[catid]}:::{bbox_scannet_cat}:{bbox_shapenet_cat}', file=f)
                        matched = True
            
            if not(matched) and list_idx == 0:
                with open(os.path.join(out_dir, f'unmatched_init_one.txt'), 'a') as f:
                    print(f'{target_scan_id}:{instance_id}:{target_shapenet_cat[catid]}:::{bbox_scannet_cat}:{bbox_shapenet_cat}', file=f)
            
            if matched:
                # 
                if list_idx == 1:
                    with open(os.path.join(out_dir, f'unmatch_1_times.txt'), 'a') as f:
                        print(f'{target_scan_id}:{instance_id}:{target_shapenet_cat[catid]}:::{bbox_scannet_cat}:{bbox_shapenet_cat}:::{str(list_idx)}', file=f)
                        for past_matched_idx in past_past_matched_idx_list:
                            print(f'     {shapenet_cat_list[past_matched_idx]}:{shapenet_cat_list[past_matched_idx]}', file=f)
                elif list_idx == 2:
                    with open(os.path.join(out_dir, f'unmatch_2_times.txt'), 'a') as f:
                        print(f'{target_scan_id}:{instance_id}:{target_shapenet_cat[catid]}:::{bbox_scannet_cat}:{bbox_shapenet_cat}:::{str(list_idx)}', file=f)
                        for past_matched_idx in past_past_matched_idx_list:
                            print(f'     {shapenet_cat_list[past_matched_idx]}:{shapenet_cat_list[past_matched_idx]}', file=f)
                elif list_idx > 2:
                    with open(os.path.join(out_dir, f'unmatch_3_times.txt'), 'a') as f:
                        print(f'{target_scan_id}:{instance_id}:{target_shapenet_cat[catid]}:::{bbox_scannet_cat}:{bbox_shapenet_cat}:::{str(list_idx)}', file=f)
                        for past_matched_idx in past_past_matched_idx_list:
                            print(f'     {shapenet_cat_list[past_matched_idx]}:{shapenet_cat_list[past_matched_idx]}', file=f)
                # 
                annotations_dict[target_scan_id]['cad'][instance_id]['shapened_semantic_id'] = int(shapenet_instance_bboxes[scannet_instance_idx, -1])
                annotations_dict[target_scan_id]['cad'][instance_id]['shapened_instance_id'] = scannet_instance_idx + 1
                with open(os.path.join(out_dir, f'match.txt'), 'a') as f:
                    print(f'{target_scan_id}:{instance_id}:{target_shapenet_cat[catid]}:::{bbox_scannet_cat}:{bbox_shapenet_cat}', file=f)
                break
            elif list_idx == len(sorted_near_instance_idx_list):
                # 
                with open(os.path.join(out_dir, f'not_matched_but_list_finished.txt'), 'a') as f:
                    print(f'{target_scan_id}:{instance_id}:{target_shapenet_cat[catid]}:::{bbox_scannet_cat}:{bbox_shapenet_cat}', file=f)

    # #########################
    # # 現シーンのデータ作成
    # #########################
    # # 保存するDirの作成
    # for instance_id in annotations_dict[target_scan_id]['cad'].keys():
    #     cat_id, ins_id, obj_idx = instance_id.split('_')
    #     obj_dir = os.path.join(out_dir, cat_id, target_scan_id, f'{ins_id}_{obj_idx}')
    #     os.makedirs(os.path.join(obj_dir), exist_ok=True)
    #     os.makedirs(os.path.join(obj_dir, 'data_dict'), exist_ok=True)
    #     os.makedirs(os.path.join(obj_dir, 'video'), exist_ok=True)

    # # 視点数のカウント用
    # obj_total_infos = {}
    # for target_instance_id in annotations_dict[target_scan_id]['cad'].keys():
    #     annotations_dict[target_scan_id]['cad'][target_instance_id]['view_num'] = 0
    #     annotations_dict[target_scan_id]['cad'][target_instance_id]['pre_img_id'] = -1
    #     annotations_dict[target_scan_id]['cad'][target_instance_id]['view_seq_id'] = 0

    #     # Viewの情報を格納
    #     obj_total_infos[target_instance_id] = {}
    #     obj_total_infos[target_instance_id]['frame'] = {}
    #     obj_total_infos[target_instance_id]['frame']['cam_pos_wrd_list'] = []
    #     obj_total_infos[target_instance_id]['frame']['w2c_list'] = []
    #     obj_total_infos[target_instance_id]['frame']['img_name_list'] = []
    #     obj_total_infos[target_instance_id]['frame']['mask_ratio_list'] = []
    #     obj_total_infos[target_instance_id]['frame']['dist_med_list'] = []
    #     # 物体のポーズ
    #     category_id, instance_id, obj_id = target_instance_id.split('_')
    #     loc = loc_dict[f'{category_id}_{instance_id}']
    #     T_o2w = np.array(annotations_dict[target_scan_id]['cad'][target_instance_id]['trs']["translation"]) # + loc # 並進を足す
    #     Q_o2w = np.array(annotations_dict[target_scan_id]['cad'][target_instance_id]['trs']["rotation"])
    #     S_o2w = np.array(annotations_dict[target_scan_id]['cad'][target_instance_id]['trs']["scale"])
    #     M_o2w = make_M_from_tqs(T_o2w, Q_o2w, S_o2w)
    #     o2w = quaternion2rotation(Q_o2w)
    #     obj_scale_wrd = S_o2w
    #     obj_pos_wrd = T_o2w - o2w @ loc * obj_scale_wrd
    #     sym_label = annotations_dict[target_scan_id]['cad'][target_instance_id]['sym']
    #     obj_total_infos[target_instance_id]['obj'] = {}
    #     obj_total_infos[target_instance_id]['obj']['obj_pos_wrd'] = obj_pos_wrd.astype(np.float32)
    #     obj_total_infos[target_instance_id]['obj']['o2w'] = o2w.astype(np.float32)
    #     obj_total_infos[target_instance_id]['obj']['obj_scale_wrd'] = obj_scale_wrd.astype(np.float32)
    #     obj_total_infos[target_instance_id]['obj']['sym_label'] = str(sym_label)

    #     # BBoxの取得
    #     box_corner_vertices = np.array([
    #                                     [-1, -1, -1],
    #                                     [ 1, -1, -1],
    #                                     [ 1,  1, -1],
    #                                     [-1,  1, -1],
    #                                     [-1, -1,  1],
    #                                     [ 1, -1,  1],
    #                                     [ 1,  1,  1],
    #                                     [-1,  1,  1],
    #                                 ], dtype=np.float64)
    #     def calc_Mbbox(model):
    #         trs_obj = model['trs']
    #         bbox_obj = np.asarray(model["bbox"], dtype=np.float64)
    #         center_obj = np.asarray(model["center"], dtype=np.float64)
    #         tcenter1 = np.eye(4)
    #         tcenter1[0:3, 3] = center_obj
    #         trans1 = np.eye(4)
    #         bbox1 = np.eye(4)
    #         bbox1[0:3, 0:3] = np.diag(bbox_obj)
    #         M = tcenter1.dot(bbox1)
    #         return M
    #     Mbbox = calc_Mbbox(annotations_dict[target_scan_id]['cad'][target_instance_id])
    #     v_list = []
    #     for v in box_corner_vertices:
    #         v1 = np.array([v[0], v[1], v[2], 1])
    #         v1 = np.dot(Mbbox, v1)[0:3]
    #         v_list.append(v1)
    #     v_list = np.array(v_list) + loc
    #     obj_total_infos[target_instance_id]['obj']['bbox'] = v_list.astype(np.float32)

    # MAX_IM_NUM = len(glob.glob(os.path.join(SCANNET_DIR_unzipped, target_scan_id, 'depth/*')))

    # # スキャンデータの読み込み
    # filename = os.path.join(SCANNET_DIR, target_scan_id, target_scan_id + '.sens')
    # tgt_version = 4
    # COMPRESSION_TYPE_COLOR = {-1:'unknown', 0:'raw', 1:'png', 2:'jpeg'}
    # COMPRESSION_TYPE_DEPTH = {-1:'unknown', 0:'raw_ushort', 1:'zlib_ushort', 2:'occi_ushort'}
    # print('start_loading')
    # with open(filename, 'rb') as f:
    #     version = struct.unpack('I', f.read(4))[0]
    #     assert tgt_version == version
    #     strlen = struct.unpack('Q', f.read(8))[0]
    #     sensor_name = b''.join(struct.unpack('c'*strlen, f.read(strlen)))
    #     intrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
    #     extrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
    #     intrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
    #     extrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
    #     color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
    #     depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
    #     color_width = struct.unpack('I', f.read(4))[0]
    #     color_height =  struct.unpack('I', f.read(4))[0]
    #     depth_width = struct.unpack('I', f.read(4))[0]
    #     depth_height =  struct.unpack('I', f.read(4))[0]
    #     depth_shift =  struct.unpack('f', f.read(4))[0]
    #     num_frames =  struct.unpack('Q', f.read(8))[0]
    #     frames = []
    #     for i in tqdm.tqdm(range(num_frames)):
    #         frame = RGBDFrame()
    #         frame.load(f)
    #         frames.append(frame)
    # MAX_IM_NUM = num_frames

    # # インスタンスマスクの解凍
    # if len(glob.glob(os.path.join(SCANNET_DIR, target_scan_id, f'instance-filt/*.png'))) == 0:
    #     print('Start unzip ...')
    #     shutil.unpack_archive(os.path.join(SCANNET_DIR, target_scan_id, f'{target_scan_id}_2d-instance-filt.zip'), os.path.join(SCANNET_DIR, target_scan_id))
    #     print('Finished unzip !')

    # # 画像でループを回す
    # for img_cnt, img_id in enumerate(tqdm.tqdm(range(0, MAX_IM_NUM, 10))): # 0始まり

    #     # データの読み込み
    #     frame_infos = frames[img_id]
    #     depth_data = frame_infos.decompress_depth(depth_compression_type)
    #     total_depth = np.fromstring(depth_data, dtype=np.uint16).reshape(depth_height, depth_width) / 1000
    #     total_rgb = frame_infos.decompress_color(color_compression_type)
    #     pose = frame_infos.camera_to_world
    #     total_ins_mask = cv2.imread(os.path.join(SCANNET_DIR, target_scan_id, f'instance-filt/{img_id}.png'))[:, :, 0]
    #     # # データの読み込み
    #     # total_rgb = cv2.imread(os.path.join(SCANNET_DIR_unzipped, target_scan_id, f'color/{img_id}.jpg'))[..., [2, 1, 0]]
    #     # total_ins_mask = cv2.imread(os.path.join(SCANNET_DIR, target_scan_id, f'instance-filt/{img_id}.png'))[:, :, 0]
    #     # total_depth = np.load(os.path.join(SCANNET_DIR_unzipped, target_scan_id, f'depth/{img_id}.npy')) / 1000
    #     # pose = load_matrix_from_txt(os.path.join(SCANNET_DIR_unzipped, target_scan_id, f'pose/{img_id}.txt'))
    #     # intrinsic_depth = load_matrix_from_txt(os.path.join(SCANNET_DIR_unzipped, target_scan_id, 'intrinsic/intrinsic_depth.txt'))

    #     # Ray方向の取得
    #     H, W, C = total_rgb.shape
    #     d_H, d_W = total_depth.shape
    #     u_coord = np.tile(np.arange(0, d_W)[None, :], (d_H, 1))
    #     v_coord = np.tile(np.arange(0, d_H)[:, None], (1, d_W))
    #     fx = intrinsic_depth[0, 0]
    #     fy = intrinsic_depth[1, 1]
    #     cx = intrinsic_depth[0, 2]
    #     cy = intrinsic_depth[1, 2]
    #     total_rays_d_cam_z1 = np.stack([(u_coord - cx) / fx, (v_coord - cy) / fy, np.ones((d_H, d_W))], axis=-1)
    #     total_distance = total_depth * np.linalg.norm(total_rays_d_cam_z1, axis=-1)
    #     total_rays_d_cam = total_rays_d_cam_z1 / np.linalg.norm(total_rays_d_cam_z1, axis=-1)[:, :, None]

    #     # 回転行列の取得
    #     M_c2w = Mscan @ pose
    #     R_c2w = M_c2w[:3, :3] / M_c2w[3, 3]
    #     T_c2w = M_c2w[:3, 3] / M_c2w[3, 3]

    #     # Resize Depth map and UV map.
    #     total_depth = cv2.resize(total_depth.astype(np.float64), (W, H), interpolation=cv2.INTER_NEAREST)
    #     total_distance = cv2.resize(total_distance.astype(np.float64), (W, H), interpolation=cv2.INTER_NEAREST)
    #     total_rays_d_cam = cv2.resize(total_rays_d_cam.astype(np.float64), (W, H), interpolation=cv2.INTER_NEAREST)

    #     # インスタンスでループを回す
    #     total_ids = np.unique(total_ins_mask)
    #     for target_instance_id in annotations_dict[target_scan_id]['cad'].keys():
    #         category_id, instance_id, obj_id = target_instance_id.split('_')
            
    #         # 同じインスタンスラベルが画像内に存在するか？
    #         instance_label = annotations_dict[target_scan_id]['cad'][target_instance_id]['shapened_instance_id']
    #         if (total_ids==instance_label).any():
    #             cat_id, ins_id, obj_idx = target_instance_id.split('_')
    #             total_mask = total_ins_mask==instance_label

    #             mask_H = total_mask.shape[0]
    #             mask_W = total_mask.shape[1]
    #             x_coord = np.tile(np.arange(0, mask_W)[None, :], (mask_H, 1))
    #             y_coord = np.tile(np.arange(0, mask_H)[:, None], (1, mask_W))
    #             image_coord = np.stack([y_coord, x_coord], axis=-1)
    #             masked_image_coord = image_coord[total_mask]
    #             max_y, max_x = masked_image_coord.max(axis=0)
    #             min_y, min_x = masked_image_coord.min(axis=0)
    #             bbox = np.array([[max_x, max_y], [min_x, min_y]])

    #             H_x = max_x - min_x
    #             H_y = max_y - min_y
    #             bbox_H_xy = np.array([H_x, H_y])

    #             # 正方形でクロップし直す
    #             if max(H_x, H_y) < min(mask_W, mask_H):
    #                 square_H = max(H_x, H_y)
    #                 diff_H_xy = (square_H - bbox_H_xy) / 2
    #                 bbox = bbox + np.stack([diff_H_xy, -diff_H_xy], axis=0)
    #                 # はみ出したら戻す
    #                 border_xy = np.array([[mask_W-1., mask_H-1.], [0., 0.]])
    #                 outside_xy = border_xy - bbox
    #                 outside_xy[0, :][outside_xy[0, :] > .0] = 0. # 値が負ならMaxがはみ出た -> ずれを引く
    #                 outside_xy[1, :][outside_xy[1, :] < .0] = 0. # 値が正ならMinがはみ出た -> ずれを足す
    #                 bbox = bbox + outside_xy.sum(axis=0)
    #             else:
    #                 up_side, down_side = total_mask.sum(axis=1)[[0, -1]] # 下側と上側どちらにマスクが張り付いているか ?
    #                 if up_side >= down_side: # if up_side > 0:
    #                     bbox = np.array([[max_x, H_x], [min_x, 0]])
    #                 else: # if down_side > up_side: # elif down_side > 0:
    #                     bbox = np.array([[max_x, H_y], [min_x, H_y-H_x]])
    #                 # import pdb; pdb.set_trace()

    #             # 整数値に直す
    #             max_xy, min_xy = bbox
    #             max_x, max_y = max_xy
    #             min_x, min_y = min_xy
    #             max_x = min(-int(-max_x), mask_W-1)
    #             max_y = min(-int(-max_y), mask_H-1)
    #             min_x = max(int(min_x), 0)
    #             min_y = max(int(min_y), 0)

    #             # BBox内で画像のクロップ
    #             mask = total_mask[min_y:max_y, min_x:max_x]
    #             ins_mask = total_ins_mask[min_y:max_y, min_x:max_x]
    #             rgb = total_rgb[min_y:max_y, min_x:max_x]
    #             depth = total_depth[min_y:max_y, min_x:max_x]
    #             distance = total_distance[min_y:max_y, min_x:max_x]
    #             rays_d_cam = total_rays_d_cam[min_y:max_y, min_x:max_x]

                # enough_bbox_size = mask.sum() > 50**2 # マスクの大きさが一定以上でＯＫ
                # have_distance = (distance[mask] > 0).sum() > 50**2 # 内部にちゃんと移っている
                # if enough_bbox_size and have_distance:

    #                 # Fore Gound の作成（暫定）
    #                 blur_mask = cv2.GaussianBlur(mask.astype(np.float32), (25,25), 0)
    #                 blur_mask = blur_mask > 0 # 
    #                 fore_ground_mask = np.full_like(mask, False)
    #                 target_meddist = np.median(distance[mask])
    #                 masked_total_ids = np.unique(ins_mask)
    #                 for nontarget_id in masked_total_ids:
    #                     nontarget_mask = ins_mask == nontarget_id
    #                     nontarget_distance = distance[nontarget_mask]
    #                     if (nontarget_distance > 0).any():
    #                         nontarget_distance = nontarget_distance[nontarget_distance > 0]
    #                         # nontarget_meddist = np.median(nontarget_distance)
    #                         nontarget_meddist = np.mean(nontarget_distance)
    #                         if nontarget_meddist < target_meddist: # 手前なら（距離の中央値がより近ければ
    #                             fore_ground_mask[nontarget_mask] = True
    #                 # for nontarget_id in masked_total_ids:
    #                 #     # 広げた物体マスクと、周辺物体マスクの重なり
    #                 #     nontarget_mask = ins_mask == nontarget_id
    #                 #     edge_nontarget_mask = np.logical_and(blur_mask, nontarget_mask)
    #                 #     # 物体マスクと、広げた周辺物体マスクの重なり
    #                 #     blur_nontarget_mask = cv2.GaussianBlur(nontarget_mask.astype(np.float32), (25,25), 0)
    #                 #     blur_nontarget_mask = blur_nontarget_mask > 0
    #                 #     edge_target_mask = np.logical_and(mask, blur_nontarget_mask)
    #                 #     # check_map_np(blur_mask, 'tes_blur_mask.png')
    #                 #     # check_map_np(nontarget_mask, 'tes_nontarget_mask.png')
    #                 #     # check_map_np(edge_nontarget_mask, 'tes_edge_nontarget_mask.png')
    #                 #     # check_map_np(blur_nontarget_mask, 'tes_blur_nontarget_mask.png')
    #                 #     # check_map_np(mask, 'tes_mask.png')
    #                 #     # check_map_np(edge_target_mask, 'tes_edge_target_mask.png')
    #                 #     get_fore_mask = False
    #                 #     if (edge_target_mask).any() and (edge_nontarget_mask).any():
    #                 #         edge_target_depth = depth[edge_target_mask]
    #                 #         edge_nontarget_depth = depth[blur_nontarget_mask]
    #                 #         nonzero_edge_target_depth_mask = edge_target_depth > 0
    #                 #         nonzero_edge_nontarget_depth_mask = edge_nontarget_depth > 0
    #                 #         if (nonzero_edge_target_depth_mask).any() and (nonzero_edge_nontarget_depth_mask).any():
    #                 #             edge_target_meddep = np.median(edge_target_depth[nonzero_edge_target_depth_mask])
    #                 #             edge_nontarget_meddep = np.median(edge_nontarget_depth[nonzero_edge_nontarget_depth_mask])
    #                 #             if edge_target_meddep < edge_nontarget_meddep: # 手前なら（距離の中央値がより近ければ
    #                 #                 fore_ground_mask[nontarget_mask] = True
    #                 #                 get_fore_mask = True
    #                 #     if not get_fore_mask:
    #                 #         nontarget_meddist = np.median(distance[nontarget_mask])
    #                 #         if nontarget_meddist < target_meddist: # 手前なら（距離の中央値がより近ければ
    #                 #             fore_ground_mask[nontarget_mask] = True

    #                 # Resize_images:
    #                 clopped_H = 128 # 256
    #                 mask = cv2.resize(mask.astype(np.float64), (clopped_H, clopped_H), interpolation=cv2.INTER_NEAREST) > .5
    #                 fore_ground_mask = cv2.resize(fore_ground_mask.astype(np.float64), (clopped_H, clopped_H), interpolation=cv2.INTER_NEAREST) > .5
    #                 distance = cv2.resize(distance, (clopped_H, clopped_H), interpolation=cv2.INTER_NEAREST)
    #                 depth = cv2.resize(depth, (clopped_H, clopped_H), interpolation=cv2.INTER_NEAREST)
    #                 rays_d_cam = cv2.resize(rays_d_cam, (clopped_H, clopped_H), interpolation=cv2.INTER_NEAREST)
    #                 rgb = cv2.resize(rgb, (clopped_H, clopped_H), interpolation=cv2.INTER_NEAREST)

    #                 # 画像名の作成
    #                 if annotations_dict[target_scan_id]['cad'][target_instance_id]['pre_img_id'] > 0:
    #                     if annotations_dict[target_scan_id]['cad'][target_instance_id]['pre_img_id'] + 1 < img_cnt:
    #                         annotations_dict[target_scan_id]['cad'][target_instance_id]['view_seq_id'] += 1
    #                 instance_img_id = annotations_dict[target_scan_id]['cad'][target_instance_id]['view_num']
    #                 instance_seq_id = annotations_dict[target_scan_id]['cad'][target_instance_id]['view_seq_id']
    #                 annotations_dict[target_scan_id]['cad'][target_instance_id]['pre_img_id'] = img_cnt
    #                 instance_data_name = '{}_{}_{}'.format(str(instance_img_id).zfill(5), str(instance_seq_id).zfill(5), str(img_cnt).zfill(5))

    #                 # カメラポーズなどの取得
    #                 loc = loc_dict[f'{category_id}_{instance_id}']
    #                 T_o2w = np.array(annotations_dict[target_scan_id]['cad'][target_instance_id]['trs']["translation"]) # + loc # 並進を足す
    #                 Q_o2w = np.array(annotations_dict[target_scan_id]['cad'][target_instance_id]['trs']["rotation"])
    #                 S_o2w = np.array(annotations_dict[target_scan_id]['cad'][target_instance_id]['trs']["scale"])
    #                 M_o2w = make_M_from_tqs(T_o2w, Q_o2w, S_o2w)
    #                 o2w = quaternion2rotation(Q_o2w)
    #                 obj_scale_wrd = S_o2w
    #                 obj_pos_wrd = T_o2w - o2w @ loc * obj_scale_wrd
    #                 cam_pos_wrd = M_c2w[:3, 3]
    #                 w2c = M_c2w[:3, :3].T
    #                 sym_label = annotations_dict[target_scan_id]['cad'][target_instance_id]['sym']
    #                 bbox_list = np.array([[max_x, max_y], [min_x, min_y]]).astype(np.float32)
    #                 bbox_list[:, 0] = bbox_list[:, 0] / (0.5 * mask_W) - 1 # change range [-1, 1]
    #                 bbox_list[:, 1] = bbox_list[:, 1] / (0.5 * mask_H) - 1 # change range [-1, 1]
    #                 bbox_diagonal = (bbox_list[0, 0] - bbox_list[1, 0]) / 2

    #                 # 結果の保存
    #                 data_dict = {}
    #                 data_dict['clopped_mask'] = mask # .astype(np.bool)
    #                 data_dict['clopped_ins_mask'] = ins_mask # .astype(np.bool)
    #                 data_dict['clopped_fore_mask'] = fore_ground_mask # .astype(np.bool)
    #                 data_dict['clopped_distance'] = distance.astype(np.float32)
    #                 data_dict['clopped_depth'] = depth.astype(np.float32)
    #                 data_dict['clopped_rays_d_cam'] = rays_d_cam.astype(np.float32)
    #                 data_dict['cam_pos_wrd'] = cam_pos_wrd.astype(np.float32)
    #                 data_dict['w2c'] = w2c.astype(np.float32)
    #                 data_dict['bbox_list'] = bbox_list.astype(np.float32)
    #                 data_dict['bbox_diagonal'] = bbox_diagonal.astype(np.float32)
    #                 data_dict['obj_pos_wrd'] = obj_pos_wrd.astype(np.float32)
    #                 data_dict['o2w'] = o2w.astype(np.float32)
    #                 data_dict['obj_scale_wrd'] = obj_scale_wrd.astype(np.float32)
    #                 data_dict['sym_label'] = str(sym_label)
    #                 data_dict['pose'] = M_c2w
    #                 pickle_dump(data_dict, os.path.join(out_dir, cat_id, target_scan_id, f'{ins_id}_{obj_idx}', 'data_dict', instance_data_name+'.pickle'))

    #                 # 確認用のビデオ
    #                 d_max = depth[depth>0].max()
    #                 d_min = depth[depth>0].min()
    #                 vis_d = (depth - d_min) / (d_max - d_min)
    #                 vis_d = (255 * np.clip(vis_d, 0.0, 1.0)).astype(np.uint8)
    #                 vis_d = cv2.applyColorMap(vis_d, cv2.COLORMAP_JET)
    #                 vis_mask = np.zeros_like(rgb)
    #                 vis_mask[fore_ground_mask] = np.array([255, 255, 0]) # c
    #                 vis_mask[mask] = np.array([147, 20, 255]) # m
    #                 vis_ins_mask = cv2.resize(ins_mask, (clopped_H, clopped_H), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    #                 m_max = vis_ins_mask.max()
    #                 m_min = vis_ins_mask.min()
    #                 vis_ins_mask = (vis_ins_mask - m_min) / (m_max - m_min)
    #                 vis_ins_mask = np.tile(255 * vis_ins_mask[..., None], (1, 1, 3)).astype(np.uint8)
    #                 video_frame = np.concatenate([cv2.resize(total_rgb[..., [2, 1, 0]], (int(W/H*clopped_H), clopped_H)), rgb[..., [2, 1, 0]], vis_d, vis_ins_mask, vis_mask], axis=1)
    #                 cv2.imwrite('tes.png', video_frame)
    #                 cv2.imwrite(os.path.join(out_dir, cat_id, target_scan_id, f'{ins_id}_{obj_idx}', 'video', instance_data_name+'.png'), video_frame)

    #                 # 画像数のカウント
    #                 annotations_dict[target_scan_id]['cad'][target_instance_id]['view_num'] += 1
    #                 annotations_dict[target_scan_id]['cad'][target_instance_id]['pre_img_id'] = img_cnt

    #                 # カメラポーズを記録
    #                 mask_ratio = total_mask.sum() / (mask_H * mask_W)
    #                 mask_ratio_int = int(mask_ratio * 10000)
    #                 target_meddist_int = int(target_meddist * 1000)
    #                 obj_total_infos[target_instance_id]['frame']['cam_pos_wrd_list'].append(cam_pos_wrd.astype(np.float32))
    #                 obj_total_infos[target_instance_id]['frame']['w2c_list'].append(w2c.astype(np.float32))
    #                 obj_total_infos[target_instance_id]['frame']['img_name_list'].append(instance_data_name)
    #                 obj_total_infos[target_instance_id]['frame']['mask_ratio_list'].append(mask_ratio_int)
    #                 obj_total_infos[target_instance_id]['frame']['dist_med_list'].append(target_meddist_int)

    # # カメラポーズのみの結果
    # for target_instance_id in annotations_dict[target_scan_id]['cad'].keys():
    #     cat_id, ins_id, obj_idx = target_instance_id.split('_')
    #     tgtobj_total_infos = obj_total_infos[target_instance_id]
    #     tgtobj_total_infos['frame']['cam_pos_wrd_list'] = np.array(tgtobj_total_infos['frame']['cam_pos_wrd_list'])
    #     tgtobj_total_infos['frame']['w2c_list'] = np.array(tgtobj_total_infos['frame']['w2c_list'])
    #     tgtobj_total_infos['frame']['img_name_list'] = np.array(tgtobj_total_infos['frame']['img_name_list'])
    #     tgtobj_total_infos['frame']['mask_ratio_list'] = np.array(tgtobj_total_infos['frame']['mask_ratio_list'])
    #     tgtobj_total_infos['frame']['dist_med_list'] = np.array(tgtobj_total_infos['frame']['dist_med_list'])
    #     pickle_dump(tgtobj_total_infos, os.path.join(out_dir, cat_id, target_scan_id, f'{ins_id}_{obj_idx}', 'total_infos.pickle'))

    # # 視点数のあるインスタンスを取得
    # has_view_instances = []
    # for target_instance_id in annotations_dict[target_scan_id]['cad'].keys():
    #     cat_id, ins_id, obj_idx = target_instance_id.split('_')
    #     if annotations_dict[target_scan_id]['cad'][target_instance_id]['view_num'] > 0:
    #         has_view_instances.append(os.path.join(target_scan_id, f'{ins_id}_{obj_idx}'))
    #     else:
    #         out_dir = os.path.join(out_dir, cat_id, target_scan_id, f'{ins_id}_{obj_idx}')
    #         shutil.rmtree(out_dir)

    # # そのシーンにカテゴリがなかったら、無駄なもの消す
    # for cat_id in target_shapenet_cat.keys():
    #     if len(glob.glob(os.path.join(out_dir, cat_id, target_scan_id, '*'))) == 0:
    #         shutil.rmtree(os.path.join(out_dir, cat_id, target_scan_id))
    #     else:
    #         # scene_infos
    #         scene_total_infos = {}
    #         scene_total_infos['dir_list'] = has_view_instances
    #         pickle_dump(scene_total_infos, os.path.join(out_dir, cat_id, target_scan_id, 'scene_total_infos.pickle'))
