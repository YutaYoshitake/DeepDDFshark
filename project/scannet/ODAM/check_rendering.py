import json
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import glob
import tqdm
import pylab
from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
import cv2
from segmentation_color import get_colored_segmap
from often_use import *
from load_scannet_data import export
# import quaternion



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

def loadOBJ(fliePath):
    vertices = []
    for line in open(fliePath, "r"):
        vals = line.split()
        if len(vals) == 0:
            continue
        if vals[0] == "v":
            v = np.array([float(v_i) for v_i in vals[1:4]])
            vertices.append(v)
    return np.array(vertices)



# Full scan2cad annotations
annotations_json_open = open('data2/scan2cad/full_annotations.json', 'r')
annotations_json_load = json.load(annotations_json_open)

# simple scan2cad annotations
appearances_json_open = open('data2/scan2cad/cad_appearances.json')
appearances_json_load = json.load(appearances_json_open)

# Prepare Annotation dict.
annotations_dict = {}
for annotation_i in annotations_json_load:
    scan_id = annotation_i['id_scan']
    annotations_dict[scan_id] = {}
    annotations_dict[scan_id]['trs'] = annotation_i['trs'] # <-- transformation from scan space to world space
    annotations_dict[scan_id]['cad'] = {}
    appearances_scene_i = appearances_json_load[scan_id]
    # if scan_id == 'scene0000_00':
    #     import pdb; pdb.set_trace()
    for cad_i in annotation_i['aligned_models']:
        catid_cad = cad_i['catid_cad']
        id_cad = cad_i['id_cad']
        total_id = f'{catid_cad}_{id_cad}'
        annotations_dict[scan_id]['cad'][total_id] = {}
        annotations_dict[scan_id]['cad'][total_id]['trs'] = cad_i['trs'] # <-- transformation from CAD space to world space 
        annotations_dict[scan_id]['cad'][total_id]['sym'] = cad_i['sym']
        annotations_dict[scan_id]['cad'][total_id]['appnum'] = appearances_scene_i[total_id]



SCANNET_DIR = 'data2/scans'
LABEL_MAP_FILE = 'data2/meta_data/scannetv2-labels.combined.tsv'

# ScanNet Scene_id
scan_id = 0
sub_scene_id = 0
scene_id = f'scene{str(0).zfill(4)}_{str(0).zfill(2)}'

# Vis Seg Mesh
mesh_file = os.path.join(SCANNET_DIR, scene_id, scene_id + '_vh_clean_2.ply')
agg_file = os.path.join(SCANNET_DIR, scene_id, scene_id + '.aggregation.json')
seg_file = os.path.join(SCANNET_DIR, scene_id, scene_id + '_vh_clean_2.0.010000.segs.json')
meta_file = os.path.join(SCANNET_DIR, scene_id, scene_id + '.txt') # includes axisAlignment info for the train set scans.   
scene_vertices, semantic_labels, vertices_instance_labels, shapenet_instance_bboxes, instance2semantic = \
    export(mesh_file, agg_file, seg_file, meta_file, LABEL_MAP_FILE, None)
scene_trs = annotations_dict[scene_id]['trs']
Mscan = make_M_from_tqs(scene_trs["translation"], scene_trs["rotation"], scene_trs["scale"]) # <-- transformation from scan space to world space

vertices = scene_vertices[..., :3]
vertices = np.sum(np.concatenate([vertices, np.ones_like(vertices)[..., :1]], axis=-1)[..., None, :]*Mscan[None, :, :], axis=-1)[..., :3]
scene_vertices[..., :3] = vertices.copy()
# scene_vertices_check = scene_vertices[..., :3].copy()
# point_cloud = []
# for v in scene_vertices_check: 
#     v1 = np.array([v[0], v[1], v[2], 1])
#     v[:3] = np.dot(Mscan, v1)[:3]
#     point_cloud.append(v)
# scene_vertices_check = np.array(point_cloud)
# import pdb; pdb.set_trace()


# # Vis Seg Map.
# img_id = 3872 # 4150, 3872
# sample_depth_path = f'{SCANNET_DIR}/scene0000_00/depth/{img_id}.png'
# instance_segmentation_map = cv2.imread(f'{SCANNET_DIR}/scene0000_00/instance-filt/{img_id}.png')
# result_map = get_colored_segmap(instance_segmentation_map)
# cv2.imwrite('tes_seg.png', result_map[:, :, [2, 1, 0]])
# import pdb; pdb.set_trace()


# ScanNet instance to Scan2CAD instance.
target_shapenet_cat = {'03001627': 'chair'}
# target_shapenet_cat = {'04379243': 'table', 
#                        '03001627': 'chair', 
#                        '03211117': 'display', 
#                        '02933112': 'cabinet', }
vertices = shapenet_instance_bboxes[..., :3]
vertices = np.sum(np.concatenate([vertices, np.ones_like(vertices)[..., :1]], axis=-1)[..., None, :]*Mscan[None, :, :], axis=-1)[..., :3]
shapenet_centers_wrd = vertices.copy()# (cx,cy,cz) and (dx,dy,dz) and label id

annotations_dict_tmp = {}
scan2cad_centers_wrd = []
for instance_id, instance_infos in annotations_dict[scene_id]['cad'].items():
    if instance_id.split('_')[0] in target_shapenet_cat.keys():
        Mcad = make_M_from_tqs(instance_infos['trs']["translation"], instance_infos['trs']["rotation"], instance_infos['trs']["scale"])
        scan2cad_centers_wrd.append(np.dot(Mcad, np.array([0., 0., 0., 1.]))[0:3])
        annotations_dict_tmp[instance_id] = instance_infos
annotations_dict[scene_id]['cad'] = annotations_dict_tmp
scan2cad_centers_wrd = np.array(scan2cad_centers_wrd)

center_distances = np.linalg.norm(shapenet_centers_wrd[:, None, :] - scan2cad_centers_wrd[None, :, :], axis=-1)
scan2cad2scannet = {}
for idx, (instance_id, instance_infos) in enumerate(annotations_dict[scene_id]['cad'].items()):
    scannet_instance_idx = np.argmin(center_distances[:, idx])
    annotations_dict[scene_id]['cad'][instance_id]['shapened_semantic_id'] = int(shapenet_instance_bboxes[scannet_instance_idx, -1])
    annotations_dict[scene_id]['cad'][instance_id]['shapened_instance_id'] = scannet_instance_idx + 1


#########################
#   Check Point Cloud   #
#########################
for target_instance_id in annotations_dict[scene_id]['cad'].keys():
    # Check point clouds.
    # catid_cad = '03001627'
    # id_cad = '95d082c78ea0b1befe52ffd0e748a1ab' # "3289bcc9bf8f5dab48d8ff57878739ca"
    # target_instance_id = f'{catid_cad}_{id_cad}'
    instance_label = annotations_dict[scene_id]['cad'][target_instance_id]['shapened_instance_id']
    catid_cad, id_cad = target_instance_id.split('_')
    cad_file = f'/d/workspace/yyoshitake/ShapeNet/ShapeNetCore.v2/{catid_cad}/{id_cad}/models/model_normalized.obj'
    cad_vertices = loadOBJ(cad_file) 
    vertices = cad_vertices[..., :3]
    cad_trs = annotations_dict[scene_id]['cad'][target_instance_id]['trs']
    Mcad = make_M_from_tqs(cad_trs["translation"], cad_trs["rotation"], cad_trs["scale"]) # <-- transformation from CAD space to world space 
    vertices = np.sum(np.concatenate([vertices, np.ones_like(vertices)[..., :1]], axis=-1)[..., None, :]*Mcad[None, :, :], axis=-1)[..., :3]
    cad_vertices[..., :3] = vertices.copy()
    # cad_point_cloud = []
    # for v in cad_vertices:
    #     vi = tuple(np.dot(Mcad, np.array([v[0], v[1], v[2], 1]))[0:3])
    #     cad_point_cloud.append(vi)
    # cad_point_cloud = np.array(cad_point_cloud)
    # import pdb; pdb.set_trace()

    fig = pylab.figure(figsize = [10, 10])
    ax = fig.add_subplot(projection='3d')
    point = cad_vertices[::3, :]
    ax.scatter(point[:, 0], point[:, 1], point[:, 2], c='c', s=0.5)
    # point = shapenet_centers_wrd[:, :3]
    # ax.scatter(point[:, 0], point[:, 1], point[:, 2], c='m', s=10.0)
    # point = scan2cad_centers_wrd[:, :3]
    # ax.scatter(point[:, 0], point[:, 1], point[:, 2], c='g', s=10.0)
    mask = vertices_instance_labels==instance_label
    point = scene_vertices[mask]
    color = get_colored_segmap(vertices_instance_labels[mask][:, None, None])[:, 0, :] / 255.0
    # ax.scatter3D(point[:, 0], point[:, 1], point[:, 2], c=color, cmap='rgb', s=0.5)
    ax.scatter3D(point[:, 0], point[:, 1], point[:, 2], c='m', s=0.5)
    ax.set_xlabel('x_value')
    ax.set_ylabel('y_value')
    ax.set_zlabel('z_value')
    # ax.set_xlim(3, 5)
    # ax.set_ylim(-1, 1)
    # ax.set_zlim(1, 3)
    ax.view_init(elev=0, azim=90)
    fig.savefig('tes.png', dpi=300)
    pylab.close()


#########################
#       Check Img       #
#########################
for instance_id in annotations_dict[scene_id]['cad'].keys():
    os.makedirs(os.path.join(SCANNET_DIR, scene_id, instance_id), exist_ok=True)
    os.makedirs(os.path.join(SCANNET_DIR, scene_id, instance_id, 'distance'), exist_ok=True)
    os.makedirs(os.path.join(SCANNET_DIR, scene_id, instance_id, 'rays_d_cam'), exist_ok=True)
    os.makedirs(os.path.join(SCANNET_DIR, scene_id, instance_id, 'pose'), exist_ok=True)
    os.makedirs(os.path.join(SCANNET_DIR, scene_id, instance_id, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(SCANNET_DIR, scene_id, instance_id, 'color'), exist_ok=True)
    os.makedirs(os.path.join(SCANNET_DIR, scene_id, instance_id, 'depth'), exist_ok=True)
    os.makedirs(os.path.join(SCANNET_DIR, scene_id, instance_id, 'video'), exist_ok=True)

for target_instance_id in annotations_dict[scene_id]['cad'].keys():
    annotations_dict[scene_id]['cad'][target_instance_id]['view_num'] = 0

MAX_IM_NUM = len(glob.glob(os.path.join(SCANNET_DIR, scene_id, f'color/*.jpg')))
for img_id in range(5000, 5245): # , MAX_IM_NUM):
# for img_id in range(3800, 3920): # , MAX_IM_NUM):

    total_rgb = cv2.imread(os.path.join(SCANNET_DIR, scene_id, f'color/{img_id}.jpg'))
    total_mask = cv2.imread(os.path.join(SCANNET_DIR, scene_id, f'instance-filt/{img_id}.png'))[:, :, 0]
    H, W, C = total_rgb.shape
    total_depth = np.load(os.path.join(SCANNET_DIR, scene_id, f'depth/{img_id}.npy')) / 1000
    def load_matrix_from_txt(path, shape=(4, 4)):
        with open(path) as f:
            txt = f.readlines()
        txt = ''.join(txt).replace('\n', ' ')
        matrix = [float(v) for v in txt.split()]
        return np.array(matrix).reshape(shape)
    pose = load_matrix_from_txt(os.path.join(SCANNET_DIR, scene_id, f'pose/{img_id}.txt'))
    intrinsic_depth = load_matrix_from_txt(os.path.join(SCANNET_DIR, scene_id, 'intrinsic/intrinsic_depth.txt'))

    d_H = total_depth.shape[0]
    d_W = total_depth.shape[1]
    u_coord = np.tile(np.arange(0, d_W)[None, :], (d_H, 1))
    v_coord = np.tile(np.arange(0, d_H)[:, None], (1, d_W))
    fx = intrinsic_depth[0, 0]
    fy = intrinsic_depth[1, 1]
    cx = intrinsic_depth[0, 2]
    cy = intrinsic_depth[1, 2]
    total_rays_d_cam_z1 = np.stack([(u_coord - cx) / fx, (v_coord - cy) / fy, np.ones((d_H, d_W))], axis=-1)
    total_distance = total_depth * np.linalg.norm(total_rays_d_cam_z1, axis=-1)
    total_rays_d_cam = total_rays_d_cam_z1 / np.linalg.norm(total_rays_d_cam_z1, axis=-1)[:, :, None]

    M_c2w = Mscan @ pose
    R_c2w = M_c2w[:3, :3] / M_c2w[3, 3]
    T_c2w = M_c2w[:3, 3] / M_c2w[3, 3]

    # Resize Depth map and UV map.
    total_depth = cv2.resize(total_depth.astype(np.float64), (W, H), interpolation=cv2.INTER_NEAREST)
    total_distance = cv2.resize(total_distance.astype(np.float64), (W, H), interpolation=cv2.INTER_NEAREST)
    total_rays_d_cam = cv2.resize(total_rays_d_cam.astype(np.float64), (W, H), interpolation=cv2.INTER_NEAREST)

    for target_instance_id in annotations_dict[scene_id]['cad'].keys():
        instance_label = annotations_dict[scene_id]['cad'][target_instance_id]['shapened_instance_id']
        if (total_mask==instance_label).any():
            mask = total_mask==instance_label

            mask_H = mask.shape[0]
            mask_W = mask.shape[1]
            x_coord = np.tile(np.arange(0, mask_W)[None, :], (mask_H, 1))
            y_coord = np.tile(np.arange(0, mask_H)[:, None], (1, mask_W))
            image_coord = np.stack([y_coord, x_coord], axis=-1)
            masked_image_coord = image_coord[mask]
            max_y, max_x = masked_image_coord.max(axis=0)
            min_y, min_x = masked_image_coord.min(axis=0)
            bbox = np.array([[max_x, max_y], [min_x, min_y]])

            H_x = max_x - min_x
            H_y = max_y - min_y
            bbox_H_xy = np.array([H_x, H_y])

            # 正方形でクロップし直す
            if H_x < mask_H:
                square_H = max(H_x, H_y)
                diff_H_xy = (square_H - bbox_H_xy) / 2
                bbox = bbox + np.stack([diff_H_xy, -diff_H_xy], axis=0)
                # はみ出したら戻す
                border_xy = np.array([[mask_W-1., mask_H-1.], [0., 0.]])
                outside_xy = border_xy - bbox
                outside_xy[0, :][outside_xy[0, :] > .0] = 0. # 値が負ならMaxがはみ出た -> ずれを引く
                outside_xy[1, :][outside_xy[1, :] < .0] = 0. # 値が正ならMinがはみ出た -> ずれを足す
                bbox = bbox + outside_xy.sum(axis=0)
            else:
                up_side, down_side = mask.sum(axis=1)[[0, -1]]
                if up_side > 0:
                    bbox = np.array([[max_x, H_x], [min_x, 0]])
                elif down_side > 0:
                    bbox = np.array([[max_x, H_y], [min_x, H_y-H_x]])
                else:
                    bbox = np.array([[max_x, H_x-H_y/2], [min_x, -H_y/2]])
                import pdb; pdb.set_trace()

            # 整数値に直す
            max_xy, min_xy = bbox
            max_x, max_y = max_xy
            min_x, min_y = min_xy
            max_x = min(-int(-max_x), mask_W-1)
            max_y = min(-int(-max_y), mask_H-1)
            min_x = max(int(min_x), 0)
            min_y = max(int(min_y), 0)
            mask = mask[min_y:max_y, min_x:max_x]
            ins_mask = total_mask[min_y:max_y, min_x:max_x] # Fore Gound マスク？
            rgb = total_rgb[min_y:max_y, min_x:max_x]
            depth = total_depth[min_y:max_y, min_x:max_x]
            distance = total_distance[min_y:max_y, min_x:max_x]
            rays_d_cam = total_rays_d_cam[min_y:max_y, min_x:max_x]

            # Resize_images:
            clopped_H = 256 # 128
            mask = cv2.resize(mask.astype(np.float64), (clopped_H, clopped_H), interpolation=cv2.INTER_NEAREST) > .5
            distance = cv2.resize(distance, (clopped_H, clopped_H), interpolation=cv2.INTER_NEAREST)
            rays_d_cam = cv2.resize(rays_d_cam, (clopped_H, clopped_H), interpolation=cv2.INTER_NEAREST)
            rgb = cv2.resize(rgb, (clopped_H, clopped_H), interpolation=cv2.INTER_NEAREST)
            depth = cv2.resize(depth, (clopped_H, clopped_H), interpolation=cv2.INTER_NEAREST)

            instance_img_id = annotations_dict[scene_id]['cad'][target_instance_id]['view_num']
            # np.save(os.path.join(SCANNET_DIR, scene_id, target_instance_id, 'distance', f'{instance_img_id}.npy'), distance)
            # np.save(os.path.join(SCANNET_DIR, scene_id, target_instance_id, 'rays_d_cam', f'{instance_img_id}.npy'), rays_d_cam)
            # np.save(os.path.join(SCANNET_DIR, scene_id, target_instance_id, 'mask', f'{instance_img_id}.npy'), mask)
            # np.save(os.path.join(SCANNET_DIR, scene_id, target_instance_id, 'pose', f'{instance_img_id}.npy'), M_c2w)
            # cv2.imwrite(os.path.join(SCANNET_DIR, scene_id, target_instance_id, 'color', f'{instance_img_id}.png'), rgb)
            # np.save(os.path.join(SCANNET_DIR, scene_id, target_instance_id, 'depth', f'{instance_img_id}.npy'), depth)
            # np.save(os.path.join(SCANNET_DIR, scene_id, target_instance_id, 'rays_d_cam', f'{instance_img_id}.npy'), total_rays_d_cam)
            # cv2.imwrite(os.path.join(SCANNET_DIR, scene_id, target_instance_id, 'color', f'{instance_img_id}.png'), total_rgb)
            # np.save(os.path.join(SCANNET_DIR, scene_id, target_instance_id, 'depth', f'{instance_img_id}.npy'), total_depth)
            video_frame = np.concatenate([cv2.resize(total_rgb, (int(W/H*clopped_H), clopped_H)), rgb, np.tile(mask[..., None], (1, 1, 3)).astype(np.uint8)*255], axis=1)
            cv2.imwrite(os.path.join(SCANNET_DIR, scene_id, target_instance_id, 'video', f'{str(instance_img_id).zfill(5)}.png'), video_frame)
            annotations_dict[scene_id]['cad'][target_instance_id]['view_num'] = instance_img_id + 1

            # rays_wrd = np.sum(rays_d_cam[..., None, :]*R_c2w[None, None, :], axis=-1)
            # OSMap_wrd = distance[:, :, None] * rays_wrd + T_c2w[None, None, :]
            # fig = pylab.figure(figsize = [10, 10])
            # ax = fig.add_subplot(projection='3d')
            # point = OSMap_wrd[mask].reshape(-1, 3) # np.array([x_data, y_data, z_data]).T[::10]
            # ax.scatter(point[:, 0], point[:, 1], point[:, 2], c='m', s=0.5)
            # point = scene_vertices[::10, :3]
            # ax.scatter(point[:, 0], point[:, 1], point[:, 2], c='c', s=0.5)
            # ax.set_xlabel('x_value')
            # ax.set_ylabel('y_value')
            # ax.set_zlabel('z_value')
            # ax.view_init(elev=0, azim=90)
            # fig.savefig('tes.png', dpi=300)
            # pylab.close()
            # import pdb; pdb.set_trace()

pickle_dump(annotations_dict, 'annotations_dict.pickle')
