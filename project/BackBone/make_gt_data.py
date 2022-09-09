import cv2
import sys
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm, trange

sys.path.append("../")
from parser_get_arg import *
from often_use import *
from DDF.train_pl import DDF

RESOLUTION = 128
FOV = 49.134





def get_ray_zIsOne_direction_np(size, fov):
    fov = np.deg2rad(fov)
    x_coord = np.tile(np.linspace(-np.tan(fov*.5), np.tan(fov*.5), size)[None], (size, 1))
    y_coord = x_coord.T
    ray_direction = np.stack([x_coord, y_coord, np.ones_like(x_coord)], axis=2)
    return ray_direction



def get_ray_normalized_direction_np(size, fov):
    fov = np.deg2rad(fov)
    x_coord = np.tile(np.linspace(-np.tan(fov*.5), np.tan(fov*.5), size)[None], (size, 1))
    y_coord = x_coord.T
    ray_direction = np.stack([x_coord, y_coord, np.ones_like(x_coord)], axis=2)
    ray_direction = ray_direction / np.linalg.norm(ray_direction, axis=-1)[..., None]
    return ray_direction





if __name__=='__main__':
    # NUM_VIEWS = 128
    NUM_VIEWS = 32
    # raw_gt_path = '/home/yyoshitake/works/make_depth_image/project/tmp_1/03001627/'
    raw_gt_path = '/home/yyoshitake/works/make_depth_image/project/tmp_2/03001627/'
    gt_path = '/home/yyoshitake/works/DeepSDF/project/BackBone/dataset/03001627/randn/gt/'
    pre_path = '/home/yyoshitake/works/DeepSDF/project/BackBone/dataset/03001627/randn/pre/'
    # instance_txt = '/home/yyoshitake/works/DeepSDF/project/instance_lists/kmean/kmeans0_train_list.txt' # /kmeans_list_0_check.txt
    instance_txt = '/home/yyoshitake/works/DeepSDF/project/instance_lists/kmean/kmeans0_val_list.txt' # /kmeans_list_0_check.txt
    input_H = 128
    input_W = 128
    ddf_H = 128
    x_coord = torch.arange(0, input_W)[None, :].expand(input_H, -1)
    y_coord = torch.arange(0, input_H)[:, None].expand(-1, input_W)
    image_coord = torch.stack([y_coord, x_coord], dim=-1) # [H, W, (Y and X)]
    rays_d_cam = get_ray_direction(ddf_H, FOV)
    data_instance_list = []
    raw_gt_paths = []
    with open(instance_txt, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            target_path = raw_gt_path + line.rstrip('\n')
            raw_gt_paths.append(target_path)
            os.makedirs(target_path, exist_ok=True) # make dir.



    for instance_path in tqdm(raw_gt_paths):

        instance_id = instance_path.split('/')[-1]
        os.makedirs(gt_path + '/' + instance_id, exist_ok=True)
        for view_id in range(1, NUM_VIEWS+1):

            # Get mask and distance.
            distance_path = instance_path + '/' + str(view_id).zfill(5) + '.exr'
            inverced_distance_map = cv2.imread(distance_path, -1)
            mask = inverced_distance_map > 0
            distance = 1 / inverced_distance_map[mask]
            rays_d_norm = np.linalg.norm(get_ray_zIsOne_direction_np(inverced_distance_map.shape[0], FOV), axis=-1)
            distance = distance * rays_d_norm[mask]
            distance_map = np.zeros_like(inverced_distance_map)
            distance_map[mask] = distance

            # Get poses.
            pose_path = instance_path + '/' + str(view_id).zfill(5) + '.pickle'
            cam_pose_info = pickle_load(pose_path)['camera_info_dict']
            obj_pose_info = pickle_load(pose_path)['obj_info_dict']
            cam_pos_wrd = cam_pose_info['pos']
            w2c = cam_pose_info['w2c']
            obj_pos_wrd = obj_pose_info['pos']
            o2w = obj_pose_info['w2o']
            obj_scale = obj_pose_info['scale']

            # Clop distance map.
            raw_distance_map = torch.from_numpy(distance_map.astype(np.float32)).clone()[None]
            raw_mask = torch.from_numpy(mask.astype(np.bool_)).clone()[None]
            clopped_mask, clopped_distance_map, bbox_list = clopping_distance_map(raw_mask, raw_distance_map, image_coord, input_H, input_W, ddf_H)
            clopped_mask = clopped_mask[0]
            clopped_distance_map = clopped_distance_map[0]
            clopped_rays_d_cam = get_clopped_rays_d_cam(ddf_H, bbox_list, rays_d_cam)[0]
            
            # to torch.
            cam_pos_wrd = torch.from_numpy(cam_pos_wrd.astype(np.float32)).clone()
            w2c = torch.from_numpy(w2c.astype(np.float32)).clone()
            obj_pos_wrd = torch.from_numpy(obj_pos_wrd.astype(np.float32)).clone()
            o2w = torch.from_numpy(o2w.astype(np.float32)).clone()
            obj_scale = torch.from_numpy(obj_scale.astype(np.float32)).clone()

            #  Get smap.
            OSMap_cam = clopped_distance_map[..., None] * clopped_rays_d_cam
            OSMap_wrd = torch.sum(OSMap_cam[..., None, :]*w2c.permute(1, 0)[None, None, :, :], dim=-1)
            OSMap_wrd = OSMap_wrd + cam_pos_wrd[None, None, :]
            OSMap_obj = OSMap_wrd - obj_pos_wrd[None, None, :]
            OSMap_obj = torch.sum(OSMap_obj[..., None, :]*o2w.permute(1, 0)[None, None, :, :], dim=-1)
            OSMap_obj = OSMap_obj / obj_scale[None, None, :]
            OSMap_wrd[torch.logical_not(clopped_mask)] = 0.
            OSMap_obj[torch.logical_not(clopped_mask)] = 0.
            # ##################################################
            # fig = plt.figure()
            # ax = Axes3D(fig)
            # point_1 = OSMap_obj[clopped_mask]
            # point_1 = point_1.to('cpu').detach().numpy().copy()
            # ax.scatter(point_1[::3, 0], point_1[::3, 1], point_1[::3, 2], marker="o", linestyle='None', c='c', s=0.05)
            # ax.view_init(elev=0, azim=90)
            # fig.savefig("tes_00_90.png")
            # ax.view_init(elev=0, azim=0)
            # fig.savefig("tes_00_00.png")
            # ax.view_init(elev=45, azim=45)
            # fig.savefig("tes_45_45.png")
            # plt.close()
            # ##################################################

            # tu numpy.
            clopped_mask = clopped_mask.to('cpu').detach().numpy().copy().astype(np.bool_)
            clopped_distance_map = clopped_distance_map.to('cpu').detach().numpy().copy().astype(np.float32)
            OSMap_wrd = OSMap_wrd.to('cpu').detach().numpy().copy().astype(np.float32)
            OSMap_obj = OSMap_obj.to('cpu').detach().numpy().copy().astype(np.float32)
            clopped_rays_d_cam = clopped_rays_d_cam.to('cpu').detach().numpy().copy().astype(np.float32)
            cam_pos_wrd = cam_pos_wrd.to('cpu').detach().numpy().copy().astype(np.float32)
            w2c = w2c.to('cpu').detach().numpy().copy().astype(np.float32)
            obj_pos_wrd = obj_pos_wrd.to('cpu').detach().numpy().copy().astype(np.float32)
            o2w = o2w.to('cpu').detach().numpy().copy().astype(np.float32)
            obj_scale = obj_scale.to('cpu').detach().numpy().copy().astype(np.float32)

            # save results.
            data_dict = {
                'clopped_mask':clopped_mask, 
                'clopped_distance_map':clopped_distance_map, 
                'osmap_wrd':OSMap_wrd, 
                'osmap_obj':OSMap_obj, 
                'rays_d_cam':clopped_rays_d_cam, 
                'camera_pos':cam_pos_wrd, 
                'camera_rot':w2c, 
                'obj_pos':obj_pos_wrd, 
                'obj_rot':o2w, 
                'obj_scale':obj_scale, 
                }
            path = gt_path + instance_id + '/' + str(view_id).zfill(5) + '_' + str(0).zfill(2) + '.pickle'
            pickle_dump(data_dict, path)