import matplotlib.pyplot as plt
import tqdm
import sys
import os
import numpy as np
import glob
import datetime
import cv2
sys.path.append("../")
from often_use import *
from parser_get_arg import *
from DDF.train_pl import DDF

torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_rot_views(lat_deg, freq, model):
    if not model.model_params_dtype:
        model.check_model_info()

    lat = np.deg2rad(lat_deg)
    y = np.sin(lat)
    xz_R = np.cos(lat)
    rad = np.linspace(0, 2*np.pi, freq+1)[:freq]

    pos_list = np.stack([xz_R*np.cos(rad), np.full_like(rad, y), xz_R*np.sin(rad)], axis=-1)
    up_list = np.tile(np.array([[0, 1, 0]]), (freq, 1))
    
    at_list = 0 - pos_list
    at_list = at_list/np.linalg.norm(at_list, axis=-1)[..., None]
    
    right_list = np.cross(at_list, up_list)
    right_list = right_list/np.linalg.norm(right_list, axis=-1)[..., None]
    
    up_list = np.cross(right_list, at_list)
    up_list = up_list/np.linalg.norm(up_list, axis=-1)[..., None]
    
    w2c_list = np.stack([right_list, -up_list, at_list], axis=-2)

    # Adjust list's type to the model.
    pos_list = torch.from_numpy(pos_list).clone().to(model.model_params_dtype).to(model.device)
    w2c_list = torch.from_numpy(w2c_list).clone().to(model.model_params_dtype).to(model.device)

    return pos_list, w2c_list


def render_distance_map(
        H, 
        o2c,
        obj_scale, 
        rays_d_cam, 
        input_lat_vec, 
        ddf, 
        cam_pos_obj = 'non', 
        with_invdistance_map = False, 
    ):

    # Get rays direction.
    rays_d_obj = torch.sum(rays_d_cam[..., None, :]*o2c[..., None, None, :, :].permute(0, 1, 2, 4, 3), -1)

    # # Get rays origin.
    # obj_pos_cam = torch.sum((obj_pos_wrd - cam_pos_wrd)[..., None, :]*w2c, dim=-1)
    # cam_pos_obj = - torch.sum(obj_pos_cam[..., None, :]*o2c.permute(0, 2, 1), dim=-1) / obj_scale[:, None]
    rays_o_obj = cam_pos_obj[:, None, None, :].expand(-1, H, H, -1)

    # Get rays inputs.
    rays_d = rays_d_obj
    rays_o = rays_o_obj

    # Estimating.
    est_invdistance_map_obj_scale, negative_D_mask = ddf.forward_from_far(rays_o, rays_d, input_lat_vec)
    est_invdistance_map = est_invdistance_map_obj_scale / obj_scale[:, None, None]

    # Get distance map.
    mask_under_border = 1 / (cam_pos_obj.norm(dim=-1) + .5 * obj_scale * ddf.radius) # 良いのか...？
    est_mask = [map_i > border_i for map_i, border_i in zip(est_invdistance_map, mask_under_border)]
    est_mask = torch.stack(est_mask, dim=0)
    est_distance_map = torch.zeros_like(est_invdistance_map)
    est_distance_map[est_mask] = 1. / est_invdistance_map[est_mask]

    if with_invdistance_map:
        return est_invdistance_map, est_mask, est_distance_map
    else:
        return est_mask, est_distance_map



# python ddf_inter_shape.py --config=../configs/paper_exp/chair/view5/tes.txt
if __name__=='__main__':
    # Get args
    args = get_args()
    # cat_list = ['chair']
    cat_list = ['display', 'cabinet', 'table']
    cat_list = ['cabinet']

    for cat_name in cat_list:
        
        if cat_name == 'display':
            args.ddf_model_path = '/home/yyoshitake/works/DeepSDF/project/DDF/lightning_logs/display/cat_depth_mae_normal_mae_seed0_normal001_lr00001/checkpoints/0000010000.ckpt'
            args.ddf_instance_list_txt = '/home/yyoshitake/works/make_depth_image/project/instance_list/paper/display/ddf_train.txt'
            args.N_instances = 986
        elif cat_name == 'cabinet':
            args.ddf_model_path = '/home/yyoshitake/works/DeepSDF/project/DDF/lightning_logs/cabinet/cat_depth_mae_normal_mae_seed0_normal001_lr00001/checkpoints/0000010000.ckpt'
            args.ddf_instance_list_txt = '/home/yyoshitake/works/make_depth_image/project/instance_list/paper/cabinet/ddf_train.txt'
            args.N_instances = 1465
        elif cat_name == 'table':
            args.ddf_model_path = '/home/yyoshitake/works/DeepSDF/project/DDF/lightning_logs/table/cat_depth_mae_normal_mae_seed0_normal001_lr00001/checkpoints/0000003000.ckpt'
            args.ddf_instance_list_txt = '/home/yyoshitake/works/make_depth_image/project/instance_list/paper/table/total_havibg_ddf_data.txt'
            args.N_instances = 4817
        
        # Set ddf.
        ddf = DDF(args)
        ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args)
        ddf.eval()

        H = 256 # 256
        Fov = 60 # 53
        rays_d_cam = get_ray_direction(H, Fov)

        interpolate_freq = 3
        origin_ins_num = 3
        freq = int(origin_ins_num+interpolate_freq*(origin_ins_num-1))
        lat_deg = 30
        cam_pos, w2c = get_rot_views(lat_deg, 10, ddf)

        total_map = []
        cam_pos = cam_pos[8][None, :].expand(3, -1)
        w2c = w2c[8][None, :, :].expand(3, -1, -1)
        o2c = w2c
        cam_pos_obj = cam_pos
        rays_d_cam = rays_d_cam.expand(3, -1, -1, -1)
        obj_scale = torch.ones(3)

        rs = np.random.RandomState(3)
        for instance_id in [[449, 731], [4058, 2442], [1321, 1833], [1809, 4551]]: 
            # chair : [[2993, 363], [1401, 1644], [1102, 1081], [2505, 1909], [2772, 3081]]:
            # table : [1481, 1833] [2685, 2442] [3506, 2455]
        # for sample_idx in tqdm.tqdm(range(1, 300+1)):
            # instance_id = rs.choice(np.arange(ddf.lat_vecs.num_embeddings).tolist(), 2, replace=False)
            instance_id = torch.tensor(instance_id, dtype=torch.long).to(ddf.device)
            origin_lat_vec = ddf.lat_vecs(instance_id)
            input_lat_vec = torch.stack([origin_lat_vec[0], origin_lat_vec.mean(0), origin_lat_vec[1]])
            est_normal_map, est_distance_map, est_mask = \
                render_normal_map_from_afar(H, o2c, cam_pos_obj, obj_scale, rays_d_cam, input_lat_vec, ddf)
            est_normal_map = (-est_normal_map + 1.0) / 2
            est_normal_map[torch.logical_not(est_mask)] = 1.0

            origin_size = 110
            fil = torchvision.transforms.Resize(origin_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)

            buffer = 10
            buffer__ = 30
            result_map = torch.ones(H, origin_size+H-buffer__, 3).to(est_normal_map)
            result_map[ buffer: buffer+origin_size, :origin_size] = fil(est_normal_map[2].permute(2, 0, 1)).permute(1,2,0)
            result_map[-buffer-origin_size:-buffer, :origin_size] = fil(est_normal_map[0].permute(2, 0, 1)).permute(1,2,0)
            result_map[:, -H:][est_mask[1, :, -H:]] = est_normal_map[1, :, :][est_mask[1, :, :]]
            # check_map_torch(result_map[:, :-35, :], 'tes.png')

            str_ins_1 = str(instance_id[0].item())
            str_ins_2 = str(instance_id[1].item())
            # img_path = os.path.join('ddf_results', cat_name, f'{str_ins_1}_{str_ins_2}.png')
            img_path = os.path.join('tes.png')
            result_map = (255*result_map[:, :-20, :]).to('cpu').detach().numpy().copy().astype('uint8')
            cv2.imwrite(img_path, result_map[:, :, [2, 1, 0]])
            import pdb; pdb.set_trace()
