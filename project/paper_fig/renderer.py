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



# python renderer.py --config=../configs/paper_exp/chair/view5/tes.txt
# CUDA_VISIBLE_DEVICES=7 python renderer.py --config=../configs/paper_exp/table/view5/rdn.txt
if __name__=='__main__':
    # Get args
    args = get_args()
    # import pdb; pdb.set_trace()
    # args.ddf_model_path = '/home/yyoshitake/works/DeepSDF/project/DDF/lightning_logs/table/cat_depth_mae_normal_mae_seed0_normal001_lr00001/checkpoints/0000003000.ckpt'
    H = 256 # 256
    Fov = 70 # 53
    rays_d_cam = get_ray_direction(H, Fov)
    
    # Set ddf.
    ddf = DDF(args)
    ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args)
    ddf.eval()

    ##################################################
    #                  rot_view                      #
    ##################################################
    """
        rays_d_cam : [B, H, H, 3]
        o2c : [B, 3, 3]
        obj_pos_cam : [B, 3]
        obj_scale : [B]
        input_lat_vec : [B, 256]
    """
    lat_deg = 30
    freq = 8
    cam_pos, w2c = get_rot_views(lat_deg, freq, ddf)
    o2c = w2c # torch.roll(torch.flip(w2c, dims=[0]), 1, dims=0)
    cam_pos_obj = cam_pos
    rays_d_cam = rays_d_cam.expand(freq, -1, -1, -1)
    obj_scale = torch.ones(freq)
    rs = np.random.RandomState(0)
    for sample_idx in tqdm.tqdm(range(1, 500+1)):
        instance_id = rs.choice(np.arange(ddf.lat_vecs.num_embeddings).tolist(), 2, replace=False)
        # instance_id = [instance_id[0].item(), instance_id[0].item()] # [388, 2532]
        instance_id = torch.tensor(instance_id, dtype=torch.long).to(ddf.device)
        input_lat_vec = ddf.lat_vecs(instance_id).mean(dim=0)[None, :].expand(freq, -1)
        input_lat_vec_src_01 = ddf.lat_vecs(instance_id)[0][None, :]
        input_lat_vec_src_02 = ddf.lat_vecs(instance_id)[1][None, :]
        # est_mask, est_distance_map = render_distance_map(H, o2c, obj_scale, rays_d_cam, input_lat_vec, ddf, cam_pos_obj)
        # import pdb; pdb.set_trace()
        est_normal_map, est_distance_map, est_mask = render_normal_map_from_afar(H, o2c, cam_pos_obj, obj_scale, rays_d_cam, input_lat_vec, ddf)
        est_normal_map_src_01, _, mask_01 = render_normal_map_from_afar(H, o2c[2][None], cam_pos_obj[2][None], obj_scale[2][None], rays_d_cam[2][None], input_lat_vec_src_01, ddf)
        est_normal_map_src_02, _, mask_02 = render_normal_map_from_afar(H, o2c[2][None], cam_pos_obj[2][None], obj_scale[2][None], rays_d_cam[2][None], input_lat_vec_src_02, ddf)

        est_normal_map = (-est_normal_map + 1.0) / 2
        est_normal_map[torch.logical_not(est_mask)] = 1.0
        est_normal_map_src_01 = (-est_normal_map_src_01 + 1.0) / 2
        est_normal_map_src_01[torch.logical_not(mask_01)] = 1.0
        est_normal_map_src_02 = (-est_normal_map_src_02 + 1.0) / 2
        est_normal_map_src_02[torch.logical_not(mask_02)] = 1.0

        origin_size = 110
        buffer = 10
        buffer__ = 20
        fil = torchvision.transforms.Resize(origin_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        original_map = torch.ones(H, origin_size, 3)
        original_map[ buffer: buffer+origin_size, :] = fil(est_normal_map_src_01[0].permute(2, 0, 1)).permute(1,2,0)
        original_map[-buffer-origin_size:-buffer, :] = fil(est_normal_map_src_02[0].permute(2, 0, 1)).permute(1,2,0)
        rot_map = torch.cat([map_i[:, 20:-20] for map_i in torch.roll(est_normal_map, -2, 0)], dim=1)
        result_map = torch.cat([original_map, rot_map], dim=1)
        # check_map_torch(result_map, 'tes.png')
        result_map = (255*result_map).to('cpu').detach().numpy().copy().astype('uint8')

        epo_str = args.ddf_model_path.split('/')[-1].split('.')[0]
        str_ins_1 = str(instance_id[0].item())
        str_ins_2 = str(instance_id[1].item())
        cv2.imwrite(f'ddf_results/rot_table/{str_ins_1}_{str_ins_2}.png', result_map[:, :, [2, 1, 0]])
        # cv2.imwrite(f'tes.png', result_map[:, :, [2, 1, 0]])
        # import pdb; pdb.set_trace()
        
        # est_normal_map = (-est_normal_map + 1.0) / 2
        # est_normal_map[torch.logical_not(est_mask)] = 1.0
        # est_normal_map = torch.cat([normal_map_i for normal_map_i in est_normal_map], dim=1)
        # est_normal_map = (255 * est_normal_map).to('cpu').detach().numpy().copy().astype('uint8')
        # img_path = os.path.join('/home/yyoshitake/works/DeepSDF/project/paper_fig/ddf/rot/', str(sample_idx).zfill(5)+'.png')
        # cv2.imwrite(img_path, est_normal_map[:, :, [2, 1, 0]])
        # pickle_path = os.path.join('/home/yyoshitake/works/DeepSDF/project/paper_fig/ddf/rot/pickle', str(sample_idx).zfill(5)+'.pickle')
        # pickle_dump(instance_id.tolist(), pickle_path)


    ##################################################
    #                 interpolate                    #
    ##################################################
    # """
    #     rays_d_cam : [B, H, H, 3]
    #     o2c : [B, 3, 3]
    #     obj_pos_cam : [B, 3]
    #     obj_scale : [B]
    #     input_lat_vec : [B, 256]
    # """
    # interpolate_freq = 3
    # origin_ins_num = 3
    # freq = int(origin_ins_num+interpolate_freq*(origin_ins_num-1))
    # lat_deg = 30
    # cam_pos, w2c = get_rot_views(lat_deg, 10, ddf)
    # cam_pos = cam_pos[2][None, :].expand(freq, -1)
    # w2c = w2c[2][None, :, :].expand(freq, -1, -1)
    
    # # ################################################################################
    # # # o2c = w2c
    # # # cam_pos_obj = cam_pos
    # # # rays_d_cam = rays_d_cam.expand(freq, -1, -1, -1)
    # # # obj_scale = torch.ones(freq)
    # # # for sample_idx in tqdm(range(1, 3001)):
    # # #     instance_id = random.sample(np.arange(ddf.lat_vecs.num_embeddings).tolist(), origin_ins_num) # np.array([1577, 1500, 1722]), # [1577, 1104, 1722]
    # # #     instance_id = torch.tensor(instance_id, dtype=torch.long).to(ddf.device)
    # # #     origin_lat_vec = ddf.lat_vecs(instance_id)
    # # #     input_lat_vec = [origin_lat_vec[0]]
    # # #     for ins_idx in range(origin_ins_num-1):
    # # #         interpolate_ratio_idx = [0.25, 0.50, 0.75, 1.00]
    # # #         for interpolate_idx in range(0, interpolate_freq+1):
    # # #             beta = interpolate_ratio_idx[interpolate_idx]
    # # #             input_lat_vec.append((1.0 - beta)*origin_lat_vec[ins_idx] + beta*origin_lat_vec[ins_idx+1])
    # # #     input_lat_vec = torch.stack(input_lat_vec, dim=0)
    # # #     est_normal_map, est_distance_map, est_mask = \
    # # #         render_normal_map_from_afar(H, o2c, cam_pos_obj, obj_scale, rays_d_cam, input_lat_vec, ddf)

    # # #     origin_size = int(H * 5 / 7) # Resize Origin Map
    # # #     fil = torchvision.transforms.Resize(origin_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
    # # #     est_normal_map = (-est_normal_map + 1.0) / 2
    # # #     est_normal_map[torch.logical_not(est_mask)] = 1.0
    # # #     result_map = []
    # # #     for map_idx, normal_map_i in enumerate(est_normal_map):
    # # #         if map_idx % (interpolate_freq+1) == 0:
    # # #             map_i = torch.ones(H, origin_size, 3)
    # # #             resized_normal_map = fil(normal_map_i.permute(2, 0, 1)).permute(1,2,0)
    # # #             under_border = (H-origin_size)//2
    # # #             map_i[under_border:(under_border+origin_size)] = resized_normal_map
    # # #             result_map.append(map_i)
    # # #         else:
    # # #             result_map.append(normal_map_i)
    # # #     result_map = (255 * torch.cat(result_map, dim=1)).to('cpu').detach().numpy().copy().astype('uint8')
    # # #     img_path = os.path.join('/home/yyoshitake/works/DeepSDF/project/paper_fig/ddf/int/', str(sample_idx).zfill(5)+'.png')
    # # #     cv2.imwrite(img_path, result_map[:, :, [2, 1, 0]])
    # # #     pickle_path = os.path.join('/home/yyoshitake/works/DeepSDF/project/paper_fig/ddf/int/pickle', str(sample_idx).zfill(5)+'.pickle')
    # # #     pickle_dump(instance_id.tolist(), pickle_path)
    # # ################################################################################
    # total_map = []
    # cam_pos = cam_pos[2][None, :].expand(3, -1)
    # w2c = w2c[2][None, :, :].expand(3, -1, -1)
    # o2c = w2c
    # cam_pos_obj = cam_pos
    # rays_d_cam = rays_d_cam.expand(3, -1, -1, -1)
    # obj_scale = torch.ones(3)
    # for instance_id in [[2993, 363], [1401, 1644], [1102, 1081], [2505, 1909], [2772, 3081]]:
    #     instance_id = torch.tensor(instance_id, dtype=torch.long).to(ddf.device)
    #     origin_lat_vec = ddf.lat_vecs(instance_id)
    #     input_lat_vec = torch.stack([origin_lat_vec[0], origin_lat_vec.mean(0), origin_lat_vec[1]])
    #     est_normal_map, est_distance_map, est_mask = \
    #         render_normal_map_from_afar(H, o2c, cam_pos_obj, obj_scale, rays_d_cam, input_lat_vec, ddf)
    #     est_normal_map = (-est_normal_map + 1.0) / 2
    #     est_normal_map[torch.logical_not(est_mask)] = 1.0

    #     origin_size = 110
    #     fil = torchvision.transforms.Resize(origin_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)

    #     buffer = 10
    #     buffer__ = 20
    #     result_map = torch.ones(H, origin_size+H-buffer__, 3).to(est_normal_map)
    #     result_map[ buffer: buffer+origin_size, :origin_size] = fil(est_normal_map[2].permute(2, 0, 1)).permute(1,2,0)
    #     result_map[-buffer-origin_size:-buffer, :origin_size] = fil(est_normal_map[0].permute(2, 0, 1)).permute(1,2,0)
    #     result_map[:, origin_size:] = est_normal_map[1, :, buffer__:]
    #     total_map.append(result_map)
    #     # check_map_torch(result_map, 'tes.png')
    #     # import pdb; pdb.set_trace()

    # check_map_torch(torch.cat(total_map, dim=1), 'zzz.png')
    # total_map=(255*torch.cat(total_map, dim=1)).to('cpu').detach().numpy().copy().astype('uint8')
    # import pdb; pdb.set_trace()


    # ##################################################
    # #                     Err                        #
    # ##################################################
    # result_dir = '/home/yyoshitake/works/DeepSDF/project/paper_fig/normal_map/result'
    # interpolate_freq = 3
    # origin_ins_num = 3
    # freq = 1
    # lat_deg = 30
    # cam_pos, w2c = get_rot_views(lat_deg, 10, ddf)
    # cam_pos = cam_pos[2][None, :].expand(freq, -1)
    # w2c = w2c[2][None, :, :].expand(freq, -1, -1)
    # ###############
    # o2c  = w2c
    # obj_scale = torch.ones(1)
    # est_o2c = w2c
    # ###############
    # cam_pos_obj = cam_pos
    # rays_d_cam = rays_d_cam.expand(1, -1, -1, -1)
    
    # total_map = []
    # lat_vec_list = pickle_load('normal_map/latvec/00000_02.pickle')
    # for lat_vec_idx, lat_vec in enumerate(lat_vec_list):
    #     input_lat_vec = torch.from_numpy(lat_vec.astype(np.float32)).clone()[None, ...].to(w2c)
    #     est_normal_map, est_distance_map, est_mask = \
    #         render_normal_map_from_afar(H, o2c, cam_pos_obj, obj_scale, rays_d_cam, input_lat_vec, ddf)
    #     # est_normal_map[torch.logical_not(est_mask)] = 1.0
    #     est_normal_map = est_normal_map.to('cpu').detach().numpy().copy()
    #     est_mask = est_mask.to('cpu').detach().numpy().copy()

    #     gray_scale = 0.85
    #     nz = est_normal_map[0, ..., -1]
    #     I = np.clip(-nz,0,1)**(1/2.2)
    #     I[est_mask[0]] = gray_scale * I[est_mask[0]]
    #     I[np.logical_not(est_mask[0])] = 1.
    #     I_uint8 = (255 * I).astype(np.uint8)
    #     cv2.imwrite(os.path.join(result_dir, str(lat_vec_idx).zfill(5) + '.png'), I_uint8)