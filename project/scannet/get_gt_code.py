import numpy as np
import os
import sys
import tqdm
import torchvision
from often_use import *
sys.path.append("../")
from parser_get_arg import *
from DDF.train_pl import DDF
import random





# 書き換える
# python get_gt_code.py --config=../configs/paper_exp/chair_re/view5/rdn.txt
target_cat_id = '03001627'
ins_list = txt2list(f'/home/yyoshitake/works/DeepSDF/project/scannet/gt_latent_code/{target_cat_id}.txt')
cabnonical_map_dir = f'/home/yyoshitake/works/DeepSDF/project/scannet/gt_latent_code/canonical_map/{target_cat_id}'

# Get args
args = get_args()
# args.ddf_model_path = '/home/yyoshitake/works/DeepSDF/project/DDF/lightning_logs/chair/cat_depth_mae_normal_mae_seed0_normal001_lr00001/checkpoints/0000010000.ckpt'

# Set ddf.
ddf = DDF(args)
ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args).cuda()
ddf.eval()
ddf_instance_list = txt2list(args.ddf_instance_list_txt)
init_lat_vec = ddf.lat_vecs(torch.tensor(list(range(len(ddf_instance_list))), device=ddf.device)).mean(0)



FOV = 60
RESOLUTION = 256
rays_d_cam = get_ray_direction(RESOLUTION, FOV).cuda()

for ins_name in ins_list:
    canonical_path = os.path.join(cabnonical_map_dir, ins_name + '.pickle')
    canonical_data_dict = pickle_load(canonical_path)
    canonical_distance_map = torch.from_numpy(canonical_data_dict['depth_map'].astype(np.float32)).clone().cuda()
    canonical_mask = canonical_distance_map > 0
    canonical_invdist_map = torch.zeros_like(canonical_distance_map)
    canonical_invdist_map[canonical_mask] = 1 / canonical_distance_map[canonical_mask]
    canonical_cam_pos   = torch.from_numpy(canonical_data_dict['camera_pos'].astype(np.float32)).clone().cuda()
    canonical_w2c   = torch.from_numpy(canonical_data_dict['camera_rot'].astype(np.float32)).clone().cuda()
    canonical_num = canonical_distance_map.shape[0]

    # Blurされたマスク
    transform = torchvision.transforms.GaussianBlur(kernel_size=21)
    canonical_blur_mask = transform(canonical_mask)
    # check_map_torch(canonical_blur_mask[0])
    
    # 入力の方向
    rays_d_cam = F.normalize(rays_d_cam, dim=-1) # del dummy batch
    rays_o_wrd = canonical_cam_pos[:, None, None, :].expand(-1, RESOLUTION, RESOLUTION, -1)
    rays_d_wrd = torch.sum(rays_d_cam[:, :, :, None, :] * canonical_w2c.permute(0, 2, 1)[:, None, None, :, :], -1)

    # 最適化開始
    lr = 1e-3
    latent_tensor = init_lat_vec.detach().clone().cuda()
    latent_tensor.requires_grad = True
    optimizer_latent = torch.optim.Adam([latent_tensor], lr=lr)

    ##############################
    # 最適化開始
    ##############################
    indexes = list(range(canonical_num))
    for i in tqdm.tqdm(range(200)):
        sampled_idx = random.sample(indexes, 16)
        # if i < 50:
        #     sampled_idx = random.sample(indexes, 1)
        # elif i < 100:
        #     sampled_idx = random.sample(indexes, 8)
        # else:
        #     sampled_idx = random.sample(indexes, 16)
        # if i == 199:
        #     sampled_idx = random.sample(indexes, 20)
        sampled_canonical_blur_mask = canonical_blur_mask[sampled_idx]
        sampled_canonical_invdist_map = canonical_invdist_map[sampled_idx]
        sampled_rays_o_wrd = rays_o_wrd[sampled_idx]
        sampled_rays_d_wrd = rays_d_wrd[sampled_idx]

        inp_latent_tensor = latent_tensor[None, :]
        if torch.norm(latent_tensor) > 1.0:
            inp_latent_tensor = latent_tensor / torch.norm(latent_tensor)
        masked_rays_o_wrd = sampled_rays_o_wrd[sampled_canonical_blur_mask][None, None, :, :]
        masked_rays_d_wrd = sampled_rays_d_wrd[sampled_canonical_blur_mask][None, None, :, :]

        check_idx = 5
        invdist_rendered = ddf(masked_rays_o_wrd, masked_rays_d_wrd, inp_latent_tensor)[0, 0]
        # check_map_torch(torch.cat([canonical_distance_map[check_idx], invdist_rendered[0]], dim=0), 'tes.png')

        # loss and opt
        loss_invdist = torch.abs(sampled_canonical_invdist_map[sampled_canonical_blur_mask] - invdist_rendered).mean()
        lat_reg = torch.mean(latent_tensor.pow(2))
        loss = 10 * loss_invdist + 1 * lat_reg
        
        # 最適化
        loss.backward()
        optimizer_latent.step()
        print(loss)
        print(torch.norm(inp_latent_tensor))
    

    
    # if torch.norm(latent_tensor) > 1.:
    #     latent_tensor = latent_tensor / torch.norm(latent_tensor)
    # np.save(f'/home/yyoshitake/works/DeepSDF/project/scannet/gt_latent_code/{target_cat_id}/{ins_name}', latent_tensor.to('cpu').detach().numpy().copy())

    # # 確認用
    # est_invdist_map = torch.ones_like(sampled_canonical_invdist_map) * 0
    # est_invdist_map[sampled_canonical_blur_mask] = invdist_rendered
    # result_map = torch.cat([est_invdist_map[::5].reshape(-1, 256), 
    #                         sampled_canonical_invdist_map[::5].reshape(-1, 256), 
    #                         torch.abs(est_invdist_map[::5].reshape(-1, 256) - sampled_canonical_invdist_map[::5].reshape(-1, 256))], dim=-1)
    # check_map_torch(result_map, f'/home/yyoshitake/works/DeepSDF/project/scannet/gt_latent_code/{ins_name}.png')
    


    if torch.norm(latent_tensor) > 1.:
        latent_tensor = latent_tensor / torch.norm(latent_tensor)
    np.save(f'/home/yyoshitake/works/DeepSDF/project/scannet/gt_latent_code__/{target_cat_id}/{ins_name}', latent_tensor.to('cpu').detach().numpy().copy())

    # 確認用
    est_invdist_map = torch.ones_like(sampled_canonical_invdist_map) * 0
    est_invdist_map[sampled_canonical_blur_mask] = invdist_rendered
    result_map = torch.cat([est_invdist_map[::5].reshape(-1, 256), 
                            sampled_canonical_invdist_map[::5].reshape(-1, 256), 
                            torch.abs(est_invdist_map[::5].reshape(-1, 256) - sampled_canonical_invdist_map[::5].reshape(-1, 256))], dim=-1)
    check_map_torch(result_map, f'/home/yyoshitake/works/DeepSDF/project/scannet/gt_latent_code__/{ins_name}.png')