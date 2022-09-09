import cv2
import sys
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm, trange
sys.path.append("../")
from parser_get_arg import *
from often_use import *
from DDF.train_pl import DDF

RESOLUTION = 128
FOV = 49.134
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device:', device)



def get_ray_zIsOne_direction_np(size, fov):
    fov = np.deg2rad(fov)
    x_coord = np.tile(np.linspace(-np.tan(fov*.5), np.tan(fov*.5), size)[None], (size, 1))
    y_coord = x_coord.T
    ray_direction = np.stack([x_coord, y_coord, np.ones_like(x_coord)], axis=2)
    return ray_direction





class dataset(data.Dataset):
    def __init__(
        self, 
        args, 
        gt_base_dir, 
        N_views, 
        instance_list_txt, 
        ):

        self.N_views = N_views
        self.data_list = []
        with open(instance_list_txt, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                for view_id in range(1, self.N_views+1):
                    self.data_list.append(
                        os.path.join(gt_base_dir, line.rstrip('\n'), str(view_id).zfill(5) + '_' + str(0).zfill(2) + '.pickle')
                        )

    def __getitem__(self, index):
        
        path = self.data_list[index]
        data_dict = pickle_load(path)
        clopped_mask = torch.from_numpy(data_dict['clopped_mask'].astype(np.float32)).clone()
        clopped_distance_map = torch.from_numpy(data_dict['clopped_distance_map'].astype(np.float32)).clone()
        OSMap_wrd = torch.from_numpy(data_dict['osmap_wrd'].astype(np.float32)).clone()
        OSMap_obj = torch.from_numpy(data_dict['osmap_obj'].astype(np.float32)).clone()
        clopped_rays_d_cam = torch.from_numpy(data_dict['rays_d_cam'].astype(np.float32)).clone()
        cam_pos_wrd = torch.from_numpy(data_dict['camera_pos'].astype(np.float32)).clone()
        w2c = torch.from_numpy(data_dict['camera_rot'].astype(np.float32)).clone()
        obj_pos_wrd = torch.from_numpy(data_dict['obj_pos'].astype(np.float32)).clone()
        o2w = torch.from_numpy(data_dict['obj_rot'].astype(np.float32)).clone()
        obj_scale = torch.from_numpy(data_dict['obj_scale'].astype(np.float32)).clone()

        
        return clopped_mask, clopped_distance_map, OSMap_wrd, OSMap_obj, clopped_rays_d_cam, \
        cam_pos_wrd, w2c, obj_pos_wrd, o2w, obj_scale, path, path.split('/')[-2], path.split('/')[-1].split('_')[0]

    def __len__(self):
        return len(self.data_list)





if __name__=='__main__':
    with torch.no_grad():


        # Get args
        args = get_args()
        args.gpu_num = torch.cuda.device_count() # log used gpu num.
        args.check_val_every_n_epoch = args.save_interval


        # Configs.
        NUM_VIEWS = 128
        # NUM_VIEWS = 32
        gt_base_dir = '/home/yyoshitake/works/DeepSDF/project/BackBone/dataset/03001627/randn/gt/'
        pre_base_dir = '/home/yyoshitake/works/DeepSDF/project/BackBone/dataset/03001627/randn/pre/'
        dif_base_dir = '/home/yyoshitake/works/DeepSDF/project/BackBone/dataset/03001627/randn/dif/'
        instance_txt = '/home/yyoshitake/works/DeepSDF/project/instance_lists/kmean/kmeans0_train_list.txt' # /kmeans_list_0_check.txt
        # instance_txt = '/home/yyoshitake/works/DeepSDF/project/instance_lists/kmean/kmeans0_val_list.txt' # /kmeans_list_0_check.txt
        input_H = 128
        input_W = 128
        ddf_H = 128
        x_coord = torch.arange(0, input_W)[None, :].expand(input_H, -1)
        y_coord = torch.arange(0, input_H)[:, None].expand(-1, input_W)
        image_coord = torch.stack([y_coord, x_coord], dim=-1) # [H, W, (Y and X)]
        rays_d_cam = get_ray_direction(ddf_H, FOV)
        random_axis_num = 1024
        random_axis_list = torch.from_numpy(sample_fibonacci_views(random_axis_num).astype(np.float32)).clone()


        # make ddf.
        ddf = DDF(args)
        ddf = ddf.load_from_checkpoint(checkpoint_path=args.ddf_model_path, args=args).to(device)
        ddf.eval()
        ddf_instance_list = []
        with open(args.ddf_instance_list_txt, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                ddf_instance_list.append(line.rstrip('\n'))
        data_instance_list = []
        with open(instance_txt, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                data_instance_list.append(line.rstrip('\n'))
                os.makedirs(pre_base_dir + '/' + line.rstrip('\n'), exist_ok=True) # make dir.
                os.makedirs(dif_base_dir + '/' + line.rstrip('\n'), exist_ok=True) # make dir.
        train_instance_ids = [ddf_instance_list.index(instance_i) for instance_i in data_instance_list]
        rand_Z_center = ddf.lat_vecs(torch.tensor(train_instance_ids, device=ddf.device)).mean(0).clone().detach()


        # Create dataloader
        target_dataset = dataset(args, gt_base_dir, NUM_VIEWS, instance_txt)
        target_dataloader = data.DataLoader(target_dataset, batch_size=args.N_batch, num_workers=args.num_workers)
        

        # Start
        for epoch_idx in tqdm(range(2)):
            if epoch_idx == 0:
                rand_P_range = 0.3
                rand_S_range = 0.3
                rand_R_range = 0.5 * torch.pi
                rand_Z_sigma = 0.05
            elif epoch_idx == 1:
                rand_P_range = 0.05
                rand_S_range = 0.05
                rand_R_range = 0.1 * torch.pi
                rand_Z_sigma = 0.03
            
            for batch in tqdm(target_dataloader):
                # Get gt.
                gt_clopped_mask = batch[0].to(device)
                gt_clopped_distance_map = batch[1].to(device)
                gt_OSMap_wrd = batch[2].to(device)
                gt_OSMap_obj = batch[3].to(device)
                clopped_rays_d_cam = batch[4].to(device)
                cam_pos_wrd = batch[5].to(device)
                w2c = batch[6].to(device)
                obj_pos_wrd = batch[7].to(device)
                o2w = batch[8].to(device)
                obj_scale = batch[9].to(device)
                gt_data_path = batch[10]
                instance_id = batch[11]
                str_views_id = batch[12]
                batch_size = len(instance_id)


                # Get inp.
                instance_idx = [ddf_instance_list.index(instance_id_i) for instance_id_i in instance_id]
                gt_shape_code = ddf.lat_vecs(torch.tensor(instance_idx, device=ddf.device))
                o2c = torch.bmm(w2c, o2w)
                gt_obj_axis_green_cam = o2c[:, :, 1]
                gt_obj_axis_red_cam = o2c[:, :, 0]
                shape_code = gt_shape_code
                axis_green_cam = gt_obj_axis_green_cam
                axis_red_cam = gt_obj_axis_red_cam
                

                # Get randomized ini.
                rand_P_seed = torch.rand(batch_size, 3)
                rand_S_seed = torch.rand(batch_size, 1)
                randn_theta_seed = torch.rand(batch_size)
                randn_axis_idx = np.random.choice(random_axis_num, batch_size)
                # Get initial position.
                rand_P = 2 * rand_P_range * (rand_P_seed - .5)
                ini_obj_pos =obj_pos_wrd + rand_P.to(obj_pos_wrd)
                # Get initial scale.
                rand_S = 2 * rand_S_range * (rand_S_seed - .5) + 1.
                ini_obj_scale = obj_scale * rand_S.to(obj_scale)
                # Get initial rot.
                randn_theta = 2 * rand_R_range * (randn_theta_seed - .5)
                randn_axis = random_axis_list[randn_axis_idx]
                cos_t = torch.cos(randn_theta)
                sin_t = torch.sin(randn_theta)
                n_x = randn_axis[:, 0]
                n_y = randn_axis[:, 1]
                n_z = randn_axis[:, 2]
                rand_R = torch.stack([torch.stack([cos_t+n_x*n_x*(1-cos_t), n_x*n_y*(1-cos_t)-n_z*sin_t, n_z*n_x*(1-cos_t)+n_y*sin_t], dim=-1), 
                                        torch.stack([n_x*n_y*(1-cos_t)+n_z*sin_t, cos_t+n_y*n_y*(1-cos_t), n_y*n_z*(1-cos_t)-n_x*sin_t], dim=-1), 
                                        torch.stack([n_z*n_x*(1-cos_t)-n_y*sin_t, n_y*n_z*(1-cos_t)+n_x*sin_t, cos_t+n_z*n_z*(1-cos_t)], dim=-1)], dim=1)
                ini_o2w = torch.bmm(rand_R.to(o2w), o2w)
                ini_obj_axis_green_wrd = ini_o2w[:, :, 1] # Y_w
                ini_obj_axis_red_wrd = ini_o2w[:, :, 0] # X_w
                ini_obj_axis_green_cam = torch.sum(ini_obj_axis_green_wrd[..., None, :]*w2c, dim=-1)
                ini_obj_axis_red_cam = torch.sum(ini_obj_axis_red_wrd[..., None, :]*w2c, dim=-1)
                # Get initial shape.
                randn_Z = rand_Z_sigma * torch.randn_like(gt_shape_code)
                if epoch_idx == 0:
                    ini_shape_code = rand_Z_center.unsqueeze(0).expand(batch_size, -1).to(gt_shape_code) + randn_Z
                elif epoch_idx == 1:
                    ini_shape_code = gt_shape_code + randn_Z


                # Rendering.
                est_clopped_mask, est_clopped_distance_map = render_distance_map_from_axis(
                                                                H = ddf_H, 
                                                                obj_pos_wrd = ini_obj_pos, 
                                                                axis_green = ini_obj_axis_green_cam, 
                                                                axis_red = ini_obj_axis_red_cam, 
                                                                obj_scale = ini_obj_scale[:, 0], 
                                                                input_lat_vec = ini_shape_code, 
                                                                cam_pos_wrd = cam_pos_wrd, 
                                                                rays_d_cam = clopped_rays_d_cam, 
                                                                w2c = w2c, 
                                                                ddf = ddf, 
                                                                with_invdistance_map = False)


                # Get est osmap.
                est_OSMap_cam = est_clopped_distance_map[..., None] * clopped_rays_d_cam
                est_OSMap_wrd = torch.sum(est_OSMap_cam[..., None, :]*w2c.permute(0, 2, 1)[:, None, None, :, :], dim=-1)
                est_OSMap_wrd = est_OSMap_wrd + cam_pos_wrd[:, None, None, :]
                est_OSMap_obj = est_OSMap_wrd - ini_obj_pos[:, None, None, :]
                est_OSMap_obj = torch.sum(est_OSMap_obj[..., None, :]*ini_o2w.permute(0, 2, 1)[:, None, None, :, :], dim=-1)
                est_OSMap_obj = est_OSMap_obj / ini_obj_scale[:, None, None, :]
                est_OSMap_wrd[torch.logical_not(est_clopped_mask)] = 0.
                est_OSMap_obj[torch.logical_not(est_clopped_mask)] = 0.
                ##################################################
                # fig = plt.figure()
                # ax = Axes3D(fig)
                # idx = 0
                # point_1 = est_OSMap_wrd[idx][est_clopped_mask[idx]]
                # point_1 = point_1.to('cpu').detach().numpy().copy()
                # ax.scatter(point_1[::3, 0], point_1[::3, 1], point_1[::3, 2], marker="o", linestyle='None', c='c', s=0.05)
                # ax.view_init(elev=0, azim=90)
                # fig.savefig("tes_00_90.png")
                # ax.view_init(elev=0, azim=0)
                # fig.savefig("tes_00_00.png")
                # ax.view_init(elev=45, azim=45)
                # fig.savefig("tes_45_45.png")
                # plt.close()
                ##################################################


                # save results.
                pre_path_list = []
                for batch_idx in range(batch_size):
                    dict_clopped_mask =           est_clopped_mask[batch_idx].to('cpu').detach().numpy().copy().astype(np.bool_)
                    dict_clopped_distance_map =   est_clopped_distance_map[batch_idx].to('cpu').detach().numpy().copy().astype(np.float32)
                    dict_OSMap_wrd =              est_OSMap_wrd[batch_idx].to('cpu').detach().numpy().copy().astype(np.float32)
                    dict_OSMap_obj =              est_OSMap_obj[batch_idx].to('cpu').detach().numpy().copy().astype(np.float32)
                    dict_clopped_rays_d_cam =     clopped_rays_d_cam[batch_idx].to('cpu').detach().numpy().copy().astype(np.float32)
                    dict_cam_pos_wrd =            cam_pos_wrd[batch_idx].to('cpu').detach().numpy().copy().astype(np.float32)
                    dict_w2c =                    w2c[batch_idx].to('cpu').detach().numpy().copy().astype(np.float32)
                    dict_obj_pos_wrd =            ini_obj_pos[batch_idx].to('cpu').detach().numpy().copy().astype(np.float32)
                    dict_o2w =                    ini_o2w[batch_idx].to('cpu').detach().numpy().copy().astype(np.float32)
                    dict_obj_scale =              ini_obj_scale[batch_idx].to('cpu').detach().numpy().copy().astype(np.float32)
                    data_dict = {
                        'clopped_mask':dict_clopped_mask, 
                        'clopped_distance_map':dict_clopped_distance_map, 
                        'osmap_wrd':dict_OSMap_wrd, 
                        'osmap_obj':dict_OSMap_obj, 
                        'rays_d_cam':dict_clopped_rays_d_cam, 
                        'camera_pos':dict_cam_pos_wrd, 
                        'camera_rot':dict_w2c, 
                        'obj_pos':dict_obj_pos_wrd, 
                        'obj_rot':dict_o2w, 
                        'obj_scale':dict_obj_scale, 
                        }
                    path = pre_base_dir + instance_id[batch_idx] + '/' + str(str_views_id[batch_idx]).zfill(5) + '_' + str(epoch_idx).zfill(2) + '.pickle'
                    pickle_dump(data_dict, path)
                    pre_path_list.append(path)
                    # ##################################################
                    # # Check_maps.
                    # pre_data_path = pre_base_dir + instance_id[batch_idx] + '/' + str(str_views_id[batch_idx]).zfill(5) + '_' + str(epoch_idx).zfill(2)
                    # check_map_np(dict_clopped_mask, pre_data_path + '_mask.png')
                    # check_map_np(dict_clopped_distance_map, pre_data_path + '_dpth.png')
                    # check_map_np(dict_OSMap_wrd, pre_data_path + '_wrd.png')
                    # if epoch_idx==0:
                    #     check_map_torch(gt_clopped_mask[batch_idx], gt_data_path[batch_idx].split('.')[0] + '_mask.png')
                    #     check_map_torch(gt_clopped_distance_map[batch_idx], gt_data_path[batch_idx].split('.')[0] + '_dpth.png')
                    #     check_map_torch(gt_OSMap_wrd[batch_idx], gt_data_path[batch_idx].split('.')[0] + '_wrd.png')
                    # ##################################################


                # Get diff map.
                diff_distance_map = est_clopped_distance_map - gt_clopped_distance_map
                diff_OSMap_cam = diff_distance_map[..., None] * clopped_rays_d_cam
                diff_OSMap_wrd = torch.sum(diff_OSMap_cam[..., None, :]*w2c.permute(0, 2, 1)[..., None, None, :, :], dim=-1)
                diff_OSMap_obj = torch.sum(diff_OSMap_wrd[..., None, :]*ini_o2w.permute(0, 2, 1)[..., None, None, :, :], dim=-1)
                diff_OSMap_obj = diff_OSMap_obj / ini_obj_scale[..., None, None, :]
                diff_or_mask = torch.logical_or(gt_clopped_mask, est_clopped_mask)
                diff_distance_map[torch.logical_not(diff_or_mask)] = 0.
                diff_OSMap_wrd[torch.logical_not(diff_or_mask)] = 0.
                diff_OSMap_obj[torch.logical_not(diff_or_mask)] = 0.
                diff_xor_mask = torch.logical_xor(gt_clopped_mask, est_clopped_mask)


                # save results.
                for batch_idx in range(batch_size):
                    dict_diff_xor_mask =     diff_xor_mask[batch_idx].to('cpu').detach().numpy().copy().astype(np.bool_)
                    dict_diff_distance_map = diff_distance_map[batch_idx].to('cpu').detach().numpy().copy().astype(np.float32)
                    dict_diff_OSMap_wrd =    diff_OSMap_wrd[batch_idx].to('cpu').detach().numpy().copy().astype(np.float32)
                    dict_diff_OSMap_obj =    diff_OSMap_obj[batch_idx].to('cpu').detach().numpy().copy().astype(np.float32)
                    dict_pre_path =          pre_path_list[batch_idx]

                    data_dict = {
                        'clopped_mask': dict_diff_xor_mask, 
                        'clopped_distance_map': dict_diff_distance_map, 
                        'osmap_wrd': dict_diff_OSMap_wrd, 
                        'osmap_obj': dict_diff_OSMap_obj, 
                        'pre_path': dict_pre_path, 
                        }
                    path = dif_base_dir + instance_id[batch_idx] + '/' + str(str_views_id[batch_idx]).zfill(5) + '_' + str(epoch_idx).zfill(2) + '.pickle'
                    pickle_dump(data_dict, path)
                    # ##################################################
                    # # Check_maps.
                    # dif_data_path = dif_base_dir + instance_id[batch_idx] + '/' + str(str_views_id[batch_idx]).zfill(5) + '_' + str(epoch_idx).zfill(2)
                    # check_map_np(dict_diff_xor_mask, dif_data_path + '_mask.png')
                    # check_map_np(dict_diff_distance_map, dif_data_path + '_dpth.png')
                    # check_map_np(dict_diff_OSMap_wrd, dif_data_path + '_wrd.png')
                    # ##################################################