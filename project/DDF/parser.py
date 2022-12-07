import configargparse





def get_args():

    parser = config_parser()
    args = parser.parse_args()

    if args.use_3d_code:
        if args.integrate_sampling_mode == 'CAT':
            args.latent_3d_size = 250
            args.voxel_scale = 0.5
        elif args.integrate_sampling_mode == 'TransFormer':
                if  args.integrate_TransFormer_mode == 'tf_sum':
                    args.latent_3d_size = args.voxel_ch_num
                elif args.integrate_TransFormer_mode == 'tf_cat':
                    args.latent_3d_size = args.voxel_ch_num * args.voxel_sample_num
                args.voxel_scale = 0.5
    
    args.test_path = '/home/yyoshitake/works/DeepSDF/project/dataset/dugon/single_test_train/e488826128fe3854b300c4ca2f51c01b/00010'

    return args



def config_parser():

    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=512, 
                        help='channels per layer')
    
    parser.add_argument("--task_type", type=str, default='train',
                        help='')
    parser.add_argument("--save_interval", type=int, default=1, 
                        help='')
    parser.add_argument("--N_views", type=int, default=128, 
                        help='number of train views')
    parser.add_argument("--N_rays", type=int, default=1024, 
                        help='number of train train')
    parser.add_argument("--H", type=int, default=512, 
                        help='')
    parser.add_argument("--ddf_H", type=int, default=256, 
                        help='')
    parser.add_argument("--W", type=int, default=512, 
                        help='')
    parser.add_argument("--fov", type=float, default=60,
                        help="")
    parser.add_argument("--N_batch", type=int, default=2, 
                        help='batch size')
    parser.add_argument("--N_epoch", type=int, default=100000, 
                        help='epoch size')
    parser.add_argument("--embed_type", type=str, default='non',
                        help='')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--use_world_dir", action='store_true',
                        help='')
    parser.add_argument("--model_index", type=int, default=-1,
                        help='')
    parser.add_argument("--video_name", type=str, default='test',
                        help='')
    parser.add_argument("--est_name", type=str, default='test',
                        help='')
    parser.add_argument("--latent_size", type=int, default=256,
                        help='')
    parser.add_argument("--model_lrate", type=float, default=0.0001,
                        help="")
    parser.add_argument("--vec_lrate", type=float, default=0.0005,
                        help="")
    parser.add_argument("--instance_list_txt", type=str, default='non',
                        help='')
    parser.add_argument("--code_reg_lambda", type=float, default=0.0001,
                        help='')
    parser.add_argument("--N_instances", type=int, default=1,
                        help='')
    parser.add_argument("--video_type", type=str, default='rot',
                        help='')
    parser.add_argument("--mapping_size", type=int, default=-1,
                        help='')
    parser.add_argument("--mapping_scale", type=float, default=10,
                        help='')
    parser.add_argument("--use_3d_code", action='store_true',
                        help='')
    parser.add_argument("--voxel_ch_num", type=int, default=8,
                        help='')
    parser.add_argument("--voxel_sample_num", type=int, default=0,
                        help='')
    parser.add_argument("--only_latent", type=bool, default=False,
                        help='')
    parser.add_argument("--use_grad_normal", action='store_true',
                        help='')
    parser.add_argument("--use_pixel_normal", action='store_true',
                        help='')
    parser.add_argument("--normal_weight", type=float, default=0.1,
                        help='')
    parser.add_argument("--all_normal", action='store_true',
                        help='')
    parser.add_argument("--strong_depth", action='store_true',
                        help='')
    parser.add_argument("--use_normal_data", action='store_true',
                        help='')
    parser.add_argument("--use_normal_loss", action='store_true',
                        help='')
    parser.add_argument("--use_normal", action='store_true',
                        help='')
    parser.add_argument("--instance_id", type=int, default=0,
                        help='')
    parser.add_argument("--exp_name_1", type=int, default=0,
                        help='')
    parser.add_argument("--exp_name_2", type=int, default=0,
                        help='')
    parser.add_argument("--use_margine", action='store_true',
                        help='')
    parser.add_argument("--margine_var", type=float, default=0,
                        help='')
    parser.add_argument("--use_perturb_normal", action='store_true',
                        help='')
    parser.add_argument('--pixel_diff_type', type=str, default='dir',
                        help='pos or dir')
    parser.add_argument("--pixel_diff_ratio", type=float, default=1e-3,
                        help='')
    parser.add_argument("--latent_final_dim", type=int, default=64,
                        help='')
    parser.add_argument("--blend_latent", action='store_true',
                        help='')
    parser.add_argument("--blend_instance_id", type=int, default=0,
                        help='')
    parser.add_argument("--blend_ratio", type=float, default=0.5,
                        help='')
    parser.add_argument('--integrate_sampling_mode', type=str, default='CAT',
                        help='pos or dir')
    parser.add_argument("--decoders_sampler", action='store_true',
                        help='')
    parser.add_argument("--time_series_model_dim_hidden", type=int, default=128,
                        help='')
    parser.add_argument("--time_series_model_n_layer", type=int, default=1,
                        help='')
    parser.add_argument("--num_heads", type=int, default=1,
                        help='')
    parser.add_argument("--num_workers", type=int, default=0,
                        help='')
    parser.add_argument("--exp_version", type=str, default=0,
                        help='')
    parser.add_argument("--random_sample_rays", type=bool, default=True,
                        help='')
    parser.add_argument("--sample_ratio", type=float, default=1.0,
                        help='')
    parser.add_argument("--inside_true_ratio", type=float, default=1.0,
                        help='')
    parser.add_argument("--outside_true_ratio", type=float, default=0.01,
                        help='')
    parser.add_argument("--train_data_dir", type=str, 
                        help='')
    parser.add_argument("--val_data_dir", type=str, 
                        help='')
    parser.add_argument("--N_val_views", type=int, default=100,
                        help='')
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1,
                        help='')
    parser.add_argument("--integrate_TransFormer_mode", type=str, default='tf_sum',
                        help='')
    parser.add_argument("--same_instances", type=bool, default=False,
                        help='')

    return parser
