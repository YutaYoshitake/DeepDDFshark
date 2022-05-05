import configargparse





def get_args():

    parser = config_parser()
    args = parser.parse_args()

    # ddf config.
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

    return args



def config_parser():

    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--exp_version", type=str, default=0,
                        help='')
    parser.add_argument("--train_data_dir", type=str, 
                        help='')
    parser.add_argument("--train_N_views", type=int, 
                        help='')
    parser.add_argument("--train_instance_list_txt", type=str, 
                        help='')
    parser.add_argument("--val_data_dir", type=str, 
                        help='')
    parser.add_argument("--val_N_views", type=int, 
                        help='')
    parser.add_argument("--val_instance_list_txt", type=str, 
                        help='')
    parser.add_argument("--frame_num", type=int, 
                        help='')

    # training options
    parser.add_argument("--H", type=int, default=256, 
                        help='')
    parser.add_argument("--W", type=int, default=256, 
                        help='')
    parser.add_argument("--fov", type=float, default=60,
                        help="")
    parser.add_argument("--save_interval", type=int, 
                        help='')
    parser.add_argument("--N_batch", type=int, 
                        help='batch size')
    parser.add_argument("--N_epoch", type=int, 
                        help='epoch size')
    parser.add_argument("--num_workers", type=int, default=0,
                        help='')
    parser.add_argument("--check_val_every_n_epoch", type=int, 
                        help='epoch size')
    parser.add_argument("--lr", type=float,
                        help="")

    # ddf config.
    parser.add_argument("--ddf_model_path", type=str, 
                        help='')
    parser.add_argument('--ddf_instance_list_txt', type=str,
                        help='pos or dir')
    parser.add_argument("--N_instances", type=int, default=3196,
                        help='')
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=512, 
                        help='channels per layer')
    parser.add_argument("--use_world_dir", action='store_true',
                        help='')
    parser.add_argument("--latent_size", type=int, default=256,
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
    parser.add_argument("--latent_final_dim", type=int, default=64,
                        help='')
    parser.add_argument("--only_latent", type=bool, default=False,
                        help='')
    parser.add_argument('--integrate_sampling_mode', type=str, default='CAT',
                        help='pos or dir')
    parser.add_argument("--same_instances", type=bool, default=False,
                        help='')
    parser.add_argument("--use_normal_loss", type=bool, default=False,
                        help='')
    parser.add_argument("--model_lrate", type=float, default=0.0001,
                        help="")
    parser.add_argument("--vec_lrate", type=float, default=0.0005,
                        help="")
    parser.add_argument("--code_reg_lambda", type=float, default=0.0001,
                        help='')
    parser.add_argument("--pixel_diff_ratio", type=float, default=1e-3,
                        help='')

    return parser
