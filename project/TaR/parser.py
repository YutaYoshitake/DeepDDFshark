import configargparse





def get_args():

    parser = config_parser()
    args = parser.parse_args()

    return args



def config_parser():

    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--exp_version", type=str, default=0,
                        help='')

    # Data info
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
    
    return parser
