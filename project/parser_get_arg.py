import configargparse
import os
from often_use import txt2list



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
    # args.convergence_thr = args.convergence_thr_shape = 0
    return args



def reload_args(args, sys_argv):
    # テスト用に指定されたArgsを取得
    designated_args = set([argv_i.split('--')[-1].split('=')[0] for argv_i in sys_argv if '--' in argv_i])
    designated_args |= {'gpu_num', 'model_ckpt_path', 'initnet_ckpt_path'}

    # それ以外のArgsをリロード
    print('##### RELOAD ARGS LOG #####')
    args_log_list = txt2list(os.path.join('lightning_logs', f'{args.expname}{args.exp_version}', 'args.txt'))
    args_log_key_list = []

    for args_log in args_log_list:
        args_log_key, args_log_value = args_log.split(' = ')
        args_log_key_list.append(args_log_key)
        
        if not args_log_key in designated_args:
            if hasattr(args, args_log_key):
                if type(getattr(args, args_log_key)) is bool:
                    args_log_value = args_log_value == 'True'
                elif isinstance(getattr(args, args_log_key), int):
                    args_log_value = int(args_log_value)
                elif isinstance(getattr(args, args_log_key), float):
                    args_log_value = float(args_log_value)
                elif isinstance(getattr(args, args_log_key), str):
                    args_log_value = str(args_log_value)
                elif getattr(args, args_log_key) is None:
                    args_log_value = None
                else:
                    print(f'Not yet implemented reeloading args whose type is {args_log_key}-{type(getattr(args, args_log_key))}.')
                    import pdb; pdb.set_trace()
                setattr(args, args_log_key, args_log_value)
            
            else:
                print(f'   Current args donot have {args_log_key}')
    print('')

    # 訓練を開始したときからのコード変更でリロードできなかったArgs
    print('##### LACKED ARGS LOG #####')
    for ori_args_key in [args_i for args_i in dir(args) if args_i[0] != '_']:
        if not ori_args_key in args_log_key_list:
            print(f'   Reloaded args donot have {ori_args_key} ->  {getattr(args, ori_args_key)}')
    print('')

    # if args.view_selection == 'simultaneous':
    #     args.convergence_thr = 25 # 5
    #     args.convergence_thr_shape = 50 # 25
    # if args.view_selection == 'sequential':
    #     args.convergence_thr = 50 # 30
    #     args.convergence_thr_shape = 80 # 50
    return args



def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--code_mode", type=str,
                        help='')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--exp_version", type=str, default='tes',
                        help='')
    parser.add_argument("--optimizer_type", type=str, default='origin', 
                        help='')
    parser.add_argument("--transformer_model", type=str, default='pytorch', 
                        help='')
    parser.add_argument("--input_type", type=str, default='osmap', 
                        help='')
    parser.add_argument("--output_diff_coordinate", type=str, default='img', 
                        help='')
    parser.add_argument("--num_encoder_layers", type=int, default=2, 
                        help='')
    parser.add_argument("--num_decoder_layers", type=int, default=1, 
                        help='')
    parser.add_argument("--positional_encoding_mode", type=str, default='no', 
                        help='')
    parser.add_argument("--integration_mode", type=str, default='cls_token', 
                        help='')
    parser.add_argument("--loss_timing", type=str, default='after_mean', 
                        help='')
    parser.add_argument("--hidden_dim", type=int, default=512, 
                        help='')
    parser.add_argument("--dim_feedforward", type=int, default=2048, 
                        help='')
    parser.add_argument("--dropout", type=float, default=0.1, 
                        help='')
    parser.add_argument("--num_head", type=int, default=8, 
                        help='')
    parser.add_argument("--split_into_patch", type=str, default='non', # 'store_false', 
                        help='')
    parser.add_argument("--encoder_norm_type", type=str, default='LayerNorm', 
                        help='')
    parser.add_argument("--train_data_dir", type=str, default='non', 
                        help='')
    parser.add_argument("--train_instance_list_txt", type=str, 
                        help='')
    parser.add_argument("--val_data_dir", type=str, default='non', 
                        help='')
    parser.add_argument("--layer_wise_attention", type=str, default='no', 
                        help='')
    parser.add_argument("--use_cls", type=str, default='no', 
                        help='')
    parser.add_argument("--val_instance_list_txt", type=str, default='non', 
                        help='')
    parser.add_argument("--test_data_dir", type=str, default='non', 
                        help='')
    parser.add_argument("--test_instance_list_txt", type=str, default='non', 
                        help='')
    parser.add_argument("--automatic_optimization", type=str, default='manual', 
                        help='')
    parser.add_argument("--model_ckpt_path", type=str, default='non', 
                        help='')
    parser.add_argument("--initnet_ckpt_path", type=str, default='non', 
                        help='')
    parser.add_argument("--init_mode", type=str, default='all',
                        help='')
    parser.add_argument("--init_net_name", type=str, default='non', 
                        help='')
    parser.add_argument("--val_model_epoch", type=int, default=0, 
                        help='')
    parser.add_argument("--init_net_epoch", type=int, default=0, 
                        help='')
    parser.add_argument("--backboneconfs_N_randn", type=int, default=1, 
                        help='')
    parser.add_argument("--lr_backbone", type=float, default=1.e-3, 
                        help='')
    parser.add_argument("--backbone_norms", type=str, default='with_norm', 
                        help='')
    parser.add_argument("--backboneconfs_datadir_list", type=str, default='gt', 
                        help='')
    parser.add_argument("--backbone_training_strategy", type=str, default='scratch', 
                        help='')
    parser.add_argument("--add_conf", type=str, default='Nothing', 
                        help='')
    parser.add_argument("--canonical_data_path", type=str, default='Nothing', 
                        help='')
    parser.add_argument("--canonical_fov", type=float, default=60, 
                        help='')
    parser.add_argument("--val_data_list", type=str, default='Nothing', 
                        help='')
    parser.add_argument("--tes_data_list", type=str, default='Nothing', 
                        help='')
    parser.add_argument("--pickle_to_check_qantitive_results", type=str, default='not_given', 
                        help='')
    parser.add_argument("--inp_itr_num", type=int, default=1, 
                        help='')
    parser.add_argument("--dec_inp_type", type=str, default='dif_obs', 
                        help='')
    parser.add_argument("--use_attn_mask", type=str, default='no', 
                        help='')
    parser.add_argument("--view_position", type=str, default='randn',
                        help='')
    parser.add_argument("--view_selection", type=str, default='simultaneous',
                        help='')
    parser.add_argument("--until_convergence", type=str, default='no',
                        help='')
    parser.add_argument("--itr_per_frame", type=int, default=1, 
                        help='')
    parser.add_argument("--convergence_thr", type=float, default=5, 
                        help='')
    parser.add_argument("--convergence_thr_shape", type=float, default=25, 
                        help='')
    parser.add_argument("--fine_tune", type=str, default='no', 
                        help='')
    parser.add_argument("--train_N_views", type=int, # 不要
                        help='')
    parser.add_argument("--val_N_views", type=int, # 不要
                        help='')
    parser.add_argument("--test_N_views", type=int, # 不要
                        help='')
    parser.add_argument("--frame_num", type=int, # 不要
                        help='')
    parser.add_argument("--reset_transformer_params", action='store_true', # 不要
                        help='')
    parser.add_argument("--use_sampled_txtfile", type=str, default='no', # 不要
                        help='')
    parser.add_argument("--train_txtfile", type=str, default='no', # 不要
                        help='')

    # training options
    parser.add_argument("--input_H", type=int, default=256, 
                        help='')
    parser.add_argument("--input_W", type=int, default=256, 
                        help='')
    parser.add_argument("--ddf_H_W_during_dfnet", type=int, default=256, 
                        help='')
    parser.add_argument("--fov", type=float, default=60,
                        help="")
    parser.add_argument("--save_interval", type=int, default=1, 
                        help='')
    parser.add_argument("--N_batch", type=int, default=1, 
                        help='batch size')
    parser.add_argument("--N_epoch", type=int, default=1, 
                        help='epoch size')
    parser.add_argument("--num_workers", type=int, default=32,
                        help='')
    parser.add_argument("--check_val_every_n_epoch", type=int, 
                        help='epoch size')
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="")
    parser.add_argument("--total_itr", type=int, default=5, 
                        help='')
    parser.add_argument("--grad_optim_max", type=int, default=50,
                        help='')
    parser.add_argument("--shape_code_reg", type=float, default=0.1, 
                        help='')
    parser.add_argument("--frame_sequence_num", type=int, # 不要
                        help='')
    parser.add_argument("--depth_sampling_type", type=str, # 不要
                        help='')
    parser.add_argument("--total_obs_num", type=int, default=5, 
                        help='')
    parser.add_argument("--rand_P_range", type=float, default=0.3, 
                        help='')
    parser.add_argument("--rand_S_range", type=float, default=0.3, 
                        help='')
    parser.add_argument("--rand_R_range", type=float, default=0.5, 
                        help='')
    parser.add_argument("--rand_Z_sigma", type=float, default=0.05, 
                        help='')
    parser.add_argument("--L_p", type=float, default=1e1, 
                        help='')
    parser.add_argument("--L_s", type=float, default=1e1, 
                        help='')
    parser.add_argument("--L_a", type=float, default=1e0, 
                        help='')
    parser.add_argument("--L_c", type=float, default=1e1, 
                        help='')
    parser.add_argument("--L_d", type=float, default=1e0, 
                        help='')
    parser.add_argument("--main_layers_name", type=str, default='non', 
                        help='')
    parser.add_argument("--optnet_InOut_type", type=str, # 不要
                        help='')
    parser.add_argument("--backbone_model_path", type=str, # 不要
                        help='')
    parser.add_argument("--depth_error_mode", type=str, default='non', 
                        help='')
    parser.add_argument("--integrate_mode", type=str, default='average', 
                        help='')
    parser.add_argument("--trans_integrate_mode", type=str, default='average', 
                        help='')
    

    # ddf config.
    parser.add_argument("--ddf_model_path", type=str, default='non', 
                        help='')
    parser.add_argument('--ddf_instance_list_txt', type=str, default='non', 
                        help='pos or dir')
    parser.add_argument("--N_instances", type=int, default=3196,
                        help='')
    parser.add_argument("--ddf_H", type=int, default=256, 
                        help='')
    parser.add_argument("--ddf_W", type=int, default=256, 
                        help='')
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=512, 
                        help='channels per layer')
    parser.add_argument("--use_world_dir", type=bool, default=True,
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
    parser.add_argument('--test_model', type=str, default='nomal',
                        help='pos or dir')

    return parser