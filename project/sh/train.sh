# CUDA_VISIBLE_DEVICES=2 python train_ori.py --code_mode=TRAIN --config=configs/dfnet_list0_randR05_sha.txt  --exp_version=date0721 --N_batch=10
# CUDA_VISIBLE_DEVICES=3 python train_ori.py --code_mode=TRAIN --config=configs/dfnet_list0_randR05_sha_zero_atten.txt  --exp_version=date0721 --N_batch=10
# CUDA_VISIBLE_DEVICES=4 python train_ori.py --code_mode=TRAIN --config=configs/dfnet_list0_randR05_sha_separate_atten.txt  --exp_version=date0721 --N_batch=10
# CUDA_VISIBLE_DEVICES=5 python train_ori.py --code_mode=TRAIN --config=configs/dfnet_list0_randR05_mha_avg.txt  --exp_version=date0721 --N_batch=10
# CUDA_VISIBLE_DEVICES=6 python train_ori.py --code_mode=TRAIN --config=configs/dfnet_list0_randR05_mha_avg_learned_weighted.txt  --exp_version=date0721 --N_batch=10
# CUDA_VISIBLE_DEVICES=7 python train_ori.py --code_mode=TRAIN --config=configs/dfnet_list0_randR05_origin_after.txt  --exp_version=date0721 --N_batch=10
CUDA_VISIBLE_DEVICES=8 python train_ori.py --code_mode=TRAIN --config=configs/dfnet_list0_randR05_origin_before.txt  --exp_version=date0721 --N_batch=10