CUDA_VISIBLE_DEVICES=0 python val_adam.py --config=configs/initnet_list0.txt --N_batch=1 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000500.ckpt --grad_optim_max=50 --shape_code_reg=0.01
CUDA_VISIBLE_DEVICES=3 python val_adam.py --config=configs/initnet_list0.txt --N_batch=3 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000500.ckpt --grad_optim_max=100
CUDA_VISIBLE_DEVICES=2 python train_ori.py --config=configs/initnet_list0.txt --N_batch=5 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000500.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_date0707/checkpoints/0000000700.ckpt --model_mode=only_init
CUDA_VISIBLE_DEVICES=5 python train_ori.py --config=configs/dfnet_list0_randR05.txt --N_batch=2 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000500.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_randR05_date0707/checkpoints/0000000700.ckpt
CUDA_VISIBLE_DEVICES=2 python train_ori.py --config=configs/dfnet_list0.txt --N_batch=2 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000500.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_date0707/checkpoints/0000000700.ckpt --optim_mode=progressive --init_mode=all
CUDA_VISIBLE_DEVICES=6 python train_ori.py --config=configs/dfnet_list0_randR05.txt --N_batch=5 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000200.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_randR05_date0707/checkpoints/0000000700.ckpt
CUDA_VISIBLE_DEVICES=7 python train_ori.py --config=configs/dfnet_list0.txt --N_batch=5 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000200.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_date0707/checkpoints/0000000700.ckpt
CUDA_VISIBLE_DEVICES=2 python train_ori.py --config=configs/initnet_list2.txt --N_batch=50 --val_instance_list_txt=instance_lists/kmean/kmeans2_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list2_date0705/checkpoints/0000000200.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_date0707/checkpoints/0000000300.ckpt --model_mode=only_init
CUDA_VISIBLE_DEVICES=1 python train_ini.py --config=configs/initnet_list0.txt --frame_sequence_num=5 --exp_version=date0705 --expname=DeepTaR/chair/initnet_list0
CUDA_VISIBLE_DEVICES=7 python train_ini.py --config=configs/initnet_list2.txt --frame_sequence_num=5 --exp_version=date0705 --expname=DeepTaR/chair/initnet_list2
CUDA_VISIBLE_DEVICES=1 python val_adam.py --config=configs/initnet_list0.txt --N_batch=3 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000200.ckpt
CUDA_VISIBLE_DEVICES=2 python val_adam.py --config=configs/initnet_list2.txt --N_batch=3 --val_instance_list_txt=instance_lists/kmean/kmeans2_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list2_date0705/checkpoints/0000000200.ckpt
CUDA_VISIBLE_DEVICES=5 python train_ori.py --config=configs/dfnet_list0.txt --exp_version=date0707
CUDA_VISIBLE_DEVICES=0 python train_ori.py --config=configs/dfnet_list2.txt --exp_version=date0707
CUDA_VISIBLE_DEVICES=4 python train_ori.py --config=configs/dfnet_list0_randR05.txt --exp_version=date0707
CUDA_VISIBLE_DEVICES=2 python train_ori.py --config=configs/dfnet_list0.txt --N_batch=2 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000500.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_date0707/checkpoints/0000000700.ckpt --optim_mode=optimall
CUDA_VISIBLE_DEVICES=3 python train_ori.py --config=configs/dfnet_list0.txt --N_batch=2 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000500.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_date0707/checkpoints/0000000700.ckpt --optim_mode=progressive --init_mode=all
CUDA_VISIBLE_DEVICES=5 python train_ori.py --config=configs/dfnet_list0.txt --N_batch=2 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000500.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_date0707/checkpoints/0000000700.ckpt --optim_mode=progressive --init_mode=single
CUDA_VISIBLE_DEVICES=1 python train_ori.py --config=configs/dfnet_list0_randR05.txt --N_batch=3 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000500.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_randR05_date0707/checkpoints/0000000700.ckpt --optim_mode=optimall
CUDA_VISIBLE_DEVICES=2 python train_ori.py --config=configs/dfnet_list0_randR05.txt --N_batch=3 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000500.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_randR05_date0707/checkpoints/0000000700.ckpt --optim_mode=progressive --init_mode=all
CUDA_VISIBLE_DEVICES=4 python train_ori.py --config=configs/dfnet_list0_randR05.txt --N_batch=3 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000500.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_randR05_date0707/checkpoints/0000000700.ckpt --optim_mode=progressive --init_mode=single
CUDA_VISIBLE_DEVICES=0 python train_ori.py --config=configs/dfnet_list0_randR05.txt --N_batch=3 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000200.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_randR05_date0707/checkpoints/0000000700.ckpt --optim_mode=optimall
CUDA_VISIBLE_DEVICES=1 python train_ori.py --config=configs/dfnet_list0_randR05.txt --N_batch=3 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000200.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_randR05_date0707/checkpoints/0000000700.ckpt --optim_mode=progressive --init_mode=single
CUDA_VISIBLE_DEVICES=2 python train_ori.py --config=configs/dfnet_list0_randR05.txt --N_batch=3 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000200.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_randR05_date0707/checkpoints/0000000700.ckpt --optim_mode=progressive --init_mode=single
CUDA_VISIBLE_DEVICES=0 python train_ori.py --config=configs/dfnet_list0_randR05.txt --N_batch=20 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000200.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_randR05_date0707/checkpoints/0000000700.ckpt --optim_mode=optimall
CUDA_VISIBLE_DEVICES=1 python train_ori.py --config=configs/dfnet_list0_randR05.txt --N_batch=20 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000200.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_randR05_date0707/checkpoints/0000000700.ckpt --optim_mode=progressive --init_mode=single
CUDA_VISIBLE_DEVICES=1 python val_adam.py --config=configs/initnet_list0.txt --N_batch=5 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000500.ckpt --grad_optim_max=50 --shape_code_reg=0.01
CUDA_VISIBLE_DEVICES=0 python val_adam.py --config=configs/initnet_list2.txt --N_batch=5 --val_instance_list_txt=instance_lists/kmean/kmeans2_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list2_date0705/checkpoints/0000000500.ckpt --grad_optim_max=50 --shape_code_reg=0.01
CUDA_VISIBLE_DEVICES=0 python val_adam.py --config=configs/initnet_list2.txt --N_batch=5 --val_instance_list_txt=instance_lists/kmean/kmeans2_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list2_date0705/checkpoints/0000000500.ckpt --grad_optim_max=50 --shape_code_reg=0.01
CUDA_VISIBLE_DEVICES=1 python val_adam.py --config=configs/initnet_list0.txt --N_batch=5 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000500.ckpt --grad_optim_max=50 --shape_code_reg=0.01
CUDA_VISIBLE_DEVICES=0 python train_ori.py --config=configs/dfnet_list0_randR05.txt --N_batch=1 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000500.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_depth_sample_randR05_date0713/checkpoints/0000000700.ckpt --optim_mode=progressive --init_mode=single
CUDA_VISIBLE_DEVICES=2,3,4 python train_ori.py --config=configs/dfnet_list0_depth_full_randR05.txt --exp_version=date0713
CUDA_VISIBLE_DEVICES=5 python train_ori.py --config=configs/dfnet_list0_depth_clop_randR05.txt --exp_version=date0713
CUDA_VISIBLE_DEVICES=7 python train_ori.py --config=configs/dfnet_list0_depth_sample_randR05.txt --exp_version=date0713
CUDA_VISIBLE_DEVICES=2 python train_ori.py --config=configs/dfnet_list0_randR05.txt --N_batch=10 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000500.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_depth_clop_randR05_date0713/checkpoints/0000000700.ckpt --optim_mode=progressive --init_mode=single
CUDA_VISIBLE_DEVICES=1 python train_ori.py --config=configs/dfnet_list0_randR05.txt --N_batch=10 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000500.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_depth_full_randR05_date0713/checkpoints/0000000700.ckpt --optim_mode=progressive --init_mode=single
CUDA_VISIBLE_DEVICES=0 python train_ori.py --config=configs/dfnet_list0_randR05.txt --N_batch=10 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000500.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_depth_sample_randR05_date0713/checkpoints/0000000700.ckpt --optim_mode=progressive --init_mode=single
CUDA_VISIBLE_DEVICES=5 python train_ori.py --config=configs/dfnet_list0_randR05.txt --exp_version=tes --N_batch=2 --integrate_mode=transformer_v1
# CUDA_VISIBLE_DEVICES=3,4 python train_ori.py --code_mode=TRAIN --config=configs/dfnet_list0_randR05_trans_avgv2.txt  --exp_version=date0719
# CUDA_VISIBLE_DEVICES=5,7 python train_ori.py --code_mode=TRAIN --config=configs/dfnet_list2_randR05_trans_avgv2.txt  --exp_version=date0719
# CUDA_VISIBLE_DEVICES=3,4,5,6,7 python train_ori.py --code_mode=TRAIN --config=configs/dfnet_list0_depth_raw_randR05.txt --exp_version=date0718Ld01 --L_d=0.1 # --depth_error_mode=full
# CUDA_VISIBLE_DEVICES=3,4,5,6,7 python train_ori.py --code_mode=TRAIN --config=configs/dfnet_list0_depth_raw_randR05.txt --exp_version=date0718 # --depth_error_mode=full
CUDA_VISIBLE_DEVICES=0,1 python train_ori.py --code_mode=TRAIN --config=configs/dfnet_list0_randR05_trans_mlpQ.txt --exp_version=3layer_date0719
# CUDA_VISIBLE_DEVICES=2,3 python train_ori.py --code_mode=TRAIN --config=configs/dfnet_list2_randR05_trans_avg.txt  --exp_version=date0718
# CUDA_VISIBLE_DEVICES=4,5 python train_ori.py --code_mode=TRAIN --config=configs/dfnet_list2_randR05.txt            --exp_version=date0718
# CUDA_VISIBLE_DEVICES=0 python train_ori.py --code_mode=VAL --config=configs/dfnet_list0_randR05_trans_mlpQ.txt --N_batch=12 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000500.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_randR05_trans_mlpQ_date0717/checkpoints/0000000700.ckpt --optim_mode=optimall
CUDA_VISIBLE_DEVICES=2 python train_ori.py --code_mode=VAL --config=configs/dfnet_list0_randR05_trans_avg.txt  --N_batch=2 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000500.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_randR05_trans_avgv2_date0719/checkpoints/0000000500.ckpt  --optim_mode=optimall
# CUDA_VISIBLE_DEVICES=2 python train_ori.py --code_mode=VAL --config=configs/dfnet_list0_randR05.txt            --N_batch=12 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000500.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_randR05_date0717/checkpoints/0000000700.ckpt            --optim_mode=optimall
# CUDA_VISIBLE_DEVICES=3 python train_ori.py --code_mode=VAL --config=configs/dfnet_list0_randR05_trans_mlpQ.txt --N_batch=12 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000500.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_randR05_trans_mlpQ_date0717/checkpoints/0000000600.ckpt --optim_mode=optimall
# CUDA_VISIBLE_DEVICES=4 python train_ori.py --code_mode=VAL --config=configs/dfnet_list0_randR05_trans_avg.txt  --N_batch=12 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000500.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_randR05_trans_avg_date0717/checkpoints/0000000600.ckpt  --optim_mode=optimall
# CUDA_VISIBLE_DEVICES=5 python train_ori.py --code_mode=VAL --config=configs/dfnet_list0_randR05.txt            --N_batch=12 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000500.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_randR05_date0717/checkpoints/0000000600.ckpt            --optim_mode=optimall
# CUDA_VISIBLE_DEVICES=7 python train_ori.py --code_mode=VAL --config=configs/dfnet_list2_randR05_trans_avg.txt  --N_batch=2 --val_instance_list_txt=instance_lists/kmean/kmeans2_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list2_date0705/checkpoints/0000000500.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list2_randR05_trans_avg_date0718/checkpoints/0000000500.ckpt  --optim_mode=optimall
# CUDA_VISIBLE_DEVICES=2 python train_ori.py --code_mode=VAL --config=configs/dfnet_list2_randR05.txt            --N_batch=2 --val_instance_list_txt=instance_lists/kmean/kmeans2_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list2_date0705/checkpoints/0000000500.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list2_randR05_date0718/checkpoints/0000000500.ckpt            --optim_mode=optimall