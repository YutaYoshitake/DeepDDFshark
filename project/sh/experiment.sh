# CUDA_VISIBLE_DEVICES=0 python train_ori.py --config=configs/dfnet_list0_randR05.txt --N_batch=20 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000200.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_randR05_date0707/checkpoints/0000000700.ckpt --optim_mode=optimall
# CUDA_VISIBLE_DEVICES=1 python train_ori.py --config=configs/dfnet_list0_randR05.txt --N_batch=20 --val_instance_list_txt=instance_lists/kmean/kmeans0_test_list.txt --initnet_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0705/checkpoints/0000000200.ckpt --model_ckpt_path=lightning_logs/DeepTaR/chair/dfnet_list0_randR05_date0707/checkpoints/0000000700.ckpt --optim_mode=progressive --init_mode=single