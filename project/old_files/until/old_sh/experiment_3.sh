# CUDA_VISIBLE_DEVICES=0 python val_adam_multi.py --config=configs/initnet.txt --N_batch=2 --val_instance_list_txt=instance_lists/kmean/top_256_kmeans_list_1.txt --expname=DeepTaR/chair/val --model_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0616/checkpoints/0000001500.ckpt
CUDA_VISIBLE_DEVICES=0 python val_adam_multi.py --config=configs/initnet.txt --N_batch=2 --val_instance_list_txt=instance_lists/kmean/top_256_kmeans_list_4.txt --expname=DeepTaR/chair/val --model_ckpt_path=lightning_logs/DeepTaR/chair/initnet_list0_date0616/checkpoints/0000001500.ckpt
