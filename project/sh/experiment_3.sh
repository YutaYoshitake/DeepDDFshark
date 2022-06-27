# CUDA_VISIBLE_DEVICES=7 python train_pro.py --config=configs/dfnet_wfd.txt --frame_sequence_num=5 --N_batch=7 --val_instance_list_txt=instance_lists/kmean/top_256_kmeans_list_2.txt --expname=DeepTaR/chair/val --model_ckpt_path=lightning_logs/DeepTaR/chair/progressivewfd_list0_0621/checkpoints/0000001000.ckpt
# CUDA_VISIBLE_DEVICES=7 python train_pro.py --config=configs/dfnet_wfd.txt --frame_sequence_num=5 --N_batch=7 --val_instance_list_txt=instance_lists/kmean/top_256_kmeans_list_3.txt --expname=DeepTaR/chair/val --model_ckpt_path=lightning_logs/DeepTaR/chair/progressivewfd_list0_0621/checkpoints/0000001000.ckpt
# CUDA_VISIBLE_DEVICES=7 python train_pro.py --config=configs/dfnet_wfd.txt --frame_sequence_num=5 --N_batch=7 --val_instance_list_txt=instance_lists/kmean/top_256_kmeans_list_4.txt --expname=DeepTaR/chair/val --model_ckpt_path=lightning_logs/DeepTaR/chair/progressivewfd_list0_0621/checkpoints/0000001000.ckpt
# CUDA_VISIBLE_DEVICES=7 python train_pro.py --config=configs/dfnet_wfd.txt --frame_sequence_num=5 --N_batch=7 --val_instance_list_txt=instance_lists/kmean/top_256_kmeans_list_0.txt --expname=DeepTaR/chair/val --model_ckpt_path=lightning_logs/DeepTaR/chair/progressivewfd_list0_0621/checkpoints/0000001000.ckpt
# CUDA_VISIBLE_DEVICES=7 python train_pro.py --config=configs/dfnet_wfd.txt --frame_sequence_num=5 --N_batch=7 --val_instance_list_txt=instance_lists/kmean/top_256_kmeans_list_1.txt --expname=DeepTaR/chair/val --model_ckpt_path=lightning_logs/DeepTaR/chair/progressivewfd_list0_0621/checkpoints/0000001000.ckpt

CUDA_VISIBLE_DEVICES=7 python train_pro.py --config=configs/dfnet.txt --frame_sequence_num=5 --N_batch=5 --val_instance_list_txt=instance_lists/kmean/top_256_kmeans_list_0.txt --expname=DeepTaR/chair/val --model_ckpt_path=lightning_logs/DeepTaR/chair/progressive_list0_0621/checkpoints/0000000500.ckpt
# CUDA_VISIBLE_DEVICES=1 python train_pro.py --config=configs/dfnet.txt --frame_sequence_num=5 --N_batch=7 --val_instance_list_txt=instance_lists/kmean/top_256_kmeans_list_3.txt --expname=DeepTaR/chair/val --model_ckpt_path=lightning_logs/DeepTaR/chair/progressive_list0_0621/checkpoints/0000000500.ckpt
# CUDA_VISIBLE_DEVICES=1 python train_pro.py --config=configs/dfnet.txt --frame_sequence_num=5 --N_batch=7 --val_instance_list_txt=instance_lists/kmean/top_256_kmeans_list_4.txt --expname=DeepTaR/chair/val --model_ckpt_path=lightning_logs/DeepTaR/chair/progressive_list0_0621/checkpoints/0000000500.ckpt
# CUDA_VISIBLE_DEVICES=1 python train_pro.py --config=configs/dfnet.txt --frame_sequence_num=5 --N_batch=7 --val_instance_list_txt=instance_lists/kmean/top_256_kmeans_list_0.txt --expname=DeepTaR/chair/val --model_ckpt_path=lightning_logs/DeepTaR/chair/progressive_list0_0621/checkpoints/0000000500.ckpt
# CUDA_VISIBLE_DEVICES=1 python train_pro.py --config=configs/dfnet.txt --frame_sequence_num=5 --N_batch=7 --val_instance_list_txt=instance_lists/kmean/top_256_kmeans_list_1.txt --expname=DeepTaR/chair/val --model_ckpt_path=lightning_logs/DeepTaR/chair/progressive_list0_0621/checkpoints/0000000500.ckpt

# CUDA_VISIBLE_DEVICES=7 python train_ori.py --config=configs/dfnet.txt --frame_sequence_num=5 --N_batch=5 --val_instance_list_txt=instance_lists/kmean/top_256_kmeans_list_2.txt --expname=DeepTaR/chair/val --model_ckpt_path=lightning_logs/DeepTaR/chair/original_list0_0621/checkpoints/0000000500.ckpt
# CUDA_VISIBLE_DEVICES=2 python train_ori.py --config=configs/dfnet.txt --frame_sequence_num=5 --N_batch=7 --val_instance_list_txt=instance_lists/kmean/top_256_kmeans_list_3.txt --expname=DeepTaR/chair/val --model_ckpt_path=lightning_logs/DeepTaR/chair/original_list0_0621/checkpoints/0000000500.ckpt
# CUDA_VISIBLE_DEVICES=2 python train_ori.py --config=configs/dfnet.txt --frame_sequence_num=5 --N_batch=7 --val_instance_list_txt=instance_lists/kmean/top_256_kmeans_list_4.txt --expname=DeepTaR/chair/val --model_ckpt_path=lightning_logs/DeepTaR/chair/original_list0_0621/checkpoints/0000000500.ckpt
# CUDA_VISIBLE_DEVICES=2 python train_ori.py --config=configs/dfnet.txt --frame_sequence_num=5 --N_batch=7 --val_instance_list_txt=instance_lists/kmean/top_256_kmeans_list_0.txt --expname=DeepTaR/chair/val --model_ckpt_path=lightning_logs/DeepTaR/chair/original_list0_0621/checkpoints/0000000500.ckpt
# CUDA_VISIBLE_DEVICES=2 python train_ori.py --config=configs/dfnet.txt --frame_sequence_num=5 --N_batch=7 --val_instance_list_txt=instance_lists/kmean/top_256_kmeans_list_1.txt --expname=DeepTaR/chair/val --model_ckpt_path=lightning_logs/DeepTaR/chair/original_list0_0621/checkpoints/0000000500.ckpt

# CUDA_VISIBLE_DEVICES=3 python train_ori.py --config=configs/dfnet.txt --frame_sequence_num=5 --N_batch=7 --val_instance_list_txt=instance_lists/kmean/top_256_kmeans_list_2.txt --expname=DeepTaR/chair/val --model_ckpt_path=lightning_logs/DeepTaR/chair/original_list0_0621/checkpoints/0000000500.ckpt --model_mode=only_init
# CUDA_VISIBLE_DEVICES=3 python train_ori.py --config=configs/dfnet.txt --frame_sequence_num=5 --N_batch=7 --val_instance_list_txt=instance_lists/kmean/top_256_kmeans_list_3.txt --expname=DeepTaR/chair/val --model_ckpt_path=lightning_logs/DeepTaR/chair/original_list0_0621/checkpoints/0000000500.ckpt --model_mode=only_init
# CUDA_VISIBLE_DEVICES=3 python train_ori.py --config=configs/dfnet.txt --frame_sequence_num=5 --N_batch=7 --val_instance_list_txt=instance_lists/kmean/top_256_kmeans_list_4.txt --expname=DeepTaR/chair/val --model_ckpt_path=lightning_logs/DeepTaR/chair/original_list0_0621/checkpoints/0000000500.ckpt --model_mode=only_init
# CUDA_VISIBLE_DEVICES=3 python train_ori.py --config=configs/dfnet.txt --frame_sequence_num=5 --N_batch=7 --val_instance_list_txt=instance_lists/kmean/top_256_kmeans_list_0.txt --expname=DeepTaR/chair/val --model_ckpt_path=lightning_logs/DeepTaR/chair/original_list0_0621/checkpoints/0000000500.ckpt --model_mode=only_init
# CUDA_VISIBLE_DEVICES=3 python train_ori.py --config=configs/dfnet.txt --frame_sequence_num=5 --N_batch=7 --val_instance_list_txt=instance_lists/kmean/top_256_kmeans_list_1.txt --expname=DeepTaR/chair/val --model_ckpt_path=lightning_logs/DeepTaR/chair/original_list0_0621/checkpoints/0000000500.ckpt --model_mode=only_init