python make_gt_data.py
# CUDA_VISIBLE_DEVICES=1 python make_pre_data.py --N_batch=32 --config=configs/make_data.txt
CUDA_VISIBLE_DEVICES=0 python make_pre_data.py --N_batch=128 --config=configs/make_data.txt