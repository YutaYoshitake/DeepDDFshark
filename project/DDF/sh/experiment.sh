CUDA_VISIBLE_DEVICES=0 python train_pl.py --config=configs/chair/check.txt --exp_version=tes
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_pl.py --config=configs/multi35/cat_0407.txt --exp_version=depth_mae_normal_mae_sdg
# CUDA_VISIBLE_DEVICES=0 python render_normal_rot.py --config=configs/multi35/cat_0407.txt --exp_version=depth_mae_normal_mae_seed0