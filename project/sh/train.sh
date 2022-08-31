# CUDA_VISIBLE_DEVICES=4 python train.py --code_mode=TRAIN --config=configs/list_0_randn/baseline_list0randn_randR05_inpOSMap_outObj.txt --exp_version=cnn01_after            --N_batch=20 --loss_timing=after_mean
# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TRAIN --config=configs/list_0_randn/baseline_list0randn_randR05_inpOSMap_outObj.txt --exp_version=cnn01_before           --N_batch=20 --loss_timing=before_mean
CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TRAIN --config=configs/list_0_randn/transavg_list0randn_randR05_inpOSMap_outObj.txt --exp_version=pytorch_layers2        --N_batch=20 --transformer_model=pytorch --num_encoder_layers=2
# CUDA_VISIBLE_DEVICES=8 python train.py --code_mode=TRAIN --config=configs/list_0_randn/transavg_list0randn_randR05_inpOSMap_outObj.txt --exp_version=pytorch_layers3        --N_batch=20 --transformer_model=pytorch --num_encoder_layers=3
# CUDA_VISIBLE_DEVICES=9 python train.py --code_mode=TRAIN --config=configs/list_0_randn/transavg_list0randn_randR05_inpOSMap_outObj.txt --exp_version=pytorch_layers2_lr5e-5 --N_batch=20 --transformer_model=pytorch --num_encoder_layers=2 --lr=5.e-5
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TRAIN --config=configs/list_0_fixed/baseline_list0fixed_randR05_inpOSMap_outObj.txt --exp_version=cnn01_after            --N_batch=20 --loss_timing=after_mean
# CUDA_VISIBLE_DEVICES=2 python train.py --code_mode=TRAIN --config=configs/list_0_fixed/baseline_list0fixed_randR05_inpOSMap_outObj.txt --exp_version=cnn01_before           --N_batch=20 --loss_timing=before_mean
# CUDA_VISIBLE_DEVICES=1 python train.py --code_mode=TRAIN --config=configs/list_0_fixed/transavg_list0fixed_randR05_inpOSMap_outObj.txt --exp_version=pytorch_layers2        --N_batch=20 --transformer_model=pytorch --num_encoder_layers=2






# ＊＊＊ CCUDA_VISIBLE_DEVICES=3 python train.py --code_mode=TRAIN --config=configs/list_0_randn/transavg_list0randn_randR05_inpOSMap_outObj.txt --exp_version=meanonly_nonDropout        --N_batch=20 --num_encoder_layers=0
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TRAIN --config=configs/list_0_randn/transavg_list0randn_randR05_inpOSMap_outObj.txt --exp_version=mhaonly_nonDropout         --N_batch=20 --num_encoder_layers=0
# ＊＊＊  CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TRAIN --config=configs/list_0_randn/transavg_list0randn_randR05_inpOSMap_outObj.txt --exp_version=nonNorm_nonDropout_layers1 --N_batch=20 --num_encoder_layers=1
# CUDA_VISIBLE_DEVICES=1 python train.py --code_mode=TRAIN --config=configs/list_0_randn/transavg_list0randn_randR05_inpOSMap_outObj.txt --exp_version=nonNorm_nonDropout_layers2 --N_batch=20 --num_encoder_layers=2
# CUDA_VISIBLE_DEVICES=2 python train.py --code_mode=TRAIN --config=configs/list_0_randn/transavg_list0randn_randR05_inpOSMap_outObj.txt --exp_version=nonNorm_01Dropout_layers2  --N_batch=20 --num_encoder_layers=2
# ＊＊＊ CUDA_VISIBLE_DEVICES=2 python train.py --code_mode=TRAIN --config=configs/list_0_randn/transavg_list0randn_randR05_inpOSMap_outObj.txt --exp_version=nonNorm_nonDropout_layers3 --N_batch=20 --num_encoder_layers=3
# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TRAIN --config=configs/list_0_randn/transavg_list0randn_randR05_inpOSMap_outObj.txt --exp_version=nonNorm_01Dropout_layers3  --N_batch=20 --num_encoder_layers=3

# CUDA_VISIBLE_DEVICES=9 python train.py --code_mode=TRAIN --config=configs/list_0_randn/transavg_list0randn_randR05_inpOSMap_outObj.txt --exp_version=tes  --N_batch=20 --num_encoder_layers=3
