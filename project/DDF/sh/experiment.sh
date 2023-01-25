CUDA_VISIBLE_DEVICES=0,1 python train_pl.py --config=configs/chair/check300.txt
CUDA_VISIBLE_DEVICES=8 python train_pl.py --config=configs/display/check.txt
CUDA_VISIBLE_DEVICES=2,5,6,7 python train_pl.py --config=configs/cabinet/cat.txt
CUDA_VISIBLE_DEVICES=0 python train_pl.py --config=configs/table/check.txt
CUDA_VISIBLE_DEVICES=0,1 python train_pl.py --config=configs/table/cat.txt

# CUDA_VISIBLE_DEVICES=0 python train_pl.py --config=configs/chair/cat.txt
# CUDA_VISIBLE_DEVICES=0 python train_pl.py --config=configs/chair/cat.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_pl.py --config=configs/multi35/cat_0407.txt --exp_version=depth_mae_normal_mae_sdg
CUDA_VISIBLE_DEVICES=2,3 MASTER_PORT=11000 GROUP_RANK=2 WORLD_SIZE=2 LOCAL_RANK=0 python ddp.py --gpus '0,1' --distributed_backend 'ddp'
CUDA_VISIBLE_DEVICES=2,3 MASTER_PORT=11000 GROUP_RANK=2 WORLD_SIZE=2 LOCAL_RANK=1 python ddp.py --gpus '0,1' --distributed_backend 'ddp'

MASTER_PORT=15000 GROUP_RANK=0 WORLD_SIZE=4 LOCAL_RANK=0 python ddp.py --gpus '0,1,2,3' --distributed_backend 'ddp'
MASTER_PORT=15000 GROUP_RANK=0 WORLD_SIZE=4 LOCAL_RANK=1 python ddp.py --gpus '0,1,2,3' --distributed_backend 'ddp'
MASTER_PORT=15000 GROUP_RANK=0 WORLD_SIZE=4 LOCAL_RANK=2 python ddp.py --gpus '0,1,2,3' --distributed_backend 'ddp'
MASTER_PORT=15000 GROUP_RANK=0 WORLD_SIZE=4 LOCAL_RANK=3 python ddp.py --gpus '0,1,2,3' --distributed_backend 'ddp'