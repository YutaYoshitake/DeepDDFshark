cd /home/yyoshitake/works/DeepSDF/project
CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=randn_sequential_onlymlp_enc1dec0_woFix --N_batch=32 --num_encoder_layers=1 --main_layers_name=onlymlp --view_selection=sequential
