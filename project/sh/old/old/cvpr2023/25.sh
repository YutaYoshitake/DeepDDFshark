cd /home/yyoshitake/works/DeepSDF/project
CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=randn_sequential_encoder_enc3dec0_Fix   --N_batch=32 --num_encoder_layers=3 --main_layers_name=encoder --view_selection=sequential --add_conf=T_Fixup
