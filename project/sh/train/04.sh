cd /home/yyoshitake/works/DeepSDF/project
CUDA_VISIBLE_DEVICES=3 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=encoder_enc3dec0_Fix   --N_batch=32 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=T_Fixup
