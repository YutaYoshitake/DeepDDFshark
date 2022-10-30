cd /home/yyoshitake/works/DeepSDF/project
CUDA_VISIBLE_DEVICES=8 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc5dec5_Fix   --N_batch=32 --num_encoder_layers=5 --num_decoder_layers=5 --main_layers_name=autoreg --add_conf=T_Fixup
