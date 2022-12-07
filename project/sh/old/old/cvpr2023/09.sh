cd /home/yyoshitake/works/DeepSDF/project
CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_woFix --N_batch=32 --num_encoder_layers=3 --num_decoder_layers=3 --main_layers_name=autoreg
