cd /home/yyoshitake/works/DeepSDF/project
CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=encoder_enc5dec0_woFix --N_batch=32 --num_encoder_layers=5 --num_decoder_layers=0 --main_layers_name=encoder
