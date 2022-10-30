cd /home/yyoshitake/works/DeepSDF/project
CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr02_attnmask --N_batch=32 --main_layers_name=autoreg \
--num_encoder_layers=3 --num_decoder_layers=3 --add_conf=T_Fixup --inp_itr_num=2 --use_attn_mask=yes --val_model_epoch=32