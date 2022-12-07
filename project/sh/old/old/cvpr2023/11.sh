cd /home/yyoshitake/works/DeepSDF/project
CUDA_VISIBLE_DEVICES=3 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr03_obsdif --N_batch=32 --main_layers_name=autoreg \
--num_encoder_layers=3 --num_decoder_layers=3 --add_conf=T_Fixup --inp_itr_num=3 --dec_inp_type=dif_obs --val_model_epoch=32