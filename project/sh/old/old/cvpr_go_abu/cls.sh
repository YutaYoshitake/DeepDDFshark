cd /home/yyoshitake/works/DeepSDF/project # autoreg_enc3dec3_Fix_itr01_estdif_cls
CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=tes  --N_batch=8 --main_layers_name=autoreg \
--num_encoder_layers=3 --num_decoder_layers=3 --add_conf=T_Fixup --inp_itr_num=1 --dec_inp_type=dif_est # --val_model_epoch=48