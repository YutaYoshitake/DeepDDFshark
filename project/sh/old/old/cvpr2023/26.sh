cd /home/yyoshitake/works/DeepSDF/project
CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_sequential_autoreg_enc3dec3_Fix_itr02_obsdif --N_batch=32 --main_layers_name=autoreg \
--num_encoder_layers=3 --num_decoder_layers=3 --add_conf=T_Fixup --inp_itr_num=2 --dec_inp_type=dif_obs