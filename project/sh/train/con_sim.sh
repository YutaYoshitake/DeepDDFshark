# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/seq.txt --exp_version=tes --N_batch=32 --main_layers_name=autoreg --num_encoder_layers=3 --num_decoder_layers=3 --add_conf=T_Fixup --inp_itr_num=1 --dec_inp_type=dif_est --view_selection=simultaneous --N_epoch=403 # cnt_sim_autoreg_enc3dec3_Fix_itr01_estdif
CUDA_VISIBLE_DEVICES=1 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/seq.txt --exp_version=cnt_sim_encoder_enc3dec0_Fix              --N_batch=32 --main_layers_name=encoder --num_encoder_layers=3 --num_decoder_layers=0 --add_conf=T_Fixup                                        --view_selection=simultaneous --N_epoch=403
# CUDA_VISIBLE_DEVICES=2 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/seq.txt --exp_version=cnt_sim_onlymlp_enc1dec0_woFix            --N_batch=32 --main_layers_name=onlymlp --num_encoder_layers=1 --num_decoder_layers=0                                                           --view_selection=simultaneous --N_epoch=403