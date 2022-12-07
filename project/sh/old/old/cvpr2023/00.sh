cd /home/yyoshitake/works/DeepSDF/project
CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc5dec5_Fix --N_batch=32 --main_layers_name=autoreg \
--num_encoder_layers=5 --num_decoder_layers=5 --add_conf=T_Fixup --N_epoch=401 --val_model_epoch=376
CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=encoder_enc5dec0_Fix --N_batch=32 --main_layers_name=encoder \
--num_encoder_layers=5 --num_decoder_layers=0 --add_conf=T_Fixup --N_epoch=401 --val_model_epoch=376
CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=encoder_enc5dec0_woFix --N_batch=32 --main_layers_name=encoder\
--num_encoder_layers=5 --num_decoder_layers=0 --add_conf=Nothing --N_epoch=401 --val_model_epoch=376

# cd /home/yyoshitake/works/DeepSDF/project
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix   --N_batch=32 --main_layers_name=autoreg \
#    --num_encoder_layers=3 --num_decoder_layers=3 --add_conf=T_Fixup --N_epoch=401 --val_model_epoch=376 
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr03_obsdif --N_batch=32 --main_layers_name=autoreg \
#    --num_encoder_layers=3 --num_decoder_layers=3 --add_conf=T_Fixup --inp_itr_num=3 --dec_inp_type=dif_obs --N_epoch=401 --val_model_epoch=384 
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr02_obsdif --N_batch=32 --main_layers_name=autoreg \
#    --num_encoder_layers=3 --num_decoder_layers=3 --add_conf=T_Fixup --inp_itr_num=2 --dec_inp_type=dif_obs --N_epoch=401 --val_model_epoch=384 

# cd /home/yyoshitake/works/DeepSDF/project
# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr02_attnmask --N_batch=32 --main_layers_name=autoreg \
#    --num_encoder_layers=3 --num_decoder_layers=3 --add_conf=T_Fixup --inp_itr_num=2 --use_attn_mask=yes --dec_inp_type=dif_obs --N_epoch=401 --val_model_epoch= 384 
# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=encoder_onlydecv2_Fix --N_batch=32 --main_layers_name=encoder \
#    --num_encoder_layers=3 --num_decoder_layers=0 --add_conf=onlydecv2 --N_epoch=401 --val_model_epoch=376 
# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=encoder_onlydecv3_Fix --N_batch=32 --main_layers_name=encoder \
#    --num_encoder_layers=3 --num_decoder_layers=0 --add_conf=onlydecv3 --N_epoch=401 --val_model_epoch=376 