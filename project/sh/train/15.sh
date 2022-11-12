cd /home/yyoshitake/works/DeepSDF/project
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr01_estdif --N_batch=32 --main_layers_name=autoreg \
# --num_encoder_layers=3 --num_decoder_layers=3 --add_conf=T_Fixup --inp_itr_num=1 --dec_inp_type=dif_est --val_model_epoch=384

# CUDA_VISIBLE_DEVICES=1 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr02_estdif --N_batch=32 --main_layers_name=autoreg \
# --num_encoder_layers=3 --num_decoder_layers=3 --add_conf=T_Fixup --inp_itr_num=2 --dec_inp_type=dif_est

# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr01_wpe_estdif --N_batch=32 --main_layers_name=autoreg \
# --num_encoder_layers=3 --num_decoder_layers=3 --add_conf=T_Fixup --N_epoch=403 --inp_itr_num=1 --dec_inp_type=dif_est --positional_encoding_mode=yes

# CUDA_VISIBLE_DEVICES=1 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr02_wpe_estdif --N_batch=32 --main_layers_name=autoreg \
# --num_encoder_layers=3 --num_decoder_layers=3 --add_conf=T_Fixup --N_epoch=403 --inp_itr_num=2 --dec_inp_type=dif_est --positional_encoding_mode=yes

CUDA_VISIBLE_DEVICES=2 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr02_estdif_latest --N_batch=8 --main_layers_name=autoreg \
--num_encoder_layers=3 --num_decoder_layers=3 --add_conf=T_Fixup --N_epoch=403 --inp_itr_num=2 --dec_inp_type=dif_est

# CUDA_VISIBLE_DEVICES=3 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr01_dstmap_estdif --N_batch=32 --main_layers_name=autoreg \
# --num_encoder_layers=3 --num_decoder_layers=3 --add_conf=T_Fixup --N_epoch=403 --inp_itr_num=1 --dec_inp_type=dif_est --positional_encoding_mode=yes --input_type=depthmap

# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=randn_sequential_autoreg_enc3dec3_Fix_itr01_estdif --N_batch=32 --main_layers_name=autoreg \
# --num_encoder_layers=3 --num_decoder_layers=3 --add_conf=T_Fixup --N_epoch=403 --inp_itr_num=1 --dec_inp_type=dif_est --view_selection=sequential

# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr01_wpe_OSMapcam_estdif --N_batch=32 --main_layers_name=autoreg \
# --num_encoder_layers=3 --num_decoder_layers=3 --add_conf=T_Fixup --N_epoch=403 --inp_itr_num=1 --dec_inp_type=dif_est --positional_encoding_mode=yes
