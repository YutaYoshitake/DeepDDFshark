# Encoderモデル
# CUDA_VISIBLE_DEVICES=1 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair_re/view5/rdn.txt --exp_version=rdn_encoder           --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=T_Fixup --inp_itr_num=1 --positional_encoding_mode=yes --total_itr=7
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair_re/view5/seq.txt --exp_version=seq_encoder           --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=T_Fixup --inp_itr_num=1 --positional_encoding_mode=yes --prg=yes --cnt_prg_step=80 --N_epoch=80
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair_re/view5/seq.txt --exp_version=seq_encoder_FINETUNE_ --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=T_Fixup --inp_itr_num=1 --positional_encoding_mode=yes --fine_tune=yes --val_model_epoch=80
# Autoregモデル
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair_re/view5/rdn.txt --exp_version=rdn_autoreg           --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=3 --main_layers_name=autoreg --add_conf=T_Fixup_Pad --inp_itr_num=1 --positional_encoding_mode=yes --total_itr=7
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair_re/view5/seq.txt --exp_version=seq_autoreg           --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=3 --main_layers_name=autoreg --add_conf=T_Fixup_Pad --inp_itr_num=1 --positional_encoding_mode=yes --prg=yes --cnt_prg_step=80 --N_epoch=80
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair_re/view5/seq.txt --exp_version=seq_autoreg_FINETUNE_ --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=3 --main_layers_name=autoreg --add_conf=T_Fixup_Pad --inp_itr_num=1 --positional_encoding_mode=yes --fine_tune=yes --val_model_epoch=80
# MLPモデル
# CUDA_VISIBLE_DEVICES=2 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair_re/view5/rdn.txt --exp_version=rdn_onlymlp           --N_batch=16 --num_encoder_layers=1 --num_decoder_layers=0 --main_layers_name=onlymlp --add_conf=Nothing --inp_itr_num=1 --positional_encoding_mode=yes --total_itr=7
# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair_re/view5/seq.txt --exp_version=seq_onlymlp           --N_batch=16 --num_encoder_layers=1 --num_decoder_layers=0 --main_layers_name=onlymlp --add_conf=Nothing --inp_itr_num=1 --positional_encoding_mode=yes --prg=yes --cnt_prg_step=80 --N_epoch=80
# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair_re/view5/seq.txt --exp_version=seq_onlymlp_FINETUNE_ --N_batch=16 --num_encoder_layers=1 --num_decoder_layers=0 --main_layers_name=onlymlp --add_conf=Nothing --inp_itr_num=1 --positional_encoding_mode=yes --fine_tune=yes --val_model_epoch=80
# Onlydec
# CUDA_VISIBLE_DEVICES=3 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair_re/view5/rdn.txt --exp_version=rdn_onlydec           --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=onlydec --inp_itr_num=2 --positional_encoding_mode=yes --total_itr=7
# CUDA_VISIBLE_DEVICES=4 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair_re/view5/seq.txt --exp_version=seq_onlydec           --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=onlydec --inp_itr_num=2 --positional_encoding_mode=yes --prg=yes --cnt_prg_step=80 --N_epoch=80
# CUDA_VISIBLE_DEVICES=4 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair_re/view5/seq.txt --exp_version=seq_onlydec_FINETUNE_ --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=onlydec --inp_itr_num=2 --positional_encoding_mode=yes --fine_tune=yes --val_model_epoch=80

# OnlyOnce
CUDA_VISIBLE_DEVICES=8 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair_re/view5/seq.txt --exp_version=seq_onlyonce \
--N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=only_once --inp_itr_num=1 --positional_encoding_mode=yes \
--view_selection=simultaneous --total_itr=1 --save_interval=80 --val_model_epoch=480

# デバック
CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair_re/view5/rdn.txt --exp_version=tes --N_batch=3 --num_encoder_layers=8 --num_decoder_layers=3 --main_layers_name=autoreg --add_conf=T_Fixup_Pad --inp_itr_num=1 --positional_encoding_mode=yes --total_itr=7