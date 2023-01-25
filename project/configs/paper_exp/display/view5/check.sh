# 無作為・PE無し
# CUDA_VISIBLE_DEVICES=2 python train.py --code_mode=TRAIN --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_autoreg_prgWope --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=3 --main_layers_name=autoreg --add_conf=T_Fixup_Pad --inp_itr_num=1 --total_itr=7 --positional_encoding_mode=no --prg=yes
# CUDA_VISIBLE_DEVICES=3 python train.py --code_mode=TRAIN --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_encoder_prgWope --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=T_Fixup     --inp_itr_num=1 --total_itr=7 --positional_encoding_mode=no --prg=yes --val_model_epoch=56
# CUDA_VISIBLE_DEVICES=8 python train.py --code_mode=TRAIN --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_onlydec_prgWope --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=onlydec     --inp_itr_num=2 --total_itr=7 --positional_encoding_mode=no --prg=yes

# 連続・PE無し
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TRAIN --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_autoreg_prgWope           --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=3 --main_layers_name=autoreg --add_conf=T_Fixup_Pad --inp_itr_num=1 --positional_encoding_mode=no --prg=yes --cnt_prg_step=160 --N_epoch=160
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TRAIN --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_autoreg_prgWope_FINETUNE_ --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=3 --main_layers_name=autoreg --add_conf=T_Fixup_Pad --inp_itr_num=1 --positional_encoding_mode=no --fine_tune=yes --val_model_epoch=160
# CUDA_VISIBLE_DEVICES=1 python train.py --code_mode=TRAIN --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_encoder_prgWope           --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=T_Fixup     --inp_itr_num=1 --positional_encoding_mode=no --prg=yes --cnt_prg_step=160 --N_epoch=160
# CUDA_VISIBLE_DEVICES=1 python train.py --code_mode=TRAIN --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_encoder_prgWope_FINETUNE_ --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=T_Fixup     --inp_itr_num=1 --positional_encoding_mode=no --fine_tune=yes --val_model_epoch=160
# CUDA_VISIBLE_DEVICES=4 python train.py --code_mode=TRAIN --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_onlydec_prgWope           --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=onlydec     --inp_itr_num=2 --positional_encoding_mode=no --prg=yes --cnt_prg_step=160 --N_epoch=160
# CUDA_VISIBLE_DEVICES=4 python train.py --code_mode=TRAIN --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_onlydec_prgWope_FINETUNE_ --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=onlydec     --inp_itr_num=2 --positional_encoding_mode=no --fine_tune=yes --val_model_epoch=160

# CUDA_VISIBLE_DEVICES=4 python train.py --code_mode=REVAL --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_autoreg_prg --N_batch=16
# CUDA_VISIBLE_DEVICES=4 python train.py --code_mode=REVAL --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_encoder_prg --N_batch=16
# CUDA_VISIBLE_DEVICES=4 python train.py --code_mode=REVAL --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_onlydec_prg --N_batch=16

# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TRAIN --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_autoreg_FINETUNE_ --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=3 --main_layers_name=autoreg --add_conf=T_Fixup_Pad --inp_itr_num=1 --positional_encoding_mode=yes --fine_tune=yes --val_model_epoch=80
# CUDA_VISIBLE_DEVICES=8 python train.py --code_mode=TRAIN --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_encoder_FINETUNE_ --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=T_Fixup     --inp_itr_num=1 --positional_encoding_mode=yes --fine_tune=yes --val_model_epoch=80
# CUDA_VISIBLE_DEVICES=3 python train.py --code_mode=TRAIN --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_onlydec_FINETUNE_ --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=onlydec     --inp_itr_num=2 --positional_encoding_mode=yes --fine_tune=yes --val_model_epoch=80

# 自己回帰モデル
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TRAIN --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_autoreg --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=3 --main_layers_name=autoreg --add_conf=T_Fixup_Pad --inp_itr_num=1 --positional_encoding_mode=yes --total_itr=7
# CUDA_VISIBLE_DEVICES=2 python train.py --code_mode=TRAIN --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_autoreg_prg --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=3 --main_layers_name=autoreg --add_conf=T_Fixup_Pad --inp_itr_num=1 --positional_encoding_mode=yes --total_itr=7 --prg=yes
# Encoder
# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TRAIN --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_encoder --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=T_Fixup --inp_itr_num=1 --positional_encoding_mode=yes --total_itr=7
# CUDA_VISIBLE_DEVICES=3 python train.py --code_mode=TRAIN --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_encoder_prg --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=T_Fixup --inp_itr_num=1 --positional_encoding_mode=yes --total_itr=7 --prg=yes
# Decoder
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TRAIN --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_onlydec --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=onlydec --inp_itr_num=2 --positional_encoding_mode=yes --total_itr=7
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TRAIN --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_onlydec_prg --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=onlydec --inp_itr_num=2 --positional_encoding_mode=yes --total_itr=7 --prg=yes

# CUDA_VISIBLE_DEVICES=9 python train.py --code_mode=REVAL --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_encoder --N_batch=16
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=REVAL --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_onlydec --N_batch=16
# CUDA_VISIBLE_DEVICES=9 python train.py --code_mode=REVAL --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_encoder --N_batch=16
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=REVAL --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_onlydec --N_batch=16

# OnlyOnce
# CUDA_VISIBLE_DEVICES=3 python train.py --code_mode=TRAIN --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_onlyonce           --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=only_once --inp_itr_num=1 --positional_encoding_mode=yes --view_position=randn --view_selection=simultaneous --total_itr=5 --N_epoch=80
# CUDA_VISIBLE_DEVICES=3 python train.py --code_mode=TRAIN --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_onlyonce_FINETUNE_ --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=only_once --inp_itr_num=1 --positional_encoding_mode=yes --view_position=randn --view_selection=simultaneous --total_itr=10 --fine_tune=yes --val_model_epoch=80
CUDA_VISIBLE_DEVICES=8 python train.py --code_mode=TRAIN --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_onlyonce \
--N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=only_once --inp_itr_num=1 --positional_encoding_mode=yes \
--view_selection=simultaneous --total_itr=1 --save_interval=80 \
--train_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root \
--val_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root