###############
# WPE
###############
# Display
# CUDA_VISIBLE_DEVICES=2 python train.py --code_mode=TRAIN --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_onlymlp           --N_batch=16 --num_encoder_layers=1 --num_decoder_layers=0 --main_layers_name=onlymlp --add_conf=Nothing --inp_itr_num=1 --positional_encoding_mode=yes --total_itr=7
CUDA_VISIBLE_DEVICES=4 python train.py --code_mode=TRAIN --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_onlymlp           --N_batch=16 --num_encoder_layers=1 --num_decoder_layers=0 --main_layers_name=onlymlp --add_conf=Nothing --inp_itr_num=1 --positional_encoding_mode=yes --prg=yes --cnt_prg_step=80 --N_epoch=80
CUDA_VISIBLE_DEVICES=4 python train.py --code_mode=TRAIN --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_onlymlp_FINETUNE_ --N_batch=16 --num_encoder_layers=1 --num_decoder_layers=0 --main_layers_name=onlymlp --add_conf=Nothing --inp_itr_num=1 --positional_encoding_mode=yes --fine_tune=yes --val_model_epoch=80

# Cabinet
# CUDA_VISIBLE_DEVICES=3 python train.py --code_mode=TRAIN --config=configs/paper_exp/cabinet/view5/rdn.txt --exp_version=rdn_onlymlp           --N_batch=16 --num_encoder_layers=1 --num_decoder_layers=0 --main_layers_name=onlymlp --add_conf=Nothing --inp_itr_num=1 --positional_encoding_mode=yes --total_itr=7
# CUDA_VISIBLE_DEVICES=8 python train.py --code_mode=TRAIN --config=configs/paper_exp/cabinet/view5/seq.txt --exp_version=seq_onlymlp           --N_batch=16 --num_encoder_layers=1 --num_decoder_layers=0 --main_layers_name=onlymlp --add_conf=Nothing --inp_itr_num=1 --positional_encoding_mode=yes --prg=yes --cnt_prg_step=80 --N_epoch=80
# CUDA_VISIBLE_DEVICES=8 python train.py --code_mode=TRAIN --config=configs/paper_exp/cabinet/view5/seq.txt --exp_version=seq_onlymlp_FINETUNE_ --N_batch=16 --num_encoder_layers=1 --num_decoder_layers=0 --main_layers_name=onlymlp --add_conf=Nothing --inp_itr_num=1 --positional_encoding_mode=yes --fine_tune=yes --val_model_epoch=80


###############
# WoPE
###############
# Chair

# Display
# CUDA_VISIBLE_DEVICES=? python train.py --code_mode=TRAIN --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_onlymlp_prgWope           --N_batch=16 --num_encoder_layers=1 --num_decoder_layers=0 --main_layers_name=onlymlp --add_conf=Nothing --inp_itr_num=2 --positional_encoding_mode=no --prg=yes --cnt_prg_step=160 --N_epoch=160
# CUDA_VISIBLE_DEVICES=? python train.py --code_mode=TRAIN --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_onlymlp_prgWope_FINETUNE_ --N_batch=16 --num_encoder_layers=1 --num_decoder_layers=0 --main_layers_name=onlymlp --add_conf=Nothing --inp_itr_num=2 --positional_encoding_mode=no --fine_tune=yes --val_model_epoch=160

# Cabinet
