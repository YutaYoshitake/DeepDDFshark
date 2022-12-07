cd /home/yyoshitake/works/DeepSDF/project
# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=encoder_enc3dec0_Fix --N_batch=32 --main_layers_name=encoder\
#  --num_encoder_layers=3 --num_decoder_layers=0 --add_conf=T_Fixup --val_model_epoch=376 --N_epoch=403
CUDA_VISIBLE_DEVICES=8 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=encoder_enc3dec0_Fix_wpe --N_batch=32 --main_layers_name=encoder\
 --num_encoder_layers=3 --num_decoder_layers=0 --add_conf=T_Fixup --N_epoch=403 --positional_encoding_mode=yes