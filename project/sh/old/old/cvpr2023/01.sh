cd /home/yyoshitake/works/DeepSDF/project
# CUDA_VISIBLE_DEVICES=2 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=onlymlp_enc1dec0_woFix --N_batch=32 --num_encoder_layers=1 --num_decoder_layers=0 --main_layers_name=onlymlp --val_model_epoch=184
CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=onlymlp_enc1dec0_woFix_wpe --N_batch=32 --main_layers_name=onlymlp \
--num_encoder_layers=1 --num_decoder_layers=0 --N_epoch=403 --positional_encoding_mode=yes