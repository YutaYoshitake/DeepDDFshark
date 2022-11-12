cd /home/yyoshitake/works/DeepSDF/project
CUDA_VISIBLE_DEVICES=1 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=rdn_sim_encoder_enc3dec0_Fix_dstmap                 --N_batch=32 --main_layers_name=encoder --num_encoder_layers=3 --num_decoder_layers=0 --positional_encoding_mode=yes --input_type=depthmap --N_epoch=503 --add_conf=T_Fixup
