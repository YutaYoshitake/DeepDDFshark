cd /home/yyoshitake/works/DeepSDF/project
CUDA_VISIBLE_DEVICES=3 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/seq.txt --exp_version=con_seq_encoder_enc3dec0_Fix_dstmap                 --N_batch=32 --main_layers_name=encoder --num_encoder_layers=3 --num_decoder_layers=0 --positional_encoding_mode=yes --input_type=depthmap --N_epoch=503 --add_conf=T_Fixup
