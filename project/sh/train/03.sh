cd /home/yyoshitake/works/DeepSDF/project
CUDA_VISIBLE_DEVICES=2 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/seq.txt --exp_version=con_seq_onlymlp_enc1dec0_woFix_dstmap               --N_batch=32 --main_layers_name=onlymlp --num_encoder_layers=1 --num_decoder_layers=0 --positional_encoding_mode=yes --input_type=depthmap --N_epoch=503
