cd /home/yyoshitake/works/DeepSDF/project
CUDA_VISIBLE_DEVICES=2 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_simultaneous_autoreg_enc3dec3_Fix   --N_batch=32 --num_encoder_layers=3 --num_decoder_layers=3 --main_layers_name=autoreg --add_conf=T_Fixup
