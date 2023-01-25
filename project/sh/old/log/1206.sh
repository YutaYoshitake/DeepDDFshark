CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=rndsim_autoreg_tau01img   --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=3 --main_layers_name=autoreg --add_conf=T_Fixup  --inp_itr_num=1 --total_itr=7 --positional_encoding_mode=yes --train_instance_list_txt=instance_lists/paper_exp/fultrain.txt
CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/seq.txt --exp_version=cntprg_autoreg_tau01pad                --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=3 --main_layers_name=autoreg --add_conf=T_Fixup_Pad --inp_itr_num=1 --positional_encoding_mode=yes --train_txtfile=dataset/sampled_path/chair/train/20221211035003_fultrain_continuous_1000_10_2_all --val_data_list=dataset/sampled_path/chair/val/20221211061032_val_continuous_7_10_2_all.pickle  --itr_per_frame=1 --total_itr=5 --N_epoch=80
CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/seq.txt --exp_version=cntprg_autoreg_tau01pad_FINETUNE_p2t10 --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=3 --main_layers_name=autoreg --add_conf=T_Fixup_Pad --inp_itr_num=1 --positional_encoding_mode=yes --train_txtfile=dataset/sampled_path/chair/train/20221211035003_fultrain_continuous_1000_10_2_all --val_data_list=dataset/sampled_path/chair/val/20221211061032_val_continuous_7_10_2_all.pickle  --fine_tune=yes --val_model_epoch=80

# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=rndsim_encoder_tau01_FINETUNE_lr1e5dep      --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=T_Fixup  --inp_itr_num=1 --total_itr=7 --positional_encoding_mode=yes --train_instance_list_txt=instance_lists/paper_exp/fultrain.txt --val_model_epoch=320 --fine_tune=yes --lr=1.e-5 --L_r=0.5 --L_d=1.e-2
# CUDA_VISIBLE_DEVICES=1 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=rndsim_autoreg_tau01img_FINETUNE_lr1e5dep   --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=3 --main_layers_name=autoreg --add_conf=T_Fixup  --inp_itr_num=1 --total_itr=7 --positional_encoding_mode=yes --train_instance_list_txt=instance_lists/paper_exp/fultrain.txt --val_model_epoch=312 --fine_tune=yes --lr=1.e-5 --L_r=0.5 --L_d=1.e-2
# CUDA_VISIBLE_DEVICES=2 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=rndsim_encoder_onlydec305_FINETUNE_lr1e5dep --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=onlydec3 --inp_itr_num=5 --total_itr=7 --positional_encoding_mode=yes --train_instance_list_txt=instance_lists/paper_exp/fultrain.txt --val_model_epoch=320 --fine_tune=yes --lr=1.e-5 --L_r=0.5 --L_d=1.e-2
# CUDA_VISIBLE_DEVICES=8 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=rndsim_encoder_onlydec205_FINETUNE_lr1e5dep --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=onlydec2 --inp_itr_num=5 --total_itr=7 --positional_encoding_mode=yes --train_instance_list_txt=instance_lists/paper_exp/fultrain.txt --val_model_epoch=312 --fine_tune=yes --lr=1.e-5 --L_r=0.5 --L_d=1.e-2
# CUDA_VISIBLE_DEVICES=2 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=rndsim_encoder_onlydec305_FINETUNE_lr1e5v2 --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=onlydec3 --inp_itr_num=5 --total_itr=7 --positional_encoding_mode=yes --train_instance_list_txt=instance_lists/paper_exp/fultrain.txt --val_model_epoch=392  --fine_tune=yes --lr=1.e-5 # v1 : --L_c=0.05 # v2 : なし
# CUDA_VISIBLE_DEVICES=8 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=rndsim_encoder_onlydec205_FINETUNE_lr1e5v2 --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=onlydec2 --inp_itr_num=5 --total_itr=7 --positional_encoding_mode=yes --train_instance_list_txt=instance_lists/paper_exp/fultrain.txt --val_model_epoch=400  --fine_tune=yes --lr=1.e-5 # v1 : --L_c=0.05 # v2 : なし
# CUDA_VISIBLE_DEVICES=1 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=rndsim_autoreg_tau01img_FINETUNE_lr1e5v2   --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=3 --main_layers_name=autoreg --add_conf=T_Fixup  --inp_itr_num=1 --total_itr=7 --positional_encoding_mode=yes --train_instance_list_txt=instance_lists/paper_exp/fultrain.txt --val_model_epoch=392  --fine_tune=yes --lr=1.e-5 # v1 : --L_c=0.05 # v2 : なし
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=rndsim_encoder_tau01_FINETUNE_lr1e5v2      --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=T_Fixup  --inp_itr_num=1 --total_itr=7 --positional_encoding_mode=yes --train_instance_list_txt=instance_lists/paper_exp/fultrain.txt --val_model_epoch=376  --fine_tune=yes --lr=1.e-5 # v1 : --L_c=0.05 # v2 : なし
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/txt.txt --exp_version=virndsim_encoder_tau01      --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=T_Fixup  --inp_itr_num=1 --total_itr=7 --positional_encoding_mode=yes --train_txtfile=dataset/sampled_path/chair/train/20221208032512_fultrain_randn_1000_1d0_1_top_3 --val_data_list=dataset/sampled_path/chair/val/20221208082525_val_randn_7_1d0_1_top_3.pickle --train_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_2/results --val_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_2/results
# CUDA_VISIBLE_DEVICES=1 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/txt.txt --exp_version=virndsim_autoreg_tau01img   --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=3 --main_layers_name=autoreg --add_conf=T_Fixup  --inp_itr_num=1 --total_itr=7 --positional_encoding_mode=yes --train_txtfile=dataset/sampled_path/chair/train/20221208032512_fultrain_randn_1000_1d0_1_top_3 --val_data_list=dataset/sampled_path/chair/val/20221208082525_val_randn_7_1d0_1_top_3.pickle --train_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_2/results --val_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_2/results
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/txt.txt --exp_version=bdrndsim_encoder_onlydec305itr3 --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=onlydec3 --inp_itr_num=3 --total_itr=7 --positional_encoding_mode=yes --train_txtfile=dataset/sampled_path/chair/train/20221206152812_fultrain_randn_1000_0d7 --val_data_list=dataset/sampled_path/chair/val/20221206225143_val_randn_10_0d7.pickle --train_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_2/results --val_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_2/results
# CUDA_VISIBLE_DEVICES=1 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/txt.txt --exp_version=bdrndsim_encoder_onlydec205itr3 --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=onlydec2 --inp_itr_num=3 --total_itr=7 --positional_encoding_mode=yes --train_txtfile=dataset/sampled_path/chair/train/20221206152812_fultrain_randn_1000_0d7 --val_data_list=dataset/sampled_path/chair/val/20221206225143_val_randn_10_0d7.pickle --train_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_2/results --val_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_2/results

CUDA_VISIBLE_DEVICES=1 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=rndsim_encoder_onlydec305 --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=onlydec3 --inp_itr_num=5 --total_itr=7 --positional_encoding_mode=yes --train_instance_list_txt=instance_lists/paper_exp/fultrain.txt --val_model_epoch=240
CUDA_VISIBLE_DEVICES=8 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=rndsim_encoder_onlydec205 --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=onlydec2 --inp_itr_num=5 --total_itr=7 --positional_encoding_mode=yes --train_instance_list_txt=instance_lists/paper_exp/fultrain.txt --val_model_epoch=240
CUDA_VISIBLE_DEVICES=2 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=rndsim_autoreg_tau01img   --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=3 --main_layers_name=autoreg --add_conf=T_Fixup  --inp_itr_num=1 --total_itr=7 --positional_encoding_mode=yes --train_instance_list_txt=instance_lists/paper_exp/fultrain.txt --val_model_epoch=176
CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=rndsim_encoder_tau01      --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=T_Fixup  --inp_itr_num=1 --total_itr=7 --positional_encoding_mode=yes --train_instance_list_txt=instance_lists/paper_exp/fultrain.txt --val_model_epoch=176
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=rndsim_encoder_onlydec105 --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=onlydec  --inp_itr_num=5 --total_itr=7 --positional_encoding_mode=yes --train_instance_list_txt=instance_lists/paper_exp/fultrain.txt --val_model_epoch=240
# CUDA_VISIBLE_DEVICES=1 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=rndsim_encoder_mmntau05   --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=2 --main_layers_name=encoder --add_conf=momenenc --inp_itr_num=5 --total_itr=7 --positional_encoding_mode=yes --train_instance_list_txt=instance_lists/paper_exp/fultrain.txt --val_model_epoch=176
# CUDA_VISIBLE_DEVICES=4 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=rndsim_onlymlp_tau00    --N_batch=16 --num_encoder_layers=1 --num_decoder_layers=0 --main_layers_name=onlymlp --add_conf=Nothing  --inp_itr_num=5 --total_itr=7 --positional_encoding_mode=yes --train_instance_list_txt=instance_lists/paper_exp/fultrain.txt

# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=prgrndnsimacclrem3_encoder_tau00 --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --total_itr=10 --add_conf=T_Fixup --lr=1.e-3 --val_model_epoch=160
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=subrndnsimacclrem3_encoder_tau00 --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --total_itr=10 --add_conf=T_Fixup --lr=1.e-3 --val_model_epoch=48
# CUDA_VISIBLE_DEVICES=2 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/seq.txt --exp_version=subcontseq0315ful_encoder_tau01v2 --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --total_itr=15 --itr_per_frame=3 --add_conf=T_Fixup --train_instance_list_txt=instance_lists/paper_exp/fultrain.txt
# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/seq.txt --exp_version=subcontseq0525sub_encoder_tau01v2   --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --total_itr=25 --itr_per_frame=5 --add_conf=T_Fixup --train_instance_list_txt=instance_lists/paper_exp/train.txt
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/seq.txt --exp_version=subcontseq0315_encoder_tau00v2      --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --total_itr=15 --itr_per_frame=3 --add_conf=T_Fixup
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/seq.txt --exp_version=subcontseq0315_autoreg_tau01v2      --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=3 --main_layers_name=autoreg --total_itr=15 --itr_per_frame=3 --add_conf=T_Fixup
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/seq.txt --exp_version=subcontseq0525ful_autoreg_tau01v2   --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=3 --main_layers_name=autoreg --total_itr=25 --itr_per_frame=5 --add_conf=T_Fixup --train_instance_list_txt=instance_lists/paper_exp/fultrain.txt  --positional_encoding_mode=yes
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TRAIN --config=configs/paper_exp/chair/view5/tes.txt --exp_version=subrndnsimacc_encoder_tau00 --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --total_itr=10 --add_conf=T_Fixup