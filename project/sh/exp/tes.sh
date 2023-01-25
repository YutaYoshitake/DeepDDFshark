# # Chair
CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair_re/view5/rdn.txt --exp_version=rdn_encoder           --N_batch=7 --val_model_epoch=408                --tes_data_list=dataset/sampled_path/chair/tes/20221210160130_tes_unknown_randn_8_10_1_all.pickle      --canonical_data_path=/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/paper_exp/chair/canonical
CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair_re/view5/seq.txt --exp_version=seq_encoder_FINETUNE_ --N_batch=7 --val_model_epoch=352                --tes_data_list=dataset/sampled_path/chair/tes/20221214021558_tes_unknown_continuous_8_10_2_all.pickle --canonical_data_path=/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/paper_exp/chair/canonical
CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair_re/view5/rdn.txt --exp_version=rdn_onlydec           --N_batch=7 --val_model_epoch=504                --tes_data_list=dataset/sampled_path/chair/tes/20221210160130_tes_unknown_randn_8_10_1_all.pickle      --canonical_data_path=/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/paper_exp/chair/canonical
CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair_re/view5/seq.txt --exp_version=seq_onlyonce          --N_batch=7 --val_model_epoch=4720 --total_itr=1 --tes_data_list=dataset/sampled_path/chair/tes/20221214021558_tes_unknown_continuous_8_10_2_all.pickle --canonical_data_path=/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/paper_exp/chair/canonical

# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/chair_re/view5/rdn.txt --exp_version=rdn_autoreg           --N_batch=7 --val_model_epoch=504                --tes_data_list=dataset/sampled_path/chair/tes/20221210160130_tes_unknown_randn_8_10_1_all.pickle      --canonical_data_path=/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/paper_exp/chair/canonical
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/chair_re/view5/seq.txt --exp_version=seq_autoreg_FINETUNE_ --N_batch=7 --val_model_epoch=368                --tes_data_list=dataset/sampled_path/chair/tes/20221214021558_tes_unknown_continuous_8_10_2_all.pickle --canonical_data_path=/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/paper_exp/chair/canonical
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/chair_re/view5/seq.txt --exp_version=seq_onlydec_FINETUNE_ --N_batch=7 --val_model_epoch=392                --tes_data_list=dataset/sampled_path/chair/tes/20221214021558_tes_unknown_continuous_8_10_2_all.pickle --canonical_data_path=/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/paper_exp/chair/canonical

# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair_re/view5/rdn.txt --exp_version=rdn_onlymlp           --N_batch=7 --val_model_epoch=496                --tes_data_list=dataset/sampled_path/chair/tes/20221210160130_tes_unknown_randn_8_10_1_all.pickle      --canonical_data_path=/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/paper_exp/chair/canonical
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/chair_re/view5/seq.txt --exp_version=seq_onlymlp_FINETUNE_ --N_batch=7 --val_model_epoch=416                --tes_data_list=dataset/sampled_path/chair/tes/20221214021558_tes_unknown_continuous_8_10_2_all.pickle --canonical_data_path=/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/paper_exp/chair/canonical

# Table
# onlymlp
# autoreg
# encoder
# onlydec
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/table/view5/rdn.txt --exp_version=rdn_encoder           --N_batch=7 --val_model_epoch=232 --tes_data_list=dataset/sampled_path/table/tes/20230111155153_tes_randn_16_10_2_all.pickle      --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/04379243
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/table/view5/seq.txt --exp_version=seq_encoder_FINETUNE_ --N_batch=7 --val_model_epoch=232 --tes_data_list=dataset/sampled_path/table/tes/20230111155715_tes_continuous_16_10_2_all.pickle --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/04379243
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/table/view5/seq.txt --exp_version=seq_encoder_FINETUNE_copy --N_batch=7 --val_model_epoch=288 --tes_data_list=dataset/sampled_path/table/tes/20230111155715_tes_continuous_16_10_2_all.pickle --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/04379243
# デバッグ用
# CUDA_VISIBLE_DEVICES=9 python train.py --code_mode=TRAIN --config=configs/paper_exp/table/view5/seq.txt --exp_version=seq_encoder_FINETUNE_ --N_batch=16 --num_encoder_layers=3 --num_decoder_layers=0 --main_layers_name=encoder --add_conf=T_Fixup --inp_itr_num=1 --positional_encoding_mode=yes --fine_tune=no --val_model_epoch=232
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=REVAL --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_autoreg_prg --N_batch=16



# まっぷの確認
# CUDA_VISIBLE_DEVICES=1 python train.py --code_mode=TES --config=configs/paper_exp/chair_a/view5/tes.txt --exp_version=onlymlp_enc1dec0_woFix_wpe --N_batch=7 --val_model_epoch=368  --tes_data_list=dataset/sampled_path/chair/tes/20221210084125_tes_unknown_randn_5_10_1_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/old/chair/tmp_1/squashfs-root

# # WoPE Cabinetの評価（無作為）
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/cabinet/view5/rdn.txt --exp_version=rdn_encoder_prgWope --N_batch=7           --val_model_epoch=472 --tes_data_list=dataset/sampled_path/cabinet/tes/20230106022326_tes_randn_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/cabinet/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/02933112
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/cabinet/view5/rdn.txt --exp_version=rdn_autoreg_prgWope --N_batch=7           --val_model_epoch=496 --tes_data_list=dataset/sampled_path/cabinet/tes/20230106022326_tes_randn_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/cabinet/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/02933112
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/cabinet/view5/rdn.txt --exp_version=rdn_onlydec_prgWope --N_batch=7           --val_model_epoch=504 --tes_data_list=dataset/sampled_path/cabinet/tes/20230106022326_tes_randn_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/cabinet/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/02933112
# # WoPECabinetの評価（連続）
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/cabinet/view5/seq.txt --exp_version=seq_encoder_prgWope_FINETUNE_ --N_batch=7 --val_model_epoch=272 --tes_data_list=dataset/sampled_path/cabinet/tes/20230106032512_tes_continuous_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/cabinet/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/02933112
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/cabinet/view5/seq.txt --exp_version=seq_autoreg_prgWope_FINETUNE_ --N_batch=7 --val_model_epoch=240 --tes_data_list=dataset/sampled_path/cabinet/tes/20230106032512_tes_continuous_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/cabinet/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/02933112

# # display
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_onlymlp           --N_batch=7 --val_model_epoch=448 --tes_data_list=dataset/sampled_path/display/tes/20230104015836_tes_randn_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/03211117
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_onlymlp_FINETUNE_ --N_batch=7 --val_model_epoch=368 --tes_data_list=dataset/sampled_path/display/tes/20230104021508_tes_continuous_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/03211117
# # cabinet
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/cabinet/view5/rdn.txt --exp_version=rdn_onlymlp           --N_batch=7 --val_model_epoch=408 --tes_data_list=dataset/sampled_path/cabinet/tes/20230106022326_tes_randn_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/cabinet/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/02933112
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/cabinet/view5/seq.txt --exp_version=seq_onlymlp_FINETUNE_ --N_batch=7 --val_model_epoch=320 --tes_data_list=dataset/sampled_path/cabinet/tes/20230106032512_tes_continuous_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/cabinet/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/02933112
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/cabinet/view5/seq.txt --exp_version=seq_onlydec_prgWope_FINETUNE_ --N_batch=7 --val_model_epoch=272 --tes_data_list=dataset/sampled_path/cabinet/tes/20230106032512_tes_continuous_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/cabinet/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/02933112



# # 追加の確認
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/cabinet/view5/rdn.txt --exp_version=rdn_autoreg           --N_batch=7 --val_model_epoch=504 --tes_data_list=dataset/sampled_path/cabinet/tes/20230106022326_tes_randn_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/cabinet/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/02933112

# # WoPE Displayの評価（無作為）
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_autoreg_prgWope --N_batch=7 --val_model_epoch=448 --tes_data_list=dataset/sampled_path/display/tes/20230104015836_tes_randn_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/03211117
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_encoder_prgWope --N_batch=7 --val_model_epoch=472 --tes_data_list=dataset/sampled_path/display/tes/20230104015836_tes_randn_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/03211117
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_onlydec_prgWope --N_batch=7 --val_model_epoch=496 --tes_data_list=dataset/sampled_path/display/tes/20230104015836_tes_randn_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/03211117


# # 追加の確認
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/cabinet/view5/seq.txt --exp_version=seq_onlydec_FINETUNE_ --N_batch=7 --val_model_epoch=336 --tes_data_list=dataset/sampled_path/cabinet/tes/20230106032512_tes_continuous_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/cabinet/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/02933112
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_autoreg_prgWope_FINETUNE_ --N_batch=7 --val_model_epoch=384 --tes_data_list=dataset/sampled_path/display/tes/20230104021508_tes_continuous_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/03211117

# # WoPE Displayの評価（連続）
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_autoreg_prgWope_FINETUNE_ --N_batch=7 --val_model_epoch=288 --tes_data_list=dataset/sampled_path/display/tes/20230104021508_tes_continuous_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/03211117
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_encoder_prgWope_FINETUNE_ --N_batch=7 --val_model_epoch=320 --tes_data_list=dataset/sampled_path/display/tes/20230104021508_tes_continuous_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/03211117
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_onlydec_prgWope_FINETUNE_ --N_batch=7 --val_model_epoch=240 --tes_data_list=dataset/sampled_path/display/tes/20230104021508_tes_continuous_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/03211117


# # AutoRegのバグ取った
# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TES --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_autoreg_FINETUNE_ --N_batch=7 --val_model_epoch=312 --tes_data_list=dataset/sampled_path/display/tes/20230104021508_tes_continuous_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/03211117 # 
# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TES --config=configs/paper_exp/cabinet/view5/seq.txt --exp_version=seq_autoreg_FINETUNE_ --N_batch=7 --val_model_epoch=432 --tes_data_list=dataset/sampled_path/cabinet/tes/20230106032512_tes_continuous_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/cabinet/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/02933112
# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TES --config=configs/paper_exp/cabinet/view5/seq.txt --exp_version=seq_autoreg_FINETUNE_ --N_batch=7 --val_model_epoch=432 --tes_data_list=dataset/sampled_path/cabinet/tes/20230106032512_tes_continuous_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/cabinet/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/02933112





# # # WPE Displayの評価（連続）
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_autoreg_FINETUNE_ --N_batch=7 --val_model_epoch=312 --tes_data_list=dataset/sampled_path/display/tes/20230104021508_tes_continuous_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/03211117 # 
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_encoder_FINETUNE_ --N_batch=7 --val_model_epoch=416 --tes_data_list=dataset/sampled_path/display/tes/20230104021508_tes_continuous_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/03211117 # 
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_onlydec_FINETUNE_ --N_batch=7 --val_model_epoch=408 --tes_data_list=dataset/sampled_path/display/tes/20230104021508_tes_continuous_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/03211117 # 
# # # Displayの評価（無作為）

# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_autoreg_prg --N_batch=7 --val_model_epoch=512 --tes_data_list=dataset/sampled_path/display/tes/20230104015836_tes_randn_16_10_2_all.pickle      --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/03211117
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_autoreg_prg --N_batch=7 --val_model_epoch=424 --tes_data_list=dataset/sampled_path/display/tes/20230104015836_tes_randn_16_10_2_all.pickle      --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/03211117
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_encoder_prg --N_batch=7 --val_model_epoch=504 --tes_data_list=dataset/sampled_path/display/tes/20230104015836_tes_randn_16_10_2_all.pickle      --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/03211117
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_onlydec_prg --N_batch=7 --val_model_epoch=480 --tes_data_list=dataset/sampled_path/display/tes/20230104015836_tes_randn_16_10_2_all.pickle      --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/03211117

# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=cntprg_encoder_tau01_only_once --N_batch=7 --val_model_epoch=496   --tes_data_list=/home/yyoshitake/works/DeepSDF/project/dataset/sampled_path/chair/tes/20221214021558_tes_unknown_continuous_8_10_2_all.pickle --test_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_1/results --canonical_data_path=/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/paper_exp/chair/canonical --pt_path=/home/yyoshitake/works/make_depth_image/project/point_clouds/03001627
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=rndsimPG_encoder_onlydec105itr02Wpe --N_batch=7 --val_model_epoch=448 --tes_data_list=dataset/sampled_path/chair/tes/20221210160130_tes_unknown_randn_8_10_1_all.pickle --test_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_1/results --canonical_data_path=/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/paper_exp/chair/canonical --pt_path=/home/yyoshitake/works/make_depth_image/project/point_clouds/03001627
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=cntprg_encoder_tau01_only_once --N_batch=7 --val_model_epoch=3400  --tes_data_list=/home/yyoshitake/works/DeepSDF/project/dataset/sampled_path/chair/tes/20221214021558_tes_unknown_continuous_8_10_2_all.pickle --test_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_1/results --canonical_data_path=/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/paper_exp/chair/canonical --pt_path=/home/yyoshitake/works/make_depth_image/project/point_clouds/03001627

# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_onlydec --N_batch=7 --val_model_epoch=424 --tes_data_list=dataset/sampled_path/display/tes/20230104015836_tes_randn_16_10_2_all.pickle      --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/03211117
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_autoreg --N_batch=7 --val_model_epoch=432 --tes_data_list=dataset/sampled_path/display/tes/20230104021508_tes_continuous_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/03211117
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_encoder --N_batch=7 --val_model_epoch=416 --tes_data_list=dataset/sampled_path/display/tes/20230104021508_tes_continuous_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/03211117
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_onlydec --N_batch=7 --val_model_epoch=504 --tes_data_list=dataset/sampled_path/display/tes/20230104021508_tes_continuous_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/03211117
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/display/view5/seq.txt --exp_version=seq_onlydec --N_batch=7 --val_model_epoch=432 --tes_data_list=dataset/sampled_path/display/tes/20230104021508_tes_continuous_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/03211117

# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_autoreg --N_batch=7 --val_model_epoch=288 --tes_data_list=dataset/sampled_path/display/tes/20230104015836_tes_randn_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/03211117
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_autoreg --N_batch=7 --val_model_epoch=432 --tes_data_list=dataset/sampled_path/display/tes/20230104015836_tes_randn_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/03211117
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_autoreg --N_batch=7 --val_model_epoch=504 --tes_data_list=dataset/sampled_path/display/tes/20230104015836_tes_randn_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/03211117
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/display/view5/rdn.txt --exp_version=rdn_encoder --N_batch=7 --val_model_epoch=472 --tes_data_list=dataset/sampled_path/display/tes/20230104015836_tes_randn_16_10_2_all.pickle --test_data_dir=/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/squashfs-root --canonical_data_path=/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical/tmp_map/03211117
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=cntprg_autoreg_tau01padWope_FINETUNE_p2t10   --N_batch=7 --val_model_epoch=312 --tes_data_list=/home/yyoshitake/works/DeepSDF/project/dataset/sampled_path/chair/tes/20221214021558_tes_unknown_continuous_8_10_2_all.pickle --test_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_1/results --canonical_data_path=/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/paper_exp/chair/canonical  --pt_path=/home/yyoshitake/works/make_depth_image/project/point_clouds/03001627
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=cntprg_encoder_tau01Wope_FINETUNE_p2t10      --N_batch=7 --val_model_epoch=304 --tes_data_list=/home/yyoshitake/works/DeepSDF/project/dataset/sampled_path/chair/tes/20221214021558_tes_unknown_continuous_8_10_2_all.pickle --test_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_1/results --canonical_data_path=/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/paper_exp/chair/canonical  --pt_path=/home/yyoshitake/works/make_depth_image/project/point_clouds/03001627

# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=rndsimPG_encoder_tau01Wope               --N_batch=7 --val_model_epoch=504 --tes_data_list=dataset/sampled_path/chair/tes/20221210160130_tes_unknown_randn_8_10_1_all.pickle --test_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_1/results                                             --canonical_data_path=/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/paper_exp/chair/canonical  --pt_path=/home/yyoshitake/works/make_depth_image/project/point_clouds/03001627
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=cntprg_encoder_onlydec105Wope_FINETUNE_p2t10 --N_batch=7 --val_model_epoch=344 --tes_data_list=/home/yyoshitake/works/DeepSDF/project/dataset/sampled_path/chair/tes/20221214021558_tes_unknown_continuous_8_10_2_all.pickle --test_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_1/results --canonical_data_path=/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/paper_exp/chair/canonical  --pt_path=/home/yyoshitake/works/make_depth_image/project/point_clouds/03001627
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=bdcntprg_onlymlp_tau00_FINETUNE_p2t10 --N_batch=7 --tes_data_list=dataset/sampled_path/chair/tes/20221214022353_tes_unknown_continuous_8_0d7_1_all.pickle --test_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_1/results --num_workers=7 --val_model_epoch=400
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=onlymlp_enc1dec0_woFix_wpe            --N_batch=7 --val_model_epoch=368

# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt   --exp_version=rndsim_encoder_onlydec105itr02     --N_batch=7 --val_model_epoch=400 --tes_data_list=dataset/sampled_path/chair/tes/20221210160130_tes_unknown_randn_8_10_1_all.pickle --test_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_1/results                                             --canonical_data_path=/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/paper_exp/chair/canonical  --pt_path=/home/yyoshitake/works/make_depth_image/project/point_clouds/03001627
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt   --exp_version=rndsim_encoder_onlydec105itr02Wope --N_batch=7 --val_model_epoch=480 --tes_data_list=dataset/sampled_path/chair/tes/20221210160130_tes_unknown_randn_8_10_1_all.pickle --test_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_1/results                                             --canonical_data_path=/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/paper_exp/chair/canonical  --pt_path=/home/yyoshitake/works/make_depth_image/project/point_clouds/03001627

# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=rndsimPG_autoreg_tau01imgWpe             --N_batch=7 --val_model_epoch=464 --tes_data_list=dataset/sampled_path/chair/tes/20221210160130_tes_unknown_randn_8_10_1_all.pickle --test_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_1/results --canonical_data_path=/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/paper_exp/chair/canonical --pt_path=/home/yyoshitake/works/make_depth_image/project/point_clouds/03001627
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=rndsimPG_autoreg_tau01imgWpe             --N_batch=7 --val_model_epoch=512 --tes_data_list=dataset/sampled_path/chair/tes/20221210160130_tes_unknown_randn_8_10_1_all.pickle --test_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_1/results --canonical_data_path=/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/paper_exp/chair/canonical --pt_path=/home/yyoshitake/works/make_depth_image/project/point_clouds/03001627
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=rndsimPG_autoreg_tau01imgWope            --N_batch=7 --val_model_epoch=472 --tes_data_list=dataset/sampled_path/chair/tes/20221210160130_tes_unknown_randn_8_10_1_all.pickle --test_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_1/results --canonical_data_path=/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/paper_exp/chair/canonical --pt_path=/home/yyoshitake/works/make_depth_image/project/point_clouds/03001627
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=rndsimPG_encoder_tau01Wpe                --N_batch=7 --val_model_epoch=480 --tes_data_list=dataset/sampled_path/chair/tes/20221210160130_tes_unknown_randn_8_10_1_all.pickle --test_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_1/results                                             --canonical_data_path=/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/paper_exp/chair/canonical  --pt_path=/home/yyoshitake/works/make_depth_image/project/point_clouds/03001627
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=rndsimPG_encoder_tau01Wope               --N_batch=7 --val_model_epoch=496 --tes_data_list=dataset/sampled_path/chair/tes/20221210160130_tes_unknown_randn_8_10_1_all.pickle --test_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_1/results                                             --canonical_data_path=/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/paper_exp/chair/canonical  --pt_path=/home/yyoshitake/works/make_depth_image/project/point_clouds/03001627
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=cntprg_encoder_onlydec105_FINETUNE_p2t10 --N_batch=7 --val_model_epoch=376 --tes_data_list=/home/yyoshitake/works/DeepSDF/project/dataset/sampled_path/chair/tes/20221214021558_tes_unknown_continuous_8_10_2_all.pickle --test_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_1/results --canonical_data_path=/home/yyoshitake/works/DeepSDF/project/dataset/dugon/moving_camera/paper_exp/chair/canonical  --pt_path=/home/yyoshitake/works/make_depth_image/project/point_clouds/03001627

# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=rndsim_onlymlp_tau00                  --N_batch=7 --tes_data_list=dataset/sampled_path/chair/tes/20221210160130_tes_unknown_randn_8_10_1_all.pickle       --test_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_1/results --num_workers=7 --val_model_epoch=472
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/txt.txt --exp_version=bdrndsim_onlymlp_tau00                --N_batch=7 --tes_data_list=dataset/sampled_path/chair/tes/20221214021233_tes_unknown_randn_8_0d7_2_all.pickle      --test_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_1/results --num_workers=7 --val_model_epoch=472
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=cntprg_onlymlp_tau00_FINETUNE_p2t10   --N_batch=7 --tes_data_list=dataset/sampled_path/chair/tes/20221214021558_tes_unknown_continuous_8_10_2_all.pickle  --test_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_1/results --num_workers=7 --val_model_epoch=352
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=bdcntprg_onlymlp_tau00_FINETUNE_p2t10 --N_batch=7 --tes_data_list=dataset/sampled_path/chair/tes/20221214022353_tes_unknown_continuous_8_0d7_1_all.pickle --test_data_dir=/d/workspace/yyoshitake/moving_camera/volumetric/tmp_1/results --num_workers=7 --val_model_epoch=400
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=onlymlp_enc1dec0_woFix_wpe            --N_batch=7 --val_model_epoch=368

