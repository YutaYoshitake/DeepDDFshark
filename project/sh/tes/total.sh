# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr01_estdif_cls              --N_batch=8 --val_model_epoch=360
# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=randn_sequential_onlymlp_enc1dec0_woFix            --N_batch=8 --val_model_epoch=368
# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr02_obsdif        --N_batch=8 --val_model_epoch=400 --total_itr=10 --until_convergence=yes
# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_simultaneous_encoder_enc3dec0_Fix            --N_batch=8 --val_model_epoch=336 --total_itr=10 --until_convergence=yes --convergence_thr=30 --convergence_thr_shape=50
# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_sequential_autoreg_enc3dec3_Fix_itr02_obsdif --N_batch=8 --val_model_epoch=392 --total_itr=10 --until_convergence=yes --convergence_thr=30 --convergence_thr_shape=50

# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix                     --N_batch=8 --val_model_epoch=400 --total_itr=10 --until_convergence=yes
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=randn_sequential_autoreg_enc3dec3_Fix_itr01_estdif --N_batch=8 --val_model_epoch=376
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=encoder_enc3dec0_Fix_wpe                                --N_batch=8 --val_model_epoch=368
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_sequential_autoreg_enc3dec3_Fix_itr01_estdif --N_batch=8 --val_model_epoch=392 --total_itr=10 --until_convergence=yes --convergence_thr=30 --convergence_thr_shape=50
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_simultaneous_autoreg_enc3dec3_Fix            --N_batch=8 --val_model_epoch=336 --total_itr=10 --until_convergence=yes --convergence_thr=50 --convergence_thr_shape=50

# CUDA_VISIBLE_DEVICES=8 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_sequential_autoreg_enc3dec3_Fix_itr01_estdif --N_batch=8 --val_model_epoch=392
# CUDA_VISIBLE_DEVICES=8 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=rdn_sim_encoder_enc3dec0_Fix_dstmap                     --N_batch=8 --val_model_epoch=392
# CUDA_VISIBLE_DEVICES=8 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr02_estdif        --N_batch=8 --val_model_epoch=360 --total_itr=10 --until_convergence=yes
# CUDA_VISIBLE_DEVICES=8 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=onlymlp_enc1dec0_woFix                   --N_batch=8 --val_model_epoch=360 --total_itr=10 --until_convergence=yes

# CUDA_VISIBLE_DEVICES=3 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=randn_sequential_encoder_enc3dec0_Fix              --N_batch=8 --val_model_epoch=384
# CUDA_VISIBLE_DEVICES=3 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=encoder_enc3dec0_Fix                     --N_batch=8 --val_model_epoch=400 --total_itr=10 --until_convergence=yes
# CUDA_VISIBLE_DEVICES=3 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr01_estdif        --N_batch=8 --val_model_epoch=368 --total_itr=10 --until_convergence=yes
# CUDA_VISIBLE_DEVICES=3 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_simultaneous_onlymlp_enc1dec0_woFix          --N_batch=8 --val_model_epoch=384 --total_itr=10 --until_convergence=yes --convergence_thr=30 --convergence_thr_shape=50

CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=encoder_onlydecv2_Fix --N_batch=8 --val_model_epoch=360
CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=encoder_onlydecv3_Fix --N_batch=8 --val_model_epoch=368


# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_sequential_autoreg_enc3dec3_Fix_itr02_obsdif --N_batch=8 --val_model_epoch=392 --total_itr=10 --until_convergence=yes
# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_simultaneous_autoreg_enc3dec3_Fix            --N_batch=8 --val_model_epoch=336 --total_itr=10 --until_convergence=yes
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_sequential_autoreg_enc3dec3_Fix_itr01_estdif --N_batch=8 --val_model_epoch=392 --total_itr=10 --until_convergence=yes
# CUDA_VISIBLE_DEVICES=3 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_simultaneous_encoder_enc3dec0_Fix            --N_batch=8 --val_model_epoch=336 --total_itr=10 --until_convergence=yes
# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix --N_batch=8 --val_model_epoch=400 --total_itr=10 --until_convergence=yes
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_simultaneous_onlymlp_enc1dec0_woFix          --N_batch=8 --val_model_epoch=384 --total_itr=10 --until_convergence=yes
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr02_estdif_latest --N_batch=8 --val_model_epoch=360
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr01_estdif        --N_batch=8 --val_model_epoch=368
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=randn_sequential_autoreg_enc3dec3_Fix_itr01_estdif --N_batch=8 --val_model_epoch=376
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_sequential_autoreg_enc3dec3_Fix_itr01_estdif --N_batch=8 --val_model_epoch=392
# CUDA_VISIBLE_DEVICES=7 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr01_estdif        --N_batch=8 --val_model_epoch=368 --total_itr=10 --until_convergence=yes
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr02_estdif        --N_batch=8 --val_model_epoch=360
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_simultaneous_encoder_enc3dec0_Fix            --N_batch=8 --val_model_epoch=336
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_simultaneous_autoreg_enc3dec3_Fix            --N_batch=8 --val_model_epoch=336
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr02_estdif        --N_batch=8 --val_model_epoch=360 --total_itr=10 --until_convergence=yes
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_simultaneous_autoreg_enc3dec3_Fix            --N_batch=8 --val_model_epoch=336 --total_itr=10 --until_convergence=yes --convergence_thr=50 --convergence_thr_shape=50
# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr02_obsdif        --N_batch=8 --val_model_epoch=400
# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=onlymlp_enc1dec0_woFix                   --N_batch=8 --val_model_epoch=360
# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_sequential_autoreg_enc3dec3_Fix_itr02_obsdif --N_batch=8 --val_model_epoch=392
# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_simultaneous_onlymlp_enc1dec0_woFix          --N_batch=8 --val_model_epoch=384
# CUDA_VISIBLE_DEVICES=5 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr02_obsdif        --N_batch=8 --val_model_epoch=400 --total_itr=10 --until_convergence=yes
# CUDA_VISIBLE_DEVICES=3 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=encoder_enc3dec0_Fix                     --N_batch=8 --val_model_epoch=400
# CUDA_VISIBLE_DEVICES=3 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=randn_sequential_onlymlp_enc1dec0_woFix            --N_batch=8 --val_model_epoch=368
# CUDA_VISIBLE_DEVICES=3 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=randn_sequential_encoder_enc3dec0_Fix              --N_batch=8 --val_model_epoch=384
# CUDA_VISIBLE_DEVICES=3 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=onlymlp_enc1dec0_woFix                   --N_batch=8 --val_model_epoch=360 --total_itr=10 --until_convergence=yes
# CUDA_VISIBLE_DEVICES=3 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=encoder_enc3dec0_Fix                     --N_batch=8 --val_model_epoch=400 --total_itr=10 --until_convergence=yes
# CUDA_VISIBLE_DEVICES=6 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix --N_batch=8 --val_model_epoch=400
# CUDA_VISIBLE_DEVICES=1 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_simultaneous_autoreg_enc3dec3_Fix            --N_batch=8 --val_model_epoch=336


# 無作為・同時・５回
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr02_estdif_latest --N_batch=8 --val_model_epoch=360
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=onlymlp_enc1dec0_woFix                   --N_batch=8 --val_model_epoch=360
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=encoder_enc3dec0_Fix                     --N_batch=8 --val_model_epoch=400
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix                     --N_batch=8 --val_model_epoch=400
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr02_obsdif        --N_batch=8 --val_model_epoch=400
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr01_estdif        --N_batch=8 --val_model_epoch=368
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr02_estdif        --N_batch=8 --val_model_epoch=360
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=onlymlp_enc1dec0_woFix                   --N_batch=8 --val_model_epoch=360 --total_itr=10 --until_convergence=yes
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=encoder_enc3dec0_Fix                     --N_batch=8 --val_model_epoch=400 --total_itr=10 --until_convergence=yes
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix                     --N_batch=8 --val_model_epoch=400 --total_itr=10 --until_convergence=yes
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr02_obsdif        --N_batch=8 --val_model_epoch=400 --total_itr=10 --until_convergence=yes
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr01_estdif        --N_batch=8 --val_model_epoch=368 --total_itr=10 --until_convergence=yes
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=autoreg_enc3dec3_Fix_itr02_estdif        --N_batch=8 --val_model_epoch=360 --total_itr=10 --until_convergence=yes

# 無作為・逐次・５回
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=randn_sequential_onlymlp_enc1dec0_woFix            --N_batch=8 --val_model_epoch=368
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=randn_sequential_encoder_enc3dec0_Fix              --N_batch=8 --val_model_epoch=384
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/tes.txt --exp_version=randn_sequential_autoreg_enc3dec3_Fix_itr01_estdif --N_batch=8 --val_model_epoch=376

# 連続的・逐次・５回
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_simultaneous_onlymlp_enc1dec0_woFix          --N_batch=8 --val_model_epoch=384
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_simultaneous_encoder_enc3dec0_Fix            --N_batch=8 --val_model_epoch=336
# CUDA_VISIBLE_DEVICES=3 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_simultaneous_encoder_enc3dec0_Fix            --N_batch=8 --val_model_epoch=384
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_simultaneous_autoreg_enc3dec3_Fix            --N_batch=8 --val_model_epoch=336
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_sequential_autoreg_enc3dec3_Fix_itr02_obsdif --N_batch=8 --val_model_epoch=392
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_sequential_autoreg_enc3dec3_Fix_itr01_estdif --N_batch=8 --val_model_epoch=392
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_simultaneous_onlymlp_enc1dec0_woFix          --N_batch=8 --val_model_epoch=384 --total_itr=10 --until_convergence=yes --convergence_thr=30 --convergence_thr_shape=50
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_simultaneous_encoder_enc3dec0_Fix            --N_batch=8 --val_model_epoch=336 --total_itr=10 --until_convergence=yes --convergence_thr=30 --convergence_thr_shape=50
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_simultaneous_autoreg_enc3dec3_Fix            --N_batch=8 --val_model_epoch=336 --total_itr=10 --until_convergence=yes --convergence_thr=50 --convergence_thr_shape=50
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_sequential_autoreg_enc3dec3_Fix_itr02_obsdif --N_batch=8 --val_model_epoch=392 --total_itr=10 --until_convergence=yes --convergence_thr=30 --convergence_thr_shape=50
# CUDA_VISIBLE_DEVICES=0 python train.py --code_mode=TES --config=configs/paper_exp/chair/view5/seq.txt --exp_version=continuous_sequential_autoreg_enc3dec3_Fix_itr01_estdif --N_batch=8 --val_model_epoch=392 --total_itr=10 --until_convergence=yes --convergence_thr=30 --convergence_thr_shape=50