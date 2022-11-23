export LC_ALL=C

SINGULARITY_OPTIONS="--bind /d/workspace/yyoshitake -B /d/workspace/yyoshitake/moving_camera/volumetric/tmp_1/results.sfs:/tes_sfs:image-src=/ -B /d/workspace/yyoshitake/moving_camera/volumetric/tmp_2/results.sfs:/trainval_sfs:image-src=/"

echo "Container was created $(date)"

# 初期設定ではPS1はbashrcの設定が強制的にSingularity>に上書きされる
# 履歴は./singularity_bash_historyに
# --nv : GPUの有効化
singularity exec \
${SINGULARITY_OPTIONS} \
--env PS1='${debian_chroot:+($debian_chroot)}\[\033[01;35m\]\u@\h\[\033[00m\]-singularity:\[\033[01;34m\]\w\[\033[00m\]\$ ' \
--env HISTFILE="$(cd $(dirname $0); pwd)/singularity_bash_history" \
--nv \
$(cd $(dirname $0); pwd)/deepsdf.sif bash

mv: replace '/tes_sfs/ue639c33f-d415-458c-8ff8-2ef68135af15/0000000001_0000000005_0000000001.pickle', overriding mode 0644 (rw-r--r--)? 
mv: replace '/tes_sfs/ue639c33f-d415-458c-8ff8-2ef68135af15/0000000002_0000000005_0000000002.pickle', overriding mode 0644 (rw-r--r--)? 
mv: replace '/tes_sfs/ue639c33f-d415-458c-8ff8-2ef68135af15/0000000003_0000000005_0000000003.pickle', overriding mode 0644 (rw-r--r--)? 
mv: replace '/tes_sfs/ue639c33f-d415-458c-8ff8-2ef68135af15/0000000004_0000000005_0000000004.pickle', overriding mode 0644 (rw-r--r--)? 
mv: replace '/tes_sfs/ue639c33f-d415-458c-8ff8-2ef68135af15/0000000005_0000000005_0000000005.pickle', overriding mode 0644 (rw-r--r--)? 
mv: replace '/tes_sfs/ue639c33f-d415-458c-8ff8-2ef68135af15/0000000006_0000000005_0000000006.pickle', overriding mode 0644 (rw-r--r--)? 
mv: replace '/tes_sfs/ue639c33f-d415-458c-8ff8-2ef68135af15/0000000007_0000000005_0000000007.pickle', overriding mode 0644 (rw-r--r--)? 
mv: replace '/tes_sfs/ue639c33f-d415-458c-8ff8-2ef68135af15/0000000008_0000000005_0000000008.pickle', overriding mode 0644 (rw-r--r--)? 
mv: replace '/tes_sfs/ue639c33f-d415-458c-8ff8-2ef68135af15/0000000009_0001000002_0000000001.pickle', overriding mode 0644 (rw-r--r--)? 
mv: replace '/tes_sfs/ue639c33f-d415-458c-8ff8-2ef68135af15/0000000010_0001000002_0000000002.pickle', overriding mode 0644 (rw-r--r--)? 
mv: replace '/tes_sfs/ue639c33f-d415-458c-8ff8-2ef68135af15/0000000011_0001000002_0000000003.pickle', overriding mode 0644 (rw-r--r--)? 
mv: replace '/tes_sfs/ue639c33f-d415-458c-8ff8-2ef68135af15/0000000012_0001000002_0000000004.pickle', overriding mode 0644 (rw-r--r--)? 
mv: replace '/tes_sfs/ue639c33f-d415-458c-8ff8-2ef68135af15/0000000013_0001000002_0000000005.pickle', overriding mode 0644 (rw-r--r--)?