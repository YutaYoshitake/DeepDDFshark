export LC_ALL=C

# if [ -e /disks ]; then
#     SINGULARITY_OPTIONS="
#     --bind /disks/local/
#     -B /d/workspace/yyoshitake/ShapeNet/ddf/cabinet/train_data.sfs:/d/workspace/yyoshitake/ShapeNet/ddf/cabinet/train_data:image-src=/
#     -B /disks/local/yyoshitake/ddf/chair/train_data.sfs:/disks/local/yyoshitake/ddf/chair/train_data:image-src=/
#     "
# else
# SINGULARITY_OPTIONS="
# --bind /d/workspace/yyoshitake
# -B /home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/cabinet/result_sfs.sfs:/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/cabinet/result_sfs:image-src=/
# -B /home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/result_sfs.sfs:/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/result_sfs:image-src=/
# " # /disks:/disks 
# fi
SINGULARITY_OPTIONS = "--bind /d/ 
-B /home/yyoshitake/works/DeepSDF/disks/ddf/cabinet/train_data.sfs:/home/yyoshitake/works/DeepSDF/disks/ddf/cabinet/train_data:image-src=/
-B /home/yyoshitake/works/DeepSDF/disks/ddf/display/result_sfs/results_ori.sfs:/home/yyoshitake/works/DeepSDF/disks/ddf/display/result_sfs/results_ori:image-src=/
"

echo "Container was created $(date)"

# 初期設定ではPS1はbashrcの設定が強制的にSingularity>に上書きされる
# 履歴は./singularity_bash_historyに
# --nv : GPUの有効化
singularity exec \
${SINGULARITY_OPTIONS} \
--env PS1='${debian_chroot:+($debian_chroot)}\[\033[01;35m\]\u@\h\[\033[00m\]-singularity:\[\033[01;34m\]\w\[\033[00m\]\$ ' \
--env HISTFILE="$(cd $(dirname $0); pwd)/singularity_bash_history" \
--nv \
$(cd $(dirname $0); pwd)/deepddf1222.sif bash
# $(cd $(dirname $0); pwd)/deepsdf_1204.sif bash
# $(cd $(dirname $0); pwd)/hoge.sif bash