export LC_ALL=C
SINGULARITY_OPTIONS="--bind /disks/local/yyoshitake/ -B /home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/cabinet/result_sfs.sfs:/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/cabinet/results:image-src=/"
# SINGULARITY_OPTIONS="-B /home/yyoshitake/works/DeepSDF/disks/dfnet/canonical.sfs:/home/yyoshitake/works/DeepSDF/disks/dfnet/canonical:image-src=/ 
#                      -B /home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/cabinet/result_sfs.sfs:/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/cabinet/results:image-src=/ 
#                      -B /home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/result_sfs.sfs:/home/yyoshitake/works/DeepSDF/disks/dfnet/volumetric/display/results:image-src=/ 
#                      --bind /disks/local/yyoshitake/ 
#                      --bind /d/workspace/yyoshitake"
# SINGULARITY_OPTIONS="--bind /d/workspace/yyoshitake --bind /disks/local/yyoshitake/moving_camera/volumetric/revised --bind /disks/local/yyoshitake/moving_camera/volumetric/tmp_2/visibility -B /disks/local/yyoshitake/moving_camera/volumetric/tmp_2/results.sfs:/disks/local/yyoshitake/moving_camera/volumetric/tmp_2/results:image-src=/ -B /disks/local/yyoshitake/moving_camera/volumetric/tmp_1/results.sfs:/disks/local/yyoshitake/moving_camera/volumetric/tmp_1/results:image-src=/ -B /disks/local/yyoshitake/ddf/chair/train_data.sfs:/disks/local/yyoshitake/ddf/chair/train_data:image-src=/"
echo "Container was created $(date)"
履歴は./singularity_bash_historyに
# --nv : GPUの有効化
singularity exec \
${SINGULARITY_OPTIONS} \
--env PS1='${debian_chroot:+($debian_chroot)}\[\033[01;35m\]\u@\h\[\033[00m\]-singularity:\[\033[01;34m\]\w\[\033[00m\]\$ ' \
--env HISTFILE="$(cd $(dirname $0); pwd)/singularity_bash_history" \
--nv \
$(cd $(dirname $0); pwd)/deepsdf_1204.sif bash