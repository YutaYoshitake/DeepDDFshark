export LC_ALL=C

SINGULARITY_OPTIONS="--bind /d/workspace/yyoshitake" # /disks:/disks 

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
