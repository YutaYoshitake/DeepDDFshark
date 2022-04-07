# FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04
# FROM nvidia/cudagl:11.3.0-devel-ubuntu18.04

ENV USER docker-user
ENV UID 1000
ENV GID 1000
ENV HOME /home/${USER}
ENV SHELL /bin/bash

# docker-userというユーザーを作る．
RUN useradd -u ${UID} -m ${USER}
# docker-userというグループ名を付ける．
RUN groupmod -g ${GID} ${USER}
# デフォルトシェルを変更する．
RUN chsh -s ${SHELL} ${USER}

# aptのインタラクティブな操作を利用しない．
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PYENV_ROOT="/opt/pyenv"
ENV PATH="$PYENV_ROOT/bin:${PATH}"
ENV PATH="$PYENV_ROOT/shims:${PATH}"
# Dependencies
# rm -rfしているのは，Dockerイメージの軽量化のため
# インタラクティブな操作がないDockerfileでは，apt-getが推奨．
RUN apt-get update && \
    apt-get install -y \
	vim curl x11-apps git g++ wget make unzip ssh sshfs && \
    rm -rf /var/lib/apt/lists/*

# ネットワーク関係のツール，sudo，sshサーバー
RUN apt-get update && \
    apt-get install -y \
	iputils-ping net-tools kmod sudo && \
    rm -rf /var/lib/apt/lists/*

# docker-userにパスワードなしsudo権限を与える．
RUN usermod -aG sudo,root ${USER} && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && echo 'root:password' | chpasswd
# SSHFSのマウントポイントを作って権限を与える．
RUN mkdir -p /app && chown docker-user:docker-user /app
# root以外もSSHFSを利用可能にする．
RUN echo user_allow_other >> /etc/fuse.conf



RUN apt-get update && \
    apt-get install -y \
	iputils-ping net-tools kmod sudo && \
    rm -rf /var/lib/apt/lists/*

# pyenv
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev && \
    rm -rf /var/lib/apt/lists/*
# OpenCV
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    yasm pkg-config libswscale-dev libtbb2 libtbb-dev\
    libjpeg-dev libpng-dev libtiff-dev libavformat-dev libpq-dev && \
    rm -rf /var/lib/apt/lists/*
# OpenPose
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libprotobuf-dev protobuf-compiler libopencv-dev \
    libgoogle-glog-dev libboost-all-dev libhdf5-dev libatlas-base-dev \
    libleveldb-dev libsnappy-dev libgflags-dev liblmdb-dev pciutils opencl-headers ocl-icd-opencl-dev \
    libviennacl-dev libcanberra-gtk-module && \
    rm -rf /var/lib/apt/lists/*
# Numpy & Scipy
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libblas-dev liblapack-dev libatlas-base-dev && \
    rm -rf /var/lib/apt/lists/*
# OpenGL
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
	libglu1-mesa-dev mesa-common-dev freeglut3-dev \
	libglfw3 libglfw3-dev xvfb mesa-utils && \
    rm -rf /var/lib/apt/lists/*
# OpenDR
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
	libglu1-mesa-dev freeglut3-dev mesa-common-dev libosmesa6-dev ffmpeg && \
    rm -rf /var/lib/apt/lists/*



# ここから好きなコマンドを打っていく．
# HERE

# RUN apt-get update && \
#     DEBIAN_FRONTEND=noninteractive apt-get install -y \
# 	libglew-dev cmake doxygen && \
#     rm -rf /var/lib/apt/lists/*

# WORKDIR /opt/CLI11
# RUN git clone https://github.com/CLIUtils/CLI11.git . && git checkout v1.9.1 && git submodule update --init
# # RUN git clone https://github.com/CLIUtils/CLI11.git . && git checkout 4af78beef777e313814b4daff70e2da9171a385a
# RUN mkdir build && cd build && cmake .. && cmake --build . && make && make test && make install

# WORKDIR /opt/Pangolin
# RUN git clone https://github.com/stevenlovegrove/Pangolin.git . && git checkout v0.6
# RUN mkdir build && cd build && cmake .. && cmake --build .
# # WORKDIRにもどる
# RUN cd build && make -j && make install 

# WORKDIR /opt/nanoflann
# RUN git clone https://github.com/jlblancoc/nanoflann.git . && git checkout v1.3.2 && git submodule update --init
# RUN apt-get update && \
#     DEBIAN_FRONTEND=noninteractive apt-get install -y \
#     libgtest-dev libeigen3-dev && \
#     rm -rf /var/lib/apt/lists/*
# RUN mkdir build && cd build && cmake .. && make && make test && make install
# RUN cd /usr/local/include && mkdir nanoflann && mv nanoflann.hpp nanoflann



# pyenv
RUN apt-get update && \
    apt-get install -y \
    build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev && \
    # インストール成功後にaptのキャッシュを消して容量を削減．
    rm -rf /var/lib/apt/lists/*
# Numpy & Scipy
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libblas-dev liblapack-dev libatlas-base-dev && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHON_VERSION 3.7.10
# pyenv自体のバージョンは，今回は固定しない．
RUN git clone https://github.com/pyenv/pyenv.git /opt/pyenv && pyenv install ${PYTHON_VERSION} && pyenv global ${PYTHON_VERSION}

# --no-cache-dirは，pipのキャッシュを無効化して容量を削減するオプション．
RUN pip install --upgrade pip
RUN pip install numpy==1.20.2
RUN pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install plyfile==0.7.4
RUN pip install scikit-image==0.18.1
RUN pip install trimesh==3.9.16
RUN pip install matplotlib==3.4.2
RUN pip install open3d==0.12.0
RUN pip install ConfigArgParse
RUN pip install opencv-python
# URL : https://github.com/otaheri/chamfer_distance
# RUN pip install chamferdist==1.0.0
# RUN pip install git+https://github.com/facebookresearch/pytorch3d.git@fe39cc7b806afeabe64593e154bfee7b4153f76f
# CMD ["/usr/bin/xeyes"]

RUN sudo apt-get update && sudo apt-get install -y nodejs npm
RUN npm install n -g
RUN sudo n stable

# neovim settings
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:neovim-ppa/stable
RUN apt-get update && apt-get install -y neovim

# この下の数行は最後に書く．
USER ${USER}
EXPOSE 8888
COPY entrypoint.sh /home/docker-user/entrypoint.sh
COPY setting.sh /home/docker-user/setting.sh

WORKDIR /home/docker-user
RUN git clone --depth 2 https://github.com/YutaYoshitake/nvim
RUN mkdir /home/docker-user/.config
RUN mv nvim /home/docker-user/.config/nvim
RUN sh -c 'curl -fLo "${XDG_DATA_HOME:-$HOME/.local/share}"/nvim/site/autoload/plug.vim --create-dirs \
       https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim'

