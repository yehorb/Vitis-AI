ARG CUDA_BASE_IMAGE=nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04
FROM $CUDA_BASE_IMAGE as cuda_base

SHELL ["/bin/bash", "-c"]

ENV TZ=America/Denver
ENV VAI_ROOT=/opt/vitis_ai
ENV VAI_HOME=/vitis_ai_home
ENV DEBIAN_FRONTEND=noninteractive

# Comment out nvidia repositories to prevent them from getting apt-get updated, see https://github.com/pytorch/pytorch/issues/74968
RUN sed -i 's/.*nvidia.*/# &/' $(find /etc/apt/ -type f -name "*.list")

RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update -y && \
        apt-get install -y --no-install-recommends \
            apt-transport-https \
            autoconf \
            automake \
            bc \
            build-essential \
            bzip2 \
            ca-certificates \
            curl \
            g++ \
            gdb \
            git \
            gnupg \
            locales \
            libboost-all-dev \
            libgflags-dev \
            libgoogle-glog-dev \
            libgtest-dev \
            libjsoncpp-dev \
            libssl-dev \
            libtool \
            libunwind-dev \
            make \
            cmake \
            openssh-client \
            openssl \
            python3 \
            python3-dev \
            python3-minimal \
            python3-numpy \
            python3-opencv \
            python3-pip \
            python3-setuptools \
            python3-venv \
            software-properties-common \
            sudo \
            tree \
            tzdata \
            unzip \
            vim \
            wget \
            yasm \
            zstd \
            libavcodec-dev \
            libavformat-dev \
            libeigen3-dev \
            libgstreamer-plugins-base1.0-dev \
            libgstreamer1.0-dev \
            libgtest-dev \
            libgtk-3-dev \
            libgtk2.0-dev \
            libhdf5-dev \
            libjpeg-dev \
            libopenexr-dev \
            libpng-dev \
            libswscale-dev \
            libtiff-dev \
            libwebp-dev \
            opencl-clhpp-headers \
            opencl-headers \
            pocl-opencl-icd \
            python3-opencv \
            python3-pip \
            python3-setuptools \
            python3-venv \
            software-properties-common \
            sudo \
            tree \
            unzip \
            vim \
            wget \
            yasm \
            zstd \
            libavcodec-dev \
            libavformat-dev \
            libeigen3-dev \
            libgstreamer-plugins-base1.0-dev \
            libgstreamer1.0-dev \
            libgtest-dev \
            libgtk-3-dev \
            libgtk2.0-dev \
            libhdf5-dev \
            libjpeg-dev \
            libopenexr-dev \
            libpng-dev \
            libswscale-dev \
            libtiff-dev \
            libwebp-dev \
            opencl-clhpp-headers \
            opencl-headers \
            ffmpeg \
            pocl-opencl-icd \
            pybind11-dev \
            python3-pybind11 \
            libopencv-dev \
            gcc-9 \
            gcc-10 \
            g++-9 \
            g++-10 \
            libprotobuf-c-dev \
            protobuf-compiler \
            python3-protobuf \
            swig \
            python3-flask \
            python3-setuptools \
            python3-wheel

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 10 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-9 \
        --slave /usr/bin/gcov gcov /usr/bin/gcov-9 && \
        update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 90 \
            --slave /usr/bin/g++ g++ /usr/bin/g++-10 \
            --slave /usr/bin/gcov gcov /usr/bin/gcov-10

RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
        echo "LC_ALL=en_US.UTF-8" >>/etc/environment && \
        echo "LANG=en_US.UTF-8" >/etc/locale.conf && \
        locale-gen en_US.UTF-8 && \
        localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8 && \
        dpkg-reconfigure locales && \
        ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
        echo $TZ >/etc/timezone && \
        dpkg-reconfigure -f noninteractive tzdata

RUN export JSON_C_VERSION="json-c-0.18-20240915" && \
        cd /tmp && wget --progress=dot:mega https://github.com/json-c/json-c/archive/${JSON_C_VERSION}.tar.gz && \
        tar xvf ${JSON_C_VERSION}.tar.gz && \
        cd json-c-${JSON_C_VERSION} && \
        mkdir build && \
        cd build && \
        cmake -DBUILD_SHARED_LIBS=ON .. && \
        make -j && \
        make install

RUN export GOSU_VERSION="1.19" && \
        curl -sSkLo /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-$(dpkg --print-architecture)" && \
        chmod +x /usr/local/bin/gosu

RUN groupadd vitis-ai-group && \
        useradd --shell /bin/bash -c '' -m -g vitis-ai-group vitis-ai-user && \
        passwd -d vitis-ai-user && \
        usermod -aG sudo vitis-ai-user && \
        echo 'ALL ALL=(ALL) NOPASSWD:ALL' >>/etc/sudoers && \
        echo 'Defaults secure_path="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/vitis_ai/conda/bin"' >>/etc/sudoers

RUN chmod 1777 /tmp && \
        mkdir /scratch && chmod 1777 /scratch && \
        mkdir -p ${VAI_ROOT} && \
        chown -R vitis-ai-user:vitis-ai-group ${VAI_ROOT}

FROM cuda_base as conda_base

USER vitis-ai-user

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    export HOME=~vitis-ai-user && \
        sudo chmod -R 777 /root /root/.local /root/.local/bin || true && \
        export MINIFORGE_VERSION="25.3.1-0" && \
        cd /tmp && \
        wget --progress=dot:mega https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh && \
        /bin/bash ./miniconda.sh -b -p $VAI_ROOT/conda && \
        . $VAI_ROOT/conda/etc/profile.d/conda.sh && \
        rm -fr /tmp/miniconda.sh && \
        /$VAI_ROOT/conda/bin/conda clean -y --force-pkgs-dirs && \
        echo ". $VAI_ROOT/conda/etc/profile.d/conda.sh" >>~vitis-ai-user/.bashrc && \
        sudo ln -s $VAI_ROOT/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
