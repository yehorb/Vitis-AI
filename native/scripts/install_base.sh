#!/bin/bash

set -ex

install_ubuntu() {
    if [[ ${DOCKER_TYPE} =~ .*"rocm"* && ${TARGET_FRAMEWORK} =~ .*"pytorch" ]]; then
        echo "using rocm pytorch imge"
        apt-get update -y &&
            apt-get install -y --no-install-recommends locales

    elif [[ ${DOCKER_TYPE} =~ .*"rocm"* ]]; then
        apt-get update -y &&
            apt-get install -y wget rccl lsb-release

    else
        chmod 1777 /tmp &&
            mkdir /scratch &&
            chmod 1777 /scratch &&
            apt-get update -y &&
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
                pocl-opencl-icd
    fi

    os_version=$(lsb_release -r -s)
    echo "base OS version:${os_version}"
    apt-get update -y && apt-get install -y --no-install-recommends \
        pybind11-dev python3-pybind11 libopencv-dev gcc-9 gcc-10 g++-9 g++-10 \
        libprotobuf-c-dev protobuf-compiler python3-protobuf swig

    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 10 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-9 \
        --slave /usr/bin/gcov gcov /usr/bin/gcov-9

    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 90 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-10 \
        --slave /usr/bin/gcov gcov /usr/bin/gcov-10

    apt-get install -y --no-install-recommends \
        python3-flask \
        python3-setuptools \
        python3-wheel

    export JSON_C_VERSION="json-c-0.18-20240915"
    cd /tmp && wget --progress=dot:mega https://github.com/json-c/json-c/archive/${JSON_C_VERSION}.tar.gz &&
        tar xvf ${JSON_C_VERSION}.tar.gz &&
        cd json-c-${JSON_C_VERSION} &&
        mkdir build &&
        cd build &&
        cmake -DBUILD_SHARED_LIBS=ON .. &&
        make -j &&
        make install
}

# Install base packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
DOCKER_TYPE=$1
TARGET_FRAMEWORK=$2

case "$ID" in
ubuntu)
    install_ubuntu
    ;;
*)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac
