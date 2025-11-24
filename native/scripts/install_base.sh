#!/bin/bash

set -ex

install_rocm_pytorch() {
    apt-get update -y
    apt-get install -y --no-install-recommends locales

}

install_rocm() {
    apt-get update -y
    apt-get install -y --no-install-recommends wget rccl lsb-release
}

install_base() {
    apt-get update -y
    apt-get install -y --no-install-recommends \
        apt-transport-https \
        autoconf \
        automake \
        bc \
        build-essential \
        bzip2 \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        g++ \
        gdb \
        git \
        gnupg \
        libavcodec-dev \
        libavformat-dev \
        libboost-all-dev \
        libeigen3-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer1.0-dev \
        libgtest-dev \
        libgtk-3-dev \
        libgtk2.0-dev \
        libhdf5-dev \
        libjpeg-dev \
        libjson-c-dev \
        libjsoncpp-dev \
        libopenexr-dev \
        libpng-dev \
        libssl-dev \
        libswscale-dev \
        libtiff-dev \
        libtool \
        libunwind-dev \
        libwebp-dev \
        locales \
        make \
        opencl-clhpp-headers \
        opencl-headers \
        openssh-client \
        openssl \
        pocl-opencl-icd \
        software-properties-common \
        sudo \
        tree \
        tzdata \
        unzip \
        vim \
        wget \
        yasm \
        zstd
}

install_compilers() {
    apt-get update -y
    apt-get install -y --no-install-recommends \
        pybind11-dev \
        libopencv-dev \
        gcc-9 \
        gcc-10 \
        g++-9 \
        g++-10 \
        libprotobuf-c-dev \
        protobuf-compiler \
        swig
}

update_alternatives_compilers() {
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 10 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-9 \
        --slave /usr/bin/gcov gcov /usr/bin/gcov-9

    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 90 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-10 \
        --slave /usr/bin/gcov gcov /usr/bin/gcov-10
}

install_ubuntu() {
    if [[ ${DOCKER_TYPE} =~ .*"rocm"* && ${TARGET_FRAMEWORK} =~ .*"pytorch" ]]; then
        install_rocm_pytorch
    elif [[ ${DOCKER_TYPE} =~ .*"rocm"* ]]; then
        install_rocm
    else
        install_base
    fi

    install_compilers

    update_alternatives_compilers
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
