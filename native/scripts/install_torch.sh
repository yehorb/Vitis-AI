#!/bin/bash

set -ex

CONDA_PREFIX="${CONDA_PREFIX:-/home/xilinx}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/tools/Xilinx/Downloads}"
VAI_CONDA_CHANNEL="${VAI_CONDA_CHANNEL:-https://www.xilinx.com/bin/public/openDownload?filename=conda-channel-3.5.0.tar.gz}"
VAI_CONDA_CHANNEL_NAME="${VAI_CONDA_CHANNEL_NAME:-vitis-ai-conda-channel-3.5.0}"
VAI_ROOT="${VAI_ROOT:-/tools/Xilinx/Vitis-AI}"

channel_tar_gz="${VAI_CONDA_CHANNEL_NAME}.tar.gz"
channel_file="${DOWNLOAD_DIR}/${channel_tar_gz}"
channel_dir="${CONDA_PREFIX}/channel/${VAI_CONDA_CHANNEL_NAME}"

download_vai_conda_channel() {
    wget --progress=dot:mega \
        -O "${channel_file}" \
        "${VAI_CONDA_CHANNEL}"
}

unpack_vai_conda_channel() {
    mkdir -p "${channel_dir}"
    cd "${channel_dir}"
    tar -xzvf "${channel_file}"
}

install_vai_conda_channel() {
    if [[ ! -f ${channel_file} ]]; then
        download_vai_conda_channel
        # Read-only
        chmod a-w "${channel_file}"
    fi
    if [[ ! -d ${channel_dir} ]]; then
        unpack_vai_conda_channel
        chmod -R a-w "${channel_dir}"
    fi
}

create_pytorch_env() {
    set +x
    source "${CONDA_PREFIX}/conda/etc/profile.d/conda.sh"
    source "${CONDA_PREFIX}/conda/etc/profile.d/mamba.sh"
    set -x

    mamba create \
        --channel "${VAI_CONDA_CHANNEL}" \
        --file "${VAI_ROOT}/native/gpu_conda/vitis-ai-pytorch.yml" \
        --verbose \
        --yes

    conda activate vitis-ai-pytorch

    pip install \
        --force-reinstall \
        scipy\<=1.9.3 \
        numpy\<=1.24.2 \
        protobuf==3.20.3 \
        tensorboard \
        graphviz==0.19.1 \
        imageio \
        scikit-image \
        natsort \
        nibabel \
        easydict \
        yacs \
        fire \
        numba \
        loguru \
        ninja

    # pytorch==1.13.1 is pinned by vitis-ai
    # From pytorch previous version installation instructions
    # https://pytorch.org/get-started/previous-versions/
    # CUDA 11.7 is the latest supported version for this release
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia --yes

    # Ensure compatible mkl version
    # https://github.com/pytorch/pytorch/issues/123097
    conda install mkl==2024.0.0 --yes

    mkdir -p "${VAI_ROOT}/compiler"
    cp -r "${CONDA_PREFIX}/lib/python3.8/site-packages/vaic/arch" "${VAI_ROOT}/compiler/arch"
}

install_torch() {
    if [[ ${VAI_CONDA_CHANNEL} =~ .*"tar.gz" ]]; then
        install_vai_conda_channel
    fi
    export VAI_CONDA_CHANNEL="file://${channel_dir}/conda-channel"
    create_pytorch_env
}

"$@"
