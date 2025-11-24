#!/bin/bash

set -ex

CONDA_PREFIX="${CONDA_PREFIX:-/home/xilinx}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/tools/Xilinx/Downloads}"
TORCH_CUDA_VERSION="${TORCH_CUDA_VERSION:-cu130}"
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

    mamba env create -v -f "${VAI_ROOT}/native/gpu_conda/vitis-ai-pytorch.yml"
    conda activate vitis-ai-pytorch
    pip install --force-reinstall scipy numpy==1.22 protobuf==3.20.3 tensorboard graphviz==0.19.1 imageio scikit-image natsort nibabel easydict yacs fire numba loguru
    mkdir -p $VAI_ROOT/compiler
    conda activate vitis-ai-pytorch
    torchvision_cmd="pip install torch==1.13.1+${TORCH_CUDA_VERSION} torchvision==0.14.1+${TORCH_CUDA_VERSION} --index-url https://download.pytorch.org/whl/${TORCH_CUDA_VERSION}"
    $torchvision_cmd
    cp -r $CONDA_PREFIX/lib/python3.8/site-packages/vaic/arch $VAI_ROOT/compiler/arch
}

install_torch() {
    if [[ ${VAI_CONDA_CHANNEL} =~ .*"tar.gz" ]]; then
        install_vai_conda_channel
    fi
    export VAI_CONDA_CHANNEL="file://${channel_dir}/conda-channel"
}

"$@"
