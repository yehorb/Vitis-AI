#!/bin/bash

set -ex

CONDA_PREFIX="${CONDA_PREFIX:-/home/xilinx}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/tools/Xilinx/Downloads}"
MINIFORGE_VERSION="${MINIFORGE_VERSION:-"25.9.1-0"}"

download_miniforge() {
    wget --progress=dot:mega \
        -O "${DOWNLOAD_DIR}/Miniforge3.sh" \
        "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-$(uname)-$(uname -m).sh"
}

install_miniforge() {
    "${DOWNLOAD_DIR}/Miniforge3.sh" -b -p $CONDA_PREFIX/conda
}

install_conda() {
    if [[ ! -f "${DOWNLOAD_DIR}/Miniforge3.sh" ]]; then
        download_miniforge
        # Executable, read-only
        chmod a+x,a-w "${DOWNLOAD_DIR}/Miniforge3.sh"
    fi

    if [[ ! -d "${CONDA_PREFIX}/conda" ]]; then
        install_miniforge
    fi
}

activate_conda() {
    # Stop printing commands
    set +x
    source "${CONDA_PREFIX}/conda/etc/profile.d/conda.sh"
    source "${CONDA_PREFIX}/conda/etc/profile.d/mamba.sh"
}

"$@"
