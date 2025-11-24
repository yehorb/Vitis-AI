#!/bin/bash

set -ex

CONDA_PREFIX="${CONDA_PREFIX:-/home/xilinx}"
MINIFORGE_VERSION="${MINIFORGE_VERSION:-"25.9.1-0"}"

if [[ ${DOCKER_TYPE} =~ .*'rocm'* && ${TARGET_FRAMEWORK} =~ .*"pytorch" ]]; then
    ln -s /opt/conda $CONDA_PREFIX/conda
else
    export HOME=~vitis-ai-user

    if [[ -d "/root/.local" ]]; then
        sudo chmod -R 777 /root/.local
    fi

    sudo chmod 777 /root /root/.local /root/.local/bin || true

    export MINIFORGE_VERSION="25.3.1-0"
    cd /tmp &&
        wget --progress=dot:mega https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh &&
        /bin/bash ./miniconda.sh -b -p $CONDA_PREFIX/conda &&
        . $CONDA_PREFIX/conda/etc/profile.d/conda.sh &&
        rm -fr /tmp/miniconda.sh &&
        /$CONDA_PREFIX/conda/bin/conda clean -y --force-pkgs-dirs
fi

echo ". $CONDA_PREFIX/conda/etc/profile.d/conda.sh" >>~vitis-ai-user/.bashrc
sudo ln -s $CONDA_PREFIX/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
