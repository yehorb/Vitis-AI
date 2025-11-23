ARG VAI_BASE=xilinx/vitis-ai-gpu-tf2-base:latest
FROM $VAI_BASE

ARG VAI_CONDA_CHANNEL="file:///scratch/conda-channel"
ENV VAI_CONDA_CHANNEL=$VAI_CONDA_CHANNEL
ARG VERSION
ENV VERSION=$VERSION
ARG GIT_HASH="<blank>"
ENV GIT_HASH=$GIT_HASH
ARG BUILD_DATE
ENV BUILD_DATE=$BUILD_DATE
ARG XRT_URL=https://www.xilinx.com/bin/public/openDownload?filename=xrt_202120.2.12.427_18.04-amd64-xrt.deb
ENV XRT_URL=$XRT_URL
ARG XRM_URL=https://www.xilinx.com/bin/public/openDownload?filename=xrm_202120.1.3.29_18.04-x86_64.deb
ENV XRM_URL=$XRM_URL
ARG VAI_DEB_CHANNEL=""
ENV VAI_DEB_CHANNEL=$VAI_DEB_CHANNEL
ARG VAI_WEGO_CONDA_CHANNEL="file:///scratch/conda-channel-wego"
ENV VAI_WEGO_CONDA_CHANNEL=$VAI_WEGO_CONDA_CHANNEL


WORKDIR /workspace

COPY ./common/ .
COPY ./conda /scratch
COPY conda/banner.sh /etc/
COPY conda/gpu_conda/bashrc /etc/bash.bashrc

# Install and set up a conda channel
ENV DOWNLOAD_DIR="/var/cache/docker/downloads"
ENV CHANNEL_FILE="conda-channel.tar.gz"
ENV CHANNEL_DIR="/scratch/conda-channel"
RUN --mount=type=cache,target=/var/cache/docker \
    sudo chmod -R 777 /var/cache/docker && \
        mkdir -p "${DOWNLOAD_DIR}" && \
        if [[ ! -f "${DOWNLOAD_DIR}/${CHANNEL_FILE}" ]]; then \
            wget -O "${DOWNLOAD_DIR}/${CHANNEL_FILE}" --progress=dot:mega ${VAI_CONDA_CHANNEL} && \
                sudo chmod 555 "${DOWNLOAD_DIR}/${CHANNEL_FILE}"; \
        fi && \
        cd /scratch && \
        tar -xzvf "${DOWNLOAD_DIR}/${CHANNEL_FILE}" && \
        sudo chmod -R 555 "${CHANNEL_DIR}"
ENV VAI_CONDA_CHANNEL="file://${CHANNEL_DIR}"

RUN \
    --mount=type=cache,target=/home/vitis-ai-user/.cache \
    --mount=type=cache,target=/home/vitis-ai-user/.conda/pkgs \
    --mount=type=cache,target=/opt/vitis_ai/conda/pkgs \
    --mount=type=cache,target=/var/cache/docker \
    sudo chmod -R 777 /home/vitis-ai-user/.cache && \
        sudo chmod -R 777 /home/vitis-ai-user/.conda/pkgs && \
        sudo chmod -R 777 /opt/vitis_ai/conda/pkgs && \
        sudo chmod -R 777 /var/cache/docker && \
        tensorflow_ver="tensorflow==2.12 keras==2.12" && \
        source $VAI_ROOT/conda/etc/profile.d/conda.sh && \
        mkdir -p $VAI_ROOT/conda/pkgs && \
        conda config --system --add channels ${VAI_CONDA_CHANNEL} && \
        conda config --system --remove channels conda-forge && \
        mamba env create -f /scratch/gpu_conda/vitis-ai-tensorflow2.yml && \
        conda activate vitis-ai-tensorflow2 && \
        pip install --ignore-installed ${tensorflow_ver} && \
        mamba install --no-update-deps -y pydot pyyaml jupyter ipywidgets \ \
            dill progressbar2 pytest pandas matplotlib \ \
            pillow -c ${VAI_CONDA_CHANNEL} -c conda-forge -c defaults && \
        pip install -r /scratch/pip_requirements.txt && \
        pip install transformers pycocotools scikit-learn scikit-image tqdm easydict && \
        pip install --ignore-installed ${tensorflow_ver} && \
        pip uninstall -y h5py && \
        mamba install -y --override-channels --force-reinstall h5py=2.10.0 -c conda-forge && \
        pip install --force --no-binary protobuf protobuf==3.20.3 && \
        conda clean -y --force-pkgs-dirs && \
        conda config --env --remove-key channels && \
        conda activate vitis-ai-tensorflow2 && \
        sudo mkdir -p $VAI_ROOT/compiler && \
        sudo cp -r $CONDA_PREFIX/lib/python3.8/site-packages/vaic/arch $VAI_ROOT/compiler/arch
