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
sudo chmod -R 777 /var/cache && \
    mkdir -p "${DOWNLOAD_DIR}" && \
    if [[ ! -f "${DOWNLOAD_DIR}/${CHANNEL_FILE}" ]]; then \
        wget -O "${DOWNLOAD_DIR}/${CHANNEL_FILE}" --progress=dot:mega ${VAI_CONDA_CHANNEL} && \
        sudo chmod 555 "${DOWNLOAD_DIR}/${CHANNEL_FILE}"; \
    fi && \
    cd /scratch && \
    tar -xzvf "${DOWNLOAD_DIR}/${CHANNEL_FILE}"; \
    sudo chmod -R 555 "${CHANNEL_DIR}"
ENV VAI_CONDA_CHANNEL="file://${CHANNEL_DIR}"
