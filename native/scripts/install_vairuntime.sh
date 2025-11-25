#!/bin/bash

set -ex

DOWNLOAD_DIR="${DOWNLOAD_DIR:-/tools/Xilinx/Downloads}"

# XRT/XRM URLs - kept for reference but NOT installed by default.
# If you have a full Vitis installation (e.g., /tools/Xilinx/Vitis/2025.1.1),
# XRT is already bundled. Use: source /tools/Xilinx/Vitis/2025.1.1/settings64.sh
# See also: /tools/Xilinx/Vitis/2025.1.1/Vitis/aietools/xrt
XRT_URL="${XRT_URL:-https://www.xilinx.com/bin/public/openDownload?filename=xrt_202220.2.14.418_20.04-amd64-xrt.deb}"
XRM_URL="${XRM_URL:-https://www.xilinx.com/bin/public/openDownload?filename=xrm_202220.1.5.212_20.04-x86_64.deb}"

VAI_DEB_CHANNEL="${VAI_DEB_CHANNEL:-https://www.xilinx.com/bin/public/openDownload?filename=vairuntime-3.5.0.tar.gz}"
VAI_DEB_CHANNEL_NAME="${VAI_DEB_CHANNEL_NAME:-vairuntime-3.5.0}"

xrt_file="${DOWNLOAD_DIR}/xrt.deb"
xrm_file="${DOWNLOAD_DIR}/xrm.deb"
vairuntime_tar_gz="${VAI_DEB_CHANNEL_NAME}.tar.gz"
vairuntime_file="${DOWNLOAD_DIR}/${vairuntime_tar_gz}"
vairuntime_dir="${DOWNLOAD_DIR}/${VAI_DEB_CHANNEL_NAME}"

# ------------------------------------------------------------------------------
# XRT/XRM functions - provided for standalone XRT installs (no Vitis).
# NOT included in install_all; call explicitly if needed.
# ------------------------------------------------------------------------------

download_xrt() {
    wget --progress=dot:mega \
        -O "${xrt_file}" \
        "${XRT_URL}"
}

download_xrm() {
    wget --progress=dot:mega \
        -O "${xrm_file}" \
        "${XRM_URL}"
}

# Install XRT from standalone .deb package.
# Skip if you have Vitis installed - XRT is bundled at:
#   /tools/Xilinx/Vitis/2025.1.1/Vitis/aietools/xrt
#   /tools/Xilinx/Vitis/2025.1.1/Vitis/bin/xrt_server
install_xrt() {
    if [[ ! -f "${xrt_file}" ]]; then
        download_xrt
        chmod a-w "${xrt_file}"
    fi

    sudo apt-get update -y
    sudo apt-get install -y "${xrt_file}"
}

# Install XRM (Xilinx Resource Manager).
# Only needed for multi-card or multi-user FPGA resource management.
# Most single-user development setups do not require XRM.
install_xrm() {
    if [[ ! -f "${xrm_file}" ]]; then
        download_xrm
        chmod a-w "${xrm_file}"
    fi

    sudo apt-get update -y
    sudo apt-get install -y "${xrm_file}"
}

# ------------------------------------------------------------------------------
# Vitis AI Runtime - required for on-host DPU inference
# ------------------------------------------------------------------------------

download_vairuntime() {
    wget --progress=dot:mega \
        -O "${vairuntime_file}" \
        "${VAI_DEB_CHANNEL}"
}

unpack_vairuntime() {
    mkdir -p "${vairuntime_dir}"
    cd "${vairuntime_dir}"
    tar -xzvf "${vairuntime_file}"
}

install_vairuntime() {
    if [[ ! -f "${vairuntime_file}" ]]; then
        download_vairuntime
        chmod a-w "${vairuntime_file}"
    fi

    if [[ ! -d "${vairuntime_dir}" ]]; then
        unpack_vairuntime
        chmod -R a-w "${vairuntime_dir}"
    fi

    sudo apt-get update -y
    sudo apt-get install -y "${vairuntime_dir}"/*/*.deb
    sudo ldconfig
}

# ------------------------------------------------------------------------------
# Default install target - only installs Vitis AI runtime.
# XRT/XRM are NOT included; use Vitis-bundled XRT or call install_xrt manually.
# ------------------------------------------------------------------------------

install_all() {
    install_vairuntime
}

"$@"
