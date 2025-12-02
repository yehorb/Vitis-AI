#!/bin/bash
#
# Copyright 2021 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -ex

DOWNLOAD_DIR="${DOWNLOAD_DIR:-/home/xilinx/Downloads}"
INSTALL_PATH="${INSTALL_PATH:-${HOME}/petalinux_sdk_2022.2}"
SDK_URL="${SDK_URL:-https://www.xilinx.com/bin/public/openDownload?filename=sdk-2022.2.0.0.sh}"
VITIS_AI_URL="${VITIS_AI_URL:-https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_2022.2-r3.0.0.tar.gz}"

sdk_file="${DOWNLOAD_DIR}/sdk-2022.2.0.0.sh"
vitis_ai_file="${DOWNLOAD_DIR}/vitis_ai_2022.2-r3.0.0.tar.gz"
sysroot_path="${INSTALL_PATH}/sysroots/cortexa72-cortexa53-xilinx-linux"

download_sdk() {
    if [[ ! -f "${sdk_file}" ]]; then
        wget --progress=dot:mega \
            -O "${sdk_file}" \
            "${SDK_URL}"
        chmod a+x,a-w "${sdk_file}"
    fi
}

download_vitis_ai() {
    if [[ ! -f "${vitis_ai_file}" ]]; then
        wget --progress=dot:mega \
            -O "${vitis_ai_file}" \
            "${VITIS_AI_URL}"
        chmod a-w "${vitis_ai_file}"
    fi
}

install_sdk() {
    if [[ ! -d "${INSTALL_PATH}" ]]; then
        mkdir -p "${INSTALL_PATH}"
    fi
    echo "${INSTALL_PATH}" | "${sdk_file}"
}

install_vitis_ai() {
    rm -rf "${sysroot_path}/usr/share/cmake/XRT/"
    tar -xzvf "${vitis_ai_file}" -C "${sysroot_path}/"
}

print_instructions() {
    set +x
    echo "Complete Cross Compiler installation"
    echo ""
    echo "Please run the following command to enable Cross Compiler"
    echo "    source ${INSTALL_PATH}/environment-setup-cortexa72-cortexa53-xilinx-linux"
    echo "If you run the above command failed, run the following commands to enable Cross Compiler"
    echo "    unset LD_LIBRARY_PATH"
    echo "    source ${INSTALL_PATH}/environment-setup-cortexa72-cortexa53-xilinx-linux"
    echo ""
}

install_all() {
    download_sdk
    download_vitis_ai
    install_sdk
    install_vitis_ai
    print_instructions
}

"$@"
