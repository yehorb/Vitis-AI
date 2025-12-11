Q_DIR="${Q_DIR:-build/var/yolox_tiny_stft_quantized}"

unset LD_LIBRARY_PATH
source /home/xilinx/tools/petalinux_sdk_2022.2/environment-setup-cortexa72-cortexa53-xilinx-linux
vai_c_xir \
    -x "${Q_DIR}/YOLOX_0_int.xmodel" \
    -a "${CONDA_PREFIX}/lib/python3.8/site-packages/vaic/arch/DPUCZDX8G/KV260/arch.json" \
    -o "${Q_DIR}/target" \
    -n kv260
