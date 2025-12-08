unset LD_LIBRARY_PATH
source /home/xilinx/tools/petalinux_sdk_2022.2/environment-setup-cortexa72-cortexa53-xilinx-linux
vai_c_xir \
    -x build/var/quantized_stft/YOLOX_0_int.xmodel \
    -a "${CONDA_PREFIX}/lib/python3.8/site-packages/vaic/arch/DPUCZDX8G/KV260/arch.json" \
    -o build/var/quantized_stft/target \
    -n yolox_stft_kv260
