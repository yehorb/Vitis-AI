# Copyright 2019 Xilinx Inc.
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

export CUDA_VISIBLE_DEVICES=0
GPU_NUM=1

export W_QUANT=1
export CUDA_HOME=/usr/local/cuda

BATCH=32
CFG=code/exps/example/custom/yolox_tiny_stft_relu.py
CKPT=YOLOX_outputs/yolox_tiny_stft_relu/best_ckpt.pth
Q_DIR=build/var/yolox_tiny_stft_quantized

set -ex

MODE='calib'
python -m yolox.tools.quant -f ${CFG} -c ${CKPT} -b ${BATCH} -d ${GPU_NUM} --conf 0.001 --nms 0.001 --quant_mode ${MODE} --quant_dir ${Q_DIR}

MODE='test'
python -m yolox.tools.quant -f ${CFG} -c ${CKPT} -b ${BATCH} -d ${GPU_NUM} --conf 0.001 --nms 0.001 --quant_mode ${MODE} --quant_dir ${Q_DIR}

# dump xmpdel
python -m yolox.tools.quant -f ${CFG} -c ${CKPT} -b ${BATCH} -d ${GPU_NUM} --conf 0.001 --nms 0.001 --quant_mode ${MODE} --quant_dir ${Q_DIR} --is_dump
