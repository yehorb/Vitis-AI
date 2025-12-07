export W_QUANT=1
export CUDA_HOME=/usr/local/cuda

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPU_NUM=8
BATCH=128

# Step1: QAT
CFG=code/exps/example/custom/yolox_nano_deploy_relu_qat.py
python -m yolox.tools.train -f ${CFG} -d ${GPU_NUM} -b ${BATCH} -o


export CUDA_VISIBLE_DEVICES=0
GPU_NUM=1
BATCH=32

# WORKSPACE=YOLOX_outputs/yolox_nano_deploy_relu_qat # assign with this path if you QAT yourself instead of using the released QAT model
WORKSPACE=qat

# Step2: Eval accuracy after QAT
QAT_WEIGHTS=qat/qat.pth  # assign the path to the released QAT weights
# QAT_WEIGHTS=${WORKSPACE}/best_ckpt.pth  # assign the path to your QAT weights
python -m yolox.tools.qat_tool -f ${CFG} -c ${QAT_WEIGHTS} -b ${BATCH} -d ${GPU_NUM} --conf 0.001

# Step3: Convert the QAT model to deployble model and verify the accuracy
CVT_DIR=${WORKSPACE}/convert_qat_results
python -m yolox.tools.qat_tool -f ${CFG} -c ${QAT_WEIGHTS} --cvt_dir ${CVT_DIR} -b ${BATCH} -d ${GPU_NUM} --conf 0.001

# Step4: Dump xmodel
python -m yolox.tools.qat_tool -f ${CFG} -c ${QAT_WEIGHTS} --cvt_dir ${CVT_DIR} --is_dump
