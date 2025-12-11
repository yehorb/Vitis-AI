set -ex

T_DIR="${T_DIR:-/home/petalinux/models/yolox_tiny/}"
Q_DIR="${Q_DIR:-build/var/yolox_tiny_stft_quantized}"

if [ -z "${STFT_DATASET}" ]; then
    echo "[${0}]: \$STFT_DATASET is not set"
    exit 1
fi

ssh petalinux@kria.local "mkdir -p ${T_DIR}"
scp -O "${Q_DIR}/target/kv260.xmodel" \
    src/edge/inference.py \
    "${STFT_DATASET}/meta.json" \
    "${Q_DIR}/*.npy" \
    "petalinux@kria.local:${T_DIR}"
