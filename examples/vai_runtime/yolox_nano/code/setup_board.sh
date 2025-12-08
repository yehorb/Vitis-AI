ssh petalinux@kria.local "mkdir -p /home/petalinux/models/yolox_stft"
scp -O build/var/quantized_stft/target/yolox_stft_kv260.xmodel \
    src/edge/inference.py \
    data/stft/20251207_162413/meta.json \
    build/var/quantized_stft/test.npy \
    build/var/quantized_stft/test_labels.npy \
    petalinux@kria.local:/home/petalinux/models/yolox_stft/
