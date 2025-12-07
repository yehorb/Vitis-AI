# Modified YOLOX-Nano for 2D object detection on COCO

### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Performance](#performance)
5. [Model_info](#model_info)
6. [Acknowledgement](#acknowledgement)

### Installation

1. Environment requirement
    - anaconda3
    - python 3.6
    - pytorch, torchvision, numpy etc, refer to [requirements.txt](requirements.txt) for more details.
    - vai_q_pytorch(Optional, required for quantization)
    - XIR Python frontend (Optional, required for dumping xmodel)

2. Installation with Docker

   First refer to [vitis-ai](https://github.com/Xilinx/Vitis-AI/tree/master/) to obtain the docker image.
   ```bash
   conda activate vitis-ai-pytorch
   pip install --user -r requirements.txt 
   cd code
   pip install --user -v -e .
   cd ..
   ```

### Preparation

1. Dataset description

The dataset MSCOCO2017 contains 118287 images for training and 5000 images for validation.

2. Download COCO dataset and create directories like this:
  ```plain
  └── data
       └── COCO
             ├── annotations
             |   ├── instances_train2017.json
             |   ├── instances_val2017.json
             |   └── ...
             ├── train2017
             |   ├── 000000000009.jpg
             |   ├── 000000000025.jpg
             |   ├── ...
             ├── val2017
                 ├── 000000000139.jpg
                 ├── 000000000285.jpg
             |   ├── ...
             └── test2017
                 ├── 000000000001.jpg
                 ├── 000000000016.jpg
                 └── ...
  ```

### Train/Eval/QAT

1. Evaluation
  - Execute run_eval.sh.
  ```shell
  bash code/run_eval.sh
  ```

2. Training
  ```shell
  bash code/run_train.sh
  ```

3. Model quantization and xmodel dumping
  ```shell
  bash code/run_quant.sh
  ```

4. QAT(Quantization-Aware-Training), model converting and xmodel dumping
  - Configure the variables and in `code/run_qat.sh` and `code/exps/example/custom/yolox_nano_deploy_relu_qat.py`, read the steps(including QAT, model testing, model converting and xmodel dumping) in the script and run the step you want.
  ```shell
  bash code/run_qat.sh
  ```

### Performance
|Metric | Float | Quantized | QAT |
| -     | -    | - | - |
|AP0.50:0.95|0.220|0.136|0.210|


### Model_info

1. Data preprocess
  ```
  data channels order: BGR
  keeping the aspect ratio of H/W, resize image with bilinear interpolation to make the long side to be 416, pad the image with (114,114,114) along the height side to get image with shape of (H,W)=(416,416)
  ``` 


### Acknowledgement

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX.git)
