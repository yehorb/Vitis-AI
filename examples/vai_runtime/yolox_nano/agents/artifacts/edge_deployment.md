# Edge Device Deployment Guide for YOLOX on KV260

This document describes the complete flow for deploying a trained YOLOX model to the Xilinx KV260 edge device using Vitis AI 3.0.

## Overview

The deployment pipeline consists of three main phases:

```
Training (GPU)  →  Quantization (GPU)  →  Compilation (Host)  →  Inference (KV260)
     ↓                   ↓                      ↓                      ↓
  .pth file      _int.xmodel (XIR)      .xmodel (DPU)         VART Runtime
```

## Prerequisites

### Host Environment
- Vitis AI 3.0 Docker container with PyTorch support
- CUDA-capable GPU for quantization calibration
- PetaLinux SDK 2022.2 for cross-compilation (required for `vai_c_xir`)

### Target Environment (KV260)
- Xilinx KV260 Vision AI Starter Kit
- Pre-built Vitis AI 3.0 SD card image (includes DPU overlay and VART runtime)
- Python 3.8+ with `vart` and `xir` packages installed

### Model Requirements
- Trained YOLOX checkpoint (.pth file)
- Model must use DPU-compatible operations (ReLU activation, standard convolutions)
- Experiment file that conditionally imports quantization-aware model components

---

## Phase 1: Quantization (PTQ)

**Purpose:** Convert floating-point model weights to INT8 for DPU execution.

**Input:** Trained model checkpoint (`best_ckpt.pth`)
**Output:** Quantized XIR model (`YOLOX_0_int.xmodel`)

### Script: `code/run_quant.sh`

The quantization script performs three sequential steps:

1. **Calibration (`--quant_mode calib`)**: Runs inference on a representative dataset to collect activation statistics. These statistics determine optimal quantization scale factors for each tensor.

2. **Testing (`--quant_mode test`)**: Evaluates the quantized model accuracy against the validation set. This validates that quantization did not significantly degrade model performance.

3. **Export (`--is_dump`)**: Serializes the quantized model to XIR format (`.xmodel`) for the Vitis AI compiler.

### Key Environment Variables

| Variable | Purpose |
|----------|---------|
| `W_QUANT=1` | Enables quantization-aware model loading in the experiment file |
| `CUDA_VISIBLE_DEVICES` | Specifies which GPU to use for calibration |

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `-f` | Path to experiment file (defines model architecture and data loaders) |
| `-c` | Path to trained checkpoint |
| `-b` | Batch size for calibration (larger = better statistics, more memory) |
| `--quant_dir` | Output directory for quantization artifacts |
| `--conf` | Confidence threshold for evaluation |

### Conditional Model Loading

The experiment file (`yolox_nano_stft_relu.py`) uses environment variable `W_QUANT` to switch between standard and quantization-aware model components:

- When `W_QUANT=1`: Imports from `yolox.models.yolox_q`, `yolo_pafpn_deploy_q`, `yolo_head_q`
- When `W_QUANT=0` or unset: Imports from standard `yolox.models`

The `*_q.py` modules include `QuantStub` at input, `DeQuantStub` at output, and replace `torch.cat()` with quantization-friendly `QF.Cat()`.

### Output Artifacts

After running `run_quant.sh`, the `--quant_dir` contains:

| File | Purpose |
|------|---------|
| `YOLOX_0_int.xmodel` | Quantized XIR model (input to compiler) |
| `Quant_info.json` | Quantization parameters for each tensor |
| `YOLOX.py` | Generated quantized model definition |
| `quant_info.json` | Additional quantization metadata |

---

## Phase 2: Compilation

**Purpose:** Compile the quantized XIR model for the KV260's specific DPU architecture.

**Input:** Quantized XIR model (`YOLOX_0_int.xmodel`)
**Output:** Compiled DPU model (`yolox_stft_kv260.xmodel`)

### Script: `code/run_compile.sh`

### Prerequisites

1. **Unset `LD_LIBRARY_PATH`**: The Vitis AI Docker environment may have conflicting libraries with the PetaLinux SDK.

2. **Source PetaLinux SDK**: Provides cross-compilation toolchain and `vai_c_xir` compiler.
   ```bash
   source /home/xilinx/tools/petalinux_sdk_2022.2/environment-setup-cortexa72-cortexa53-xilinx-linux
   ```

### Compiler Command: `vai_c_xir`

| Parameter | Description |
|-----------|-------------|
| `-x` | Input XIR model (from quantization) |
| `-a` | Architecture JSON for target DPU (KV260 uses DPUCZDX8G) |
| `-o` | Output directory for compiled model |
| `-n` | Output model name |

### DPU Architecture

The KV260 uses the **DPUCZDX8G** DPU IP with **B4096** configuration. The architecture file specifies:
- Supported operations
- Parallelism factors
- Memory constraints
- Input/output tensor layouts

The compiler:
1. Parses the XIR graph
2. Maps operations to DPU instructions
3. Partitions unsupported operations to CPU
4. Optimizes memory access patterns
5. Generates executable subgraphs

### Output

The compiled `.xmodel` file contains:
- DPU subgraph(s) with compiled instructions
- CPU subgraph(s) for unsupported operations
- Tensor metadata (shapes, fix-points, names)

---

## Phase 3: Edge Inference

**Purpose:** Execute the compiled model on KV260 hardware.

### Deployment Steps

1. Transfer compiled model to KV260:
   ```bash
   scp build/var/quantized_stft/target/yolox_stft_kv260.xmodel root@<KV260_IP>:/home/root/models/
   ```

2. Ensure DPU overlay is loaded (typically automatic on boot with Vitis AI image)

3. Run inference application using VART Python API

---

## Writing the Edge Inference Script

This section provides detailed guidance for implementing a clean-room inference script using the VART (Vitis AI Runtime) Python API.

### Required Imports

The inference script requires two Vitis AI Python packages:
- `xir`: For loading and parsing the compiled xmodel graph
- `vart`: For creating DPU runners and executing inference

### Step 1: Load the Compiled Model

Load the xmodel file using `xir.Graph.deserialize()`. This returns a graph object containing all subgraphs (DPU and CPU).

### Step 2: Extract DPU Subgraph

The compiled model contains multiple subgraphs. You must extract the DPU subgraph for hardware acceleration:

1. Get the root subgraph from the graph
2. Topologically sort child subgraphs
3. Filter for subgraphs with `device` attribute equal to `"DPU"`

For single-DPU models like YOLOX-nano, there will be exactly one DPU subgraph.

### Step 3: Create DPU Runner

Use `vart.Runner.create_runner()` with the DPU subgraph and execution mode `"run"`. This allocates DPU resources and prepares for inference.

### Step 4: Query Tensor Information

From the runner, obtain input and output tensor descriptors:

**Input Tensors:**
- `runner.get_input_tensors()` returns a list of input tensor objects
- Each tensor has `.dims` (shape), `.name`, and attributes like `fix_point`

**Output Tensors:**
- `runner.get_output_tensors()` returns output tensor objects
- Same attributes available

**Important:** The `fix_point` attribute is critical for quantization scaling:
- Input fix_point: multiply input by `2^fix_point` before casting to int8
- Output fix_point: multiply output by `2^(-fix_point)` to convert back to float

### Step 5: Prepare Input Data

The DPU expects **int8** input tensors in **NHWC** format (batch, height, width, channels).

Preprocessing steps:
1. Load and resize input to model's expected dimensions (128x128 for this model)
2. Normalize using the same parameters as training (vmin_db, vmax_db scaling to 0-1)
3. Scale by input quantization factor: `input_scaled = input_float * (2 ** input_fix_point)`
4. Cast to int8: `input_int8 = input_scaled.astype(np.int8)`
5. Reshape to NHWC: `input_int8.reshape(1, height, width, channels)`

**Critical:** The normalization must exactly match training. For this STFT model:
- Input values were normalized to [0, 1] range during training
- The same min/max dB values from `meta.json` must be used

### Step 6: Allocate Buffers

Create numpy arrays for input and output:
- Input buffer: `np.empty(input_shape, dtype=np.int8, order="C")`
- Output buffer: `np.empty(output_shape, dtype=np.int8, order="C")`

The `order="C"` ensures C-contiguous memory layout required by VART.

### Step 7: Execute Inference

VART uses asynchronous execution:

1. Copy preprocessed data to input buffer
2. Call `runner.execute_async(input_buffers, output_buffers)` - returns job ID
3. Call `runner.wait(job_id)` to block until completion
4. Read results from output buffer

### Step 8: Dequantize Output

The DPU returns int8 values. Convert to float:
```
output_float = output_int8.astype(np.float32) * (2 ** (-output_fix_point))
```

### Step 9: YOLOX-Specific Postprocessing

YOLOX outputs require grid-based decoding before NMS. This runs on the ARM CPU.

**Output Format:**
- Shape: `[batch, num_anchors, 5 + num_classes]`
- Per anchor: `[x_offset, y_offset, width, height, objectness, class_scores...]`

**Grid Decoding:**

YOLOX-nano uses three detection heads at strides 8, 16, and 32. For 128x128 input:
- Stride 8: 16x16 = 256 anchors
- Stride 16: 8x8 = 64 anchors
- Stride 32: 4x4 = 16 anchors
- Total: 336 anchors

For each anchor at grid position (gx, gy) with stride s:
- `x_center = (x_offset + gx) * s`
- `y_center = (y_offset + gy) * s`
- `box_width = exp(width) * s`
- `box_height = exp(height) * s`

**Confidence Calculation:**
- Apply sigmoid to objectness and class scores (if not done in model)
- Final score = objectness * class_score

**Non-Maximum Suppression:**
- Filter boxes below confidence threshold
- Apply IoU-based NMS to remove overlapping detections
- For single-class detection, class-agnostic NMS is simpler

**Coordinate Conversion:**
- Convert from center-width-height (cxcywh) to corner format (x1y1x2y2)
- Clip to image boundaries

### Performance Considerations

1. **Batch Size:** Single-image inference (batch=1) is typical for real-time applications

2. **Buffer Reuse:** Allocate input/output buffers once, reuse across inference calls

3. **Multithreading:** VART supports multiple runners for pipeline parallelism. Create separate runners for concurrent execution.

4. **Memory:** The DPU has limited on-chip memory. Very large models may require tiling.

5. **Preprocessing:** Consider implementing preprocessing on FPGA fabric for maximum throughput, or use NEON intrinsics on ARM cores.

---

## Current Project Status

| Step | Status | Artifacts |
|------|--------|-----------|
| Training | Complete | `YOLOX_outputs/yolox_nano_stft_relu/best_ckpt.pth` |
| Quantization | Complete | `build/var/quantized_stft/YOLOX_0_int.xmodel` |
| Compilation | Complete | `build/var/quantized_stft/target/yolox_stft_kv260.xmodel` |
| Edge Inference | Pending | Script to be implemented |

---

## Reference Documentation

- Vitis AI User Guide (UG1414): Comprehensive reference for quantization, compilation, and runtime
- Vitis AI Library User Guide (UG1354): High-level C++ API for common models
- VART API Reference: Python and C++ runtime API documentation
- DPUCZDX8G Product Guide (PG338): DPU IP core architecture and capabilities

---

## Troubleshooting

### Quantization Issues

**"QuantStub not found"**: Ensure `W_QUANT=1` is set and experiment file imports from `*_q.py` modules.

**"Unsupported operation"**: Some operations (SiLU, Mish) are not DPU-compatible. Use ReLU activation.

**Accuracy degradation**: Try increasing calibration batch size or using fast finetuning (`--fast_finetune`).

### Compilation Issues

**"Missing arch.json"**: Ensure PetaLinux SDK is sourced and `CONDA_PREFIX` points to valid Vitis AI installation.

**"Unsupported layer"**: Some operations run on CPU. Check compiler log for partitioning details.

### Runtime Issues

**"DPU not found"**: Ensure DPU overlay is loaded (`xmutil listapps`, `xmutil loadapp kv260-smartcam`)

**"Shape mismatch"**: Verify input preprocessing produces correct NHWC tensor shape.

**"Fix-point overflow"**: Input values exceed int8 range after scaling. Check normalization.
