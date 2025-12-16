#!/usr/bin/env python3
"""
YOLOX STFT Burst Detection - KV260 Edge Inference

Reference implementation for running quantized YOLOX on Vitis AI DPU.
Requires: vart, xir (installed on KV260 Vitis AI image)

Usage:
    python inference.py --model yolox_stft_kv260.xmodel --input spectrogram.npy

================================================================================
ARCHITECTURE OVERVIEW: Why DPU Outputs Separate Tensors
================================================================================

When running YOLOX on the host (PyTorch), the model outputs decoded bounding
boxes ready for visualization. On the DPU, we receive RAW tensors that require
manual postprocessing. This section explains why.

HOST vs EDGE Output Difference
------------------------------

On HOST (PyTorch training/evaluation):
    model(input) -> decoded boxes (N, 336, 6) with coordinates in pixels

On EDGE (DPU inference):
    model(input) -> [head0 (1,16,16,6), head1 (1,8,8,6), head2 (1,4,4,6)]
                    Raw logits, separate tensors, NHWC format

The difference stems from the model architecture designed for quantization.

Why Separate Tensors?
---------------------

1. GRAPH STRUCTURE (code/yolox/models/yolo_head_q.py lines 224-225):

   In inference mode, the quantization-ready head returns a Python list:

       def forward(self, xin, labels=None, imgs=None):
           ...
           else:  # inference mode
               return outputs  # List of 3 tensors, NOT concatenated

   The postprocess() method (lines 227-237) that concatenates and decodes
   is NOT part of the traced computational graph - it's called separately
   by the evaluator after inference.

2. DPU HARDWARE LIMITATIONS:

   The Vitis AI compiler (vai_c_xir) analyzes the graph and determines what
   can run on DPU hardware. The DPU excels at:
   - Convolutions, pooling, element-wise ops
   - Fixed tensor shapes

   It cannot efficiently handle:
   - Dynamic reshape across different spatial sizes
   - Concatenation of tensors with different H/W dimensions

   The 3 detection heads have different spatial sizes:
   - Head 0: 16x16 (stride 8,  for small objects)
   - Head 1: 8x8   (stride 16, for medium objects)
   - Head 2: 4x4   (stride 32, for large objects)

   Merging these requires flatten + concat + permute, which would run on CPU
   anyway. So the compiler places the output boundary at the last conv layer
   of each head.

3. COMPILATION BOUNDARY:

   PyTorch Graph:
       Backbone -> FPN -> Head[k] -> stem -> cls_conv -> cls_pred -+
                                  -> reg_conv -> reg_pred ----------+-> q_cat -> DeQuantStub -> OUTPUT[k]
                                             -> obj_pred -----------+

   The q_cat (quantized concat of reg/obj/cls within same head) IS on DPU
   because all inputs have identical spatial dimensions. But cross-head
   merging would require CPU operations.

Host-Side Postprocessing Reference
----------------------------------

On the host, the evaluator (code/yolox/evaluators/coco_evaluator_q.py) calls
postprocess() explicitly after getting raw outputs:

    outputs = float_model.module.head.postprocess(outputs)  # line 180

This postprocess() method in yolo_head_q.py (lines 227-237) does:
    1. Flatten each head: x.flatten(start_dim=2) for x in outputs
    2. Concatenate: torch.cat(..., dim=2)
    3. Permute to (batch, n_anchors, channels)
    4. Apply sigmoid to confidence scores
    5. Decode coordinates via decode_outputs()

This script replicates that logic for edge deployment.

================================================================================
DECODING PROCESS: From Raw Logits to Bounding Boxes
================================================================================

The DPU outputs raw network predictions that must be decoded to pixel coords.

Channel Layout (from yolo_head_q.py line 186)
---------------------------------------------

Each detection head output has 6 channels (for 1-class detection):

    output = q_cat([reg_output, obj_output, cls_output], dim=1)
    #              [  0:4     ,    4:5    ,    5:6    ]

    Channel 0-1: x, y offsets (raw, not activated)
    Channel 2-3: w, h predictions (raw, apply exp())
    Channel 4:   objectness logit (apply sigmoid)
    Channel 5+:  class logits (apply sigmoid)

Grid-Based Coordinate Decoding
------------------------------

YOLOX uses anchor-free detection with grid-based coordinate prediction.
Each spatial location (i, j) in the feature map predicts one detection.

Reference: yolo_head_q.py decode_outputs() (lines 259-274)

For a feature map of size (H, W) with stride S:

    # Build coordinate grid (meshgrid of cell indices)
    grid[i, j] = (j, i)  # (x_cell, y_cell)

    # Decode center coordinates
    x_center = (x_offset + grid_x) * stride
    y_center = (y_offset + grid_y) * stride

    # Decode width/height (exponential to ensure positive)
    width  = exp(w_raw) * stride
    height = exp(h_raw) * stride

This corresponds to yolo_head_q.py lines 272-273:
    outputs[..., :2] = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides

Confidence Score Computation
----------------------------

Final detection confidence combines objectness and class probability:

    obj_conf = sigmoid(obj_logit)      # P(object exists)
    cls_conf = sigmoid(cls_logit)      # P(class | object)
    score = obj_conf * cls_conf        # P(class)

Reference: yolo_head_q.py line 233:
    outputs[..., 4:] = outputs[..., 4:].sigmoid()

Box Format Conversion
---------------------

Network predicts center-based format (cx, cy, w, h).
We convert to corner format (x1, y1, x2, y2) for NMS:

    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2

================================================================================
TENSOR FORMAT: NCHW vs NHWC
================================================================================

PyTorch (host):  NCHW - (batch, channels, height, width)
DPU (edge):      NHWC - (batch, height, width, channels)

The Vitis AI compiler automatically transposes tensors for DPU efficiency.
This script handles NHWC format in decode_single_head().

Example for 128x128 input with stride 8:
    PyTorch output: (1, 6, 16, 16) - NCHW
    DPU output:     (1, 16, 16, 6) - NHWC

================================================================================
COMPLETE PROCESSING PIPELINE
================================================================================

1. PREPROCESSING (preprocess_spectrogram):
   - Normalize dB values to [0, 1] using vmin/vmax from training
   - Reshape to NHWC format (1, H, W, 1)

2. DPU INFERENCE (DpuRunner.run):
   - Quantize float input to int8 using fix-point scale
   - Execute on DPU hardware
   - Dequantize int8 outputs back to float32
   - Returns list of 3 tensors (one per detection head)

3. POSTPROCESSING (postprocess -> decode_yolox_outputs):
   For each detection head:
   a. Determine stride from spatial size: stride = input_size / feature_size
   b. Build coordinate grid for this head
   c. Decode (x, y) centers: (offset + grid) * stride
   d. Decode (w, h): exp(raw) * stride
   e. Apply sigmoid to objectness and class logits
   f. Compute final score: obj_conf * cls_conf
   g. Convert cxcywh to xyxy format

4. FILTERING AND NMS:
   - Filter detections by confidence threshold
   - Apply Non-Maximum Suppression to remove duplicates
   - Return final list of (box, score) tuples

================================================================================
FILE REFERENCES
================================================================================

Training/Quantization (host):
    code/yolox/models/yolo_head_q.py     - Detection head with quantization stubs
        forward() lines 149-225          - Returns list of raw tensors
        postprocess() lines 227-237      - Concat + sigmoid + decode
        decode_outputs() lines 259-274   - Grid-based coordinate decoding

    code/yolox/evaluators/coco_evaluator_q.py
        lines 180-182                    - Calls head.postprocess() after inference

    code/yolox/models/yolo_pafpn_deploy_q.py  - Feature pyramid network
    code/yolox/models/darknet_deploy_q.py    - Backbone network

Edge Inference (this file):
    decode_single_head()      - Equivalent to decode_outputs() for one head
    decode_yolox_outputs()    - Iterates heads, equivalent to postprocess()
    sigmoid()                 - Equivalent to .sigmoid() in PyTorch
    nms()                     - Non-Maximum Suppression

================================================================================
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from functools import wraps
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

# Vitis AI Runtime imports (available on KV260)
import vart
import xir

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ModelConfig:
    """Model configuration matching training parameters."""

    input_height: int = 128
    input_width: int = 128
    input_channels: int = 1
    num_classes: int = 1
    strides: Tuple[int, ...] = (8, 16, 32)
    conf_threshold: float = 0.5
    nms_threshold: float = 0.45


# =============================================================================
# VART Runtime
# =============================================================================


def get_dpu_subgraph(graph: xir.Graph) -> xir.Subgraph:
    """
    Extract the DPU subgraph from a compiled xmodel.

    The compiled model contains multiple subgraphs for different devices.
    We need the one marked with device="DPU" for hardware acceleration.
    """
    root = graph.get_root_subgraph()
    if root.is_leaf:
        raise RuntimeError("Graph has no subgraphs")

    child_subgraphs = root.toposort_child_subgraph()
    dpu_subgraphs = [
        sg
        for sg in child_subgraphs
        if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU"
    ]

    if len(dpu_subgraphs) != 1:
        raise RuntimeError(f"Expected 1 DPU subgraph, found {len(dpu_subgraphs)}")

    return dpu_subgraphs[0]


@dataclass
class OutputTensorInfo:
    """Information about a single output tensor."""

    shape: Tuple[int, ...]
    fixpoint: int
    scale: float
    name: str


class DpuRunner:
    """
    Wrapper for VART DPU runner with buffer management.

    Handles:
    - Model loading and subgraph extraction
    - Input/output buffer allocation (supports multiple outputs)
    - Quantization scale factors (fix-points)
    - Synchronous inference execution

    Why Multiple Outputs?
    ---------------------
    The Vitis AI compiler partitions the YOLOX graph such that each detection
    head becomes a separate output tensor. This happens because:

    1. The quantization-ready model (yolo_head_q.py) returns a list of tensors
       in inference mode, not a concatenated tensor (see forward() line 225)

    2. The DPU cannot efficiently concatenate tensors with different spatial
       dimensions (16x16, 8x8, 4x4) - this would require reshape operations
       that run on CPU anyway

    3. The compiler places output boundaries at the last DPU-executable op
       in each branch (the DeQuantStub after q_cat for each head)

    Typical output configuration for YOLOX-nano on 128x128 input:
        Output[0]: (1, 16, 16, 6) - stride 8 head,  256 anchors
        Output[1]: (1, 8, 8, 6)   - stride 16 head, 64 anchors
        Output[2]: (1, 4, 4, 6)   - stride 32 head, 16 anchors

    Note: Shapes are NHWC (DPU format), not NCHW (PyTorch format)
    """

    def __init__(self, xmodel_path: str):
        """
        Initialize DPU runner from compiled xmodel.

        Args:
            xmodel_path: Path to compiled .xmodel file
        """
        # Load graph and extract DPU subgraph
        self.graph = xir.Graph.deserialize(xmodel_path)
        subgraph = get_dpu_subgraph(self.graph)

        # Create runner
        self.runner = vart.Runner.create_runner(subgraph, "run")

        # Get tensor descriptors
        self.input_tensors = self.runner.get_input_tensors()
        self.output_tensors = self.runner.get_output_tensors()

        # Input info (single input expected)
        self.input_shape = tuple(self.input_tensors[0].dims)
        self.input_fixpoint = self.input_tensors[0].get_attr("fix_point")
        self.input_scale = float(2**self.input_fixpoint)

        # Output info (may have multiple outputs for YOLOX heads)
        self.output_info: List[OutputTensorInfo] = []
        for tensor in self.output_tensors:
            fixpoint = tensor.get_attr("fix_point")
            self.output_info.append(
                OutputTensorInfo(
                    shape=tuple(tensor.dims),
                    fixpoint=fixpoint,
                    scale=float(2 ** (-fixpoint)),
                    name=tensor.name,
                )
            )

        # Pre-allocate buffers (C-contiguous for VART)
        self.input_buffer = [np.empty(self.input_shape, dtype=np.int8, order="C")]
        self.output_buffers = [
            np.empty(info.shape, dtype=np.int8, order="C") for info in self.output_info
        ]

    def run(self, input_float: np.ndarray) -> List[np.ndarray]:
        """
        Execute inference on DPU.

        Args:
            input_float: Preprocessed input, float32, shape matching model input
                        Values should be normalized to [0, 1] range

        Returns:
            List of dequantized outputs as float32 numpy arrays
        """
        # Quantize input: scale and convert to int8
        input_scaled = input_float * self.input_scale
        input_int8 = np.clip(input_scaled, -128, 127).astype(np.int8)

        # Copy to buffer (must preserve C-contiguous layout)
        np.copyto(self.input_buffer[0], input_int8.reshape(self.input_shape))

        # Execute on DPU (async API, but we wait immediately)
        job_id = self.runner.execute_async(self.input_buffer, self.output_buffers)
        self.runner.wait(job_id)

        # Dequantize outputs
        outputs = []
        for buf, info in zip(self.output_buffers, self.output_info):
            output_float = buf.astype(np.float32) * info.scale
            outputs.append(output_float)

        return outputs


# =============================================================================
# Preprocessing
# =============================================================================


def preprocess_spectrogram(
    spectrogram: np.ndarray,
    vmin_db: float,
    vmax_db: float,
    target_shape: Tuple[int, int] = (128, 128),
) -> np.ndarray:
    """
    Preprocess STFT spectrogram for inference.

    Must match training preprocessing exactly:
    1. Normalize dB values to [0, 1] range
    2. Reshape to NHWC format

    Args:
        spectrogram: Raw spectrogram in dB scale, shape (H, W)
        vmin_db: Minimum dB value (from training meta.json)
        vmax_db: Maximum dB value (from training meta.json)
        target_shape: Expected (height, width)

    Returns:
        Preprocessed array, shape (1, H, W, 1), float32, range [0, 1]
    """
    # Validate shape
    if spectrogram.shape != target_shape:
        raise ValueError(
            f"Spectrogram shape {spectrogram.shape} != expected {target_shape}"
        )

    # Normalize to [0, 1] - must match training normalization
    normalized = (spectrogram - vmin_db) / (vmax_db - vmin_db)
    normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)

    # Reshape to NHWC: (1, height, width, channels)
    return normalized.reshape(1, target_shape[0], target_shape[1], 1)


# =============================================================================
# Postprocessing (YOLOX-specific)
# =============================================================================


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def decode_single_head(
    output: np.ndarray,
    stride: int,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode a single YOLOX detection head output.

    This function is the NumPy equivalent of the PyTorch decoding in:
        code/yolox/models/yolo_head_q.py :: decode_outputs() (lines 259-274)

    The key difference is that we process one head at a time (since DPU outputs
    separate tensors), whereas the PyTorch version concatenates first then decodes.

    Decoding formulas (from yolo_head_q.py lines 272-273):
        x_center = (x_offset + grid_x) * stride
        y_center = (y_offset + grid_y) * stride
        width    = exp(w_raw) * stride
        height   = exp(h_raw) * stride

    Channel layout (from yolo_head_q.py line 186):
        output = concat([reg_output, obj_output, cls_output], dim=1)
        channels: [x, y, w, h, obj, cls0, cls1, ...]
                  [0, 1, 2, 3, 4,   5,    6,    ...]

    Args:
        output: Head output in NHWC format, shape (1, H, W, 5+num_classes)
                Note: DPU outputs NHWC, PyTorch uses NCHW
        stride: Detection stride for this head (8, 16, or 32)
                Determines the scale of coordinate predictions
        num_classes: Number of detection classes

    Returns:
        boxes: Decoded boxes in xyxy format, shape (H*W, 4)
               Coordinates are in pixel space of the input image
        scores: Confidence scores, shape (H*W,)
                Combined objectness * class_confidence
    """
    batch, h, w, channels = output.shape
    assert batch == 1, "Batch size must be 1"

    # Build grid for this head
    yv, xv = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    grid = np.stack((xv, yv), axis=-1).astype(np.float32)  # (H, W, 2)

    # Reshape output to (H*W, channels)
    output_flat = output[0].reshape(-1, channels)  # (H*W, 5+num_classes)

    # Extract components
    xy = output_flat[:, :2]  # (H*W, 2)
    wh = output_flat[:, 2:4]  # (H*W, 2)
    obj_conf = output_flat[:, 4:5]  # (H*W, 1)
    cls_conf = output_flat[:, 5 : 5 + num_classes]  # (H*W, num_classes)

    # Flatten grid
    grid_flat = grid.reshape(-1, 2)  # (H*W, 2)

    # Decode coordinates
    # x_center = (x_offset + grid_x) * stride
    # y_center = (y_offset + grid_y) * stride
    xy_decoded = (xy + grid_flat) * stride
    wh_decoded = np.exp(np.clip(wh, -10, 10)) * stride  # Clip to prevent overflow

    # Apply sigmoid to confidence scores
    obj_conf = sigmoid(obj_conf)
    cls_conf = sigmoid(cls_conf)

    # Final score = objectness * class_confidence
    scores = (obj_conf * cls_conf).max(axis=1)  # (H*W,)

    # Convert cxcywh to xyxy
    boxes = np.zeros((h * w, 4), dtype=np.float32)
    boxes[:, 0] = xy_decoded[:, 0] - wh_decoded[:, 0] / 2  # x1
    boxes[:, 1] = xy_decoded[:, 1] - wh_decoded[:, 1] / 2  # y1
    boxes[:, 2] = xy_decoded[:, 0] + wh_decoded[:, 0] / 2  # x2
    boxes[:, 3] = xy_decoded[:, 1] + wh_decoded[:, 1] / 2  # y2

    return boxes, scores


def decode_yolox_outputs(
    outputs: List[np.ndarray],
    config: ModelConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode all YOLOX detection head outputs.

    This function is the NumPy equivalent of:
        code/yolox/models/yolo_head_q.py :: postprocess() (lines 227-237)

    Key difference from PyTorch version:
    - PyTorch: Concatenates heads first, then decodes all at once
    - Here: Decodes each head separately, then concatenates results

    This is necessary because the DPU outputs separate tensors for each head
    (see module docstring for explanation of why DPU separates tensors).

    For a 128x128 input image, the three heads produce:
        Head 0: 16x16 grid, stride 8  -> 256 anchors (small objects)
        Head 1: 8x8 grid,   stride 16 -> 64 anchors  (medium objects)
        Head 2: 4x4 grid,   stride 32 -> 16 anchors  (large objects)
        Total: 336 anchor predictions

    Args:
        outputs: List of head outputs in NHWC format from DPU
                 Typically 3 tensors with shapes like:
                 [(1,16,16,6), (1,8,8,6), (1,4,4,6)]
        config: Model configuration with input dimensions

    Returns:
        boxes: All decoded boxes in xyxy format, shape (336, 4)
        scores: All confidence scores, shape (336,)
    """
    all_boxes = []
    all_scores = []

    # Determine strides based on output shapes
    # For 128x128 input: stride 8 -> 16x16, stride 16 -> 8x8, stride 32 -> 4x4
    for output in outputs:
        h = output.shape[1]
        stride = config.input_height // h

        boxes, scores = decode_single_head(output, stride, config.num_classes)
        all_boxes.append(boxes)
        all_scores.append(scores)

    # Concatenate all heads
    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)

    return boxes, scores


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """
    Non-Maximum Suppression.

    Args:
        boxes: Bounding boxes, shape (N, 4), format xyxy
        scores: Confidence scores, shape (N,)
        iou_threshold: IoU threshold for suppression

    Returns:
        Indices of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # Sort by score descending
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        # Compute IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h

        iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)

        # Keep boxes with IoU below threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)


def postprocess(
    outputs: List[np.ndarray],
    config: ModelConfig,
) -> List[Tuple[np.ndarray, float]]:
    """
    Full postprocessing pipeline: decode, filter, NMS.

    This is the edge equivalent of the host-side postprocessing chain:
        1. code/yolox/models/yolo_head_q.py :: postprocess() - decode raw outputs
        2. code/yolox/utils/demo_utils.py :: postprocess() - NMS and filtering

    On the host, these are called by the evaluator:
        code/yolox/evaluators/coco_evaluator_q.py (lines 180-191)

    Pipeline:
        DPU outputs (3 raw tensors)
            |
            v
        decode_yolox_outputs() - grid decode, sigmoid, concat
            |
            v
        Confidence filtering - remove low-score detections
            |
            v
        NMS - remove overlapping duplicates
            |
            v
        Final detections [(box, score), ...]

    Args:
        outputs: List of raw model outputs from DPU (one per detection head)
                 These are dequantized float32 tensors in NHWC format
        config: Model configuration with thresholds

    Returns:
        List of (box, score) tuples for detected bursts
        Each box is [x1, y1, x2, y2] in pixel coordinates (0 to input_size)
    """
    # Decode all heads
    boxes, scores = decode_yolox_outputs(outputs, config)

    # Filter by confidence threshold
    mask = scores > config.conf_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]

    if len(filtered_boxes) == 0:
        return []

    # Apply NMS
    keep_indices = nms(filtered_boxes, filtered_scores, config.nms_threshold)

    # Collect results
    detections = []
    for idx in keep_indices:
        box = filtered_boxes[idx]
        score = float(filtered_scores[idx])
        detections.append((box, score))

    return detections


# =============================================================================
# Main Inference Pipeline
# =============================================================================


class YoloxInference:
    """
    Complete inference pipeline for YOLOX STFT burst detection on KV260.

    Combines:
    - DPU runner for hardware-accelerated inference
    - Preprocessing (normalization)
    - Postprocessing (decode + NMS)
    """

    def __init__(
        self,
        xmodel_path: str,
        vmin_db: float,
        vmax_db: float,
        config: Optional[ModelConfig] = None,
    ):
        """
        Initialize inference pipeline.

        Args:
            xmodel_path: Path to compiled .xmodel
            vmin_db: Normalization minimum (from training meta.json)
            vmax_db: Normalization maximum (from training meta.json)
            config: Model configuration
        """
        self.runner = DpuRunner(xmodel_path)
        self.vmin_db = vmin_db
        self.vmax_db = vmax_db
        self.config = config if config else ModelConfig()

        print(f"Model loaded: {xmodel_path}")
        print(f"  Input shape: {self.runner.input_shape}")
        print(f"  Input fix-point: {self.runner.input_fixpoint}")
        print(f"  Number of output heads: {len(self.runner.output_info)}")
        for i, info in enumerate(self.runner.output_info):
            print(f"  Output[{i}]: shape={info.shape}, fix-point={info.fixpoint}")

    def detect(self, spectrogram: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        Run detection on a single spectrogram.

        Args:
            spectrogram: STFT spectrogram in dB, shape (128, 128)

        Returns:
            List of (box, score) detections
        """
        # Preprocess
        input_tensor = timing(preprocess_spectrogram)(
            spectrogram,
            self.vmin_db,
            self.vmax_db,
            (self.config.input_height, self.config.input_width),
        )

        # Run DPU inference
        outputs = timing(self.runner.run)(input_tensor)

        # Postprocess
        detections = timing(postprocess)(outputs, self.config)

        return detections

    def benchmark(self, spectrogram: np.ndarray, iterations: int = 100) -> float:
        """
        Benchmark inference throughput.

        Returns:
            Average inference time in milliseconds
        """
        # Warmup
        for _ in range(10):
            self.detect(spectrogram)

        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            self.detect(spectrogram)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / iterations) * 1000
        print(f"Average inference time: {avg_ms:.2f} ms ({1000/avg_ms:.1f} FPS)")
        return avg_ms


def visualize_predictions(
    spectrogram: npt.NDArray[np.float32],
    detections: List[Tuple[npt.NDArray[np.float32], float]],
    vmin: float,
    vmax: float,
    cmap: str = "magma",
    title: str = "YOLOX Burst Detection (Edge)",
):
    """
    Visualize spectrogram with detection bounding boxes.

    Args:
        spectrogram: Input spectrogram, shape (H, W)
        detections: List of (box, score) tuples from postprocess()
                    Box format is xyxy: [x1, y1, x2, y2]
        vmin: Min value for colormap normalization
        vmax: Max value for colormap normalization
        cmap: Matplotlib colormap name
        title: Plot title
        save_path: If provided, save figure to this path instead of showing
    """

    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    assert isinstance(ax, plt.Axes)

    im = ax.imshow(
        spectrogram,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    # Draw bounding boxes (xyxy format)
    for box, score in detections:
        if box.shape[0] != 4:
            continue
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle(
            (x1, y1),
            width,
            height,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)
        # Add score label
        ax.text(
            x1,
            y2 + 2,
            f"{score:.2f}",
            color="lime",
            fontsize=8,
            fontweight="bold",
            verticalalignment="bottom",
        )

    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Frequency bin")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Magnitude (dB)")

    plt.tight_layout()
    plt.show()


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="YOLOX STFT Burst Detection on KV260")
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Path to compiled .xmodel file",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input spectrogram (.npy file, shape 128x128)",
    )
    parser.add_argument(
        "--meta",
        "-M",
        default=None,
        help="Path to meta.json with normalization params (optional)",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=-90.0,
        help="Min dB for normalization (default: -90.0)",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=-20.0,
        help="Max dB for normalization (default: -20.0)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--nms",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run inference benchmark",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot input spectrogram and boxes using matplotlib",
    )

    args = parser.parse_args()

    # Load normalization params from meta.json if provided
    vmin_db, vmax_db = args.vmin, args.vmax
    if args.meta:
        with open(args.meta) as f:
            meta = json.load(f)
        vmin_db = meta.get("render", {}).get("vmin_db", vmin_db)
        vmax_db = meta.get("render", {}).get("vmax_db", vmax_db)
        print(f"Loaded normalization from {args.meta}: vmin={vmin_db}, vmax={vmax_db}")

    # Configure model
    config = ModelConfig(
        conf_threshold=args.conf,
        nms_threshold=args.nms,
    )

    # Initialize inference pipeline
    inference = YoloxInference(args.model, vmin_db, vmax_db, config)

    # Load input
    spectrogram = np.load(args.input)
    print(f"Input shape: {spectrogram.shape}, dtype: {spectrogram.dtype}")

    if args.benchmark:
        inference.benchmark(spectrogram)
    else:
        # Run detection
        detections = inference.detect(spectrogram)

        print(f"\nDetected {len(detections)} burst(s):")
        for i, (box, score) in enumerate(detections):
            print(
                f"  [{i}] score={score:.3f} box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]"
            )

        if args.plot:
            visualize_predictions(spectrogram, detections, vmin_db, vmax_db)


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.perf_counter()
        result = f(*args, **kw)
        te = time.perf_counter()
        elapsed = (te - ts) * 1000
        print("func:%r took: %2.4f ms" % (f.__name__, elapsed))
        return result

    return wrap


if __name__ == "__main__":
    main()
