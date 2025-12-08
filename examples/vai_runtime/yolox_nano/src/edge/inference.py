#!/usr/bin/env python3
"""
YOLOX STFT Burst Detection - KV260 Edge Inference

Minimal reference implementation for running quantized YOLOX on Vitis AI DPU.
Requires: vart, xir (installed on KV260 Vitis AI image)

Usage:
    python inference.py --model yolox_stft_kv260.xmodel --input spectrogram.npy
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

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


class DpuRunner:
    """
    Wrapper for VART DPU runner with buffer management.

    Handles:
    - Model loading and subgraph extraction
    - Input/output buffer allocation
    - Quantization scale factors (fix-points)
    - Synchronous inference execution
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

        # Extract shapes (VART uses NHWC format)
        self.input_shape = tuple(self.input_tensors[0].dims)
        self.output_shape = tuple(self.output_tensors[0].dims)

        # Extract quantization fix-points
        # Input: multiply by 2^fix_point before int8 conversion
        # Output: multiply by 2^(-fix_point) for dequantization
        self.input_fixpoint = self.input_tensors[0].get_attr("fix_point")
        self.output_fixpoint = self.output_tensors[0].get_attr("fix_point")

        self.input_scale = float(2**self.input_fixpoint)
        self.output_scale = float(2 ** (-self.output_fixpoint))

        # Pre-allocate buffers (C-contiguous for VART)
        self.input_buffer = [np.empty(self.input_shape, dtype=np.int8, order="C")]
        self.output_buffer = [np.empty(self.output_shape, dtype=np.int8, order="C")]

    def run(self, input_float: np.ndarray) -> np.ndarray:
        """
        Execute inference on DPU.

        Args:
            input_float: Preprocessed input, float32, shape matching model input
                        Values should be normalized to [0, 1] range

        Returns:
            Dequantized output as float32 numpy array
        """
        # Quantize input: scale and convert to int8
        input_scaled = input_float * self.input_scale
        input_int8 = np.clip(input_scaled, -128, 127).astype(np.int8)

        # Copy to buffer (must preserve C-contiguous layout)
        np.copyto(self.input_buffer[0], input_int8.reshape(self.input_shape))

        # Execute on DPU (async API, but we wait immediately)
        job_id = self.runner.execute_async(self.input_buffer, self.output_buffer)
        self.runner.wait(job_id)

        # Dequantize output
        output_float = self.output_buffer[0].astype(np.float32) * self.output_scale

        return output_float


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


def build_grid(
    input_size: int,
    strides: Tuple[int, ...],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build YOLOX anchor grid for decoding.

    Args:
        input_size: Input image size (assumes square)
        strides: Detection head strides (e.g., [8, 16, 32])

    Returns:
        grids: Grid coordinates, shape (1, total_anchors, 2)
        expanded_strides: Stride per anchor, shape (1, total_anchors, 1)
    """
    grids = []
    expanded_strides = []

    for stride in strides:
        grid_size = input_size // stride
        yv, xv = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
        grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2).astype(np.float32)
        grids.append(grid)
        expanded_strides.append(
            np.full((1, grid_size * grid_size, 1), stride, dtype=np.float32)
        )

    return np.concatenate(grids, axis=1), np.concatenate(expanded_strides, axis=1)


def decode_yolox_output(
    output: np.ndarray,
    config: ModelConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode raw YOLOX output to bounding boxes and scores.

    Args:
        output: Raw model output, shape (1, num_anchors, 5 + num_classes)
                Format per anchor: [x, y, w, h, objectness, class_scores...]
        config: Model configuration

    Returns:
        boxes: Decoded boxes in xyxy format, shape (num_anchors, 4)
        scores: Confidence scores, shape (num_anchors,)
    """
    # Build anchor grid
    grids, strides = build_grid(config.input_height, config.strides)

    # Extract components
    # Note: output layout is [x_offset, y_offset, w, h, obj_conf, cls_conf...]
    xy = output[..., :2]
    wh = output[..., 2:4]
    obj_conf = output[..., 4:5]
    cls_conf = output[..., 5 : 5 + config.num_classes]

    # Decode box coordinates
    # x_center = (x_offset + grid_x) * stride
    # y_center = (y_offset + grid_y) * stride
    # width = exp(w) * stride
    # height = exp(h) * stride
    xy_decoded = (xy + grids) * strides
    wh_decoded = np.exp(wh) * strides

    # Apply sigmoid to confidence scores
    obj_conf = sigmoid(obj_conf)
    cls_conf = sigmoid(cls_conf)

    # Final score = objectness * class_confidence
    scores = (obj_conf * cls_conf).squeeze(-1).squeeze(0)  # (num_anchors,)

    # Convert cxcywh to xyxy
    boxes = np.zeros((output.shape[1], 4), dtype=np.float32)
    boxes[:, 0] = xy_decoded[0, :, 0] - wh_decoded[0, :, 0] / 2  # x1
    boxes[:, 1] = xy_decoded[0, :, 1] - wh_decoded[0, :, 1] / 2  # y1
    boxes[:, 2] = xy_decoded[0, :, 0] + wh_decoded[0, :, 0] / 2  # x2
    boxes[:, 3] = xy_decoded[0, :, 1] + wh_decoded[0, :, 1] / 2  # y2

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

        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        # Keep boxes with IoU below threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)


def postprocess(
    output: np.ndarray,
    config: ModelConfig,
) -> List[Tuple[np.ndarray, float]]:
    """
    Full postprocessing pipeline: decode, filter, NMS.

    Args:
        output: Raw model output from DPU
        config: Model configuration

    Returns:
        List of (box, score) tuples for detected bursts
        Each box is [x1, y1, x2, y2] in pixel coordinates
    """
    # Decode all anchors
    boxes, scores = decode_yolox_output(output, config)

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
        config: ModelConfig = ModelConfig(),
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
        self.config = config

        print(f"Model loaded: {xmodel_path}")
        print(f"  Input shape: {self.runner.input_shape}")
        print(f"  Output shape: {self.runner.output_shape}")
        print(f"  Input fix-point: {self.runner.input_fixpoint}")
        print(f"  Output fix-point: {self.runner.output_fixpoint}")

    def detect(self, spectrogram: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        Run detection on a single spectrogram.

        Args:
            spectrogram: STFT spectrogram in dB, shape (128, 128)

        Returns:
            List of (box, score) detections
        """
        # Preprocess
        input_tensor = preprocess_spectrogram(
            spectrogram,
            self.vmin_db,
            self.vmax_db,
            (self.config.input_height, self.config.input_width),
        )

        # Run DPU inference
        output = self.runner.run(input_tensor)

        # Postprocess
        detections = postprocess(output, self.config)

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
        default=-80.0,
        help="Min dB for normalization (default: -80.0)",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=0.0,
        help="Max dB for normalization (default: 0.0)",
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

    args = parser.parse_args()

    # Load normalization params from meta.json if provided
    vmin_db, vmax_db = args.vmin, args.vmax
    if args.meta:
        with open(args.meta) as f:
            meta = json.load(f)
        vmin_db = meta.get("vmin_db", vmin_db)
        vmax_db = meta.get("vmax_db", vmax_db)
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


if __name__ == "__main__":
    main()
