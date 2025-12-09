#!/usr/bin/env python3
"""
Visualize Host Evaluation Results

Runs inference on a specific tile using the trained float model with the same
postprocessing chain as evaluator.py, then visualizes the results.

Use this to manually compare with edge (DPU) inference results.

Host postprocessing chain (from evaluator.py):
    1. model(imgs) -> raw outputs (list of 3 tensors)
    2. float_model.head.postprocess() -> concat + sigmoid + decode
    3. yolox.utils.postprocess() -> NMS and filtering

Usage:
    # By split + index (same as export_tile.py):
    python tools/visualize_eval.py \
        --exp exps/example/custom/yolox_nano_stft_relu.py \
        --ckpt YOLOX_outputs/yolox_nano_stft_relu/best_ckpt.pth \
        --h5 ../data/stft/20251207_162413/tensors/tiles.h5 \
        --split ../data/stft/20251207_162413/splits/val.txt \
        --index 42

    # By tile ID:
    python tools/visualize_eval.py \
        --exp exps/example/custom/yolox_nano_stft_relu.py \
        --ckpt YOLOX_outputs/yolox_nano_stft_relu/best_ckpt.pth \
        --h5 ../data/stft/20251207_162413/tensors/tiles.h5 \
        --tile-id rec_20251207_162413_002
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from stft_dataset import LoadSplit, Matlab, StftDataset
from yolox.exp.build import get_exp
from yolox.utils import postprocess as yolox_postprocess


def load_tile(
    h5_path: Path,
    split_path: Optional[Path],
    tile_id: Optional[str],
    index: int,
) -> Tuple[np.ndarray, List[np.ndarray], str]:
    """
    Load spectrogram and labels from STFT dataset using stft_dataset tools.

    Uses the same loading mechanism as export_tile.py.

    Returns:
        spectrogram: Shape (H, W), float32, dB scale
        ground_truth: List of boxes in xyxy pixel format
        tile_id: Tile identifier string
    """

    # Build dataset based on source type
    if tile_id:
        # Direct tile ID access
        dataset = StftDataset(Matlab(h5_path, [tile_id]))
        idx = 0
    else:
        # Split file access
        assert split_path is not None
        dataset = StftDataset(LoadSplit(Matlab(h5_path, []), split_path))
        idx = index

        if idx < 0 or idx >= len(dataset):
            raise ValueError(
                f"Index {idx} out of range. Dataset has {len(dataset)} tiles."
            )

    # Get the tile: img is [1, H, W], labels is [N, 5] with (cls, cx, cy, w, h)
    img, labels, tile_id_out = dataset[idx]

    # Remove channel dimension: [1, H, W] -> [H, W]
    spectrogram = img[0]

    # Convert labels from cxcywh (pixel coords) to xyxy
    ground_truth = []
    for label in labels:
        _, cx, cy, w, h = label
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        ground_truth.append(np.array([x1, y1, x2, y2]))

    return spectrogram, ground_truth, tile_id_out


def load_normalization(h5_path: Path) -> Tuple[float, float]:
    """
    Load normalization params from meta.json in dataset root.

    Returns:
        vmin, vmax in dB
    """
    # meta.json is typically at dataset_root/meta.json
    # h5_path is typically dataset_root/tensors/tiles.h5
    dataset_root = h5_path.parent.parent
    meta_path = dataset_root / "meta.json"

    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        vmin = meta.get("render", {}).get("vmin_db", -90.0)
        vmax = meta.get("render", {}).get("vmax_db", -20.0)
        return vmin, vmax

    # Fallback defaults
    return -90.0, -20.0


def load_model(exp_path: str, ckpt_path: str, device: str = "cuda"):
    """Load quantized model."""
    exp = get_exp(exp_path, None)
    model = exp.get_model()
    model.eval()

    ckpt = torch.load(ckpt_path, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    return model, exp


def run_inference(
    model: torch.nn.Module,
    spectrogram: np.ndarray,
    vmin: float,
    vmax: float,
    conf_threshold: float,
    nms_threshold: float,
    device: str = "cuda",
) -> List[Tuple[np.ndarray, float]]:
    """
    Run inference using host postprocessing chain (same as evaluator.py).

    Returns:
        List of (box_xyxy, score) tuples
    """
    # Normalize to [0, 1]
    normalized = (spectrogram - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)

    # To tensor: (1, 1, H, W)
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        # Step 1: model forward (returns list of 3 raw tensors)
        raw_outputs = model(tensor)

        # Step 2: head.postprocess() - concat + sigmoid + decode
        # This is what evaluator.py does via params.postprocess()
        decoded = model.head.postprocess(raw_outputs)

        # Step 3: yolox.utils.postprocess() - NMS and filtering
        # Returns list of tensors, one per batch image
        # Each tensor is [N, 7]: x1, y1, x2, y2, obj_conf, cls_conf, cls_id
        results = yolox_postprocess(
            decoded,
            num_classes=1,
            conf_thre=conf_threshold,
            nms_thre=nms_threshold,
            class_agnostic=True,
        )

    # Extract detections
    detections = []
    if results[0] is not None:
        for det in results[0]:
            box = det[:4].cpu().numpy()
            score = float(det[4].cpu() * det[5].cpu())  # obj_conf * cls_conf
            detections.append((box, score))

    return detections


def visualize(
    spectrogram: np.ndarray,
    detections: List[Tuple[np.ndarray, float]],
    ground_truth: Optional[List[np.ndarray]],
    vmin: float,
    vmax: float,
    title: str,
    save_path: Optional[str] = None,
):
    """Visualize spectrogram with predictions and ground truth."""
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        spectrogram,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )

    # Draw ground truth boxes (cyan, dashed)
    if ground_truth:
        for i, gt_box in enumerate(ground_truth):
            x1, y1, x2, y2 = gt_box
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="cyan",
                facecolor="none",
                linestyle="--",
                label="Ground Truth" if i == 0 else None,
            )
            ax.add_patch(rect)

    # Draw predicted boxes (lime, solid)
    for i, (box, score) in enumerate(detections):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
            label="Prediction" if i == 0 else None,
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            y2 + 2,
            f"{score:.2f}",
            color="lime",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Frequency bin")
    ax.set_title(title)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Magnitude (dB)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize host evaluation results for comparison with edge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--exp", "-e", required=True, help="Experiment file path")
    parser.add_argument("--ckpt", "-c", required=True, help="Quantized checkpoint path")
    parser.add_argument(
        "--h5",
        required=True,
        type=Path,
        help="Path to tiles.h5 file",
    )

    # Mutually exclusive: either (--split + optional --index) or --tile-id
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--split",
        type=Path,
        help="Path to split file (e.g., val.txt). Use with --index.",
    )
    source_group.add_argument(
        "--tile-id",
        type=str,
        help="Specific tile ID to export (e.g., rec_20251207_162413_002)",
    )

    parser.add_argument(
        "--index", "-i", type=int, default=0, help="Tile index (with --split)"
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold")
    parser.add_argument("--save", "-s", help="Save to file instead of showing")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")

    args = parser.parse_args()

    # Load tile using stft_dataset tools
    spectrogram, ground_truth, tile_id = load_tile(
        args.h5, args.split, args.tile_id, args.index
    )
    vmin, vmax = load_normalization(args.h5)

    print(f"Loaded tile: {tile_id}")
    print(f"  Shape: {spectrogram.shape}")
    print(f"  Value range: [{spectrogram.min():.1f}, {spectrogram.max():.1f}] dB")
    print(f"  Normalization: vmin={vmin}, vmax={vmax}")
    print(f"  Ground truth boxes: {len(ground_truth)}")

    # Load model
    # Ensure we use quantized model components
    os.environ["W_QUANT"] = "1"
    model, exp = load_model(args.exp, args.ckpt, args.device)
    print(f"Loaded model from {args.ckpt}")

    # Run inference
    detections = run_inference(
        model, spectrogram, vmin, vmax, args.conf, args.nms, args.device
    )

    # Print results
    print(f"\nDetected {len(detections)} burst(s):")
    for i, (box, score) in enumerate(detections):
        print(
            f"  [{i}] score={score:.3f} box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]"
        )

    # Visualize
    title = f"Host Evaluation: {tile_id} (conf={args.conf})"
    visualize(spectrogram, detections, ground_truth, vmin, vmax, title, args.save)


if __name__ == "__main__":
    main()
