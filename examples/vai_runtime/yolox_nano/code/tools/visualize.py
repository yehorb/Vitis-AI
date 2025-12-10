#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Unified STFT Spectrogram Visualization Tool.

Visualize YOLOX detection results on STFT spectrograms with support for:
- Ground truth boxes only (dataset exploration)
- Model predictions with ground truth (inference evaluation)
- Raw dB or normalized [0,1] display modes

This tool replaces the older demo_stft.py and visualize_eval.py scripts.

Usage Examples:
    # Visualize dataset samples with GT only (no model)
    python -m tools.visualize \\
        -f exps/example/custom/yolox_nano_stft_relu.py \\
        --split val \\
        --num-samples 5

    # Visualize with model predictions
    python -m tools.visualize \\
        -f exps/example/custom/yolox_nano_stft_relu.py \\
        -c YOLOX_outputs/yolox_nano_stft_relu/best_ckpt.pth \\
        --split val \\
        --num-samples 5 \\
        --conf 0.25

    # Single tile by ID with raw dB display
    python -m tools.visualize \\
        -f exps/example/custom/yolox_nano_stft_relu.py \\
        -c YOLOX_outputs/yolox_nano_stft_relu/best_ckpt.pth \\
        --tile-id rec_20251207_162413_002 \\
        --display raw

    # Save visualizations to directory
    python -m tools.visualize \\
        -f exps/example/custom/yolox_nano_stft_relu.py \\
        -c YOLOX_outputs/yolox_nano_stft_relu/best_ckpt.pth \\
        --split val \\
        --num-samples 10 \\
        --save ./outputs/
"""

from __future__ import annotations

import argparse
import os
import sys
import typing as t
from pathlib import Path

import numpy as np
import torch
from stft_dataset import LoadSplit, Matlab, StftDataset
from stft_dataset.normalization import load_normalization_params, Normalize
from stft_dataset.vis import labels_to_xyxy, visualize_detections
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess

if t.TYPE_CHECKING:
    import numpy.typing as npt
    from torch import nn

    # (box_xyxy, score)
    DetectionResult = t.Tuple[npt.NDArray[np.float32], float]


def make_parser() -> argparse.ArgumentParser:
    """Create argument parser for visualization tool."""
    parser = argparse.ArgumentParser(
        description="Unified STFT spectrogram visualization tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required: experiment file
    parser.add_argument(
        "-f",
        "--exp-file",
        required=True,
        type=str,
        help="Experiment description file (e.g., exps/.../yolox_nano_stft.py)",
    )

    # Optional: checkpoint for model inference
    parser.add_argument(
        "-c",
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint file path. If not provided, only GT boxes are shown.",
    )
    parser.add_argument(
        "-q",
        "--quant",
        action="store_true",
        help="Load quantized model (sets W_QUANT=1)",
    )

    # Data source options
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val"],
        help="Dataset split to use (default: val)",
    )
    parser.add_argument(
        "--num-samples",
        default=5,
        type=int,
        help="Number of samples to visualize (default: 5)",
    )
    parser.add_argument(
        "--tile-id",
        type=str,
        default=None,
        help="Specific tile ID to visualize (overrides --num-samples and --split)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Specific index in the split to visualize (overrides --num-samples)",
    )

    # Display options
    parser.add_argument(
        "--display",
        default="normalized",
        choices=["raw", "normalized"],
        help="Display mode: 'raw' shows dB values, 'normalized' shows [0,1] range (default: normalized)",
    )
    parser.add_argument(
        "--cmap",
        default="magma",
        type=str,
        help="Matplotlib colormap (default: magma)",
    )

    # Inference options
    parser.add_argument(
        "--conf",
        default=0.25,
        type=float,
        help="Confidence threshold for predictions (default: 0.25)",
    )
    parser.add_argument(
        "--nms",
        default=0.45,
        type=float,
        help="NMS threshold (default: 0.45)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on (default: cuda)",
    )
    parser.add_argument(
        "--fuse",
        action="store_true",
        help="Fuse conv and bn layers for faster inference",
    )

    # Output options
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Directory to save visualizations (default: display only)",
    )

    return parser


def load_model(
    exp: t.Any,
    ckpt_path: str,
    device: str,
    fuse: bool = False,
    quant: bool = False,
) -> nn.Module:
    """
    Load trained model from checkpoint.

    Parameters
    ----------
    exp :
        Experiment configuration object.
    ckpt_path : str
        Path to checkpoint file.
    device : str
        Device to load model on ('cuda' or 'cpu').
    fuse : bool
        Whether to fuse conv and bn layers.

    Returns
    -------
    model : nn.Module
        Loaded model in eval mode.
    """
    if quant:
        os.environ["W_QUANT"] = "1"
    model = exp.get_model()

    if device == "cuda" and torch.cuda.is_available():
        model.cuda()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    if fuse:
        model = fuse_model(model)

    model.eval()
    return model


def run_inference(
    model: nn.Module,
    img: npt.NDArray[np.float32],
    num_classes: int,
    conf_thre: float,
    nms_thre: float,
    device: str,
) -> t.List[DetectionResult]:
    """
    Run inference on a single spectrogram.

    Parameters
    ----------
    model : nn.Module
        YOLOX model.
    img : ndarray
        Normalized spectrogram [1, H, W] in dB scale.
        Should be sourced directly from Normalized(StftDataset())
    vmin_db : float
        Minimum dB value for normalization.
    vmax_db : float
        Maximum dB value for normalization.
    num_classes : int
        Number of detection classes.
    conf_thre : float
        Confidence threshold.
    nms_thre : float
        NMS threshold.
    device : str
        Device to run on.

    Returns
    -------
    detections : list of (box, score) tuples
        Each box is [4] xyxy format, score is float.
    """

    # Add dimension for batch size (of 1)
    # img_tensor.shape ~ [1,          1,        W, H]
    #                    [batch_size, channels, W, H]
    img_tensor = torch.from_numpy(img).unsqueeze(0)

    if device == "cuda" and torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    with torch.no_grad():
        outputs = model(img_tensor)
        outputs = postprocess(
            outputs,
            num_classes=num_classes,
            conf_thre=conf_thre,
            nms_thre=nms_thre,
            class_agnostic=True,
        )

    # Extract detections
    detections: t.List[DetectionResult] = []
    result = outputs[0]
    if result is not None:
        for det in result:  # type: ignore[union-attr]
            box = det[:4].cpu().numpy()
            score = float(det[4].cpu() * det[5].cpu())  # obj_conf * cls_conf
            detections.append((box, score))

    return detections


def main() -> None:
    args = make_parser().parse_args()

    # Load experiment config
    exp = get_exp(args.exp_file, None)

    # Get paths from experiment
    h5_path = Path(exp.h5_path)
    meta_path = Path(exp.meta_path)
    train_split = Path(exp.train_split)
    val_split = Path(exp.val_split)

    # Load normalization params
    vmin_db, vmax_db = load_normalization_params(meta_path)
    print(f"Normalization: vmin={vmin_db:.1f} dB, vmax={vmax_db:.1f} dB")

    # Build dataset based on source type
    if args.tile_id:
        # Direct tile ID access
        dataset = StftDataset(Matlab(h5_path, [args.tile_id]))
        indices: t.List[int] = [0]
        print(f"Loading specific tile: {args.tile_id}")
    else:
        # Split file access
        split_file = train_split if args.split == "train" else val_split
        dataset = StftDataset(LoadSplit(Matlab(h5_path, []), split_file))

        if args.index is not None:
            # Specific index
            if args.index < 0 or args.index >= len(dataset):
                print(
                    f"Error: Index {args.index} out of range. "
                    f"Dataset has {len(dataset)} tiles."
                )
                sys.exit(1)
            indices = [args.index]
        else:
            # Multiple samples
            indices = list(range(min(args.num_samples, len(dataset))))

        print(f"Dataset: {len(dataset)} samples from {args.split} split")
    normalized = Normalize(dataset, vmin_db, vmax_db)

    # Load model if checkpoint provided
    model: t.Optional[nn.Module] = None
    if args.ckpt:
        print(f"Loading model from: {args.ckpt}")
        model = load_model(exp, args.ckpt, args.device, args.fuse, args.quant)
        print("Model loaded successfully")
    else:
        print("No checkpoint provided - showing ground truth only")

    # Create save directory if needed
    save_dir: t.Optional[Path] = None
    if args.save:
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving visualizations to: {save_dir}")

    # Process each tile
    for idx in indices:
        # Load tile: img is [1, H, W] in dB, labels is [N, 5]
        raw_img, labels, tile_id = dataset[idx]
        norm_img = normalized._normalize(raw_img)

        # Convert labels to xyxy format for visualization
        gt_boxes = labels_to_xyxy(labels)

        print(f"\nTile: {tile_id}")
        print(f"  Shape: {raw_img.shape}")
        print(f"  Value range: [{raw_img.min():.1f}, {raw_img.max():.1f}] dB")
        print(f"  Ground truth boxes: {len(gt_boxes)}")

        # Run inference if model is available
        predictions: t.Optional[t.List[DetectionResult]] = None
        if model is not None:
            predictions = run_inference(
                model,
                norm_img,
                exp.num_classes,
                args.conf,
                args.nms,
                args.device,
            )
            print(f"  Predictions: {len(predictions)}")

        # Build title
        n_gt = len(gt_boxes)
        n_pred = len(predictions) if predictions else 0
        if model is not None:
            title = f"{tile_id} | GT: {n_gt} | Pred: {n_pred} (conf>{args.conf:.2f})"
        else:
            title = f"{tile_id} | GT: {n_gt}"

        # Determine save path
        save_path: t.Optional[str] = None
        if save_dir is not None:
            suffix = f"_{args.display}"
            save_path = str(save_dir / f"{tile_id}{suffix}.png")

        if args.display == "raw":
            spectrogram = raw_img[0]
        elif args.display == "normalized":
            spectrogram = norm_img[0]
        else:
            raise ValueError(f"Invalid display mode: {args.display}")

        # Visualize
        visualize_detections(
            spectrogram=spectrogram,
            gt_boxes=gt_boxes if len(gt_boxes) > 0 else None,
            predictions=predictions,
            display_mode=args.display,  # type: ignore[arg-type]
            vmin_db=vmin_db,
            vmax_db=vmax_db,
            title=title,
            cmap=args.cmap,
            save_path=save_path,
        )


if __name__ == "__main__":
    main()
