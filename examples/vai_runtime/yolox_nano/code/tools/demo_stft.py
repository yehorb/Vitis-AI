#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
STFT spectrogram detection demo.

Loads a trained YOLOX model and runs inference on STFT tiles from the dataset.
Visualizes predictions (red) alongside ground truth boxes (green).

Usage:
    python -m yolox.tools.demo_stft \
        -f exps/example/custom/yolox_nano_stft_relu.py \
        -c YOLOX_outputs/yolox_nano_stft_relu/best_ckpt.pth \
        --split val \
        --num-samples 5 \
        --conf 0.25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess


def make_parser():
    parser = argparse.ArgumentParser("STFT Detection Demo")
    parser.add_argument(
        "-f", "--exp_file", required=True, type=str, help="Experiment description file"
    )
    parser.add_argument(
        "-c", "--ckpt", required=True, type=str, help="Checkpoint file path"
    )
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
        "--conf",
        default=0.25,
        type=float,
        help="Confidence threshold for predictions (default: 0.25)",
    )
    parser.add_argument(
        "--nms", default=0.45, type=float, help="NMS threshold (default: 0.45)"
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
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Directory to save visualizations (default: display only)",
    )
    parser.add_argument(
        "--tile-id",
        type=str,
        default=None,
        help="Specific tile ID to visualize (overrides --num-samples)",
    )
    return parser


def load_model(exp, ckpt_path: str, device: str, fuse: bool = False):
    """Load trained model from checkpoint."""
    model = exp.get_model()

    if device == "cuda" and torch.cuda.is_available():
        model.cuda()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    if fuse:
        model = fuse_model(model)

    model.eval()
    return model


def run_inference(
    model, img: np.ndarray, exp, conf_thre: float, nms_thre: float, device: str
):
    """
    Run inference on a single image.

    Parameters
    ----------
    model : YOLOX model
    img : np.ndarray
        Image tensor [1, H, W] float32, normalized to [0, 1]
    exp : Experiment config
    conf_thre : float
        Confidence threshold
    nms_thre : float
        NMS threshold
    device : str
        'cuda' or 'cpu'

    Returns
    -------
    predictions : np.ndarray or None
        [N, 7] array: x1, y1, x2, y2, obj_conf, cls_conf, cls_id
        Returns None if no detections
    """
    # Add batch dimension: [1, H, W] -> [1, 1, H, W]
    img_tensor = torch.from_numpy(img).unsqueeze(0).float()

    if device == "cuda" and torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    with torch.no_grad():
        outputs = model(img_tensor)
        outputs = postprocess(
            outputs,
            num_classes=exp.num_classes,
            conf_thre=conf_thre,
            nms_thre=nms_thre,
            class_agnostic=True,
        )

    if outputs[0] is None:
        return None

    return outputs[0].cpu().numpy()


def visualize_predictions(
    img: np.ndarray,
    gt_boxes: np.ndarray,
    pred_boxes: Optional[np.ndarray],
    tile_id: str,
    conf_thre: float,
    save_path: Optional[str] = None,
):
    """
    Visualize STFT tile with GT boxes (green) and predictions (red).

    Parameters
    ----------
    img : np.ndarray
        Image [1, H, W] float32
    gt_boxes : np.ndarray
        Ground truth boxes [N, 5] as (class_id, cx, cy, w, h)
    pred_boxes : np.ndarray or None
        Predictions [M, 7] as (x1, y1, x2, y2, obj_conf, cls_conf, cls_id)
    tile_id : str
        Tile identifier for title
    conf_thre : float
        Confidence threshold used
    save_path : str or None
        If provided, save figure to this path instead of displaying
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # Display spectrogram
    im = ax.imshow(
        img.squeeze(), origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1
    )

    # Draw ground truth boxes (green)
    n_gt = 0
    for box in gt_boxes:
        if box.sum() == 0:  # Skip zero-padded boxes
            continue
        cls_id, cx, cy, w, h = box
        x1 = cx - w / 2
        y1 = cy - h / 2
        rect = patches.Rectangle(
            (x1, y1),
            w,
            h,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
            label="GT" if n_gt == 0 else None,
        )
        ax.add_patch(rect)
        n_gt += 1

    # Draw predictions (red)
    n_pred = 0
    if pred_boxes is not None:
        for box in pred_boxes:
            x1, y1, x2, y2, obj_conf, cls_conf, cls_id = box
            score = obj_conf * cls_conf
            w = x2 - x1
            h = y2 - y1
            rect = patches.Rectangle(
                (x1, y1),
                w,
                h,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                linestyle="--",
                label="Pred" if n_pred == 0 else None,
            )
            ax.add_patch(rect)
            # Add confidence label
            ax.text(
                x1,
                y1 - 2,
                f"{score:.2f}",
                color="red",
                fontsize=8,
                verticalalignment="bottom",
            )
            n_pred += 1

    ax.set_title(
        f"Tile: {tile_id} | GT: {n_gt} | Pred: {n_pred} (conf>{conf_thre:.2f})"
    )
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Frequency bin")
    ax.legend(loc="upper right")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized magnitude")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()


def main():
    args = make_parser().parse_args()

    # Load experiment config
    exp = get_exp(args.exp_file, None)
    exp.test_conf = args.conf
    exp.nmsthre = args.nms

    print(f"Loading model from: {args.ckpt}")
    model = load_model(exp, args.ckpt, args.device, args.fuse)
    print("Model loaded successfully")

    # Import dataset components
    from stft_dataset import LoadSplit, Matlab, StftDataset
    from stft_dataset.normalization import Normalize, load_normalization_params

    # Determine split file
    split_file = exp.train_split if args.split == "train" else exp.val_split

    # Load normalization params
    vmin_db, vmax_db = load_normalization_params(exp.meta_path)

    # Build dataset pipeline
    dataset = Normalize(
        StftDataset(LoadSplit(Matlab(exp.h5_path, []), split_file)),
        vmin_db,
        vmax_db,
    )

    print(f"Dataset: {len(dataset)} samples from {args.split} split")

    # Create save directory if needed
    if args.save:
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Determine which samples to visualize
    if args.tile_id:
        # Find specific tile
        indices = []
        for i in range(len(dataset)):
            _, _, tid = dataset[i]
            if tid == args.tile_id:
                indices = [i]
                break
        if not indices:
            print(f"Tile ID '{args.tile_id}' not found in dataset")
            sys.exit(1)
    else:
        # Use first N samples
        indices = list(range(min(args.num_samples, len(dataset))))

    # Run inference and visualize
    for idx in indices:
        img, gt_boxes, tile_id = dataset[idx]

        print(f"\nProcessing tile: {tile_id}")
        print(f"  Image shape: {img.shape}")
        print(f"  GT boxes: {(gt_boxes.sum(axis=1) != 0).sum()}")

        # Run inference
        predictions = run_inference(model, img, exp, args.conf, args.nms, args.device)

        if predictions is not None:
            print(f"  Predictions: {len(predictions)}")
        else:
            print(f"  Predictions: 0")

        # Visualize
        save_path = None
        if args.save:
            save_path = str(save_dir / f"{tile_id}.png")

        visualize_predictions(img, gt_boxes, predictions, tile_id, args.conf, save_path)


if __name__ == "__main__":
    main()
