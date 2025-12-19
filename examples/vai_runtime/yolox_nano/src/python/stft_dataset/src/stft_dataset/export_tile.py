#!/usr/bin/env python3
"""
Export a spectrogram tile from the STFT dataset for edge inference testing.

Usage:
    # Export from split file by index:
    stft-export-tile \
        --h5 data/stft/20251207_162413/tensors/tiles.h5 \
        --split data/stft/20251207_162413/splits/val.txt \
        --index 0 \
        --output test_spectrogram.npy

    # Export by specific tile ID:
    stft-export-tile \
        --h5 data/stft/20251207_162413/tensors/tiles.h5 \
        --tile-id rec_20251207_162413_002 \
        --output test_spectrogram.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stft_dataset import LoadSplit, Matlab, StftDataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a spectrogram tile for edge inference testing"
    )
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

    count_group = parser.add_mutually_exclusive_group(required=True)
    count_group.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of tile to export when using --split (default: 0)",
    )
    count_group.add_argument(
        "--all",
        action="store_true",
        help="Export all tiles in the split when using --split",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("test_spectrogram.npy"),
        help="Output .npy file path (default: test_spectrogram.npy)",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="Optional: path to meta.json to include normalization info in output",
    )
    parser.add_argument(
        "--with-labels",
        action="store_true",
        help="Also export ground truth labels as <output>_labels.npy",
    )

    args = parser.parse_args()

    # Build dataset based on source type
    if args.tile_id:
        # Direct tile ID access
        dataset = StftDataset(Matlab(args.h5, [args.tile_id]))
        index = 0
    else:
        # Split file access
        dataset = StftDataset(LoadSplit(Matlab(args.h5, []), args.split))
        index = args.index

        # Validate index
        if index < 0 or index >= len(dataset):
            raise ValueError(
                f"Index {index} out of range. Dataset has {len(dataset)} tiles."
            )

    if not args.all:
        # Get the tile
        img, labels, tile_id = dataset[index]

        # img shape is [1, H, W] from StftDataset, we want [H, W] for edge inference
        spectrogram = img[0]  # Remove channel dimension
    else:
        img, _, _ = dataset[0]
        _, H, W = img.shape
        num_tiles = len(dataset)
        spectrogram = np.zeros((num_tiles, H, W), dtype=np.float32)
        for i in range(num_tiles):
            img, _, _ = dataset[i]
            spectrogram[i, :] = img[0]
        tile_id = "all"

    print(f"Tile ID: {tile_id}")
    print(f"Shape: {spectrogram.shape}")
    print(f"Dtype: {spectrogram.dtype}")
    print(f"Value range: [{spectrogram.min():.2f}, {spectrogram.max():.2f}] dB")

    # Save spectrogram
    np.save(args.output, spectrogram)
    print(f"Saved spectrogram: {args.output}")

    # Optionally save labels
    if args.all:
        print("--all is not compatible with --with-labels")
        print("labels will not be saved")
    if args.with_labels and not args.all:
        labels_path = args.output.with_name(args.output.stem + "_labels.npy")
        np.save(labels_path, labels)
        print(f"Saved labels: {labels_path}")

        # Print ground truth info
        n_boxes = len(labels)
        print(f"\nGround truth: {n_boxes} box(es)")
        for i, label in enumerate(labels):
            cls_id, cx, cy, w, h = label
            # Convert back to x0, y0, x1, y1 for readability
            x0, y0 = cx - w / 2, cy - h / 2
            x1, y1 = cx + w / 2, cy + h / 2
            print(
                f"  [{i}] class={int(cls_id)} box=[{x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}]"
            )

    # Show normalization info if meta.json provided
    if args.meta:
        with open(args.meta) as f:
            meta = json.load(f)
        vmin_db = meta.get("render", {}).get("vmin_db", "N/A")
        vmax_db = meta.get("render", {}).get("vmax_db", "N/A")
        print(f"\nNormalization (from {args.meta}):")
        print(f"  vmin_db: {vmin_db}")
        print(f"  vmax_db: {vmax_db}")


if __name__ == "__main__":
    main()
