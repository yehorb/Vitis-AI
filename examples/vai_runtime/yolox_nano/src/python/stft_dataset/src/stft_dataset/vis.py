from __future__ import annotations

import typing as t

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

if t.TYPE_CHECKING:
    from matplotlib.figure import Figure
    import numpy.typing as npt

    # (box_xyxy, score)
    DetectionResult = t.Tuple[npt.NDArray[np.float32], float]


def visualize_detections(
    spectrogram: npt.NDArray[np.floating[t.Any]],
    gt_boxes: t.Optional[npt.NDArray[np.floating[t.Any]]] = None,
    predictions: t.Optional[t.List[DetectionResult]] = None,
    display_mode: t.Literal["raw", "normalized"] = "normalized",
    vmin_db: float = -90.0,
    vmax_db: float = -20.0,
    title: str = "",
    cmap: str = "magma",
    save_path: t.Optional[str] = None,
) -> Figure:
    """
    Visualize STFT spectrogram with ground truth and predicted bounding boxes.

    Supports two display modes:
    - "raw": Display spectrogram in dB scale with original values
    - "normalized": Display spectrogram normalized to [0, 1] range

    Parameters
    ----------
    spectrogram : ndarray
        2D spectrogram array [H, W]. Can be raw dB values or normalized [0,1].
    gt_boxes : ndarray, optional
        Ground truth boxes [N, 4] in xyxy pixel format.
    predictions : list of (box, score) tuples, optional
        Predicted detections where box is [4] xyxy and score is float.
    display_mode : {"raw", "normalized"}
        How to display the spectrogram:
        - "raw": Use vmin_db/vmax_db as color limits, show dB values
        - "normalized": Normalize to [0,1] range for display
    vmin_db : float
        Minimum dB value for normalization/display (default: -90.0).
    vmax_db : float
        Maximum dB value for normalization/display (default: -20.0).
    title : str
        Title for the plot.
    cmap : str
        Matplotlib colormap name (default: "magma").
    save_path : str, optional
        If provided, save figure to this path instead of displaying.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    assert isinstance(ax, plt.Axes)

    # Prepare display data based on mode
    if display_mode == "normalized":
        # Normalize to [0, 1]
        display_data = (spectrogram - vmin_db) / (vmax_db - vmin_db)
        display_data = np.clip(display_data, 0.0, 1.0)
        vmin_display, vmax_display = 0.0, 1.0
        colorbar_label = "Normalized magnitude"
    else:  # raw
        display_data = spectrogram
        vmin_display, vmax_display = vmin_db, vmax_db
        colorbar_label = "Magnitude (dB)"

    # Display spectrogram
    im = ax.imshow(
        display_data,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin_display,
        vmax=vmax_display,
    )

    n_gt = 0
    n_pred = 0

    # Draw ground truth boxes (lime, solid)
    if gt_boxes is not None:
        for i, box in enumerate(gt_boxes):
            if len(box) < 4:
                continue
            x1, y1, x2, y2 = box[:4]
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
                label="Ground Truth" if i == 0 else None,
            )
            ax.add_patch(rect)
            n_gt += 1

    # Draw predicted boxes (red, dashed)
    if predictions is not None:
        for i, (box, score) in enumerate(predictions):
            x1, y1, x2, y2 = box[:4]
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                linestyle="--",
                label="Prediction" if i == 0 else None,
            )
            ax.add_patch(rect)
            # Add confidence label
            ax.text(
                x1,
                y2 + 2,
                f"{score:.2f}",
                color="red",
                fontsize=9,
                fontweight="bold",
            )
            n_pred += 1

    # Build title
    if title:
        full_title = title
    else:
        full_title = f"GT: {n_gt} | Pred: {n_pred}"

    ax.set_title(full_title)
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Frequency bin")

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()

    return fig  # type: ignore[return-value]


def labels_to_xyxy(
    labels: npt.NDArray[np.floating[t.Any]],
) -> npt.NDArray[np.float32]:
    """
    Convert YOLOX labels from (cls, cx, cy, w, h) to xyxy format.

    Parameters
    ----------
    labels : ndarray
        Labels array [N, 5] with columns (class_id, cx, cy, w, h).

    Returns
    -------
    boxes : ndarray
        Boxes array [N, 4] with columns (x1, y1, x2, y2).
    """
    if len(labels) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    # Filter out zero-padded labels
    valid_mask = labels.sum(axis=1) != 0
    valid_labels = labels[valid_mask]

    if len(valid_labels) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    cx = valid_labels[:, 1]
    cy = valid_labels[:, 2]
    w = valid_labels[:, 3]
    h = valid_labels[:, 4]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
