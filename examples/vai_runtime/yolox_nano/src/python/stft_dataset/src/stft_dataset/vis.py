from __future__ import annotations

import typing as t

import matplotlib.patches as patches
import matplotlib.pyplot as plt

if t.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from torch import Tensor


def visualize_stft_sample(
    sample: t.Tuple[Tensor, Tensor, str],
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "magma",
) -> None:
    """
    Visualize a single STFT tile with bounding boxes.

    Parameters
    ----------
    sample :
        A single item from StftDataset: (spec, boxes, tile_id).
        - spec:  [H, W] torch.Tensor, float32 (dB)
        - boxes: [N, 4] torch.Tensor, int/long, (x0, y0, w, h), 0-based
        - tile_id: str
    vmin, vmax :
        Optional color limits for the spectrogram. If None, use data min/max.
    cmap :
        Matplotlib colormap name.
    """
    spec, boxes, tile_id = sample

    # Move to CPU + numpy
    spec_np: npt.NDArray[np.float32] = spec.detach().cpu().numpy()
    boxes_np: npt.NDArray[np.int32] = boxes.detach().cpu().numpy()

    if vmin is None:
        vmin = float(spec_np.min())
    if vmax is None:
        vmax = float(spec_np.max())

    fig, ax = plt.subplots(figsize=(6, 5))
    assert isinstance(ax, plt.Axes)

    im = ax.imshow(
        spec_np,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    # Draw bounding boxes
    for b in boxes_np:
        print(b)
        if b.shape[0] != 4:
            continue
        x0, y0, w, h = b
        rect = patches.Rectangle(
            (x0, y0),
            w,
            h,
            linewidth=1.5,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)

    _ = ax.set_title(f"Tile: {tile_id} (boxes={boxes_np.shape[0]})")
    _ = ax.set_xlabel("Time (frames)")
    _ = ax.set_ylabel("Frequency bin")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Magnitude (dB)")

    plt.tight_layout()
    plt.show()
