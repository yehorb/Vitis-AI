"""
Normalization utilities for STFT spectrograms.
"""

from __future__ import annotations

import json
import pathlib
import typing as t

import numpy as np
from torch.utils.data import Dataset

if t.TYPE_CHECKING:
    import numpy.typing as npt


def normalize_db_scalar(x: float, vmin_db: float, vmax_db: float) -> float:
    """
    Normalize a single dB value to [0, 1] range.

    Parameters
    ----------
    x : float
        Input value in dB.
    vmin_db : float
        Minimum dB value (maps to 0).
    vmax_db : float
        Maximum dB value (maps to 1).

    Returns
    -------
    float
        Normalized value clamped to [0, 1].
    """
    scale = 1.0 / (vmax_db - vmin_db)
    y = (x - vmin_db) * scale
    if y < 0.0:
        y = 0.0
    elif y > 1.0:
        y = 1.0
    return y


def normalize_db_array(
    x: npt.NDArray[np.floating[t.Any]],
    vmin_db: float,
    vmax_db: float,
) -> npt.NDArray[np.float32]:
    """
    Normalize an array of dB values to [0, 1] range.

    Parameters
    ----------
    x : ndarray
        Input array in dB (any shape).
    vmin_db : float
        Minimum dB value (maps to 0).
    vmax_db : float
        Maximum dB value (maps to 1).

    Returns
    -------
    ndarray
        Normalized array clamped to [0, 1], dtype float32.
    """
    scale: float = 1.0 / (vmax_db - vmin_db)
    y = (x - vmin_db) * scale
    y = np.clip(y, 0.0, 1.0)
    return y.astype(np.float32)


def load_normalization_params(
    meta_path: t.Union[str, pathlib.Path],
) -> t.Tuple[float, float]:
    """
    Load normalization parameters from meta.json.

    Parameters
    ----------
    meta_path : str or Path
        Path to meta.json file.

    Returns
    -------
    vmin_db : float
        Minimum dB value for normalization.
    vmax_db : float
        Maximum dB value for normalization.
    """
    if isinstance(meta_path, str):
        meta_path = pathlib.Path(meta_path)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    vmin_db: float = meta["render"]["vmin_db"]
    vmax_db: float = meta["render"]["vmax_db"]

    return vmin_db, vmax_db


T = t.TypeVar("T", bound=t.Tuple[t.Any, ...])


class Normalize(Dataset[T], t.Generic[T]):
    """
    Dataset decorator that applies dB normalization to spectrograms.

    Wraps any dataset that yields (img, ...) tuples where img is
    a numpy array of dB values. The normalized image replaces the
    original in the output tuple.

    Parameters
    ----------
    dataset : Dataset[T]
        Underlying dataset yielding tuples with img as first element.
    vmin_db : float
        Minimum dB value (maps to 0).
    vmax_db : float
        Maximum dB value (maps to 1).

    Example
    -------
    >>> base = StftDataset(Matlab("tiles.h5", tile_ids))
    >>> vmin, vmax = load_normalization_params("meta.json")
    >>> normalized = Normalize(base, vmin, vmax)
    >>> img, labels, tile_id = normalized[0]  # img is now in [0, 1]
    """

    def __init__(
        self,
        dataset: Dataset[T],
        vmin_db: float,
        vmax_db: float,
    ):
        self.dataset: Dataset[T] = dataset
        self.vmin_db: float = vmin_db
        self.vmax_db: float = vmax_db
        self.scale: float = 1.0 / (vmax_db - vmin_db)

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> T:
        item = self.dataset[index]
        img = item[0]
        img_normalized = self._normalize(img)
        return (img_normalized, *item[1:])  # type: ignore[return-value]

    def _normalize(
        self,
        img: npt.NDArray[np.floating[t.Any]],
    ) -> npt.NDArray[np.float32]:
        """Apply normalization using precomputed scale."""
        y = (img - self.vmin_db) * self.scale
        y = np.clip(y, 0.0, 1.0)
        return y.astype(np.float32)


class RandomHorizontalFlip(Dataset[T], t.Generic[T]):
    """
    Dataset decorator that applies random horizontal flip augmentation.

    Flips both the spectrogram and bounding box x-coordinates. This is
    equivalent to time-reversal, which is a valid augmentation for
    signal detection in spectrograms.

    Parameters
    ----------
    dataset : Dataset[T]
        Underlying dataset yielding (img, labels, ...) tuples.
        - img: [C, H, W] array
        - labels: [N, 5] array with columns [class_id, cx, cy, w, h]
    flip_prob : float
        Probability of applying horizontal flip (default: 0.5).
    img_width : int
        Width of the image, used to compute flipped x-coordinates.

    Example
    -------
    >>> base = Normalize(StftDataset(...), vmin, vmax)
    >>> augmented = RandomHorizontalFlip(base, flip_prob=0.5, img_width=128)
    >>> img, labels, tile_id = augmented[0]
    """

    def __init__(
        self,
        dataset: Dataset[T],
        flip_prob: float = 0.5,
        img_width: int = 128,
    ):
        self.dataset: Dataset[T] = dataset
        self.flip_prob: float = flip_prob
        self.img_width: int = img_width

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> T:
        item = self.dataset[index]
        img = item[0]
        labels = item[1]

        if np.random.random() < self.flip_prob:
            img, labels = self._flip(img, labels)

        return (img, labels, *item[2:])  # type: ignore[return-value]

    def _flip(
        self,
        img: npt.NDArray[np.float32],
        labels: npt.NDArray[np.float32],
    ) -> t.Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Apply horizontal flip to image and labels."""
        # Flip image along width axis: [C, H, W] -> flip W
        img_flipped = np.ascontiguousarray(img[:, :, ::-1])

        # Flip box x-coordinates: cx_new = (img_width - 1) - cx
        # For 128px image: cx=0 -> 127, cx=127 -> 0
        if len(labels) > 0:
            labels_flipped = labels.copy()
            labels_flipped[:, 1] = (self.img_width - 1) - labels[:, 1]
        else:
            labels_flipped = labels

        return img_flipped, labels_flipped
