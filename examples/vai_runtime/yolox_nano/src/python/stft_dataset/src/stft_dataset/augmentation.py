"""
Data augmentation utilities for STFT spectrograms.
"""

from __future__ import annotations

import typing as t

import numpy as np
from torch.utils.data import Dataset

if t.TYPE_CHECKING:
    import numpy.typing as npt


T = t.TypeVar("T", bound=t.Tuple[t.Any, ...])


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
