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


class RandomVerticalFlip(Dataset[T], t.Generic[T]):
    """
    Dataset decorator that applies random vertical flip augmentation.

    Flips both the spectrogram and bounding box y-coordinates. This is
    equivalent to frequency mirroring. Use with caution - may not be
    physically meaningful for all signal types.

    Parameters
    ----------
    dataset : Dataset[T]
        Underlying dataset yielding (img, labels, ...) tuples.
        - img: [C, H, W] array
        - labels: [N, 5] array with columns [class_id, cx, cy, w, h]
    flip_prob : float
        Probability of applying vertical flip (default: 0.5).
    img_height : int
        Height of the image, used to compute flipped y-coordinates.

    Example
    -------
    >>> base = Normalize(StftDataset(...), vmin, vmax)
    >>> augmented = RandomVerticalFlip(base, flip_prob=0.5, img_height=128)
    >>> img, labels, tile_id = augmented[0]
    """

    def __init__(
        self,
        dataset: Dataset[T],
        flip_prob: float = 0.5,
        img_height: int = 128,
    ):
        self.dataset: Dataset[T] = dataset
        self.flip_prob: float = flip_prob
        self.img_height: int = img_height

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
        """Apply vertical flip to image and labels."""
        # Flip image along height axis: [C, H, W] -> flip H
        img_flipped = np.ascontiguousarray(img[:, ::-1, :])

        # Flip box y-coordinates: cy_new = (img_height - 1) - cy
        if len(labels) > 0:
            labels_flipped = labels.copy()
            labels_flipped[:, 2] = (self.img_height - 1) - labels[:, 2]
        else:
            labels_flipped = labels

        return img_flipped, labels_flipped


class GaussianNoiseAugmentation(Dataset[T], t.Generic[T]):
    """
    Dataset decorator that adds Gaussian noise to spectrograms.

    This helps the model generalize to lower SNR signals by training
    on noisier versions of the data.

    Parameters
    ----------
    dataset : Dataset[T]
        Underlying dataset yielding (img, labels, ...) tuples.
        - img: [C, H, W] array, normalized to [0, 1]
        - labels: [N, 5] array with columns [class_id, cx, cy, w, h]
    noise_prob : float
        Probability of applying noise augmentation (default: 0.3).
    std_range : tuple of float
        Range of noise standard deviation (min, max).
        Values are sampled uniformly from this range.

    Example
    -------
    >>> base = Normalize(StftDataset(...), vmin, vmax)
    >>> augmented = GaussianNoiseAugmentation(base, noise_prob=0.3, std_range=(0.01, 0.05))
    >>> img, labels, tile_id = augmented[0]
    """

    def __init__(
        self,
        dataset: Dataset[T],
        noise_prob: float = 0.3,
        std_range: t.Tuple[float, float] = (0.01, 0.05),
    ):
        self.dataset: Dataset[T] = dataset
        self.noise_prob: float = noise_prob
        self.std_min: float = std_range[0]
        self.std_max: float = std_range[1]

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> T:
        item = self.dataset[index]
        img = item[0]

        if np.random.random() < self.noise_prob:
            img = self._add_noise(img)

        return (img, *item[1:])  # type: ignore[return-value]

    def _add_noise(
        self,
        img: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Add Gaussian noise to image."""
        # Sample noise std uniformly from range
        std = np.random.uniform(self.std_min, self.std_max)

        # Add Gaussian noise
        noise = np.random.randn(*img.shape).astype(np.float32) * std
        img_noisy = img + noise

        # Clip to valid range [0, 1]
        img_noisy = np.clip(img_noisy, 0.0, 1.0)

        return img_noisy.astype(np.float32)
