from __future__ import annotations

import typing as t
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

if t.TYPE_CHECKING:
    import numpy.typing as npt
    import stft_dataset


def stft_collate_fn(
    batch: t.List[stft_dataset.YoloxDataPoint],
    max_labels: int = 50,
) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for batching STFT samples.

    Stacks images and pads labels to a fixed size. Zero-padded label rows
    are detected by YOLOX using sum(row) == 0.

    Parameters
    ----------
    batch : list of tuples
        Each tuple is (img, labels, tile_id) from StftDataset.
    max_labels : int
        Maximum number of labels per image. Labels beyond this are truncated.

    Returns
    -------
    imgs : torch.Tensor
        Shape [B, 1, H, W], float32
    labels : torch.Tensor
        Shape [B, max_labels, 5], float32
        Zero-padded for images with fewer labels.
    """
    imgs_list: t.Tuple[npt.NDArray[np.float32], ...] = ()
    labels_list: t.Tuple[npt.NDArray[np.float32], ...] = ()
    imgs_list, labels_list, _ = zip(*batch)

    # Stack images: list of [1, H, W] -> [B, 1, H, W]
    imgs = from_numpy(np.stack(imgs_list, axis=0))

    # Pad labels to [B, max_labels, 5]
    B = len(labels_list)
    padded_labels = torch.zeros(B, max_labels, 5, dtype=torch.float32)
    for i, lbl in enumerate(labels_list):
        n = min(len(lbl), max_labels)
        if n > 0:
            padded_labels[i, :n] = from_numpy(lbl[:n])

    return imgs, padded_labels


def from_numpy(ndarray: npt.NDArray[t.Any]) -> torch.Tensor:
    return torch.from_numpy(ndarray)  # pyright: ignore[reportUnknownMemberType]


class StftDataLoader:
    """
    Wrapper around PyTorch DataLoader for STFT dataset.

    Provides the close_mosaic() method expected by the YOLOX trainer.
    """

    def __init__(
        self,
        dataset: Dataset[stft_dataset.YoloxDataPoint],
        max_labels: int,
        **kwargs,
    ):
        """
        Create a DataLoader for STFT dataset.

        Parameters
        ----------
        dataset : StftDataset
            The dataset to load from.
        max_labels : int
            Maximum number of labels per image (for padding).
        """
        self.dataset = dataset
        self._loader = DataLoader(
            dataset=dataset,
            collate_fn=partial(stft_collate_fn, max_labels=max_labels),
            **kwargs,
        )

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)

    def close_mosaic(self) -> None:
        """
        No-op for YOLOX trainer compatibility.

        The YOLOX trainer calls this method when switching from mosaic
        augmentation to regular training. Since we don't use mosaic
        augmentation, this is a no-op.
        """
        pass
