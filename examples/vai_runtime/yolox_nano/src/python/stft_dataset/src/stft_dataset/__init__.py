from __future__ import annotations

import pathlib
import threading
import typing as t

import h5py
import numpy as np
from torch.utils.data import Dataset

if t.TYPE_CHECKING:
    import numpy.typing as npt

    T_co = t.TypeVar("T_co", covariant=True)
    MatlabDataPoint = t.Tuple[npt.NDArray[np.float32], npt.NDArray[np.int32], str]
    YoloxDataPoint = t.Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], str]


class Matlab(Dataset["MatlabDataPoint"]):
    """
    Multiprocess-safe HDF5 dataset for STFT spectrograms.

    Uses thread-local storage to ensure each DataLoader worker gets its own
    HDF5 file handle. File handles are opened lazily on first access within
    each worker process.

    Parameters
    ----------
    h5_path : str or Path
        Path to tiles.h5.
    tile_ids : sequence of str, optional
        List of tile_ids to use (e.g. from splits/train.txt).
        If None, tile_ids are inferred from f['/S_db'].keys().
    """

    def __init__(
        self,
        h5_path: t.Union[str, pathlib.Path],
        tile_ids: t.Sequence[str],
    ):
        if isinstance(h5_path, str):
            h5_path = pathlib.Path(h5_path)
        self.h5_path: pathlib.Path = h5_path
        self.tile_ids: t.List[str] = list(tile_ids)
        # Thread-local storage for file handles (one per worker)
        self._local = threading.local()

    def __len__(self):
        return len(self.tile_ids)

    def _get_h5(self) -> h5py.File:
        """
        Get HDF5 file handle for current worker.

        Each DataLoader worker process gets its own file handle stored in
        thread-local storage. This avoids sharing file handles across
        fork boundaries which causes segfaults.
        """
        # Check if we have a handle for this worker
        if not hasattr(self._local, "h5") or self._local.h5 is None:
            self._local.h5 = h5py.File(self.h5_path, "r")
        return self._local.h5

    def __getitem__(self, index: int) -> MatlabDataPoint:
        h5 = self._get_h5()
        tile_id = self.tile_ids[index]

        # --- Spectrogram tile: float32 [H, W] ---
        if not isinstance(s_db_dataset := h5[f"/S_db/{tile_id}"], h5py.Dataset):
            raise TypeError(f"Expected /S_db/{tile_id} to be a Dataset")
        s_db = t.cast("npt.NDArray[np.float32]", s_db_dataset[...])

        # --- Boxes: int32 [N, 4] (x0, y0, w, h) ---
        # Load and transpose Matlab boxes
        if f"/boxes/{tile_id}" in h5:
            if not isinstance(boxes_dataset := h5[f"/boxes/{tile_id}"], h5py.Dataset):
                raise TypeError(f"Expected /boxes/{tile_id} to be a Dataset")
            boxes = t.cast("npt.NDArray[np.int32]", boxes_dataset[...])
        else:
            boxes = np.zeros((4, 0), dtype=np.int32)

        return s_db, boxes, tile_id

    def close(self):
        if hasattr(self._local, "h5") and self._local.h5 is not None:
            self._local.h5.close()
            self._local.h5 = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        """
        Prepare for pickling (used when DataLoader spawns workers).

        Exclude the thread-local storage and file handle - each worker
        will create its own after unpickling.
        """
        state = self.__dict__.copy()
        state["_local"] = None
        return state

    def __setstate__(self, state):
        """Restore from pickle, reinitialize thread-local storage."""
        self.__dict__.update(state)
        self._local = threading.local()


class DiscoverTileIds(Dataset["MatlabDataPoint"]):
    """
    Dataset wrapper that discovers tile IDs from the HDF5 file.

    Usage:
        m = Matlab("tiles.h5", [])
        d = DiscoverTileIds(m)  # populates m.tile_ids from HDF5 keys
    """

    def __init__(self, dataset: Matlab):
        self.dataset: Matlab = dataset
        with h5py.File(dataset.h5_path, "r") as f:
            tiles = f["/S_db"]
            if not isinstance(tiles, h5py.Group):
                raise TypeError("loaded tile[s] of incorrect type")
            self.dataset.tile_ids = sorted(tiles.keys())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> MatlabDataPoint:
        return self.dataset[index]

    def close(self):
        return self.dataset.close()

    def __del__(self):
        self.close()


class LoadSplit(Dataset["MatlabDataPoint"]):
    """
    Dataset wrapper that loads tile IDs from a split file.

    Split files contain one tile ID per line (e.g., train.txt, val.txt, test.txt).

    Usage:
        m = Matlab("tiles.h5", [])
        d = LoadSplit(m, "splits/train.txt")  # populates m.tile_ids from file

    Parameters
    ----------
    dataset : Matlab
        The underlying Matlab dataset to populate.
    split_path : str or Path
        Path to the split file containing tile IDs, one per line.
    """

    def __init__(
        self,
        dataset: Matlab,
        split_path: t.Union[str, pathlib.Path],
    ):
        self.dataset: Matlab = dataset
        if isinstance(split_path, str):
            split_path = pathlib.Path(split_path)
        self.split_path: pathlib.Path = split_path

        # Load tile IDs from split file
        with open(split_path, "r") as f:
            tile_ids = [line.strip() for line in f if line.strip()]
        self.dataset.tile_ids = tile_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> MatlabDataPoint:
        return self.dataset[index]

    def close(self):
        return self.dataset.close()

    def __del__(self):
        self.close()


class StftDataset(Dataset["YoloxDataPoint"]):
    """
    A PyTorch Dataset that loads STFT spectrogram tiles for YOLOX training.

    Expected HDF5 layout:
      /S_db/<tile_id>  -> float32 [W, H]
      /boxes/<tile_id> -> int32 [4, N]  (x0, y0, w, h)

    Source data is in Matlab format - column major - so it is transposed to
    NumPy format - row major.
      S_db  transposed to [H, W]
      boxes transposed to [N, 4]

    Output format (YOLOX-compatible):
      img: np.ndarray [1, H, W] float32, raw dB values
      labels: np.ndarray [N, 5] float32, each row is [class_id, cx, cy, w, h]
      tile_id: str
    """

    def __init__(
        self,
        dataset: Dataset[MatlabDataPoint],
    ):
        self.dataset: Dataset[MatlabDataPoint] = dataset

    def __len__(self):
        return len(self.dataset)  # pyright: ignore[reportArgumentType]

    def __getitem__(self, index: int) -> YoloxDataPoint:
        m_s_db, m_boxes, tile_id = self.dataset[index]

        # --- Spectrogram tile: float32 [H, W] ---
        s_db = m_s_db.T
        # Add channel dimension: [H, W] -> [1, H, W]
        s_db = np.expand_dims(s_db, axis=0).astype(np.float32)

        # --- Boxes: int32 [N, 4] (x0, y0, w, h) ---
        # Transpose Matlab boxes
        boxes = m_boxes.T

        n_boxes = boxes.shape[0]
        labels = np.zeros((n_boxes, 5), dtype=np.float32)

        # Convert to YOLOX format: [N, 5] (class_id, cx, cy, w, h)
        if n_boxes > 0:
            x0 = boxes[:, 0].astype(np.float32)
            y0 = boxes[:, 1].astype(np.float32)
            w = boxes[:, 2].astype(np.float32)
            h = boxes[:, 3].astype(np.float32)

            # Convert (x0, y0, w, h) -> (cx, cy, w, h)
            cx = x0 + w / 2.0
            cy = y0 + h / 2.0

            labels[:, 0] = 0  # class_id (single class: QPSK)
            labels[:, 1] = cx
            labels[:, 2] = cy
            labels[:, 3] = w
            labels[:, 4] = h

        return s_db, labels, tile_id
