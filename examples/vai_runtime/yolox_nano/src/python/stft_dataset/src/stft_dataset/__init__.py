from __future__ import annotations

import pathlib
import typing as t

import h5py
import numpy as np
from torch.utils.data import Dataset

if t.TYPE_CHECKING:
    import numpy.typing as npt

    DataPoint = t.Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], str]


class StftDataset(Dataset["DataPoint"]):
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
        tile_ids: t.Union[None, t.Sequence[str]] = None,
    ):
        if isinstance(h5_path, str):
            h5_path = pathlib.Path(h5_path)
        self.h5_path: pathlib.Path = h5_path
        self._h5: t.Union[None, h5py.File] = None

        if tile_ids is None:
            with h5py.File(h5_path, "r") as f:
                tiles = f["/S_db"]
                if not isinstance(tiles, h5py.Group):
                    raise TypeError("loaded tile[s] of incorrect type")
                self.tile_ids: t.List[str] = sorted(tiles.keys())
        else:
            self.tile_ids = list(tile_ids)

    def __len__(self):
        return len(self.tile_ids)

    def _get_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, index: int) -> DataPoint:
        h5 = self._get_h5()
        tile_id = self.tile_ids[index]

        # --- Spectrogram tile: float32 [H, W] ---
        s_db: npt.NDArray[np.float32] = h5[f"/S_db/{tile_id}"][...]
        s_db = s_db.T
        # Add channel dimension: [H, W] -> [1, H, W]
        s_db = np.expand_dims(s_db, axis=0).astype(np.float32)

        # --- Boxes: int32 [N, 4] (x0, y0, w, h) ---
        # Load and transpose Matlab boxes
        if f"/boxes/{tile_id}" in h5:
            boxes_np: npt.NDArray[np.int32] = h5[f"/boxes/{tile_id}"][...]
            boxes_np = boxes_np.T
        else:
            boxes_np = np.zeros((0, 4), dtype=np.int32)

        n_boxes = boxes_np.shape[0]
        labels = np.zeros((n_boxes, 5), dtype=np.float32)

        # Convert to YOLOX format: [N, 5] (class_id, cx, cy, w, h)
        if n_boxes > 0:
            x0 = boxes_np[:, 0].astype(np.float32)
            y0 = boxes_np[:, 1].astype(np.float32)
            w = boxes_np[:, 2].astype(np.float32)
            h = boxes_np[:, 3].astype(np.float32)

            # Convert (x0, y0, w, h) -> (cx, cy, w, h)
            cx = x0 + w / 2.0
            cy = y0 + h / 2.0

            labels[:, 0] = 0  # class_id (single class: QPSK)
            labels[:, 1] = cx
            labels[:, 2] = cy
            labels[:, 3] = w
            labels[:, 4] = h

        return s_db, labels, tile_id

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self):
        self.close()
