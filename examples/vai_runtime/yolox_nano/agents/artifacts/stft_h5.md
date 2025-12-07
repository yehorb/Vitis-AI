# HDF5 file structure (`tiles.h5`)

Location: `tensors/tiles.h5` inside the run directory.

The HDF5 file has:

1. **Root-level datasets**

   - `"/S_db_full"`
     - Type: `float32` (`single` in MATLAB).
     - Shape: `[H, total_frames] = [fft_size, number_of_STFT_frames]`.
     - Content: full spectrogram magnitude in dB for the entire recording.

   - `"/F"`
     - Type: `float32`.
     - Shape: `[H, 1]`.
     - Content: frequency bin centers in Hz for each STFT row.
     - Typically ranges approximately from `−Fs/2` to `+Fs/2` when `centered = true`.

   - `"/T"`
     - Type: `float32`.
     - Shape: `[1, total_frames]`.
     - Content: time stamp (seconds) of each STFT frame.

2. **Root-level attributes**

   Attached to `/` (the root group) with types as written below:

   - `sample_rate_hz` : double (`Fs`).
   - `fft_size` : 32‑bit integer.
   - `window_type` : string (e.g. `"hann"`).
   - `window_length_samples` : int32.
   - `overlap_samples` : int32.
   - `frame_hop_samples` : int32 (`window_length_samples − overlap_samples`).
   - `frame_hop_seconds` : double (`frame_hop_samples / Fs`).
   - `freq_bin_hz` : double (`Fs / fft_size`).
   - `centered` : uint8 flag (0 or 1).
   - `vmin_db` : double (`render.vmin_db`).
   - `vmax_db` : double (`render.vmax_db`).

   These mirror the `meta.json` contents in an HDF5-friendly form.

3. **Per-tile datasets under `"/S_db"` and `"/boxes"`**

   Tiles are indexed from 0 to `n_tiles−1`, where `n_tiles = ceil(total_frames / W)` and `W = frames_per_image = 128`.

   For each tile `k`:

   - **Tile ID**:
     - `tile_id = sprintf("rec_%s_%03d", run_id, k)` (k is zero-based).
     - All following dataset paths use this name.

   - **Spectrogram tile dataset**: `"/S_db/<tile_id>"`
     - Example: `"/S_db/rec_20250928_231045_000"`.
     - Type: `float32`.
     - Shape: `[H, W] = [128, 128]`.
       - If the last tile would be shorter in time (near the end of `T`), it is right-padded by repeating the last column so that width is always exactly `W`.
     - Content: `SdBk` sub-matrix of `S_db_full` in dB, corresponding to this tile’s time slice.

   - **Bounding box dataset**: `"/boxes/<tile_id>"`
     - Example: `"/boxes/rec_20250928_231045_000"`.
     - Type: `int32`.
     - Shape: `[N_boxes, 4]`, where `N_boxes` is the number of pulse boxes overlapping this tile.
       - If `N_boxes == 0`, an empty `(0,4)` matrix is created (no data written).
     - Layout for each row (same as JSON bboxes):
       - `[x, y, width, height]`, with:
         - `x` in `[0..W-1]` (frame index inside tile).
         - `y` in `[0..H-1]` (frequency bin index).
         - `width` ≥ 1, `height` ≥ 1, both counts of bins/frames.
     - Boxes are computed by:
       - Mapping global time/frequency box `(t0_gt, t1_gt, fmin_gt, fmax_gt)` to nearest indices in `T` and `F`.
       - Converting to 0‑based pixel indices:
         - `x0 = c1 − idx0`, `x1 = c2 − idx0`, where `idx0` is the first frame index of the tile.
         - `y0 = r1 − 1`, `y1 = r2 − 1`.
       - Rejecting non-intersecting boxes before clipping.
       - Clipping `x0..x1` into `[0..W−1]` and `y0..y1` into `[0..H−1]`.
       - Converting to `[x, y, width, height]` with `width = x1c − x0c + 1`, etc.

4. **Data types summary (HDF5)**

- `S_db_full`, all `/S_db/<tile_id>`: `float32` (single-precision).
- `F`, `T`: `float32`.
- `/boxes/<tile_id>`: `int32` (each row is a bounding box).
- Root attributes: mix of `double`, `int32`, `uint8`, and `string` as noted above.
