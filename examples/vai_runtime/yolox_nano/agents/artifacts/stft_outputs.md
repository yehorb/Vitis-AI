# Output artifacts: meaning and data formats

All outputs are placed under a run-specific directory:

- Root directory: `../../data/stft/<run_id>` where `run_id = YYYYMMDD_HHMMSS` timestamp.

1. **Raw IQ binaries**
   - `rec_<run_id>_iq_int16.bin`
     - Format: sequence of 16‑bit signed integers (platform endianness, typically little-endian).
     - Length: `2 * N_total` samples.
     - Layout: interleaved I and Q:
       - Index 0: I[0], index 1: Q[0], index 2: I[1], index 3: Q[1], etc.
     - Scaling:
       - Original complex `x` is scaled so that real/imag components are clipped to `[-1, 1)` then mapped to signed `int16` range via multiplication by `intmax('int16')` and clamping.

   - `rec_<run_id>_iq_float32.bin`
     - Format: 32‑bit IEEE754 floats (single precision), platform endianness.
     - Length: `2 * N_total` samples.
     - Layout: same interleaved pattern [I0, Q0, I1, Q1, …].
     - Values: direct cast from the real and imaginary parts of `x` to `single`, without additional scaling.

2. **Per-tile rendered spectrogram images (PNG)**
   - Directory: `images_tiles/`.
   - Files: `rec_<run_id>_<NNN>.png` where `NNN` is zero-based tile index (3 digits).
   - Size: `H × W = 128 × 128` pixels.
   - Data type: 16‑bit grayscale (`GRAY16`).
   - Conversion from dB to pixel:
     - For each tile `SdBk` (float32 in dB), clamp to `[vmin_db, vmax_db]` with `vmin_db = -90`, `vmax_db = -20`.
     - Map linearly to the range `[0, 65535]`:
       - `I16 = round(65535 * (clamp(SdBk) − vmin) / (vmax − vmin))`.
   - Purpose: human-viewable training images and inputs for YOLO-style detectors.

3. **Per-tile JSON annotations**
   - Directory: `ann_tiles/`.
   - Files: `rec_<run_id>_<NNN>.json`.
   - Structure (roughly COCO-like):
     - `image`:
       - `id`: tile ID string, e.g. `"rec_20250928_231045_000"`.
       - `file_name`: relative PNG path, e.g. `"images_tiles/rec_..._000.png"`.
       - `width`: `W` (int, 128).
       - `height`: `H` (int, 128).
     - `annotations`: array of objects, each:
       - `id`: integer index of the underlying pulse (matches `gt` index).
       - `category_id`: `1` (only class is QPSK).
       - `bbox`: `[x, y, width, height]`
         - `x`, `y`: integer top-left pixel coordinates in 0‑based indexing:
           - `x`: time axis, frame index within the tile (`0..W-1`).
           - `y`: frequency bin index (`0..H-1`).
         - `width`, `height`: positive integers (number of pixels in time and frequency).
       - `iscrowd`: `0`.

4. **Dataset metadata (`meta.json`)**
   - Path: `meta.json` under the run root.
   - JSON fields:
     - `sample_rate_hz`: double (`Fs`).
     - `fft_size`: int.
     - `window_type`: string (`"hann"`, etc.).
     - `window_length_samples`: int.
     - `overlap_samples`: int.
     - `frame_hop_samples`: int (`window_length_samples - overlap_samples`).
     - `frame_hop_seconds`: float (`frame_hop_samples / Fs`).
     - `freq_bin_hz`: float (`Fs / fft_size`).
     - `centered`: boolean flag.
     - `render`: object with:
       - `encoding`: string (`"GRAY16"`).
       - `scale`: string (`"dB"`).
       - `vmin_db`, `vmax_db`: floats.
     - `frames_per_image`: int (tile width, `W`).
     - `run_id`: string.

5. **Label map (`label_map.json`)**
   - Path: `label_map.json` under the run root.
   - JSON schema:
     - `classes`: array with a single entry:
       - `{ "id": 1, "name": "QPSK" }`.

6. **Train/Val/Test splits**
   - Directory: `splits/`.
   - Files: `train.txt`, `val.txt`, `test.txt`.
   - Contents:
     - Each file is a newline-separated list of **tile IDs** (without extension), e.g.:
       - `rec_20250928_231045_000`
       - `rec_20250928_231045_001`
       - …
   - Split proportions:
     - 80% train, 10% val, 10% test (rounded via `round` and remainder in test), using a random permutation of tile indices.

7. **Optional Figures**
   - Directory: `fig_full/`.
   - If `save_figures` is set to `true` (default is `false`), saves:
     - `stft_full.fig` / `.png`: full STFT heatmap.
     - `stft_full_boxes.fig` / `.png`: same with ground-truth rectangles overlayed in time–frequency coordinates (seconds and Hz).
