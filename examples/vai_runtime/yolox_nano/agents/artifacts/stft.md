# Data generation algorithm

- **Overall goal**
  Synthesize a single complex baseband IQ recording containing additive white noise plus multiple short QPSK bursts (pulses) with randomized parameters (center frequency, bandwidth, SNR, duration, repetition period). Then compute an STFT, cut it into fixed-size time–frequency tiles, and generate labels (bounding boxes) for each burst in each tile.

- **Parameters (key ones)**
  - Sample rate `Fs = 10 MHz`.
  - STFT FFT size and image height: `fft_size = 128` → each spectrogram column has 128 frequency bins.
  - STFT window: Hann, length 128, overlap 64 → hop 64 samples.
  - Image tile width (time): `frames_per_image = 128` STFT frames.
  - Total recording duration: `total_duration = 0.025 s` → `N_total ≈ 250,000` samples.
  - Noise average power (linear): `Pn = 10^(noise_power_db/10)` with `noise_power_db = -80 dB`.
  - Pulse parameter ranges:
    - Center frequency `f0` uniform in `[freq_min, freq_max] = [-5e6, 5e6] Hz`, but later clipped to stay within the STFT band.
    - SNR (per pulse) uniform in `[5, 40] dB`.
    - Pulse width `pw` uniform in `[20e-6, 120e-6] s`.
    - Repetition period `per` uniform in `[150e-6, 600e-6] s`.
    - Nominal occupied bandwidth `B` uniform in `[0.1e6, 2e6] Hz`.

- **Base noise record**
  1. Compute `N_total = round(total_duration * Fs)`.
  2. Seed MATLAB’s RNG with `cfg.seed` for reproducibility.
  3. Generate complex white Gaussian noise:
     - `x[n] = sqrt(Pn/2) * (randn + j*randn)` for `n = 0..N_total-1`.
     - This yields circular complex noise with mean power approximately `Pn`.

- **Iterative pulse generation and placement**
  1. Initialize current time `t0 = 0` and an empty array of ground-truth boxes `gt`.
  2. While `t0 < total_duration - pw_min`:
     - Randomly draw: `f0`, `snr_db`, `pw`, `per`, `B` from their respective ranges.
     - Compute pulse sample length `Np = max(1, round(pw * Fs))`.
     - Convert start/end times to sample indices:
       - `n0 = round(t0 * Fs) + 1`, `n1 = min(n0 + Np - 1, N_total)`.
       - Time axis for the pulse: `t_p = (0..(n1-n0)) / Fs`.

- **QPSK symbol rate and occupied bandwidth**
  1. Define raised-cosine roll-off `alpha = 0.35`.
  2. Approximate symbol rate from target bandwidth:
     - Start with `Rs ≈ B / (1 + alpha)` (min 1 kHz), then choose integer samples per symbol `sps = max(2, round(Fs / Rs))`.
     - Recompute exact symbol rate `Rs = Fs / sps`.
     - Actual occupied bandwidth `B_occ = (1 + alpha) * Rs`.
  3. Correct `f0` so that the band `[f0 - B_occ/2, f0 + B_occ/2]` remains safely within ±0.9×Nyquist:
     - `f0` is clipped to `[-0.9*Fs/2 + B_occ/2, +0.9*Fs/2 - B_occ/2]`.

- **QPSK symbol sequence and pulse shaping**
  1. For each pulse, reseed RNG: `seed = qpsk.seed_base + pulse_idx` for reproducible but distinct symbol sequences.
  2. Generate random symbol indices `m ∈ {0..3}` with length `ceil(numel(t_p)/sps)`.
  3. Map symbols to standard unit-energy QPSK constellation:
     - Mapping array: `(1/√2) * [1+1j; -1+1j; -1-1j; 1-1j]`.
  4. If raised-root-cosine (RRC) is enabled (it is by default):
     - Design root-raised cosine filter `h = rcosdesign(alpha, span_sym, sps, 'sqrt')` with `span_sym = 8`.
     - Upsample and filter: `y = upfirdn(symbols, h, sps, 1)`.
     - Remove group delay `gd = span_sym * sps / 2`, keep only `y(gd+1:end-gd)`.
     - Zero-pad or truncate `y` to exactly `numel(t_p)` samples.
  5. If RRC were disabled, it would use simple nearest-neighbor upsampling: repeat each symbol `sps` times.

- **Frequency translation and amplitude envelope**
  1. Upconvert baseband QPSK `y` to center frequency `f0`:
     - `pulse[n] = y[n] * exp(j * 2π f0 * t_p[n])`.
  2. Apply smooth rise/fall edges:
     - Rise and fall durations (seconds) configured as `edge.rise_s = edge.fall_s = 10e-6`.
     - Convert to samples: `Nr = round(edge.rise_s * Fs)`, `Nf = round(edge.fall_s * Fs)`.
     - Create Hann-like fade-in/fade-out windows and a flat middle region:
       - `rise` and `fall` are cosine ramps, concatenated with ones to form `env` of length `numel(pulse)`.
     - Multiply pointwise: `pulse ← pulse .* env`.

- **SNR-controlled scaling**
  1. Compute current pulse power: `Ppulse = mean(|pulse|^2) + eps`.
  2. Desired linear SNR: `10^(snr_db/10)`.
  3. Scale factor to achieve target SNR against noise floor `Pn`:
     - `scale = sqrt((Pn * 10^(snr_db/10)) / Ppulse)`.
  4. Apply: `pulse ← pulse * scale`.

- **Add pulse to noise record and record ground-truth**
  1. Add the pulse into the composite record segment: `x[n0:n1] += pulse`.
  2. Append a ground-truth box to `gt` with:
     - Time bounds:
       - `t0_gt = (n0-1)/Fs`, `t1_gt = (n1-1)/Fs`.
     - Frequency bounds derived from `f0` and `B_occ`:
       - `fmin_gt = f0 - B_occ/2`, `fmax_gt = f0 + B_occ/2`.

- **Overlap control for subsequent pulses**
  1. If overlap is disabled: `t0 ← t0 + per`.
  2. If enabled (default `overlap.enable = true` and `overlap.mode = 'uniform'`):
     - Uniform mode:
       - Random jitter factor `jit` clamped to [0,1].
       - Inter-burst spacing `dt ~ Uniform((1−jit)*per, per)`.
     - Poisson mode (not used by default):
       - Mean step `tau = per` or overridden by `overlap.step_s`.
       - `dt` drawn from exponential distribution: `dt = −tau * log(u)`.
  3. Set `t0 ← t0 + dt` and repeat until `t0` exceeds limit.

- **STFT computation**
  1. Build window according to `window_type` (Hann by default).
  2. Compute spectrogram with or without frequency centering:
     - `S, F, T = spectrogram(x, win, overlap_samples, fft_size, Fs, 'centered')`.
     - Output:
       - `S`: complex matrix of size `[H, total_frames]`, `H = fft_size`.
       - `F`: vector length `H` of frequency bin centers (Hz).
       - `T`: vector length `total_frames` of frame center times (seconds).
  3. Convert to power in decibels:
     - `SdB = 20*log10(max(|S|, 1e-12))` and cast to single-precision float.
