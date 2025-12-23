function generate_training_signal(varargin)
p = inputParser;
addParameter(p, 'OutputDirectory', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'RunId', '');
% RNG for reproducible signal
addParameter(p, 'Seed', 1);
% seconds, approx 5000 tiles
addParameter(p, 'TotalDuration', 6.0);
% 5 db is a heuristic "threshold" for CFAR detector
addParameter(p, 'SnrMax', 5);
addParameter(p, 'SnrMin', -15);
addParameter(p, 'NTrain', 0.8);
addParameter(p, 'NVal', 0.1);
parse(p, varargin{:});
args = p.Results;

% run/session identifier for filenames
if ~isempty(args.RunId)
    run_id = string(args.RunId);
else
    run_id = string(datetime('now','Format','yyyyMMdd_HHmmss'));  % e.g., 20250928_231045
end
if ~isempty(args.OutputDirectory)
    output_dir = string(args.OutputDirectory);
else
    output_dir = "../../data/stft/";
end
if ~endsWith(output_dir, "/")
    output_dir = output_dir + "/";
end

% ------------------------- DATASET PATHS -------------------------
ds.root        = output_dir + run_id;
ds.images_dir  = fullfile(ds.root, 'images_tiles');
ds.ann_dir     = fullfile(ds.root, 'ann_tiles');
ds.splits_dir  = fullfile(ds.root, 'splits');
ds.meta_path   = fullfile(ds.root, 'meta.json');
ds.labelmap    = fullfile(ds.root, 'label_map.json');
ds.fig_dir     = fullfile(ds.root, 'fig_full');
ds.tensors_dir = fullfile(ds.root, 'tensors');
ds.h5_path     = fullfile(ds.tensors_dir, 'tiles.h5');

% ------------------------- RENDER (PNG) --------------------------
render.encoding = 'GRAY16';
render.scale    = 'dB';
render.vmin_db  = -90;
render.vmax_db  = -20;

% ------------------------- STFT / TILE ---------------------------
stft.sample_rate_hz        = 10e6;
stft.fft_size              = 128;
stft.window_type           = 'hann';
stft.window_length_samples = stft.fft_size;
stft.overlap_samples       = stft.fft_size/2;
stft.centered              = true;
stft.frames_per_image      = 128;  % tile width (W); tile height (H)=fft_size

% ------------------------- SIGNAL GENERATION ---------------------
cfg.seed           = args.Seed;
cfg.total_duration = args.TotalDuration;
cfg.noise_power_db = -80;  % mean noise power in dB

% Add a slight margin to avoid generating signals right on an edge
rngs.freq_min = -4.5e6; rngs.freq_max = 4.5e6;
rngs.snr_min = args.SnrMin; rngs.snr_max = args.SnrMax;

rngs.pw_min  = 20e-6;  rngs.pw_max  = 120e-6;
rngs.per_min = 150e-6; rngs.per_max = 400e-6;
rngs.bw_min  = 0.1e6;  rngs.bw_max  = 2e6;

% explicit overlap policy
overlap.enable = true;
overlap.mode   = 'uniform';  % 'uniform' | 'poisson'
overlap.step_s = [];         % for 'poisson' ([] => use drawn 'per')
overlap.jitter = 1.0;        % for 'uniform': dt ~ U((1-j)*per, per), j in [0..1]

% pulse shaping & modulation
edge.rise_s = 10e-6; edge.fall_s = 10e-6;

qpsk.rrc_enable   = true;
qpsk.rolloff      = 0.35;
qpsk.span_sym     = 8;
qpsk.seed_base    = 1234;

% ------------------------- PREPARE DIRS --------------------------
if ~exist(ds.images_dir,  'dir'), mkdir(ds.images_dir); end
if ~exist(ds.ann_dir,     'dir'), mkdir(ds.ann_dir); end
if ~exist(ds.splits_dir,  'dir'), mkdir(ds.splits_dir); end
if ~exist(ds.fig_dir,     'dir'), mkdir(ds.fig_dir); end
if ~exist(ds.tensors_dir, 'dir'), mkdir(ds.tensors_dir); end

if exist(ds.h5_path,'file'), delete(ds.h5_path); end

% ------------------------- NOISE + PULSES ------------------------
Fs      = stft.sample_rate_hz;
N_total = round(cfg.total_duration * Fs);
rng(cfg.seed);

% ------------------------- LOG -----------------------------------
cfg.how_often_to_log_signal = 1000;
cfg.how_often_to_log_tiles = 100;

Pn = 10^(cfg.noise_power_db/10);
x = sqrt(Pn/2) * (randn(N_total,1) + 1j*randn(N_total,1));  % flat noise over whole record

gt = struct('t0',{},'t1',{},'fmin',{},'fmax',{});
t0 = 0; pulse_idx = 0;
num_it_log = cfg.how_often_to_log_signal;

while t0 < (cfg.total_duration - rngs.pw_min)
    f0  = rngs.freq_min + (rngs.freq_max - rngs.freq_min)*rand;
    snr_db = rngs.snr_min + (rngs.snr_max - rngs.snr_min)*rand;

    pw  = rngs.pw_min + (rngs.pw_max - rngs.pw_min)*rand;
    per = rngs.per_min + (rngs.per_max - rngs.per_min)*rand;
    B   = rngs.bw_min + (rngs.bw_max - rngs.bw_min)*rand;

    Np = max(1, round(pw*Fs));
    n0 = round(t0*Fs) + 1; if n0 > N_total, break; end
    n1 = min(n0 + Np - 1, N_total);
    t_p = ((0:(n1-n0)).')/Fs;

    alpha = qpsk.rolloff;
    Rs = max(B/(1+alpha), 1e3);
    sps = max(2, round(Fs/Rs));
    Rs = Fs/sps;
    B_occ = (1+alpha) * Rs;

    f0 = min(max(f0, -Fs/2*0.9 + B_occ/2), Fs/2*0.9 - B_occ/2);

    pulse_idx = pulse_idx + 1; rng(qpsk.seed_base + pulse_idx);
    m = randi([0 3], ceil(numel(t_p)/sps), 1);
    map = (1/sqrt(2))*[1+1j; -1+1j; -1-1j; 1-1j];
    s = map(m+1);

    if qpsk.rrc_enable
        h  = rcosdesign(alpha, qpsk.span_sym, sps, 'sqrt');
        y  = upfirdn(s, h, sps, 1);
        gd = (qpsk.span_sym*sps)/2;
        y  = y(gd+1:end-gd);
        if numel(y) < numel(t_p), y = [y; zeros(numel(t_p)-numel(y),1)]; end
        y  = y(1:numel(t_p));
        pulse = y .* exp(1j*2*pi*f0*t_p);
    else
        s_up  = repelem(s, sps); s_up = s_up(1:numel(t_p));
        pulse = s_up .* exp(1j*2*pi*f0*t_p);
    end

    Nr = max(0, round(edge.rise_s*Fs));
    Nf = max(0, round(edge.fall_s*Fs));
    if Nr + Nf >= numel(pulse), Nr = floor((numel(pulse)-1)/2); Nf = numel(pulse)-Nr-1; end
    rise = []; fall = [];
    if Nr>0, n=(0:Nr-1).'; rise = 0.5*(1 - cos(pi*(n+1)/Nr)); end
    if Nf>0, n=(0:Nf-1).'; fall = flipud(0.5*(1 - cos(pi*(n+1)/Nf))); end
    env = [rise; ones(numel(pulse)-Nr-Nf,1); fall];
    pulse = pulse .* env;

    Ppulse = mean(abs(pulse).^2) + eps;
    scale  = sqrt((Pn * 10^(snr_db/10)) / Ppulse);
    pulse  = pulse * scale;

    x(n0:n1) = x(n0:n1) + pulse;

    gi = numel(gt)+1;
    gt(gi).t0   = (n0-1)/Fs;
    gt(gi).t1   = (n1-1)/Fs;
    gt(gi).fmin = f0 - B_occ/2;
    gt(gi).fmax = f0 + B_occ/2;

    if ~overlap.enable
        t0 = t0 + per;
    else
        switch lower(overlap.mode)
            case 'uniform'
                jit = max(0, min(1, overlap.jitter));
                dt  = (1-jit)*per + jit*per*rand;
                t0  = t0 + dt;
            case 'poisson'
                tau = per; if ~isempty(overlap.step_s), tau = overlap.step_s; end
                u   = rand; if u==0, u = eps; end
                dt  = -tau*log(u);
                t0  = t0 + dt;
            otherwise
                t0 = t0 + per;
        end
    end

    if num_it_log == cfg.how_often_to_log_signal
        fprintf('Generating signal. Time=%.2f | Total duration=%.2f | run_id=%s\n', ...
            t0, cfg.total_duration, run_id);
        num_it_log = 1;
    else
        num_it_log = num_it_log + 1;
    end
end

% ------------------------- SAVE IQ RECORD (int16 + float32) -----
iq_fname_i16 = fullfile(ds.root, sprintf('rec_%s_iq_int16.bin', run_id));
maxVal = double(intmax('int16'));
I_i16 = int16(max(min(real(x) * maxVal, maxVal), -maxVal));
Q_i16 = int16(max(min(imag(x) * maxVal, maxVal), -maxVal));
IQ_i16 = zeros(2*numel(I_i16), 1, 'int16');
IQ_i16(1:2:end) = I_i16;
IQ_i16(2:2:end) = Q_i16;
fid_i16 = fopen(iq_fname_i16, 'w'); fwrite(fid_i16, IQ_i16, 'int16'); fclose(fid_i16);

iq_fname_f32 = fullfile(ds.root, sprintf('rec_%s_iq_float32.bin', run_id));
I_f32 = single(real(x)); Q_f32 = single(imag(x));
IQ_f32 = zeros(2*numel(I_f32), 1, 'single');
IQ_f32(1:2:end) = I_f32; IQ_f32(2:2:end) = Q_f32;
fid_f32 = fopen(iq_fname_f32, 'w'); fwrite(fid_f32, IQ_f32, 'single'); fclose(fid_f32);

% ------------------------- STFT (S, F, T) -----------------------
switch lower(stft.window_type)
    case 'hann',     win = hann(stft.window_length_samples);
    case 'hamming',  win = hamming(stft.window_length_samples);
    case 'blackman', win = blackman(stft.window_length_samples);
    otherwise,       win = hann(stft.window_length_samples);
end
if stft.centered
    [S,F,T] = spectrogram(x, win, stft.overlap_samples, stft.fft_size, Fs, 'centered');
else
    [S,F,T] = spectrogram(x, win, stft.overlap_samples, stft.fft_size, Fs);
end
SdB = single(20*log10(max(abs(S), 1e-12)));  % float32

% helper: map dB -> uint16 GRAY in fixed [vmin..vmax]
vmin = render.vmin_db; vmax = render.vmax_db;
to_uint16_db = @(M) uint16(round(65535 * (min(max(M,vmin),vmax)-vmin) / (vmax - vmin + eps)));

% ------------------------- INIT HDF5 MASTER ----------------------
H = stft.fft_size;
total_frames = numel(T);
chunkW = min(1024, total_frames);

h5create(ds.h5_path, '/S_db_full', [H, total_frames], 'Datatype','single','ChunkSize',[H, max(1,chunkW)],'Deflate',4);
h5write( ds.h5_path, '/S_db_full', SdB );

h5create(ds.h5_path, '/F', [H,1], 'Datatype','single');
h5write( ds.h5_path, '/F', single(F(:)) );

h5create(ds.h5_path, '/T', [1,total_frames], 'Datatype','single','ChunkSize',[1, max(1,chunkW)],'Deflate',4);
h5write( ds.h5_path, '/T', single(T(:)') );

h5writeatt_safe(ds.h5_path, '/', 'sample_rate_hz',        Fs);
h5writeatt_safe(ds.h5_path, '/', 'fft_size',              int32(stft.fft_size));
h5writeatt_safe(ds.h5_path, '/', 'window_type',           stft.window_type);
h5writeatt_safe(ds.h5_path, '/', 'window_length_samples', int32(stft.window_length_samples));
h5writeatt_safe(ds.h5_path, '/', 'overlap_samples',       int32(stft.overlap_samples));
h5writeatt_safe(ds.h5_path, '/', 'frame_hop_samples',     int32(stft.window_length_samples - stft.overlap_samples));
h5writeatt_safe(ds.h5_path, '/', 'frame_hop_seconds',     (stft.window_length_samples - stft.overlap_samples)/Fs);
h5writeatt_safe(ds.h5_path, '/', 'freq_bin_hz',           Fs/stft.fft_size);
h5writeatt_safe(ds.h5_path, '/', 'centered',              uint8(stft.centered));
h5writeatt_safe(ds.h5_path, '/', 'vmin_db',               render.vmin_db);
h5writeatt_safe(ds.h5_path, '/', 'vmax_db',               render.vmax_db);

% ------------------------- TILES + JSON + HDF5 per-tile ----------
cfg.write_png = false;
cfg.write_json_annotations = false;

W = stft.frames_per_image;
n_tiles = ceil(total_frames / W);
image_ids = strings(0);
num_it_log = cfg.how_often_to_log_tiles;

% ------------------------- PRECOMPUTE GT INDICES -----------------
% Convert ground truth from physical units (seconds, Hz) to STFT grid
% indices (frame number, bin number). This avoids repeated find() calls
% inside the tile loop.
%
% For each pulse gi:
%   gt_c1(gi): first STFT frame index where pulse is present
%   gt_c2(gi): last STFT frame index where pulse is present
%   gt_r1(gi): first frequency bin index covering the pulse
%   gt_r2(gi): last frequency bin index covering the pulse
%
% T is the time vector from spectrogram (1 x total_frames)
% F is the frequency vector from spectrogram (H x 1)

n_gt = numel(gt);
gt_c1 = zeros(n_gt, 1);
gt_c2 = zeros(n_gt, 1);
gt_r1 = zeros(n_gt, 1);
gt_r2 = zeros(n_gt, 1);

for gi = 1:n_gt
    % Map pulse time bounds [t0, t1] to frame indices [c1, c2]
    % find(..., 'first') returns the smallest index where condition is true
    gt_c1(gi) = find(T >= gt(gi).t0, 1, 'first');
    gt_c2(gi) = find(T <= gt(gi).t1, 1, 'last');

    % Map pulse frequency bounds [fmin, fmax] to bin indices [r1, r2]
    gt_r1(gi) = find(F >= gt(gi).fmin, 1, 'first');
    gt_r2(gi) = find(F <= gt(gi).fmax, 1, 'last');
end

% Extract time bounds as arrays for vectorized tile-overlap filtering
pulse_start_time = [gt.t0];
pulse_end_time = [gt.t1];

% ------------------------- OPEN HDF5 FOR TILE WRITES -----------------
% Use low-level HDF5 API for performance: open file once, write all tiles,
% then close. High-level h5write opens/closes the file on each call.

% Create groups for tiles using high-level API (only done once)
h5_file_id = H5F.open(ds.h5_path, 'H5F_ACC_RDWR', 'H5P_DEFAULT');

s_db_group_id = H5G.create(h5_file_id, '/S_db', 'H5P_DEFAULT', 'H5P_DEFAULT', 'H5P_DEFAULT');
boxes_group_id = H5G.create(h5_file_id, '/boxes', 'H5P_DEFAULT', 'H5P_DEFAULT', 'H5P_DEFAULT');

H5G.close(s_db_group_id);
H5G.close(boxes_group_id);

% ------------------------- TILE LOOP ----------------------------------
for k = 1:n_tiles
    idx0 = (k-1)*W + 1; idx1 = min(k*W, total_frames);
    tile_id = sprintf('rec_%s_%03d', run_id, k-1);  % no seed in name
    image_ids(end+1) = tile_id;

    SdBk = SdB(:, idx0:idx1);
    if size(SdBk,2) < W
        SdBk = [SdBk, repmat(SdBk(:,end), 1, W-size(SdBk,2))];
    end

    if cfg.write_png
        I16 = to_uint16_db(double(SdBk));
        imwrite(I16, fullfile(ds.images_dir, [tile_id '.png']));
    end

    if cfg.write_json_annotations
        ann.image = struct('id',tile_id,'file_name',fullfile('images_tiles',[tile_id '.png']),...
            'width',W,'height',H);
        ann.annotations = [];
    end
    boxes_for_h5 = [];

    % Find tile time bounds for vectorized pulse filtering
    tile_start_time = T(idx0);
    tile_end_time = T(idx1);

    % Vectorized selection: find pulses that overlap this tile temporally
    % A pulse overlaps if: pulse_end >= tile_start AND pulse_start <= tile_end
    overlap_mask = (pulse_end_time >= tile_start_time) & (pulse_start_time <= tile_end_time);
    gi_list = find(overlap_mask);

    % Only iterate over pulses that overlap this tile (typically 3-5 per tile)
    for gi = gi_list
        % Use precomputed frame/bin indices
        c1 = gt_c1(gi);
        c2 = gt_c2(gi);
        r1 = gt_r1(gi);
        r2 = gt_r2(gi);

        % Convert global frame indices to tile-local x coordinates
        x0 = c1 - idx0;
        x1 = c2 - idx0;

        % Convert bin indices to 0-based y coordinates
        y0 = r1 - 1;
        y1 = r2 - 1;

        % Clip box coordinates to tile boundaries [0, W-1] x [0, H-1]
        x0c = max(0, min(W-1, x0));
        x1c = max(0, min(W-1, x1));
        y0c = max(0, min(H-1, y0));
        y1c = max(0, min(H-1, y1));
        if x1c < x0c || y1c < y0c
            continue;
        end

        box = [x0c, y0c, (x1c - x0c + 1), (y1c - y0c + 1)];
        if cfg.write_json_annotations
            a = struct('id',gi,'category_id',1,'bbox',box,'iscrowd',0);
            ann.annotations = [ann.annotations, a];
        end
        boxes_for_h5 = [boxes_for_h5; box]; %#ok<AGROW>
    end

    if cfg.write_json_annotations
        fid = fopen(fullfile(ds.ann_dir, [tile_id '.json']), 'w');
        fwrite(fid, jsonencode(ann, 'PrettyPrint', true));
        fclose(fid);
    end

    dset_S = char(tile_id);
    dset_B = char(tile_id);

    % Write spectrogram tile using low-level API
    h5_write_dataset(h5_file_id, '/S_db', dset_S, SdBk, 'single', 4);

    % Only create boxes dataset if there are boxes
    if ~isempty(boxes_for_h5)
        boxes_mat = int32(boxes_for_h5);
        h5_write_dataset(h5_file_id, '/boxes', dset_B, boxes_mat, 'int32', 1);
    end

    if num_it_log == cfg.how_often_to_log_tiles
        fprintf('Writing tiles. Tiles=%d | Total=%d | run_id=%s\n', ...
            k, n_tiles, run_id);
        num_it_log = 1;
    else
        num_it_log = num_it_log + 1;
    end
end

% Close HDF5 file after all tile writes
H5F.close(h5_file_id);

% ------------------------- META + LABEL MAP (JSON) ---------------
meta = struct( ...
    'sample_rate_hz', Fs, ...
    'fft_size', stft.fft_size, ...
    'window_type', stft.window_type, ...
    'window_length_samples', stft.window_length_samples, ...
    'overlap_samples', stft.overlap_samples, ...
    'frame_hop_samples', stft.window_length_samples - stft.overlap_samples, ...
    'frame_hop_seconds', (stft.window_length_samples - stft.overlap_samples)/Fs, ...
    'freq_bin_hz', Fs/stft.fft_size, ...
    'centered', stft.centered, ...
    'render', struct('encoding',render.encoding,'scale',render.scale,'vmin_db',vmin,'vmax_db',vmax), ...
    'frames_per_image', stft.frames_per_image, ...
    'run_id', run_id ...
    );
fid = fopen(ds.meta_path,'w'); fwrite(fid, jsonencode(meta,'PrettyPrint',true)); fclose(fid);

label_map = struct('classes', {struct('id',1,'name','QPSK')});
fid = fopen(ds.labelmap,'w'); fwrite(fid, jsonencode(label_map,'PrettyPrint',true)); fclose(fid);

% ------------------------- SPLITS -------------------------------
N = numel(image_ids);
idx = randperm(N);
n_train = round(args.NTrain*N);
n_val   = round(args.NVal*N);
train_ids = image_ids(idx(1:n_train));
val_ids   = image_ids(idx(n_train+1:n_train+n_val));
test_ids  = image_ids(idx(n_train+n_val+1:end));
write_list(fullfile(ds.splits_dir, 'train.txt'), train_ids);
write_list(fullfile(ds.splits_dir, 'val.txt'),   val_ids);
write_list(fullfile(ds.splits_dir, 'test.txt'),  test_ids);

% ------------------------- FULL-STFT FIGS (.fig + .png) ---------
save_figures = false;

if save_figures
    SdB_disp = min(max(double(SdB), vmin), vmax);
    set(0,'DefaultFigureVisible','on');

    f1 = figure('Name','STFT Full');
    imagesc(T, F, SdB_disp); axis xy; colormap parula; colorbar;
    xlabel('Time (s)'); ylabel('Frequency (Hz)'); title('STFT Full');
    savefig(f1, fullfile(ds.fig_dir,'stft_full.fig'));
    saveas(f1, fullfile(ds.fig_dir,'stft_full.png'));

    f2 = figure('Name','STFT Full + GT Boxes');
    imagesc(T, F, SdB_disp); axis xy; colormap parula; colorbar; hold on;
    for k = 1:numel(gt)
        rectangle('Position',[gt(k).t0, gt(k).fmin, max(gt(k).t1-gt(k).t0,eps), max(gt(k).fmax-gt(k).fmin,eps)], ...
            'EdgeColor',[1 0 0],'LineWidth',3);
    end
    hold off; xlabel('Time (s)'); ylabel('Frequency (Hz)'); title('STFT Full + GT Boxes');
    savefig(f2, fullfile(ds.fig_dir,'stft_full_boxes.fig'));
    saveas(f2, fullfile(ds.fig_dir,'stft_full_boxes.png'));
end

fprintf('Done. Tiles=%d | train=%d val=%d test=%d | overlap=%d (%s) | run_id=%s\n', ...
    N, numel(train_ids), numel(val_ids), numel(test_ids), overlap.enable, overlap.mode, run_id);
end

% ------------------------- HELPERS -------------------------------
function write_list(path, arr)
if isempty(arr), return; end
fid = fopen(path, 'w');
for i=1:numel(arr), fprintf(fid, '%s\n', arr(i)); end
fclose(fid);
end

function h5writeatt_safe(h5path, loc, key, val)
if islogical(val), val = uint8(val); end
if isa(val,'string'), val = char(val); end
h5writeatt(h5path, loc, key, val);
end

function h5_write_dataset(file_id, group_path, dset_name, data, dtype, deflate_level)
% Write a dataset to an open HDF5 file using low-level API.
% This avoids the overhead of h5write which opens/closes the file each call.
%
% Parameters:
%   file_id      - HDF5 file identifier (from H5F.open)
%   group_path   - Parent group path (e.g., '/S_db')
%   dset_name    - Dataset name within the group
%   data         - Data to write
%   dtype        - 'single' or 'int32'
%   deflate_level - Compression level (0-9), 0 = no compression

% Get HDF5 type
switch dtype
    case 'single'
        h5_type = 'H5T_NATIVE_FLOAT';
    case 'int32'
        h5_type = 'H5T_NATIVE_INT32';
    otherwise
        error('Unsupported dtype: %s', dtype);
end

% Data dimensions (MATLAB is column-major, HDF5 is row-major, so flip)
dims = fliplr(size(data));

% Create dataspace
space_id = H5S.create_simple(numel(dims), dims, dims);

% Create dataset creation property list with chunking and compression
dcpl_id = H5P.create('H5P_DATASET_CREATE');
if deflate_level > 0
    H5P.set_chunk(dcpl_id, dims);
    H5P.set_deflate(dcpl_id, deflate_level);
end

% Open parent group and create dataset
group_id = H5G.open(file_id, group_path);
dset_id = H5D.create(group_id, dset_name, h5_type, space_id, dcpl_id);

% Write data
H5D.write(dset_id, 'H5ML_DEFAULT', 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT', data);

% Cleanup
H5D.close(dset_id);
H5G.close(group_id);
H5P.close(dcpl_id);
H5S.close(space_id);
end
