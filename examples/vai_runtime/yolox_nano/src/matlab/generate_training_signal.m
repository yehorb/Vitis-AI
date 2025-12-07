clear; clc;

% run/session identifier for filenames
run_id = datestr(now,'yyyymmdd_HHMMSS');  % e.g., 20250928_231045

% ------------------------- DATASET PATHS -------------------------
ds.root        = "../../build/dataset/" + run_id;
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
stft.frames_per_image      = 128;    % tile width (W); tile height (H)=fft_size

% ------------------------- SIGNAL GENERATION ---------------------
cfg.seed           = 1;          % RNG for reproducible signal
cfg.total_duration = 0.025;      % seconds
cfg.noise_power_db = -80;        % mean noise power in dB

rngs.freq_min = -5e6; rngs.freq_max = 5e6;
rngs.snr_min  = 5;    rngs.snr_max  = 40;
rngs.pw_min   = 20e-6;  rngs.pw_max = 120e-6; 
rngs.per_min  = 150e-6; rngs.per_max = 600e-6;
rngs.bw_min   = 0.1e6;  rngs.bw_max  = 2e6;    

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
if ~exist(ds.images_dir, 'dir'), mkdir(ds.images_dir); end
if ~exist(ds.ann_dir,    'dir'), mkdir(ds.ann_dir);  end
if ~exist(ds.splits_dir, 'dir'), mkdir(ds.splits_dir); end
if ~exist(ds.fig_dir,    'dir'), mkdir(ds.fig_dir);   end
if ~exist(ds.tensors_dir,'dir'), mkdir(ds.tensors_dir); end
if exist(ds.h5_path,'file'), delete(ds.h5_path); end

% ------------------------- NOISE + PULSES ------------------------
Fs      = stft.sample_rate_hz;
N_total = round(cfg.total_duration * Fs);
rng(cfg.seed);

Pn = 10^(cfg.noise_power_db/10);
x  = sqrt(Pn/2) * (randn(N_total,1) + 1j*randn(N_total,1));  % flat noise over whole record

gt = struct('t0',{},'t1',{},'fmin',{},'fmax',{});
t0 = 0; pulse_idx = 0;

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
    Rs  = Fs/sps;
    B_occ = (1+alpha) * Rs;

    f0 = min(max(f0, -Fs/2*0.9 + B_occ/2), Fs/2*0.9 - B_occ/2);

    pulse_idx = pulse_idx + 1; rng(qpsk.seed_base + pulse_idx);
    m = randi([0 3], ceil(numel(t_p)/sps), 1);
    map = (1/sqrt(2))*[1+1j; -1+1j; -1-1j; 1-1j];
    s  = map(m+1);

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
                u = rand; if u==0, u = eps; end
                dt = -tau*log(u);
                t0 = t0 + dt;
            otherwise
                t0 = t0 + per;
        end
    end
end

% ------------------------- SAVE IQ RECORD (int16 + float32) -----
iq_fname_i16 = fullfile(ds.root, sprintf('rec_%s_iq_int16.bin', run_id));
maxVal = double(intmax('int16'));
I_i16 = int16(max(min(real(x) * maxVal,  maxVal), -maxVal));
Q_i16 = int16(max(min(imag(x) * maxVal,  maxVal), -maxVal));
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
    case 'hann',    win = hann(stft.window_length_samples);
    case 'hamming', win = hamming(stft.window_length_samples);
    case 'blackman',win = blackman(stft.window_length_samples);
    otherwise,      win = hann(stft.window_length_samples);
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
W = stft.frames_per_image;
n_tiles = ceil(total_frames / W);
image_ids = strings(0);

for k = 1:n_tiles
    idx0 = (k-1)*W + 1; idx1 = min(k*W, total_frames);
    tile_id = sprintf('rec_%s_%03d', run_id, k-1);  % no seed in name
    image_ids(end+1) = tile_id;

    SdBk = SdB(:, idx0:idx1);
    if size(SdBk,2) < W
        SdBk = [SdBk, repmat(SdBk(:,end), 1, W-size(SdBk,2))];
    end

    I16 = to_uint16_db(double(SdBk));
    imwrite(I16, fullfile(ds.images_dir, [tile_id '.png']));

    ann.image = struct('id',tile_id,'file_name',fullfile('images_tiles',[tile_id '.png']),...
                       'width',W,'height',H);
    ann.annotations = [];
    boxes_for_h5 = [];

    for gi = 1:numel(gt)
        c1 = find(T >= gt(gi).t0, 1, 'first');  if isempty(c1), continue; end
        c2 = find(T <= gt(gi).t1, 1, 'last');   if isempty(c2), continue; end
        r1 = find(F >= gt(gi).fmin, 1, 'first'); if isempty(r1), continue; end
        r2 = find(F <= gt(gi).fmax, 1, 'last');  if isempty(r2), continue; end

        x0 = c1 - idx0; x1 = c2 - idx0;     % frames
        y0 = r1 - 1;    y1 = r2 - 1;       % bins (0-based)

        % reject non-intersecting BEFORE clipping
        if x1 < 0 || x0 > (W-1) || y1 < 0 || y0 > (H-1)
            continue;
        end

        % clip into tile
        x0c = max(0, min(W-1, x0));
        x1c = max(0, min(W-1, x1));
        y0c = max(0, min(H-1, y0));
        y1c = max(0, min(H-1, y1));
        if x1c < x0c || y1c < y0c
            continue;
        end

        box = [x0c, y0c, (x1c - x0c + 1), (y1c - y0c + 1)];
        a = struct('id',gi,'category_id',1,'bbox',box,'iscrowd',0);
        ann.annotations = [ann.annotations, a];
        boxes_for_h5 = [boxes_for_h5; box]; %#ok<AGROW>
    end

    fid = fopen(fullfile(ds.ann_dir, [tile_id '.json']), 'w');
    fwrite(fid, jsonencode(ann, 'PrettyPrint', true));
    fclose(fid);

    dset_S = ['/S_db/' char(tile_id)];
    dset_B = ['/boxes/' char(tile_id)];
    h5create_safe(ds.h5_path, dset_S, size(SdBk), 'single', [H, min(W, size(SdBk,2))], 4);
    h5write(ds.h5_path, dset_S, SdBk);

    if isempty(boxes_for_h5), boxes_mat = int32(zeros(0,4));
    else,                    boxes_mat = int32(boxes_for_h5);
    end
    h5create_safe(ds.h5_path, dset_B, size(boxes_mat), 'int32', [max(1,size(boxes_mat,1)), 4], 1);
    if ~isempty(boxes_mat)
        h5write(ds.h5_path, dset_B, boxes_mat);
    end
end

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
n_train = round(0.8*N);
n_val   = round(0.1*N);
train_ids = image_ids(idx(1:n_train));
val_ids   = image_ids(idx(n_train+1:n_train+n_val));
test_ids  = image_ids(idx(n_train+n_val+1:end));
write_list(fullfile(ds.splits_dir, 'train.txt'), train_ids);
write_list(fullfile(ds.splits_dir, 'val.txt'),   val_ids);
write_list(fullfile(ds.splits_dir, 'test.txt'),  test_ids);

% ------------------------- FULL-STFT FIGS (.fig + .png) ---------
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

fprintf('Done. Tiles=%d | train=%d val=%d test=%d | overlap=%d (%s) | run_id=%s\n', ...
    N, numel(train_ids), numel(val_ids), numel(test_ids), overlap.enable, overlap.mode, run_id);

% ------------------------- HELPERS -------------------------------
function write_list(path, arr)
fid = fopen(path, 'w');
for i=1:numel(arr), fprintf(fid, '%s\n', arr(i)); end
fclose(fid);
end

function h5writeatt_safe(h5path, loc, key, val)
if islogical(val), val = uint8(val); end
if isa(val,'string'), val = char(val); end
h5writeatt(h5path, loc, key, val);
end

function h5create_safe(h5path, dset, sz, dtype, chunks, deflate)
if exist(h5path,'file')
    try, h5info(h5path, dset); return; catch, end
end
args = {'Datatype', dtype};
if ~isempty(chunks), args = [args, {'ChunkSize', chunks}]; end
if ~isempty(deflate) && deflate>0, args = [args, {'Deflate', deflate}]; end
h5create(h5path, dset, sz, args{:});
end
