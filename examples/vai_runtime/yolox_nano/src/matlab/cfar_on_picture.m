clear; close all;

%% ------------------------------------------------------------------------
%  User parameters
% -------------------------------------------------------------------------
% Path to one spectrogram tile from your dataset
imgPath = "dataset20251105_153340/images/train/rec_20251105_153340_018.png";

% Must match dataset generator
render.vmin_db = -90;
render.vmax_db = -20;

% 1D OS-CFAR configuration (per column, along rows => process frequency bins)
guardPerSide = 10;     % guard cells on each side of CUT
trainPerSide = 40;     % training cells on each side of CUT
pfa          = 1e-5;   % desired probability of false alarm

% OS-CFAR rank as fraction of training cells (0..1)
rankFrac     = 0.6;

% Morphology / ROI extraction (for boxes)
closeKernel  = [3 3];  % morphological closing kernel (rows x cols)
roiMinArea   = 2;      % min ROI area in pixels (set 1 to disable)


%% ------------------------------------------------------------------------
%  Load spectrogram tile and convert uint16 → dB → linear power
% -------------------------------------------------------------------------
if ~isfile(imgPath)
    error('Image file not found: %s', imgPath);
end

I16 = imread(imgPath);               % uint16, H x W

vmin = render.vmin_db;
vmax = render.vmax_db;

% Decode back to dB (inverse of dataset encoding)
SdB = vmin + double(I16) * (vmax - vmin) / 65535;   % H x W, in dB

% dB -> linear power
% SdB = 20*log10(|S|) → |S|^2 (power) = 10^(SdB/10)
Slin = 10.^(SdB/10);                 % H x W

[H, W] = size(Slin);

%% ------------------------------------------------------------------------
%  Configure OS-CFAR detector (1D)
% -------------------------------------------------------------------------
numTrainingCells = 2 * trainPerSide;   % total training cells
numGuardCells    = 2 * guardPerSide;   % total guard cells

% Compute OS rank from fraction
rank = max(1, min(numTrainingCells, round(rankFrac * numTrainingCells)));

cfar1d = phased.CFARDetector( ...
    'Method',               'OS', ...
    'NumTrainingCells',     numTrainingCells, ...
    'NumGuardCells',        numGuardCells, ...
    'ThresholdFactor',      'Auto', ...
    'Rank',                 rank, ...
    'ProbabilityFalseAlarm',pfa, ...
    'OutputFormat',         'CUT result', ...
    'ThresholdOutputPort',  true);     % also output threshold

% Padding size at top/bottom so training+guard fit for all original rows
padSize = trainPerSide + guardPerSide;   % cells to pad on each side

% CUT indices in the padded column corresponding to original rows 1..H
CUTIdxPadded = (1:H) + padSize;         % length H


%% ------------------------------------------------------------------------
%  Run 1D OS-CFAR on all columns at once (vectorized, with block padding)
% -------------------------------------------------------------------------
% Block padding by copying top/bottom blocks for all columns:
% topPad: first padSize rows, bottomPad: last padSize rows
topPad    = Slin(1:padSize,    :);           % padSize x W
bottomPad = Slin(end-padSize+1:end, :);      % padSize x W

SlinPad = [topPad; Slin; bottomPad];             % (H + 2*padSize) x W

% Single CFAR call for all columns:
[detSub, thrSub] = cfar1d(SlinPad, CUTIdxPadded);   % each H x W

% detSub/thrSub already correspond to original rows (via CUTIdxPadded)
detMask      = logical(detSub);      % H x W
thrLinMatrix = thrSub;               % H x W (power)

%% ------------------------------------------------------------------------
%  Morphology and ROI mask
% -------------------------------------------------------------------------
detMaskClean = imclose(detMask, strel('rectangle', closeKernel));
detMaskClean = imfill(detMaskClean, 'holes');

if roiMinArea > 1
    detMaskForRoi = bwareaopen(detMaskClean, roiMinArea);
else
    detMaskForRoi = detMaskClean;
end

%% ------------------------------------------------------------------------
%  Extract bounding boxes from OS-CFAR mask (vectorized)
% -------------------------------------------------------------------------
stats = regionprops(detMaskForRoi, 'BoundingBox');

if isempty(stats)
    bboxes = zeros(0,4);
else
    % Collect all BoundingBox rows into one matrix [x y w h]
    bb = vertcat(stats.BoundingBox);          % N x 4

    x0 = floor(bb(:,1));
    y0 = floor(bb(:,2));
    w  = ceil(bb(:,3));
    h  = ceil(bb(:,4));

    % Clamp to image bounds (vectorized)
    x0 = max(0, min(W-1, x0));
    y0 = max(0, min(H-1, y0));
    w  = max(1, min(W - x0, w));
    h  = max(1, min(H - y0, h));

    bboxes = [x0, y0, w, h];                  % N x 4
end


%% ------------------------------------------------------------------------
%  Convert threshold to dB, prepare grids
% -------------------------------------------------------------------------
thr_dB = 10 * log10(thrLinMatrix + eps);  % threshold in dB

% Time and frequency bin indices
timeBins = 1:W;        % columns
freqBins = 1:H;        % rows (1 = lowest freq at bottom, H = highest at top)
[T, F]   = meshgrid(timeBins, freqBins);

%% ------------------------------------------------------------------------
%  Visualization: 2D + combined 3D (signal + threshold) + 2D boxes
% -------------------------------------------------------------------------
figure('Name','OS-CFAR detection results');

% (1) 2D view with OS-CFAR raw detections
subplot(2,2,1);
imagesc(timeBins, freqBins, SdB);
axis xy;
colormap parula; colorbar;
title('Detections (raw)');
xlabel('Time bins'); ylabel('Frequency bins');
hold on;
% Overlay CFAR detections as red points
[rowDet, colDet] = find(detMask);   % indices where CUT > threshold
plot(colDet, rowDet, 'r.', 'MarkerSize', 6);
hold off;

% 2D view with OS-CFAR detections (contour)
subplot(2,2,2);
imagesc(timeBins, freqBins, SdB);
axis xy;
colormap parula; colorbar;
title('Detection (raw -> morphology -> contours)');
xlabel('Time bins'); ylabel('Frequency bins');
hold on;
contour(timeBins, freqBins, detMaskForRoi, [0.5 0.5], 'w', 'LineWidth', 1);
hold off;

% Combined 3D: signal (surface) + OS-CFAR threshold (mesh)
subplot(2,2,3);
surf(T, F, SdB, 'EdgeColor', 'none');           % signal surface
hold on;
mesh(T, F, thr_dB, 'EdgeColor', [0.2 0.2 0.2], ...
     'LineStyle', '-', 'LineWidth', 0.5);       % threshold mesh (dark gray)
hold off;
axis tight;
xlabel('Time bins'); ylabel('Frequency bins'); zlabel('Level (dB)');
title('Signal and OS-CFAR threshold (3D)');
colormap parula;
view(45, 30);

% Reserved subplot
subplot(2,2,4);
axis off;

% 2D detections with red bounding boxes
subplot(2,2,4);
imagesc(timeBins, freqBins, SdB);
axis xy;
colormap parula; colorbar;
title('Detection (bounding boxes)');
xlabel('Time bins'); ylabel('Frequency bins');
hold on;
for i = 1:size(bboxes,1)
    rectangle('Position', bboxes(i,:), ...
        'EdgeColor','r', 'LineWidth', 1.5);
end
hold off;
