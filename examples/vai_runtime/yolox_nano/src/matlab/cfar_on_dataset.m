clear; close all;

%% ------------------------------------------------------------------------
%  Paths
% -------------------------------------------------------------------------
imgDir = "dataset20251105_153340/images/train";   % folder with spectrogram PNGs
lblDir = "dataset20251105_153340/labels/train";   % folder with YOLO .txt annotations
outDir = "cfar_results";                          % output folder for images with BBs

if ~isfolder(imgDir)
    error('Image directory not found: %s', imgDir);
end
if ~isfolder(lblDir)
    error('Label directory not found: %s', lblDir);
end
if ~isfolder(outDir)
    mkdir(outDir);
end

% Must match dataset generator
render.vmin_db = -90;
render.vmax_db = -20;

%% ------------------------------------------------------------------------
%  1D OS-CFAR configuration (per column, along rows = frequency bins)
% -------------------------------------------------------------------------
guardPerSide = 10;     % guard cells on each side of CUT
trainPerSide = 40;     % training cells on each side of CUT
pfa          = 1e-5;   % desired probability of false alarm

% OS-CFAR rank as fraction of training cells (0..1) (~0.6–0.8 typical)
rankFrac     = 0.6;

% Morphology / ROI extraction (for CFAR boxes)
closeKernel  = [3 3];  % morphological closing kernel (rows x cols)
roiMinArea   = 2;      % min ROI area in pixels (set 1 to disable)

% Object-level matching tolerance (in bins/pixels)
edgeTol = 5;           % max allowed diff for each box edge (left/right/top/bottom)

%% ------------------------------------------------------------------------
%  Configure OS-CFAR detector (1D)
% -------------------------------------------------------------------------
numTrainingCells = 2 * trainPerSide;
numGuardCells    = 2 * guardPerSide;

rank = max(1, min(numTrainingCells, round(rankFrac * numTrainingCells)));

cfar1d = phased.CFARDetector( ...
    'Method',               'OS', ...
    'NumTrainingCells',     numTrainingCells, ...
    'NumGuardCells',        numGuardCells, ...
    'ThresholdFactor',      'Auto', ...
    'Rank',                 rank, ...
    'ProbabilityFalseAlarm',pfa, ...
    'OutputFormat',         'CUT result', ...
    'ThresholdOutputPort',  true);

fprintf('OS-CFAR config: trainPerSide=%d, guardPerSide=%d, rank=%d (%.0f%%), Pfa=%.1e\n', ...
    trainPerSide, guardPerSide, rank, 100*rank/numTrainingCells, pfa);

%% ------------------------------------------------------------------------
%  CFAR padding (block padding)
% -------------------------------------------------------------------------
padSize = trainPerSide + guardPerSide;   % cells to pad on each side

%% ------------------------------------------------------------------------
%  Object-level stats: TP / FP / FN over all images
% -------------------------------------------------------------------------
totalGT = 0;   % total number of GT objects (signals); GT = Ground Truth
totalTP = 0;   % number of correctly detected GT objects; TP = True Positive
totalFP = 0;   % number of extra CFAR detections; FP = False Positive
totalFN = 0;   % number of missed GT objects; FN = False Negative

%% ------------------------------------------------------------------------
%  Enumerate images
% -------------------------------------------------------------------------
imgFiles = dir(fullfile(imgDir, "*.png"));
if isempty(imgFiles)
    error('No PNG images found in %s', imgDir);
end

for idx = 1:numel(imgFiles)
    imgName = imgFiles(idx).name;
    imgPath = fullfile(imgDir, imgName);

    %% ------------------------------------------------------------
    %  Load spectrogram and convert uint16 → dB → linear power
    % -------------------------------------------------------------
    I16 = imread(imgPath);               % uint16, H x W

    vmin = render.vmin_db;
    vmax = render.vmax_db;

    SdB = vmin + double(I16) * (vmax - vmin) / 65535;   % H x W, dB
    Slin = 10.^(SdB/10);                                % H x W

    [H, W] = size(Slin);

    %% ------------------------------------------------------------
    %  Vectorized OS-CFAR with block padding (along frequency)
    % -------------------------------------------------------------
    topPad    = Slin(1:padSize,        :);          % padSize x W
    bottomPad = Slin(end-padSize+1:end, :);         % padSize x W

    SlinPad = [topPad; Slin; bottomPad];                % (H + 2*padSize) x W
    CUTIdxPadded = (1:H) + padSize;

    [detSub, thrSub] = cfar1d(SlinPad, CUTIdxPadded);   % each H x W

    detMask      = logical(detSub);                     % H x W
    thrLinMatrix = thrSub;                              % H x W (power)

    %% ------------------------------------------------------------
    %  Morphology → CFAR ROI mask + CFAR boxes (bboxesCFAR)
    % -------------------------------------------------------------
    detMaskClean = imclose(detMask, strel('rectangle', closeKernel));
    detMaskClean = imfill(detMaskClean, 'holes');

    if roiMinArea > 1
        detMaskForRoi = bwareaopen(detMaskClean, roiMinArea);
    else
        detMaskForRoi = detMaskClean;
    end

    statsCFAR = regionprops(detMaskForRoi, 'BoundingBox');

    if isempty(statsCFAR)
        bboxesCFAR = zeros(0,4);
    else
        bb = vertcat(statsCFAR.BoundingBox);           % N x 4
        x0 = floor(bb(:,1));
        y0 = floor(bb(:,2));
        w  = ceil(bb(:,3));
        h  = ceil(bb(:,4));

        x0 = max(0, min(W-1, x0));
        y0 = max(0, min(H-1, y0));
        w  = max(1, min(W - x0, w));
        h  = max(1, min(H - y0, h));

        bboxesCFAR = [x0, y0, w, h];                  % [x y w h]
    end

    %% ------------------------------------------------------------
    %  Read YOLO GT labels and build GT boxes in pixels (bboxesGT)
    % -------------------------------------------------------------
    [~, baseName, ~] = fileparts(imgName);
    lblPath = fullfile(lblDir, baseName + ".txt");

    bboxesGT = zeros(0,4);  % [x y w h]

    if isfile(lblPath)
        fid = fopen(lblPath, 'r');
        gtList = [];
        while true
            line = fgetl(fid);
            if ~ischar(line), break; end
            if isempty(strtrim(line)), continue; end

            vals = sscanf(line, '%f');
            if numel(vals) < 5
                continue;
            end
            % YOLO: class xc yc w h (normalized 0..1)
            xc = vals(2); yc = vals(3);
            bw = vals(4); bh = vals(5);

            boxW = bw * W;
            boxH = bh * H;
            xCenter = xc * W;
            yCenter = yc * H;

            x0 = xCenter - boxW/2;
            y0 = yCenter - boxH/2;

            % Convert to integer pixel coordinates
            x1 = max(1, floor(x0) + 1);
            y1 = max(1, floor(y0) + 1);
            x2 = min(W, ceil(x0 + boxW));
            y2 = min(H, ceil(y0 + boxH));

            if x2 >= x1 && y2 >= y1
                gtList = [gtList; x1, y1, (x2-x1+1), (y2-y1+1)];
            end
        end
        fclose(fid);
        bboxesGT = gtList;
    end

    numGT  = size(bboxesGT,  1);
    numDet = size(bboxesCFAR,1);

    totalGT = totalGT + numGT;

    %% ------------------------------------------------------------
    %  Object-level matching: CFAR boxes vs GT boxes with edgeTol
    % -------------------------------------------------------------
    % For each GT box we try to find ONE matching CFAR box such that
    %   |left_det - left_gt|   <= edgeTol
    %   |right_det - right_gt| <= edgeTol
    %   |top_det - top_gt|     <= edgeTol
    %   |bottom_det - bot_gt|  <= edgeTol

    gtMatched  = false(numGT,1);
    detMatched = false(numDet,1);

    TP_img = 0;
    FN_img = 0;
    FP_img = 0;

    % Precompute GT edges
    if numGT > 0
        xg = bboxesGT(:,1);
        yg = bboxesGT(:,2);
        wg = bboxesGT(:,3);
        hg = bboxesGT(:,4);

        left_g   = xg;
        top_g    = yg;
        right_g  = xg + wg - 1;
        bottom_g = yg + hg - 1;
    else
        left_g = []; top_g = []; right_g = []; bottom_g = [];
    end

    % Precompute CFAR edges
    if numDet > 0
        xd = bboxesCFAR(:,1);
        yd = bboxesCFAR(:,2);
        wd = bboxesCFAR(:,3);
        hd = bboxesCFAR(:,4);

        left_d   = xd;
        top_d    = yd;
        right_d  = xd + wd - 1;
        bottom_d = yd + hd - 1;
    else
        left_d = []; top_d = []; right_d = []; bottom_d = [];
    end

    for g = 1:numGT
        matchIdx = 0;
        for d = 1:numDet
            if detMatched(d)
                continue;
            end
            if abs(left_d(d)   - left_g(g))   <= edgeTol && ...
               abs(right_d(d)  - right_g(g))  <= edgeTol && ...
               abs(top_d(d)    - top_g(g))    <= edgeTol && ...
               abs(bottom_d(d) - bottom_g(g)) <= edgeTol
                matchIdx = d;
                break;
            end
        end
        if matchIdx > 0
            gtMatched(g)  = true;
            detMatched(matchIdx) = true;
            TP_img = TP_img + 1;
        else
            FN_img = FN_img + 1;
        end
    end

    FP_img = sum(~detMatched);

    totalTP = totalTP + TP_img;
    totalFN = totalFN + FN_img;
    totalFP = totalFP + FP_img;

    %% ------------------------------------------------------------
    %  Save annotated image with CFAR & GT BBs
    % -------------------------------------------------------------
    timeBins = 1:W;
    freqBins = 1:H;

    fig = figure('Visible','off');
    imagesc(timeBins, freqBins, SdB);
    axis xy;
    colormap parula; colorbar;
    title(sprintf('CFAR vs GT: %s', imgName), 'Interpreter','none');
    xlabel('Time bins'); ylabel('Frequency bins');
    hold on;

    % CFAR boxes (red)
    for i = 1:size(bboxesCFAR,1)
        rectangle('Position', bboxesCFAR(i,:), ...
            'EdgeColor','r', 'LineWidth', 1.5);
    end

    % GT boxes (green)
    for i = 1:size(bboxesGT,1)
        rectangle('Position', bboxesGT(i,:), ...
            'EdgeColor','g', 'LineWidth', 1.2);
    end

    hold off;

    [~, baseName, ~] = fileparts(imgName);
    outPath = fullfile(outDir, baseName + "_cfar_obj.png");
    exportgraphics(fig, outPath, 'Resolution',150);
    close(fig);

    fprintf('Processed %3d/%3d: %s | GT=%d, TP=%d, FN=%d, FP=%d\n', ...
        idx, numel(imgFiles), imgName, numGT, TP_img, FN_img, FP_img);
end

%% ------------------------------------------------------------------------
%  Final object-level Pd / Pfd results
% -------------------------------------------------------------------------
if totalGT == 0
    warning('No GT objects found in dataset. Pd/Pfd undefined.');
    Pd_obj  = NaN;
    Pfd_obj = NaN;
else
    Pd_obj  = totalTP / totalGT;
    Pfd_obj = totalFP / totalGT;
end

fprintf('\n=== CFAR object-level performance over dataset ===\n');
fprintf('Total GT objects  : %d\n', totalGT);
fprintf('Total TP (matched): %d\n', totalTP);
fprintf('Total FN (missed) : %d\n', totalFN);
fprintf('Total FP (extra)  : %d\n', totalFP);
fprintf('Pd_object  = TP / GT  = %.6f\n', Pd_obj);
fprintf('Pfd_object = FP / GT  = %.6f\n', Pfd_obj);
