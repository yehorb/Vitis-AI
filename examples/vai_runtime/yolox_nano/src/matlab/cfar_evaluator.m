function [recall, precision, summary] = cfar_evaluator(dataset_root, varargin)
%CFAR_EVALUATOR Evaluate 2D CFAR detector on STFT dataset.
%
%   [R, P, SUMMARY] = CFAR_EVALUATOR(DATASET_ROOT) runs a 2D CA-CFAR
%   detector on the STFT tiles stored under DATASET_ROOT (as produced by
%   generate_training_signal.m), using the test split, and computes
%   precision/recall/F1 in the same way as the YOLOX evaluator.
%
%   Optional name-value pairs:
%       'Pfa'            - Probability of false alarm (default 1e-4)
%       'GuardBandSize'  - [rows cols] guard band (default [4 4])
%       'TrainingBandSize' - [rows cols] training band (default [16 16])
%       'IoUThreshold'   - IoU threshold for matching (default 0.5)
%       'MinArea'        - Minimum blob area in pixels (default 3)

    p = inputParser;
    addParameter(p, 'Pfa', 1e-4, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'GuardBandSize', [4 4], @(x) isnumeric(x) && numel(x) == 2);
    addParameter(p, 'TrainingBandSize', [16 16], @(x) isnumeric(x) && numel(x) == 2);
    addParameter(p, 'IoUThreshold', 0.5, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'MinArea', 3, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    parse(p, varargin{:});
    cfg = p.Results;

    if isstring(dataset_root) || ischar(dataset_root)
        dataset_root = char(dataset_root);
    else
        error('dataset_root must be a string or char path.');
    end

    h5_path = fullfile(dataset_root, 'tensors', 'tiles.h5');
    splits_dir = fullfile(dataset_root, 'splits');
    test_list_path = fullfile(splits_dir, 'test.txt');

    if ~isfile(h5_path)
        error('HDF5 file not found: %s', h5_path);
    end
    if ~isfile(test_list_path)
        error('Test split file not found: %s', test_list_path);
    end

    test_ids = read_id_list(test_list_path);
    n_images = numel(test_ids);
    if n_images == 0
        error('Test split is empty.');
    end

    % Simple progress logging
    log_every = max(1, floor(n_images / 20));  % ~5%% steps
    fprintf('CFAR evaluator: %d tiles in test split. Processing...\n', n_images);

    boxes_info = h5info(h5_path, '/boxes');
    box_names = string({boxes_info.Datasets.Name});

    cfar2d = phased.CFARDetector2D( ...
        'Method', 'CA', ...
        'GuardBandSize',    cfg.GuardBandSize, ...
        'TrainingBandSize', cfg.TrainingBandSize, ...
        'ProbabilityFalseAlarm', cfg.Pfa, ...
        'OutputFormat', 'CUT result', ...
        'ThresholdOutputPort', false);

    total_tp = 0;
    total_fp = 0;
    total_fn = 0;
    total_gt = 0;
    total_pred = 0;

    inference_time = 0.0;

    for k = 1:n_images
        if mod(k, log_every) == 0 || k == 1 || k == n_images
            fprintf('  CFAR evaluator: tile %d / %d (%.1f%%%%)\n', ...
                k, n_images, 100 * k / n_images);
        end

        tile_id = strtrim(test_ids(k));
        tile_id = char(tile_id);

        s_db = h5read(h5_path, ['/S_db/' tile_id]);
        s_db = double(s_db);
        s_power = 10.^(s_db / 10.0);

        t_start = tic;
        det_mask = cfar2d(s_power);
        det_mask = logical(det_mask);

        if cfg.MinArea > 1
            det_mask = bwareaopen(det_mask, round(cfg.MinArea));
        end

        cc = bwconncomp(det_mask);
        if cc.NumObjects == 0
            pred_boxes = zeros(0, 4);
        else
            stats = regionprops(cc, 'BoundingBox');
            bb = vertcat(stats.BoundingBox);
            H = size(det_mask, 1);
            W = size(det_mask, 2);
            x0 = bb(:, 1) - 1;
            y0 = bb(:, 2) - 1;
            w  = bb(:, 3);
            h  = bb(:, 4);
            x0 = round(x0);
            y0 = round(y0);
            w  = round(w);
            h  = round(h);
            x0 = max(0, min(W - 1, x0));
            y0 = max(0, min(H - 1, y0));
            w  = max(1, min(W - x0, w));
            h  = max(1, min(H - y0, h));
            pred_boxes = [x0, y0, w, h];
        end

        tile_time = toc(t_start);
        inference_time = inference_time + tile_time;

        if any(box_names == string(tile_id))
            gt_boxes = h5read(h5_path, ['/boxes/' tile_id]);
            gt_boxes = double(gt_boxes);
            if size(gt_boxes, 2) ~= 4 && size(gt_boxes, 1) == 4
                gt_boxes = gt_boxes.';
            end
        else
            gt_boxes = zeros(0, 4);
        end

        [tp, fp, fn, ngt, npred] = eval_tile_boxes(gt_boxes, pred_boxes, cfg.IoUThreshold);

        total_tp = total_tp + tp;
        total_fp = total_fp + fp;
        total_fn = total_fn + fn;
        total_gt = total_gt + ngt;
        total_pred = total_pred + npred;
    end

    precision = total_tp / (total_tp + total_fp);
    if ~isfinite(precision)
        precision = 0.0;
    end
    recall = total_tp / (total_tp + total_fn);
    if ~isfinite(recall)
        recall = 0.0;
    end
    if (precision + recall) > 0
        f1 = 2 * precision * recall / (precision + recall);
    else
        f1 = 0.0;
    end

    avg_time_ms = 1000 * inference_time / n_images;

    summary = sprintf([ ...
        'STFT Evaluation Results\n', ...
        '-----------------------\n', ...
        'Total GT boxes:   %d\n', ...
        'Total predictions:%d\n', ...
        'TP: %d, FP: %d, FN: %d\n', ...
        'Precision: %.4f\n', ...
        'Recall:    %.4f\n', ...
        'F1 Score:  %.4f\n', ...
        'Avg inference time: %.2f ms/image\n' ...
        ], total_gt, total_pred, total_tp, total_fp, total_fn, precision, recall, f1, avg_time_ms);

    fprintf('%s\n', summary);
end


function ids = read_id_list(path)
    txt = fileread(path);
    if isempty(txt)
        ids = strings(0, 1);
    else
        parts = regexp(txt, '\r?\n', 'split');
        parts = parts(~cellfun('isempty', parts));
        ids = string(parts(:));
    end
end


function [tp, fp, fn, n_gt, n_pred] = eval_tile_boxes(gt_boxes, pred_boxes, iou_thr)
    if isempty(gt_boxes)
        n_gt = 0;
    else
        n_gt = size(gt_boxes, 1);
    end
    if isempty(pred_boxes)
        n_pred = 0;
    else
        n_pred = size(pred_boxes, 1);
    end

    if n_pred == 0 && n_gt == 0
        tp = 0; fp = 0; fn = 0;
        return;
    elseif n_pred == 0
        tp = 0; fp = 0; fn = n_gt;
        return;
    elseif n_gt == 0
        tp = 0; fp = n_pred; fn = 0;
        return;
    end

    iou_mat = compute_iou_matrix(gt_boxes, pred_boxes);

    tp = 0;
    matched_gt = false(n_gt, 1);
    for j = 1:n_pred
        ious = iou_mat(:, j);
        [best_iou, best_idx] = max(ious);
        if best_iou >= iou_thr && ~matched_gt(best_idx)
            tp = tp + 1;
            matched_gt(best_idx) = true;
        end
    end

    fp = n_pred - tp;
    fn = n_gt - sum(matched_gt);
end


function iou_mat = compute_iou_matrix(gt_boxes, pred_boxes)
    n_gt = size(gt_boxes, 1);
    n_pred = size(pred_boxes, 1);
    iou_mat = zeros(n_gt, n_pred);
    for i = 1:n_gt
        for j = 1:n_pred
            iou_mat(i, j) = bbox_iou_xywh(gt_boxes(i, :), pred_boxes(j, :));
        end
    end
end


function iou = bbox_iou_xywh(a, b)
    ax1 = a(1);
    ay1 = a(2);
    ax2 = a(1) + a(3) - 1;
    ay2 = a(2) + a(4) - 1;

    bx1 = b(1);
    by1 = b(2);
    bx2 = b(1) + b(3) - 1;
    by2 = b(2) + b(4) - 1;

    ix1 = max(ax1, bx1);
    iy1 = max(ay1, by1);
    ix2 = min(ax2, bx2);
    iy2 = min(ay2, by2);

    iw = max(0, ix2 - ix1 + 1);
    ih = max(0, iy2 - iy1 + 1);
    inter = iw * ih;

    area_a = a(3) * a(4);
    area_b = b(3) * b(4);
    union_ab = area_a + area_b - inter;

    if union_ab <= 0
        iou = 0.0;
    else
        iou = inter / union_ab;
    end
end
