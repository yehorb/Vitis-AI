"""
Simple IoU-based evaluator for STFT dataset.

Computes precision, recall, and F1 for single-class detection.
Optimized for high recall (sensitivity) as the primary metric.
"""

from __future__ import annotations

import dataclasses
import time
import typing as t

import torch
from loguru import logger
from yolox import utils

if t.TYPE_CHECKING:
    from stft_dataset.loader import StftDataLoader


@dataclasses.dataclass
class EvaluatorParams:
    model: torch.nn.Module
    distributed: None = None
    half: bool = False
    trt_file: None = None
    decoder: None = None
    test_size: None = None
    return_outputs: bool = False

    def get_model(self):
        return self.model.eval()

    def to_device(self, imgs: torch.Tensor, dtype: torch.dtype):
        return imgs.to(dtype=dtype, device="cuda")

    def should_dump_xmodel(self):
        return False

    def exit_early(self, params: t.Dict[int, t.Any]):
        return (0, 0, None), params

    def postprocess(self, is_parallel: bool, outputs: t.Any):
        del is_parallel  # unused
        return outputs


@dataclasses.dataclass
class QEvaluatorParams:
    quant_model: torch.nn.Module
    float_model: torch.nn.Module
    distributed: None = None
    half: bool = False
    trt_file: None = None
    decoder: None = None
    test_size: None = None
    is_dump: bool = False
    device: torch.device = torch.device("cuda")
    return_outputs: bool = False

    def get_model(self):
        return self.quant_model.eval()

    def to_device(self, imgs: torch.Tensor, dtype: torch.dtype):
        return imgs.to(dtype=dtype, device=self.device)

    def should_dump_xmodel(self):
        return self.is_dump

    def exit_early(self, params: t.Dict[int, t.Any]):
        if self.return_outputs:
            return (0, 0, None), params
        return (0, 0, None), ""

    def postprocess(self, is_parallel: bool, outputs: t.Any):
        if is_parallel:
            return self.float_model.module.head.postprocess(outputs)
        else:
            return self.float_model.head.postprocess(outputs)


class StftEvaluator:
    """
    Simple IoU-based evaluator for STFT burst detection.

    Returns recall as primary metric (first return value) since
    high sensitivity is critical for low-SNR signal detection.
    """

    def __init__(
        self,
        dataloader: StftDataLoader,
        img_size: t.Tuple[int, int],
        confthre: float,
        nmsthre: float,
        num_classes: int = 1,
        iou_thre: float = 0.5,
    ):
        """
        Initialize evaluator.

        Parameters
        ----------
        dataloader : DataLoader
            Validation dataloader yielding (imgs, labels) batches.
        img_size : tuple
            Image size (H, W) for inference.
        confthre : float
            Confidence threshold for filtering detections.
        nmsthre : float
            IoU threshold for NMS.
        num_classes : int
            Number of classes (default 1 for QPSK).
        iou_thre : float
            IoU threshold for matching predictions to ground truth.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.iou_thre = iou_thre

    def evaluate(self, *args: t.Any, **kwargs: t.Any) -> t.Union[
        t.Tuple[float, float, str],
        t.Tuple[t.Tuple[float, float, str], t.Dict[int, t.Any]],
    ]:
        """
        Run evaluation on validation set.

        Parameters
        ----------
        model : nn.Module
            Model to evaluate.
        distributed : bool
            Whether running distributed (not supported, ignored).
        half : bool
            Whether to use FP16 inference.
        trt_file : str, optional
            TensorRT file (not supported, ignored).
        decoder : optional
            Output decoder (not supported, ignored).
        test_size : tuple, optional
            Test image size (uses self.img_size if None).
        return_outputs : bool
            Whether to return per-image outputs.

        Returns
        -------
        recall : float
            TP / (TP + FN), primary metric for model selection.
        precision : float
            TP / (TP + FP).
        summary : str
            Human-readable summary of results.
        outputs : dict, optional
            Per-image predictions (if return_outputs=True).
        """

        # Test mode:
        #  code/tools/eval.py, line 197
        if len(args) == 6:
            params = EvaluatorParams(*args)  # pyright: ignore[reportAny]
        # Quantization mode:
        #  code/tools/quant.py, line 238
        elif len(args) == 9:
            params = QEvaluatorParams(*args)  # pyright: ignore[reportAny]
        # Training mode:
        #   code/yolox/core/trainer.py, line 330
        #   code/yolox/exp/yolox_base.py, line 322
        elif len(args) == 3 and len(kwargs) == 1:  # pyright: ignore[reportAny]
            params = EvaluatorParams(
                args[0],
                args[1],
                args[2],
                None,
                None,
                None,
                kwargs["return_outputs"],
            )
        else:
            raise AttributeError(
                f"Unexpected call to evaluate() with {len(args)} params"
            )

        model = params.get_model()
        if params.half:
            model = model.half()
        dtype = torch.float16 if params.half else torch.float32
        is_parallel = utils.is_parallel(model)

        inference_time = 0.0
        n_samples = 0

        inference_results: t.List[t.Tuple[torch.Tensor, ...]] = []
        output_data: t.Dict[int, t.Any] = {}

        with torch.no_grad():
            # 2 values to unpack, tile_ids are dropped in dataloader
            for batch_idx, (imgs_batch, labels_batch) in enumerate(self.dataloader):
                imgs_batch = params.to_device(imgs_batch, dtype)
                batch_size = imgs_batch.shape[0]
                n_samples += batch_size

                # Inference
                # Network predicts center-based format (cx, cy, w, h)
                # We convert to corner format (x1, y1, x2, y2) for NMS
                # `outputs` contents: [x1, y1, x2, y2, confidence]

                start = time.time()
                # Default float model returns final bboxes [x1, y1, x2, y2, confidence] here
                outputs = model(imgs_batch)
                inference_time += time.time() - start

                if params.should_dump_xmodel():
                    return params.exit_early(output_data)

                # Quantization-aware model decodes individual head outputs into bboxes here
                outputs = params.postprocess(is_parallel, outputs)

                # Everything down from here is just post-processing
                inference_results.append((labels_batch, outputs))

            stats = PredictionStats()
            for batch_idx, batch in enumerate(inference_results):

                # Match predictions to ground truth for each image
                batch_stats, batch_output = evaluate_batch(
                    batch_idx,
                    batch,
                    EvaluationConfig(
                        self.num_classes, self.confthre, self.nmsthre, self.iou_thre
                    ),
                )
                stats += batch_stats

                if params.return_outputs:
                    output_data = {**output_data, **batch_output}

        # Compute metrics
        precision = (
            stats.tp / (stats.tp + stats.fp) if (stats.tp + stats.fp) > 0 else 0.0
        )
        recall = stats.tp / (stats.tp + stats.fn) if (stats.tp + stats.fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        avg_time_ms = 1000 * inference_time / n_samples if n_samples > 0 else 0.0

        summary = (
            f"STFT Evaluation Results\n"
            f"-----------------------\n"
            f"Total GT boxes:   {stats.gt}\n"
            f"Total predictions:{stats.pred}\n"
            f"TP: {stats.tp}, FP: {stats.fp}, FN: {stats.fn}\n"
            f"Precision: {precision:.4f}\n"
            f"Recall:    {recall:.4f}\n"
            f"F1 Score:  {f1:.4f}\n"
            f"Avg inference time: {avg_time_ms:.2f} ms/image\n"
        )

        logger.info(f"Eval: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")

        if params.return_outputs:
            return (recall, precision, summary), output_data
        return recall, precision, summary


def extract_gt_boxes(labels: torch.Tensor) -> torch.Tensor:
    """
    Extract ground truth boxes from labels tensor.

    Parameters
    ----------
    labels : torch.Tensor
        Shape [max_labels, 5] with format (class_id, cx, cy, w, h).
        Zero-padded rows have sum == 0.

    Returns
    -------
    boxes : torch.Tensor
        Shape [N, 4] in xyxy format, on same device as input.
    """
    # Filter out zero-padded rows
    valid_mask = labels.sum(dim=1) > 0
    valid_targets = labels[valid_mask]

    if len(valid_targets) == 0:
        return torch.empty((0, 4), device=labels.device)

    # Convert cxcywh to xyxy
    cx = valid_targets[:, 1]
    cy = valid_targets[:, 2]
    w = valid_targets[:, 3]
    h = valid_targets[:, 4]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return torch.stack([x1, y1, x2, y2], dim=1)


def extract_pred_boxes(detections: t.Optional[torch.Tensor]) -> torch.Tensor:
    """
    Extract prediction boxes from postprocess output.

    Parameters
    ----------
    detections : torch.Tensor or None
        Shape [N, 7] with format (x1, y1, x2, y2, obj_conf, class_conf, class_pred).
        None if no detections.

    Returns
    -------
    boxes : torch.Tensor
        Shape [N, 4] in xyxy format.
    """
    if detections is None or len(detections) == 0:
        return torch.empty((0, 4), device="cuda")

    return detections[:, :4]


def match_boxes(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    cfg: EvaluationConfig,
):
    """
    Match predictions to ground truth using greedy IoU matching.

    Parameters
    ----------
    pred_boxes : torch.Tensor
        Shape [M, 4] predicted boxes in xyxy format.
    gt_boxes : torch.Tensor
        Shape [N, 4] ground truth boxes in xyxy format.

    Returns
    -------
    tp : int
        True positives (matched predictions).
    fp : int
        False positives (unmatched predictions).
    fn : int
        False negatives (unmatched ground truth).
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    if n_pred == 0 and n_gt == 0:
        return PredictionStats(0, 0, 0)
    if n_pred == 0:
        return PredictionStats(0, 0, n_gt)
    if n_gt == 0:
        return PredictionStats(0, n_pred, 0)

    # Compute IoU matrix [M, N]
    # Ensure both tensors are on the same device
    gt_boxes = gt_boxes.to(pred_boxes.device)
    iou_matrix = utils.bboxes_iou(pred_boxes, gt_boxes, xyxy=True)

    # Greedy matching: for each prediction, find best GT match
    matched_gt = set()
    tp = 0

    for pred_idx in range(n_pred):
        ious = iou_matrix[pred_idx]
        best_gt_idx = int(ious.argmax().item())
        best_iou = float(ious[best_gt_idx].item())

        if best_iou >= cfg.iouthre and best_gt_idx not in matched_gt:
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = n_pred - tp
    fn = n_gt - len(matched_gt)

    return PredictionStats(tp, fp, fn, n_gt, n_pred)


@dataclasses.dataclass
class EvaluationConfig:
    num_classes: int
    confthre: float
    nmsthre: float
    iouthre: float


@dataclasses.dataclass
class PredictionStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    gt: int = 0
    pred: int = 0

    def __add__(self, other):
        cls = self.__class__
        if not isinstance(other, cls):
            raise NotImplementedError
        return cls(
            self.tp + other.tp,
            self.fp + other.fp,
            self.fn + other.fn,
            self.gt + other.gt,
            self.pred + other.pred,
        )


def evaluate_batch(
    batch_idx: int, batch: t.Tuple[torch.Tensor, ...], cfg: EvaluationConfig
):
    labels_batch, outputs = batch
    batch_size = labels_batch.shape[0]

    # Apply NMS
    outputs = utils.postprocess(
        outputs,
        cfg.num_classes,
        cfg.confthre,
        cfg.nmsthre,
        class_agnostic=True,
    )

    stats = PredictionStats()
    output_data: t.Dict[int, t.Any] = {}
    for i in range(batch_size):
        gt_boxes = extract_gt_boxes(labels_batch[i])
        pred_boxes = extract_pred_boxes(outputs[i])
        sample_stats = match_boxes(pred_boxes, gt_boxes, cfg)
        stats += sample_stats

        img_id = batch_idx * batch_size + i
        output_data[img_id] = {
            "pred_boxes": (
                pred_boxes.cpu().numpy().tolist() if len(pred_boxes) > 0 else []
            ),
            "gt_boxes": (gt_boxes.cpu().numpy().tolist() if len(gt_boxes) > 0 else []),
            "tp": sample_stats.tp,
            "fp": sample_stats.fp,
            "fn": sample_stats.fn,
        }

    return stats, output_data
