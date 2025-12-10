#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import random
import warnings

import torch
import torch.backends.cudnn as cudnn
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import (
    configure_module,
    configure_nccl,
    fuse_model,
    get_local_rank,
    get_model_info,
    setup_logger,
)


def parse_range(range_str: str) -> "list[float]":
    """Parse 'start,end,step' string into list of float values."""
    parts = range_str.split(",")
    if len(parts) != 3:
        raise ValueError(f"Expected 'start,end,step' format, got: {range_str}")
    start, end, step = float(parts[0]), float(parts[1]), float(parts[2])
    values = []
    v = start
    while v <= end + 1e-9:  # small epsilon for float comparison
        values.append(round(v, 4))
        v += step
    return values


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    # Threshold tuning arguments
    parser.add_argument(
        "--tune",
        dest="tune",
        default=False,
        action="store_true",
        help="Enable threshold tuning mode. Sweeps over conf/nms ranges.",
    )
    parser.add_argument(
        "--conf-range",
        type=str,
        default="0.1,0.5,0.1",
        help="Conf threshold range for tuning: start,end,step (default: 0.1,0.5,0.1)",
    )
    parser.add_argument(
        "--nms-range",
        type=str,
        default="0.3,0.7,0.1",
        help="NMS threshold range for tuning: start,end,step (default: 0.3,0.7,0.1)",
    )
    parser.add_argument(
        "--tune-metric",
        type=str,
        default="f1",
        choices=["f1", "recall", "precision"],
        help="Metric to optimize during tuning (default: f1)",
    )
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=False,
        action="store_true",
        help="Print more information about the model.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    configure_nccl()
    cudnn.benchmark = True

    rank = get_local_rank()

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()

    if args.verbose:
        info = get_model_info(model, exp.test_size, getattr(exp, "input_channels", 3))
        logger.info("Model Summary: {}".format(info))
        logger.info("Model Structure:\n{}".format(str(model)))

    evaluator = exp.get_evaluator(
        args.batch_size, is_distributed, args.test, args.legacy
    )
    evaluator.per_class_AP = True
    evaluator.per_class_AR = True

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if not args.speed and not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint from {}".format(ckpt_file))
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert (
            not args.fuse and not is_distributed and args.batch_size == 1
        ), "TensorRT model is not support model fusing and distributed inferencing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    # Tuning mode: sweep over conf/nms combinations
    if args.tune:
        import typing as t

        from stft_dataset.evaluator import StftEvaluator
        from tabulate import tabulate

        if not isinstance(evaluator, StftEvaluator):
            logger.error(
                "Tuning mode requires StftEvaluator. "
                "Make sure your experiment uses StftEvaluator."
            )
            return

        conf_values = parse_range(args.conf_range)
        nms_values = parse_range(args.nms_range)

        logger.info(
            f"Tuning mode: {len(conf_values)} conf x {len(nms_values)} nms = "
            f"{len(conf_values) * len(nms_values)} combinations"
        )
        logger.info(f"  conf: {conf_values}")
        logger.info(f"  nms: {nms_values}")
        logger.info(f"  metric: {args.tune_metric}")

        # Phase 1: Run inference and cache outputs
        logger.info("Phase 1: Running inference...")

        model = model.eval()
        if args.fp16:
            model = model.half()
        dtype = torch.float16 if args.fp16 else torch.float32

        with torch.no_grad():
            inference_results = evaluator.run_inference(model, dtype)
            logger.info(f"Cached {len(inference_results)} batches")

            # Phase 2: Sweep thresholds
            logger.info("Phase 2: Sweeping thresholds...")
            best_result, all_results = evaluator.tune_thresholds(
                inference_results,
                conf_values,
                nms_values,
                logger,
                metric=args.tune_metric,
            )

        # Format results table
        table_data: t.List[t.List[str]] = []
        for r in all_results:
            is_best = r.conf == best_result.conf and r.nms == best_result.nms
            marker = " *" if is_best else ""
            table_data.append(
                [
                    f"{r.conf:.2f}",
                    f"{r.nms:.2f}",
                    f"{r.precision * 100:.2f}",
                    f"{r.recall * 100:.2f}",
                    f"{r.f1 * 100:.2f}{marker}",
                ]
            )

        headers = ["conf", "nms", "P (%)", "R (%)", "F1 (%)"]
        results_table = tabulate(table_data, headers=headers, tablefmt="pipe")

        logger.info("\n" + "=" * 60)
        logger.info("Threshold Tuning Results")
        logger.info("=" * 60)
        logger.info("\n" + results_table)
        logger.info("\n" + "=" * 60)
        logger.info(
            f"Best ({args.tune_metric}): conf={best_result.conf:.2f}, "
            f"nms={best_result.nms:.2f}, "
            f"P={best_result.precision * 100:.2f}%, "
            f"R={best_result.recall * 100:.2f}%, "
            f"F1={best_result.f1 * 100:.2f}%"
        )
        logger.info("=" * 60)

        return

    # start evaluate
    *_, summary = evaluator.evaluate(
        model, is_distributed, args.fp16, trt_file, decoder, exp.test_size
    )
    logger.info("\n" + summary)


if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args, num_gpu),
    )
