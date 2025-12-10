#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
YOLOX-tiny experiment v3 for STFT spectrogram burst detection.

LARGER MODEL - trains from scratch (NOT compatible with v1/v2 checkpoints).

Architecture changes from v1/v2:
- width=0.375 (YOLOX-tiny) vs 0.25 (nano) - 1.5x wider
- ~5M params vs ~0.9M params
- Better feature extraction capacity for low-SNR signals

Training improvements (same as v2):
- Extended training (100 epochs)
- L1 loss in final epochs for better localization
- Gaussian noise augmentation for low-SNR robustness
- Save history checkpoints for analysis
"""

import os
import pathlib

import torch.distributed as dist
import torch.nn as nn
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        # ---------------- model config ---------------- #
        # YOLOX-tiny architecture (larger than nano)
        self.num_classes = 1  # Single class: QPSK burst
        self.depth = 0.33  # Same depth as nano (sufficient for 128x128)
        self.width = 0.375  # TINY width (1.5x wider than nano's 0.25)
        self.act = "relu"  # DPU-compatible activation
        self.input_channels = 1  # Grayscale STFT spectrogram

        # ---------------- dataloader config ---------------- #
        self.input_size = (128, 128)  # STFT tile size
        self.test_size = (128, 128)
        self.data_num_workers = 2  # Limited by HDF5 file handle constraints

        # Disable multi-scale training (fixed size spectrograms)
        self.multiscale_range = 0

        # ---------------- data paths ---------------- #
        self.data_dir = pathlib.Path(os.environ["STFT_DATASET"])
        self.h5_path = self.data_dir / "tensors" / "tiles.h5"
        self.meta_path = self.data_dir / "meta.json"
        self.train_split = self.data_dir / "splits" / "train.txt"
        self.val_split = self.data_dir / "splits" / "val.txt"

        # ---------------- augmentation config ---------------- #
        # Disable augmentations - use specialized augs later
        self.mosaic_prob = 0.0
        self.mixup_prob = 0.0
        self.hsv_prob = 0.0
        self.flip_prob = 0.0
        self.enable_mixup = False

        # ---------------- training config ---------------- #
        self.max_epoch = 100  # Extended training
        self.warmup_epochs = 5  # Slightly longer warmup for fresh training
        self.no_aug_epochs = 10  # Last 10 epochs: L1 loss + no augmentation
        self.eval_interval = 5  # Evaluate every 5 epochs
        self.print_interval = 10
        self.save_history_ckpt = True  # Save checkpoints for analysis

        # Learning rate - standard for training from scratch
        self.basic_lr_per_img = 0.001 / 64.0  # Standard YOLOX LR

        # Minimum LR ratio for cosine schedule
        self.min_lr_ratio = 0.01  # Lower minimum for longer training

        # ---------------- testing config ---------------- #
        # Initial thresholds (will tune after training)
        self.test_conf = 0.4
        self.nmsthre = 0.2

        # ---------------- experiment name ---------------- #
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_model(self, sublinear=False):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if "model" not in self.__dict__:
            if os.environ.get("W_QUANT", "0") == "1":
                from yolox.models.yolo_head_q import YOLOXHead
                from yolox.models.yolo_pafpn_deploy_q import YOLOPAFPN
                from yolox.models.yolox_q import YOLOX
            else:
                from yolox.models import YOLOX, YOLOXHead
                from yolox.models.yolo_pafpn_deploy import YOLOPAFPN

            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth,
                self.width,
                in_channels=in_channels,
                act=self.act,
                depthwise=True,
                input_channels=self.input_channels,
            )
            head = YOLOXHead(
                self.num_classes,
                self.width,
                in_channels=in_channels,
                act=self.act,
                depthwise=True,
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_data_loader(
        self, batch_size, is_distributed, no_aug=False, cache_img=False
    ):
        del cache_img  # unused

        from stft_dataset import LoadSplit, Matlab, StftDataset
        from stft_dataset.loader import StftDataLoader
        from stft_dataset.normalization import (
            Normalize,
            load_normalization_params,
        )
        from yolox.data import InfiniteSampler

        # Import the noise augmentation
        from stft_dataset import augmentation as aug

        # Load normalization parameters
        vmin_db, vmax_db = load_normalization_params(self.meta_path)

        # Build dataset pipeline
        dataset = Normalize(
            StftDataset(LoadSplit(Matlab(self.h5_path, []), self.train_split)),
            vmin_db,
            vmax_db,
        )

        # Time reversal (has physical sense)
        self.hflip_prob = 0.5
        # Frequency reversal? (has less physical sense) - but good for training
        self.vflip_prob = 0.5
        # Gaussian noise augmentation for low-SNR robustness
        self.noise_aug_prob = 0.3
        self.noise_std_range = (0.01, 0.05)  # Range of noise std (normalized space)

        # Apply flip augmentation (time-reversal) unless disabled
        if not no_aug:
            dataset = aug.RandomHorizontalFlip(
                dataset,
                flip_prob=self.hflip_prob,
                img_width=self.input_size[1],
            )
            dataset = aug.RandomVerticalFlip(
                dataset,
                flip_prob=self.vflip_prob,
                img_height=self.input_size[0],
            )
            dataset = aug.GaussianNoiseAugmentation(
                dataset,
                noise_prob=self.noise_aug_prob,
                std_range=self.noise_std_range,
            )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(dataset), seed=self.seed if self.seed else 0)

        # Create dataloader
        train_loader = StftDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.data_num_workers,
            pin_memory=True,
            drop_last=True,
            max_labels=50,
            sampler=sampler,
        )

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        del is_distributed, testdev, legacy  # unused

        from stft_dataset import LoadSplit, Matlab, StftDataset
        from stft_dataset.loader import StftDataLoader
        from stft_dataset.normalization import Normalize, load_normalization_params

        # Load normalization parameters
        vmin_db, vmax_db = load_normalization_params(self.meta_path)

        # Build dataset pipeline
        dataset = Normalize(
            StftDataset(LoadSplit(Matlab(self.h5_path, []), self.val_split)),
            vmin_db,
            vmax_db,
        )

        # Create dataloader (no shuffle for validation)
        val_loader = StftDataLoader(
            dataset=dataset,
            max_labels=50,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.data_num_workers,
            pin_memory=True,
            drop_last=False,
        )

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from stft_dataset.evaluator import StftEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = StftEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
        return evaluator

    def preprocess(self, inputs, targets, tsize):
        # No preprocessing needed - images are already normalized and correct size
        return inputs, targets

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        # Fixed size - no random resizing
        return self.input_size
