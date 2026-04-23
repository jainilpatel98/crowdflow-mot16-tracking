#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.collate import mot_collate_fn
from datasets.mot16_dataset import build_mot16_datasets
from engine.trainer import DistillationTrainer
from models.adapters import MultiScaleFeatureAdapters
from models.student_jde import StudentJDE
from models.teacher_wrapper import TeacherWrapper
from utils.assigners import PyramidAssigner
from utils.config import load_yaml
from utils.distributed import cleanup_distributed, init_distributed_mode, is_main_process
from utils.logger import setup_logger
from utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the distilled student detector and embedding model.")
    parser.add_argument("--config", default="configs/student_distill.yaml")
    return parser.parse_args()


def build_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    dataset_cfg = load_yaml(config["dataset"]["config"])

    seed_everything(config.get("seed", 1337))
    is_distributed, local_rank, world_size = init_distributed_mode()
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if is_distributed else config["training"]["device"])
    else:
        device = torch.device("cpu")
    output_dir = Path(config["training"]["output_dir"])
    logger = setup_logger(f"student_distill_rank{local_rank}", output_dir / "train.log") if is_main_process() else None

    train_dataset, val_dataset = build_mot16_datasets(
        root=dataset_cfg["dataset"]["root"],
        input_size=tuple(dataset_cfg["dataset"]["input_size"]),
        train_sequences=dataset_cfg["dataset"]["train_sequences"],
        val_sequences=dataset_cfg["dataset"]["val_sequences"],
        augmentation_config=dataset_cfg["augmentation"],
        person_class_id=dataset_cfg["dataset"].get("person_class_id", 1),
        visibility_threshold=dataset_cfg["dataset"].get("visibility_threshold", 0.0),
        include_ignore_regions=dataset_cfg["dataset"].get("include_ignore_regions", True),
        teacher_cache_root=config["teacher"]["cache_root"] if config["teacher"].get("use_cache", False) else None,
    )

    batch_size = config["dataset"].get("batch_size", 16)
    num_workers = dataset_cfg["dataset"].get("num_workers", 8)
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        shuffle = False
    else:
        train_sampler = (
            WeightedRandomSampler(
                weights=train_dataset.sequence_sample_weights(),
                num_samples=len(train_dataset),
                replacement=config.get("sampler", {}).get("replacement", True),
            )
            if config.get("sampler", {}).get("balance_sequences", False)
            else None
        )
        val_sampler = None
        shuffle = not config.get("sampler", {}).get("balance_sequences", False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=mot_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=mot_collate_fn,
    )

    student = StudentJDE(
        backbone_name=config["student"]["backbone_name"],
        emb_dim=config["student"]["emb_dim"],
        num_classes=1,
        fpn_channels=config["student"]["fpn_channels"],
        pretrained_backbone=config["student"].get("pretrained_backbone", False),
        num_id_classes=train_dataset.num_identities,
    ).to(device)
    teacher = None
    if not config["teacher"].get("use_cache", False):
        teacher = TeacherWrapper(
            ckpt_path=config["teacher"]["ckpt_path"],
            device=str(device),
            person_class=config["teacher"].get("person_class", 0),
            emb_dim=config["student"]["emb_dim"],
        )
    feature_adapters = MultiScaleFeatureAdapters(
        student_channels=student.feature_channels,
        teacher_channels=config["teacher"]["feature_channels"],
    ).to(device)

    if is_distributed:
        student = DDP(student, device_ids=[local_rank] if device.type == "cuda" else None, find_unused_parameters=True)
        feature_adapters = DDP(feature_adapters, device_ids=[local_rank] if device.type == "cuda" else None, find_unused_parameters=True)

    student_for_optim = student.module if hasattr(student, "module") else student
    feature_for_optim = feature_adapters.module if hasattr(feature_adapters, "module") else feature_adapters

    backbone_params = list(student_for_optim.backbone.parameters())
    head_params = [
        parameter
        for name, parameter in student_for_optim.named_parameters()
        if not name.startswith("backbone.")
    ]
    if feature_for_optim is not None:
        head_params += list(feature_for_optim.parameters())

    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": config["optimizer"]["lr_backbone"]},
            {"params": head_params, "lr": config["optimizer"]["lr_heads"]},
        ],
        weight_decay=config["optimizer"]["weight_decay"],
        betas=tuple(config["optimizer"]["betas"]),
    )

    total_steps = config["training"]["epochs"] * max(1, len(train_loader))
    warmup_steps = config["scheduler"]["warmup_epochs"] * max(1, len(train_loader))
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=config["scheduler"]["min_lr_ratio"],
    )

    assigner = PyramidAssigner(
        strides={level: int(value) for level, value in config["assigner"]["strides"].items()},
        area_ranges={level: tuple(value) for level, value in config["assigner"]["area_ranges"].items()},
        center_radius=float(config["assigner"]["center_radius"]),
    )

    trainer = DistillationTrainer(
        student_model=student,
        teacher_model=teacher,
        feature_adapters=feature_adapters,
        assigner=assigner,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        phases=config["phases"],
        output_dir=output_dir,
        amp=config["training"].get("amp", True),
        grad_clip=float(config["training"].get("grad_clip", 1.0)),
        temperature=float(config["training"].get("temperature", 2.0)),
        use_teacher_cache=config["teacher"].get("use_cache", False),
        embedding_min_visibility=float(dataset_cfg["dataset"].get("embedding_visibility_threshold", 0.25)),
        id_min_visibility=float(dataset_cfg["dataset"].get("id_visibility_threshold", 0.25)),
        logger=logger,
    )
    try:
        trainer.fit(train_loader, val_loader, epochs=config["training"]["epochs"])
    finally:
        if is_distributed:
            cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
