#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch import nn
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
from models.teacher_wrapper import TeacherROIProjector, TeacherWrapper
from utils.assigners import PyramidAssigner
from utils.checkpoint import validate_checkpoint_shapes
from utils.config import load_yaml
from utils.distributed import cleanup_distributed, init_distributed_mode, is_main_process
from utils.logger import setup_logger
from utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the distilled student detector and embedding model.")
    parser.add_argument("--config", default="configs/student_distill.yaml")
    parser.add_argument(
        "--backbone",
        default=None,
        help="Override student backbone (e.g. resnet50, mobilenetv3_large). Default: from config.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=False,
        help=(
            "Use pre-computed teacher cache instead of running the teacher live. "
            "Note: cache was computed on deterministic (non-augmented) images, so "
            "feature/box KD losses will have a spatial mismatch with the augmented student input. "
            "Default: OFF (live teacher, same augmented input for both)."
        ),
    )
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume from.")
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

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    use_cache = args.use_cache or config["teacher"].get("use_cache", False)
    cache_root = config["teacher"].get("cache_root") if use_cache else None

    train_dataset, val_dataset = build_mot16_datasets(
        root=dataset_cfg["dataset"]["root"],
        input_size=tuple(dataset_cfg["dataset"]["input_size"]),
        train_sequences=dataset_cfg["dataset"]["train_sequences"],
        val_sequences=dataset_cfg["dataset"]["val_sequences"],
        augmentation_config=dataset_cfg["augmentation"],
        person_class_id=dataset_cfg["dataset"].get("person_class_id", 1),
        visibility_threshold=dataset_cfg["dataset"].get("visibility_threshold", 0.0),
        include_ignore_regions=dataset_cfg["dataset"].get("include_ignore_regions", True),
        teacher_cache_root=cache_root,
    )

    batch_size  = config["dataset"].get("batch_size", 12)
    num_workers = dataset_cfg["dataset"].get("num_workers", 8)

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
        val_sampler   = DistributedSampler(val_dataset,   shuffle=False, drop_last=False)
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
        shuffle = train_sampler is None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                              sampler=train_sampler, num_workers=num_workers,
                              pin_memory=device.type == "cuda", collate_fn=mot_collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              sampler=val_sampler, num_workers=num_workers,
                              pin_memory=device.type == "cuda", collate_fn=mot_collate_fn)

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    backbone_name = args.backbone or config["student"].get("backbone_name", "resnet50")
    fpn_channels  = config["student"]["fpn_channels"]
    emb_dim       = config["student"]["emb_dim"]

    student = StudentJDE(
        backbone_name=backbone_name,
        emb_dim=emb_dim,
        num_classes=1,
        fpn_channels=fpn_channels,
        pretrained_backbone=config["student"].get("pretrained_backbone", True),
        num_id_classes=train_dataset.num_identities,
    ).to(device)

    teacher = None
    if not use_cache:
        teacher = TeacherWrapper(
            ckpt_path=config["teacher"]["ckpt_path"],
            device=str(device),
            person_class=config["teacher"].get("person_class", 0),
            emb_dim=emb_dim,
        )

    feature_adapters = MultiScaleFeatureAdapters(
        student_channels=student.feature_channels,
        teacher_channels=config["teacher"]["feature_channels"],
    ).to(device)

    # Teacher ROI projector  (Issue 2 — trainable, same arch as student's ROIProjector)
    teacher_p3_channels = config["teacher"]["feature_channels"]["p3"]
    teacher_roi_projector = TeacherROIProjector(
        in_channels=teacher_p3_channels,
        emb_dim=emb_dim,
    ).to(device)
    teacher_id_classifier = nn.Linear(emb_dim, train_dataset.num_identities).to(device)

    # ------------------------------------------------------------------
    # Resume checkpoint (with shape validation)
    # ------------------------------------------------------------------
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        validate_checkpoint_shapes(ckpt, student, feature_adapters, teacher_roi_projector)
        student.load_state_dict(ckpt["student"])
        if ckpt.get("feature_adapters") is not None:
            feature_adapters.load_state_dict(ckpt["feature_adapters"])
        if ckpt.get("teacher_roi_projector") is not None:
            teacher_roi_projector.load_state_dict(ckpt["teacher_roi_projector"])
        if ckpt.get("teacher_id_classifier") is not None:
            teacher_id_classifier.load_state_dict(ckpt["teacher_id_classifier"])
        start_epoch = ckpt.get("epoch", 0) + 1
        if logger:
            logger.info("Resumed from %s (epoch %d)", args.resume, start_epoch - 1)

    # ------------------------------------------------------------------
    # DDP wrapping
    # ------------------------------------------------------------------
    if is_distributed:
        student               = DDP(student,               device_ids=[local_rank] if device.type == "cuda" else None, find_unused_parameters=True)
        feature_adapters      = DDP(feature_adapters,      device_ids=[local_rank] if device.type == "cuda" else None)
        teacher_roi_projector = DDP(teacher_roi_projector, device_ids=[local_rank] if device.type == "cuda" else None)
        teacher_id_classifier = DDP(teacher_id_classifier, device_ids=[local_rank] if device.type == "cuda" else None)

    # ------------------------------------------------------------------
    # Optimizer — backbone at lower LR, everything else (heads + projector) at head LR
    # ------------------------------------------------------------------
    student_inner  = student.module           if hasattr(student, "module")               else student
    adapters_inner = feature_adapters.module  if hasattr(feature_adapters, "module")      else feature_adapters
    proj_inner     = teacher_roi_projector.module if hasattr(teacher_roi_projector, "module") else teacher_roi_projector
    tea_id_inner   = teacher_id_classifier.module if hasattr(teacher_id_classifier, "module") else teacher_id_classifier

    backbone_params = list(student_inner.backbone.parameters())
    head_params = (
        [p for n, p in student_inner.named_parameters() if not n.startswith("backbone.")]
        + list(adapters_inner.parameters())
        + list(proj_inner.parameters())
        + list(tea_id_inner.parameters())
    )

    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": config["optimizer"]["lr_backbone"]},
            {"params": head_params,     "lr": config["optimizer"]["lr_heads"]},
        ],
        weight_decay=config["optimizer"]["weight_decay"],
        betas=tuple(config["optimizer"]["betas"]),
    )

    total_steps  = config["training"]["epochs"] * max(1, len(train_loader))
    warmup_steps = config["scheduler"]["warmup_epochs"] * max(1, len(train_loader))
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps,
                                 config["scheduler"]["min_lr_ratio"])

    # ------------------------------------------------------------------
    # Assigner
    # ------------------------------------------------------------------
    assigner = PyramidAssigner(
        strides    ={lv: int(v) for lv, v in config["assigner"]["strides"].items()},
        area_ranges={lv: tuple(v) for lv, v in config["assigner"]["area_ranges"].items()},
        center_radius=float(config["assigner"]["center_radius"]),
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = DistillationTrainer(
        student_model=student,
        teacher_model=teacher,
        feature_adapters=feature_adapters,
        teacher_roi_projector=teacher_roi_projector,
        teacher_id_classifier=teacher_id_classifier,
        assigner=assigner,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        phases=config["phases"],
        output_dir=output_dir,
        amp=config["training"].get("amp", True),
        grad_clip=float(config["training"].get("grad_clip", 1.0)),
        temperature=float(config["training"].get("temperature", 2.0)),
        use_teacher_cache=use_cache,
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
