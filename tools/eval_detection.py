#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.collate import mot_collate_fn
from datasets.mot16_dataset import build_mot16_datasets
from engine.evaluator import evaluate_detection
from models.student_jde import StudentJDE
from utils.checkpoint import validate_checkpoint_shapes
from utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detection quality on the MOT16 validation sequences.")
    parser.add_argument("--config", default="configs/student_distill.yaml")
    parser.add_argument("--checkpoint", default="runs/student_distill/best.pt")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    dataset_cfg = load_yaml(config["dataset"]["config"])
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset = build_mot16_datasets(
        root=dataset_cfg["dataset"]["root"],
        input_size=tuple(dataset_cfg["dataset"]["input_size"]),
        train_sequences=dataset_cfg["dataset"]["train_sequences"],
        val_sequences=dataset_cfg["dataset"]["val_sequences"],
        augmentation_config=dataset_cfg["augmentation"],
        person_class_id=dataset_cfg["dataset"].get("person_class_id", 1),
        visibility_threshold=dataset_cfg["dataset"].get("visibility_threshold", 0.0),
        include_ignore_regions=dataset_cfg["dataset"].get("include_ignore_regions", True),
    )
    loader = DataLoader(
        val_dataset,
        batch_size=config["dataset"].get("batch_size", 16),
        shuffle=False,
        num_workers=dataset_cfg["dataset"].get("num_workers", 8),
        collate_fn=mot_collate_fn,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    id_classes = checkpoint["student"]["id_classifier.weight"].shape[0] if "id_classifier.weight" in checkpoint["student"] else 0
    model = StudentJDE(
        backbone_name=config["student"]["backbone_name"],
        emb_dim=config["student"]["emb_dim"],
        num_classes=1,
        fpn_channels=config["student"]["fpn_channels"],
        pretrained_backbone=False,
        num_id_classes=id_classes,
    ).to(device)
    validate_checkpoint_shapes(checkpoint, model, feature_adapters=None)
    model.load_state_dict(checkpoint["student"])

    summary = evaluate_detection(
        model,
        loader,
        device=device,
        strides={level: int(value) for level, value in config["assigner"]["strides"].items()},
        score_threshold=float(config["inference"]["score_threshold"]),
        nms_iou_threshold=float(config["inference"]["nms_iou_threshold"]),
    )
    print(f"mAP@0.5: {summary.map50:.4f}")
    print(f"mAP@0.5:0.95: {summary.map50_95:.4f}")
    print(f"Precision: {summary.precision:.4f}")
    print(f"Recall: {summary.recall:.4f}")
    print(f"Mean IoU: {summary.mean_iou:.4f}")
    print(f"Predictions: {summary.num_predictions}")
    print(f"Targets: {summary.num_targets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
