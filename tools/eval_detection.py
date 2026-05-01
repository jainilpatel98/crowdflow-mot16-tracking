#!/usr/bin/env python3
"""Evaluate detection quality on the MOT16 validation sequences.

Single-threshold mode (default):
    python tools/eval_detection.py --config configs/student_distill_resnet50.yaml \
        --checkpoint runs/student_distill_resnet50/best.pt

Override threshold from CLI (ignores config value):
    python tools/eval_detection.py ... --score-threshold 0.10

Threshold sweep mode (model inference runs ONCE; all thresholds are analytical):
    python tools/eval_detection.py ... --sweep-threshold
    python tools/eval_detection.py ... --sweep-threshold \
        --threshold-min 0.05 --threshold-max 0.50 --threshold-step 0.02
"""
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
from engine.evaluator import (
    collect_raw_predictions,
    compute_metrics_at_threshold,
    evaluate_detection,
)
from models.teacher_wrapper import TeacherWrapper
from models.student_jde import StudentJDE
from utils.checkpoint import validate_checkpoint_shapes
from utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate detection quality on the MOT16 validation sequences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default="configs/student_distill.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-type", choices=("student", "teacher"), default="student")

    # --- Single-threshold override ---
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Override inference.score_threshold from config (single-threshold mode).",
    )

    # --- Sweep mode ---
    parser.add_argument(
        "--sweep-threshold",
        action="store_true",
        help="Sweep score thresholds and print a comparison table. "
             "Model inference runs only once; all thresholds are evaluated analytically.",
    )
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.05,
        metavar="FLOAT",
        help="Lowest threshold to evaluate in sweep mode (default: 0.05).",
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.50,
        metavar="FLOAT",
        help="Highest threshold to evaluate in sweep mode (default: 0.50).",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.02,
        metavar="FLOAT",
        help="Step size between thresholds in sweep mode (default: 0.02).",
    )
    return parser.parse_args()


class TeacherForDetectionEval(torch.nn.Module):
    def __init__(self, teacher: TeacherWrapper) -> None:
        super().__init__()
        self.teacher = teacher

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> dict[str, object]:
        teacher_outputs = self.teacher(images)
        return {
            "boxes": teacher_outputs["boxes"],
            "scores": teacher_outputs["scores"],
        }


def _build_model_and_loader(args, config, dataset_cfg, device):
    """Load dataset, build model, load checkpoint. Shared by both modes."""
    _, val_dataset = build_mot16_datasets(
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

    if args.model_type == "teacher":
        checkpoint_path = args.checkpoint or config["teacher"]["ckpt_path"]
        teacher = TeacherWrapper(
            ckpt_path=checkpoint_path,
            device=str(device),
            person_class=config["teacher"].get("person_class", 0),
            emb_dim=config["student"].get("emb_dim", 128),
        )
        model = TeacherForDetectionEval(teacher).to(device)
    else:
        checkpoint_path = args.checkpoint or str(Path(config["training"]["output_dir"]) / "best.pt")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        id_classes = (
            checkpoint["student"]["id_classifier.weight"].shape[0]
            if "id_classifier.weight" in checkpoint["student"]
            else 0
        )
        model = StudentJDE(
            backbone_name=config["student"]["backbone_name"],
            emb_dim=config["student"]["emb_dim"],
            num_classes=1,
            fpn_channels=config["student"]["fpn_channels"],
            pretrained_backbone=False,
            num_id_classes=id_classes,
            tower_layers=int(config["student"].get("tower_layers", 2)),
        ).to(device)
        validate_checkpoint_shapes(checkpoint, model, feature_adapters=None)
        model.load_state_dict(checkpoint["student"])
    return model, loader


def _print_single(summary) -> None:
    print(f"mAP@0.5:       {summary.map50:.4f}")
    print(f"mAP@0.5:0.95:  {summary.map50_95:.4f}")
    print(f"Precision:     {summary.precision:.4f}")
    print(f"Recall:        {summary.recall:.4f}")
    print(f"F1:            {summary.f1:.4f}")
    print(f"Mean IoU:      {summary.mean_iou:.4f}")
    print(f"Predictions:   {summary.num_predictions}")
    print(f"Targets:       {summary.num_targets}")


def _sweep_rank(summary, threshold: float) -> tuple[float, float, float, float]:
    # Threshold selection is an operating-point problem, so use F1 first.
    # AP is still reported for reference, but it is ranking-based and does
    # not meaningfully choose a score threshold.
    return (summary.f1, summary.map50_95, summary.precision, threshold)


def _run_sweep(args, model, loader, device, strides, nms_iou_threshold: float) -> None:
    """Collect predictions once, then sweep thresholds analytically."""
    print("Collecting raw predictions (single inference pass)…")
    raw_preds, gt_by_image, gt_count = collect_raw_predictions(
        model, loader, device=device,
        strides=strides,
        nms_iou_threshold=nms_iou_threshold,
        model_type=args.model_type,
    )
    raw_candidate_count = sum(item["boxes"].shape[0] for item in raw_preds.values())
    print(
        f"Collected {raw_candidate_count:,} raw candidates over {len(gt_by_image)} images "
        f"({gt_count:,} GT boxes).\n"
    )

    # Build threshold list with float precision rounding
    thresholds: list[float] = []
    t = args.threshold_min
    while t <= args.threshold_max + 1e-9:
        thresholds.append(round(t, 6))
        t += args.threshold_step

    # Header
    col_w = 7
    header = (
        f"{'Thresh':>7}  {'mAP@.5':>{col_w}}  {'mAP@.5:.95':>{col_w+3}}  "
        f"{'Precision':>{col_w+2}}  {'Recall':>{col_w}}  {'F1':>{col_w}}  "
        f"{'MeanIoU':>{col_w}}  {'Preds':>7}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    best_thresh = thresholds[0]
    best_rank = None
    results = []

    for thresh in thresholds:
        m = compute_metrics_at_threshold(raw_preds, gt_by_image, gt_count, thresh)
        results.append((thresh, m))
        marker = ""
        rank = _sweep_rank(m, thresh)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_thresh = thresh
            marker = " <-- best F1 so far"

        print(
            f"{thresh:7.3f}  {m.map50:{col_w}.4f}  {m.map50_95:{col_w+3}.4f}  "
            f"{m.precision:{col_w+2}.4f}  {m.recall:{col_w}.4f}  {m.f1:{col_w}.4f}  "
            f"{m.mean_iou:{col_w}.4f}  {m.num_predictions:>7}{marker}"
        )

    print(sep)
    best_summary = next(m for t, m in results if t == best_thresh)
    print(
        f"\nBest threshold: {best_thresh:.3f}  "
        f"(F1 = {best_summary.f1:.4f}, Precision = {best_summary.precision:.4f}, "
        f"Recall = {best_summary.recall:.4f})"
    )
    print("Selection rule: highest F1, with mAP@0.5:0.95 / precision / threshold as tie-breakers.")
    print("\nFull metrics at best threshold:")
    _print_single(best_summary)
    print(f"\nHint: set inference.score_threshold: {best_thresh} in your config.")


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    dataset_cfg = load_yaml(config["dataset"]["config"])
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

    model, loader = _build_model_and_loader(args, config, dataset_cfg, device)

    strides = {level: int(v) for level, v in config["assigner"]["strides"].items()}
    nms_iou_threshold = float(config["inference"]["nms_iou_threshold"])

    if args.sweep_threshold:
        _run_sweep(args, model, loader, device, strides, nms_iou_threshold)
        return 0

    # --- Single-threshold mode ---
    score_threshold = (
        args.score_threshold
        if args.score_threshold is not None
        else float(config["inference"]["score_threshold"])
    )
    summary = evaluate_detection(
        model,
        loader,
        device=device,
        strides=strides,
        score_threshold=score_threshold,
        nms_iou_threshold=nms_iou_threshold,
        model_type=args.model_type,
    )
    print(f"score_threshold: {score_threshold}")
    _print_single(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
