from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from torchvision.transforms import functional as F
from ultralytics import YOLO

GT_COLUMNS = [
    "frame",
    "id",
    "bb_left",
    "bb_top",
    "bb_width",
    "bb_height",
    "mark",
    "class",
    "visibility",
]

DEFAULT_TRAIN_SEQUENCES = ["MOT16-02", "MOT16-04", "MOT16-05", "MOT16-09", "MOT16-10"]
DEFAULT_VAL_SEQUENCES = ["MOT16-11", "MOT16-13"]
NUM_CLASSES = 2  # background + pedestrian


@dataclass
class SampleItem:
    image_path: Path
    image_rel: str
    gt_boxes: list[list[float]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_project_root(start: Path) -> Path:
    if (start / "MOT16").exists():
        return start
    if (start.parent / "MOT16").exists():
        return start.parent
    return start


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def parse_sequence_gt(gt_path: Path) -> dict[int, list[list[float]]]:
    frame_to_boxes: dict[int, list[list[float]]] = {}
    with gt_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            frame = int(float(row[0]))
            mark = int(float(row[6]))
            cls = int(float(row[7]))
            if mark != 1 or cls != 1:
                continue

            x = float(row[2])
            y = float(row[3])
            w = float(row[4])
            h = float(row[5])
            box = [x, y, x + w, y + h]
            frame_to_boxes.setdefault(frame, []).append(box)
    return frame_to_boxes


def build_mot16_index(
    project_root: Path,
    sequences: list[str],
    split: str = "train",
    keep_empty_frames: bool = False,
) -> list[SampleItem]:
    mot16_root = project_root / "MOT16"
    index: list[SampleItem] = []

    for sequence in sequences:
        seq_dir = mot16_root / split / sequence
        img_dir = seq_dir / "img1"
        gt_path = seq_dir / "gt" / "gt.txt"
        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if split == "train" and not gt_path.exists():
            raise FileNotFoundError(f"GT file not found: {gt_path}")

        frame_to_boxes = parse_sequence_gt(gt_path) if split == "train" else {}
        image_paths = sorted(img_dir.glob("*.jpg"))
        for image_path in image_paths:
            frame = int(image_path.stem)
            gt_boxes = frame_to_boxes.get(frame, [])
            if not keep_empty_frames and len(gt_boxes) == 0 and split == "train":
                continue
            index.append(
                SampleItem(
                    image_path=image_path,
                    image_rel=str(image_path.relative_to(project_root)),
                    gt_boxes=gt_boxes,
                )
            )

    return index


def parse_sequence_arg(value: str) -> list[str]:
    return [s.strip() for s in value.split(",") if s.strip()]


def choose_trainable_parameters(model: torch.nn.Module, freeze_backbone: bool) -> list[torch.nn.Parameter]:
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
    return [p for p in model.parameters() if p.requires_grad]


def model_with_student_head(pretrained_backbone: bool = True) -> torch.nn.Module:
    if pretrained_backbone:
        try:
            model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        except Exception:
            model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    else:
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model


def to_tensor_targets(gt_boxes: list[list[float]], device: torch.device) -> dict[str, torch.Tensor]:
    boxes = torch.tensor(gt_boxes, dtype=torch.float32, device=device)
    if boxes.numel() == 0:
        boxes = boxes.reshape(0, 4)
    labels = torch.ones((boxes.shape[0],), dtype=torch.int64, device=device)
    return {"boxes": boxes, "labels": labels}


def load_teacher_cache(cache_jsonl: Path) -> dict[str, dict[str, list]]:
    pseudo_map: dict[str, dict[str, list]] = {}
    with cache_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pseudo_map[row["image_rel"]] = {
                "boxes": row.get("boxes", []),
                "scores": row.get("scores", []),
            }
    return pseudo_map


class DistillDataset(Dataset):
    def __init__(
        self,
        index_items: list[SampleItem],
        pseudo_map: dict[str, dict[str, list]],
        teacher_min_conf: float,
    ):
        self.index_items = index_items
        self.pseudo_map = pseudo_map
        self.teacher_min_conf = teacher_min_conf

    def __len__(self) -> int:
        return len(self.index_items)

    def __getitem__(self, idx: int):
        item = self.index_items[idx]
        image = Image.open(item.image_path).convert("RGB")
        image_tensor = F.to_tensor(image)

        gt_target = {
            "boxes": torch.tensor(item.gt_boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones((len(item.gt_boxes),), dtype=torch.int64),
        }

        pseudo = self.pseudo_map.get(item.image_rel, {"boxes": [], "scores": []})
        pseudo_boxes = []
        for box, score in zip(pseudo.get("boxes", []), pseudo.get("scores", [])):
            if float(score) >= self.teacher_min_conf:
                pseudo_boxes.append([float(v) for v in box])

        pseudo_target = {
            "boxes": torch.tensor(pseudo_boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones((len(pseudo_boxes),), dtype=torch.int64),
        }

        return image_tensor, gt_target, pseudo_target


def collate_fn(batch):
    images, gt_targets, pseudo_targets = zip(*batch)
    return list(images), list(gt_targets), list(pseudo_targets)


def move_targets_to_device(targets: list[dict[str, torch.Tensor]], device: torch.device) -> list[dict[str, torch.Tensor]]:
    moved = []
    for target in targets:
        moved.append({k: v.to(device) for k, v in target.items()})
    return moved


def build_teacher_cache(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    sequences = parse_sequence_arg(args.sequences) if args.sequences else None
    if sequences is None:
        split_dir = project_root / "MOT16" / args.split
        sequences = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])

    index_items = build_mot16_index(
        project_root=project_root,
        sequences=sequences,
        split=args.split,
        keep_empty_frames=True,
    )
    if args.max_images is not None:
        index_items = index_items[: args.max_images]

    print(f"[cache] split={args.split} sequences={len(sequences)} images={len(index_items)}")

    model = YOLO(str(Path(args.teacher_model).resolve()))
    image_paths = [str(item.image_path) for item in index_items]
    rel_map = {str(item.image_path.resolve()): item.image_rel for item in index_items}

    count_written = 0
    with output_jsonl.open("w", encoding="utf-8") as f:
        for batch_paths in chunked(image_paths, args.batch_size):
            results = model.predict(
                source=batch_paths,
                conf=args.conf,
                iou=args.iou,
                classes=[0],  # person
                imgsz=args.imgsz,
                verbose=False,
                stream=True,
            )

            for result_idx, result in enumerate(results):
                source_path = str(Path(batch_paths[result_idx]).resolve())
                image_rel = rel_map.get(source_path)
                if image_rel is None:
                    # Fallback: skip unmatched paths instead of corrupting cache.
                    continue

                boxes_xyxy = []
                scores = []
                if result.boxes is not None and result.boxes.xyxy is not None:
                    boxes_xyxy = result.boxes.xyxy.cpu().tolist()
                    scores = result.boxes.conf.cpu().tolist() if result.boxes.conf is not None else [1.0] * len(boxes_xyxy)

                row = {
                    "image_rel": image_rel,
                    "boxes": boxes_xyxy,
                    "scores": scores,
                }
                f.write(json.dumps(row) + "\n")
                count_written += 1

    print(f"[cache] wrote {count_written} rows to {output_jsonl}")


def compute_detection_stats(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    score_thresh: float,
    iou_thresh: float,
) -> dict[str, float]:
    model.eval()
    tp = 0
    fp = 0
    fn = 0

    with torch.no_grad():
        for images, gt_targets, _pseudo_targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, gt in zip(outputs, gt_targets):
                pred_boxes = output["boxes"].detach().cpu()
                pred_scores = output["scores"].detach().cpu()
                keep = pred_scores >= score_thresh
                pred_boxes = pred_boxes[keep]

                gt_boxes = gt["boxes"].detach().cpu()
                if len(gt_boxes) == 0:
                    fp += int(len(pred_boxes))
                    continue
                if len(pred_boxes) == 0:
                    fn += int(len(gt_boxes))
                    continue

                ious = box_iou(pred_boxes, gt_boxes)
                matched_gt = set()

                for pred_idx in range(ious.shape[0]):
                    best_iou, best_gt = torch.max(ious[pred_idx], dim=0)
                    best_iou_value = float(best_iou.item())
                    best_gt_idx = int(best_gt.item())
                    if best_iou_value >= iou_thresh and best_gt_idx not in matched_gt:
                        tp += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        fp += 1

                fn += int(len(gt_boxes) - len(matched_gt))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate_loss_only(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    model.train()
    losses = []
    with torch.no_grad():
        for images, gt_targets, _pseudo_targets in dataloader:
            images = [img.to(device) for img in images]
            gt_targets = move_targets_to_device(gt_targets, device)
            loss_dict = model(images, gt_targets)
            loss = sum(loss_dict.values())
            losses.append(float(loss.item()))
    model.eval()
    return float(sum(losses) / len(losses)) if losses else math.inf


def train_student(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    project_root = Path(args.project_root).resolve()
    cache_jsonl = Path(args.cache_jsonl).resolve()
    if not cache_jsonl.exists():
        raise FileNotFoundError(f"Teacher cache not found: {cache_jsonl}")

    train_sequences = parse_sequence_arg(args.train_sequences)
    val_sequences = parse_sequence_arg(args.val_sequences)

    train_index = build_mot16_index(project_root, train_sequences, split="train", keep_empty_frames=False)
    val_index = build_mot16_index(project_root, val_sequences, split="train", keep_empty_frames=False)

    if args.max_train_images is not None:
        train_index = train_index[: args.max_train_images]
    if args.max_val_images is not None:
        val_index = val_index[: args.max_val_images]

    pseudo_map = load_teacher_cache(cache_jsonl)
    train_dataset = DistillDataset(train_index, pseudo_map=pseudo_map, teacher_min_conf=args.teacher_min_conf)
    val_dataset = DistillDataset(val_index, pseudo_map=pseudo_map, teacher_min_conf=args.teacher_min_conf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    device = resolve_device(args.device)
    model = model_with_student_head(pretrained_backbone=not args.no_pretrained_backbone).to(device)
    params = choose_trainable_parameters(model, freeze_backbone=args.freeze_backbone)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root).resolve() / f"distill_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config_out = vars(args).copy()
    config_out["device_resolved"] = str(device)
    config_out["train_images"] = len(train_dataset)
    config_out["val_images"] = len(val_dataset)
    (run_dir / "config.json").write_text(json.dumps(config_out, indent=2), encoding="utf-8")

    print(f"[train] device={device} train_images={len(train_dataset)} val_images={len(val_dataset)}")
    print(f"[train] run_dir={run_dir}")

    history: list[dict[str, float]] = []
    best_val_loss = math.inf
    best_path = run_dir / "best_student.pt"
    last_path = run_dir / "last_student.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_total = 0.0
        epoch_gt = 0.0
        epoch_pseudo = 0.0
        batches = 0

        for images, gt_targets, pseudo_targets in train_loader:
            images = [img.to(device) for img in images]
            gt_targets = move_targets_to_device(gt_targets, device)
            pseudo_targets = move_targets_to_device(pseudo_targets, device)

            optimizer.zero_grad()

            loss_dict_gt = model(images, gt_targets)
            loss_gt = sum(loss_dict_gt.values())

            if args.lambda_kd > 0:
                loss_dict_pseudo = model(images, pseudo_targets)
                loss_pseudo = sum(loss_dict_pseudo.values())
            else:
                loss_pseudo = torch.tensor(0.0, device=device)

            total_loss = loss_gt + args.lambda_kd * loss_pseudo
            total_loss.backward()
            optimizer.step()

            epoch_total += float(total_loss.item())
            epoch_gt += float(loss_gt.item())
            epoch_pseudo += float(loss_pseudo.item())
            batches += 1

        train_total = epoch_total / max(1, batches)
        train_gt = epoch_gt / max(1, batches)
        train_pseudo = epoch_pseudo / max(1, batches)

        val_loss = evaluate_loss_only(model, val_loader, device=device)
        det_stats = compute_detection_stats(
            model,
            val_loader,
            device=device,
            score_thresh=args.eval_score_thresh,
            iou_thresh=args.eval_iou_thresh,
        )

        epoch_row = {
            "epoch": float(epoch),
            "train_total_loss": float(train_total),
            "train_gt_loss": float(train_gt),
            "train_pseudo_loss": float(train_pseudo),
            "val_loss": float(val_loss),
            "val_precision": float(det_stats["precision"]),
            "val_recall": float(det_stats["recall"]),
            "val_f1": float(det_stats["f1"]),
        }
        history.append(epoch_row)

        print(
            f"[epoch {epoch}/{args.epochs}] "
            f"train_total={train_total:.4f} train_gt={train_gt:.4f} train_pseudo={train_pseudo:.4f} "
            f"val_loss={val_loss:.4f} val_f1={det_stats['f1']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "num_classes": NUM_CLASSES,
            "train_sequences": train_sequences,
            "val_sequences": val_sequences,
        }
        torch.save(checkpoint, last_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, best_path)

        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"[train] finished. best_val_loss={best_val_loss:.4f}")
    print(f"[train] best checkpoint: {best_path}")
    print(f"[train] last checkpoint: {last_path}")


def evaluate_student(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    val_sequences = parse_sequence_arg(args.val_sequences)
    val_index = build_mot16_index(project_root, val_sequences, split="train", keep_empty_frames=False)
    if args.max_val_images is not None:
        val_index = val_index[: args.max_val_images]

    empty_pseudo_map: dict[str, dict[str, list]] = {}
    dataset = DistillDataset(val_index, pseudo_map=empty_pseudo_map, teacher_min_conf=1.0)
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    device = resolve_device(args.device)
    model = model_with_student_head(pretrained_backbone=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    val_loss = evaluate_loss_only(model, loader, device=device)
    det_stats = compute_detection_stats(
        model,
        loader,
        device=device,
        score_thresh=args.eval_score_thresh,
        iou_thresh=args.eval_iou_thresh,
    )

    print(f"[eval] checkpoint={checkpoint_path}")
    print(f"[eval] val_images={len(dataset)} val_loss={val_loss:.4f}")
    print(
        f"[eval] precision={det_stats['precision']:.4f} "
        f"recall={det_stats['recall']:.4f} "
        f"f1={det_stats['f1']:.4f} "
        f"tp={int(det_stats['tp'])} fp={int(det_stats['fp'])} fn={int(det_stats['fn'])}"
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO -> Faster R-CNN distillation for MOT16 pedestrian detection.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache_parser = subparsers.add_parser("build-cache", help="Build YOLO pseudo-label cache for MOT16 images.")
    cache_parser.add_argument("--project-root", type=str, default=str(Path.cwd()))
    cache_parser.add_argument("--teacher-model", type=str, default=str(Path.cwd() / "yolo26n.pt"))
    cache_parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    cache_parser.add_argument("--sequences", type=str, default="")
    cache_parser.add_argument("--output-jsonl", type=str, default=str(Path.cwd() / "distillation" / "artifacts" / "teacher_cache_train.jsonl"))
    cache_parser.add_argument("--conf", type=float, default=0.25)
    cache_parser.add_argument("--iou", type=float, default=0.5)
    cache_parser.add_argument("--imgsz", type=int, default=1280)
    cache_parser.add_argument("--batch-size", type=int, default=64)
    cache_parser.add_argument("--max-images", type=int, default=None)

    train_parser = subparsers.add_parser("train", help="Train Faster R-CNN student with mixed supervision.")
    train_parser.add_argument("--project-root", type=str, default=str(Path.cwd()))
    train_parser.add_argument("--cache-jsonl", type=str, required=True)
    train_parser.add_argument("--output-root", type=str, default=str(Path.cwd() / "distillation" / "runs"))
    train_parser.add_argument("--train-sequences", type=str, default=",".join(DEFAULT_TRAIN_SEQUENCES))
    train_parser.add_argument("--val-sequences", type=str, default=",".join(DEFAULT_VAL_SEQUENCES))
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--eval-batch-size", type=int, default=1)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--lambda-kd", type=float, default=0.5)
    train_parser.add_argument("--teacher-min-conf", type=float, default=0.35)
    train_parser.add_argument("--seed", type=int, default=21)
    train_parser.add_argument("--device", type=str, default="auto")
    train_parser.add_argument("--freeze-backbone", action="store_true")
    train_parser.add_argument("--no-pretrained-backbone", action="store_true")
    train_parser.add_argument("--eval-score-thresh", type=float, default=0.5)
    train_parser.add_argument("--eval-iou-thresh", type=float, default=0.5)
    train_parser.add_argument("--max-train-images", type=int, default=None)
    train_parser.add_argument("--max-val-images", type=int, default=None)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained student checkpoint on validation sequences.")
    eval_parser.add_argument("--project-root", type=str, default=str(Path.cwd()))
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    eval_parser.add_argument("--val-sequences", type=str, default=",".join(DEFAULT_VAL_SEQUENCES))
    eval_parser.add_argument("--eval-batch-size", type=int, default=1)
    eval_parser.add_argument("--num-workers", type=int, default=0)
    eval_parser.add_argument("--device", type=str, default="auto")
    eval_parser.add_argument("--eval-score-thresh", type=float, default=0.5)
    eval_parser.add_argument("--eval-iou-thresh", type=float, default=0.5)
    eval_parser.add_argument("--max-val-images", type=int, default=None)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    if args.command == "build-cache":
        build_teacher_cache(args)
    elif args.command == "train":
        train_student(args)
    elif args.command == "evaluate":
        evaluate_student(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
