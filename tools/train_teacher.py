#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import load_yaml


TRAIN_SEQUENCES = ["MOT16-02", "MOT16-04", "MOT16-05", "MOT16-09", "MOT16-10", "MOT16-11", "MOT16-13"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the YOLO teacher on MOT16 person-only labels.")
    parser.add_argument("--config", default="configs/teacher_finetune.yaml")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def convert_mot16_to_yolo(root: Path, converted_root: Path, val_sequences: list[str]) -> Path:
    images_train = converted_root / "images" / "train"
    images_val = converted_root / "images" / "val"
    labels_train = converted_root / "labels" / "train"
    labels_val = converted_root / "labels" / "val"
    for directory in (images_train, images_val, labels_train, labels_val):
        directory.mkdir(parents=True, exist_ok=True)

    for sequence_name in TRAIN_SEQUENCES:
        sequence_dir = root / "train" / sequence_name
        gt_by_frame: dict[int, list[list[float]]] = defaultdict(list)
        with (sequence_dir / "gt" / "gt.txt").open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if len(row) < 9:
                    continue
                frame_id = int(float(row[0]))
                mark = int(float(row[6]))
                cls = int(float(row[7]))
                if mark != 1 or cls != 1:
                    continue
                gt_by_frame[frame_id].append([float(v) for v in row[2:6]])

        seqinfo_path = sequence_dir / "seqinfo.ini"
        seqinfo = {}
        for line in seqinfo_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                seqinfo[key.strip()] = value.strip()
        image_w = int(seqinfo["imWidth"])
        image_h = int(seqinfo["imHeight"])

        split_name = "val" if sequence_name in val_sequences else "train"
        image_dst_root = images_val if split_name == "val" else images_train
        label_dst_root = labels_val if split_name == "val" else labels_train

        for image_path in sorted((sequence_dir / "img1").glob("*.jpg")):
            frame_id = int(image_path.stem)
            dst_image = image_dst_root / f"{sequence_name}_{image_path.name}"
            dst_label = label_dst_root / f"{sequence_name}_{image_path.stem}.txt"
            _link_or_copy(image_path, dst_image)
            lines = []
            for x, y, w, h in gt_by_frame.get(frame_id, []):
                x1 = max(0.0, min(float(image_w), x))
                y1 = max(0.0, min(float(image_h), y))
                x2 = max(0.0, min(float(image_w), x + w))
                y2 = max(0.0, min(float(image_h), y + h))
                clipped_w = x2 - x1
                clipped_h = y2 - y1
                if clipped_w <= 1e-6 or clipped_h <= 1e-6:
                    continue

                cx = (x1 + clipped_w / 2.0) / image_w
                cy = (y1 + clipped_h / 2.0) / image_h
                nw = clipped_w / image_w
                nh = clipped_h / image_h
                lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            dst_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    dataset_yaml = converted_root / "mot16_person.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(converted_root.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "person"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return dataset_yaml


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    dataset_cfg = config["dataset"]
    teacher_cfg = config["teacher"]

    dataset_yaml = convert_mot16_to_yolo(
        root=Path(dataset_cfg["root"]),
        converted_root=Path(dataset_cfg["converted_root"]),
        val_sequences=dataset_cfg.get("val_sequences", []),
    )
    model = YOLO(teacher_cfg["model"])
    model.train(
        data=str(dataset_yaml),
        epochs=teacher_cfg.get("epochs", 30),
        imgsz=teacher_cfg.get("imgsz", 640),
        batch=teacher_cfg.get("batch", 16),
        workers=teacher_cfg.get("workers", 8),
        lr0=teacher_cfg.get("lr0", 0.001),
        cos_lr=teacher_cfg.get("cos_lr", True),
        patience=teacher_cfg.get("patience", 10),
        project=teacher_cfg.get("project", "runs/teacher"),
        name=teacher_cfg.get("name", "yolo26x_mot16_person"),
        device=args.device or teacher_cfg.get("device", "cuda"),
        single_cls=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
