#!/usr/bin/env python3
"""Cache teacher outputs to disk for offline student training.

.. deprecated::
    The default training mode is now **live distillation** (``use_cache=false``),
    where the teacher runs on the same augmented input as the student.  This
    ensures full spatial alignment between teacher and student signals and
    removes the need for a separate caching step.

    Use this script only if you want to trade accuracy for training speed by
    pre-computing teacher outputs on *deterministic* (non-augmented) images.
    When ``--use-cache`` is active during student training, the feature-KD and
    box-KD losses will see a spatial mismatch because the student sees augmented
    images while the cached teacher outputs were computed on canonical images.
    Detection (GT) and ID losses are unaffected by this mismatch.

Usage (only if you explicitly need cache mode)::

    python tools/cache_teacher_outputs.py --config configs/student_distill.yaml --split train
    python tools/cache_teacher_outputs.py --config configs/student_distill.yaml --split val
    # Then train with:
    python tools/train_student.py --config configs/student_distill.yaml --use-cache
"""


import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal envs
    def tqdm(iterable, **kwargs):
        return iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.collate import mot_collate_fn
from datasets.mot16_dataset import MOT16Dataset, Mot16SequenceConfig
from datasets.transforms import MotTransforms, TransformConfig
from models.teacher_wrapper import TeacherWrapper
from utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache teacher outputs for MOT16 student training.")
    parser.add_argument("--config", default="configs/student_distill.yaml")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def build_dataset(student_cfg: dict, split: str) -> MOT16Dataset:
    dataset_cfg = load_yaml(student_cfg["dataset"]["config"])
    sequences = dataset_cfg["dataset"]["train_sequences"] if split == "train" else dataset_cfg["dataset"]["val_sequences"]
    transform_cfg = TransformConfig(
        input_size=tuple(dataset_cfg["dataset"]["input_size"]),
        letterbox=dataset_cfg["augmentation"].get("letterbox", True),
    )
    return MOT16Dataset(
        Mot16SequenceConfig(
            root=Path(dataset_cfg["dataset"]["root"]),
            sequences=sequences,
            input_size=tuple(dataset_cfg["dataset"]["input_size"]),
            person_class_id=dataset_cfg["dataset"].get("person_class_id", 1),
            visibility_threshold=dataset_cfg["dataset"].get("visibility_threshold", 0.0),
            include_ignore_regions=dataset_cfg["dataset"].get("include_ignore_regions", True),
            teacher_cache_root=Path(student_cfg["teacher"]["cache_root"]),
        ),
        train=False,
        transforms=MotTransforms(transform_cfg, train=False),
    )


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    dataset = build_dataset(config, args.split)
    device = torch.device(config["teacher"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    teacher = TeacherWrapper(
        ckpt_path=config["teacher"]["ckpt_path"],
        device=str(device),
        person_class=config["teacher"].get("person_class", 0),
        emb_dim=config["student"].get("emb_dim", 128),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=mot_collate_fn,
    )

    manifest = []
    for batch in tqdm(loader, desc=f"Cache {args.split}"):
        images = batch["images"].to(device)
        targets = batch["targets"]
        teacher_outputs = teacher(images)
        image_size = tuple(targets[0]["image_size"].tolist())
        roi_embeddings = teacher.extract_roi_embeddings(
            teacher_outputs["spatial_feat"],
            boxes_per_image=[target["boxes"].to(device) for target in targets],
            image_size=image_size,
        )

        roi_offset = 0
        for sample_index, target in enumerate(targets):
            num_boxes = target["boxes"].shape[0]
            per_image_roi = roi_embeddings[roi_offset : roi_offset + num_boxes].cpu()
            roi_offset += num_boxes

            cache_path = Path(target["teacher_cache_path"])
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "features": {level: teacher_outputs["features"][level][sample_index : sample_index + 1].cpu() for level in ("p3", "p4", "p5")},
                "logits": {level: teacher_outputs["logits"][level][sample_index : sample_index + 1].cpu() for level in ("p3", "p4", "p5")},
                "raw_boxes": {level: teacher_outputs["raw_boxes"][level][sample_index : sample_index + 1].cpu() for level in ("p3", "p4", "p5")},
                "boxes": teacher_outputs["boxes"][sample_index].cpu(),
                "scores": teacher_outputs["scores"][sample_index].cpu(),
                "spatial_feat": teacher_outputs["spatial_feat"][sample_index : sample_index + 1].cpu(),
                "roi_embeddings": per_image_roi,
                "track_labels": target["track_labels"].cpu(),
                "visibilities": target["visibilities"].cpu(),
                "image_path": target["image_path"],
                "sequence_name": target["sequence_name"],
                "frame_idx": target["frame_idx"],
            }
            torch.save(payload, cache_path)
            manifest.append({"cache_path": str(cache_path), "image_path": target["image_path"]})

    manifest_path = Path(config["teacher"]["cache_root"]) / f"{args.split}_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
