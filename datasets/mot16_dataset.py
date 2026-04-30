from __future__ import annotations

import configparser
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.transforms import MotTransforms, TransformConfig, letterbox_image


# ---------------------------------------------------------------------------
# Sequence configuration
# ---------------------------------------------------------------------------

@dataclass
class Mot16SequenceConfig:
    root: Path
    sequences: list[str]
    input_size: tuple[int, int] = (640, 640)
    person_class_id: int = 1
    visibility_threshold: float = 0.0
    include_ignore_regions: bool = True
    teacher_cache_root: Path | None = None


# ---------------------------------------------------------------------------
# GT annotation parsing
# ---------------------------------------------------------------------------

def _parse_gt(
    gt_path: Path,
    person_class_id: int,
    visibility_threshold: float,
    include_ignore_regions: bool,
) -> dict[int, dict[str, Any]]:
    """Parse MOT16 gt.txt into a per-frame dict.

    MOT16 gt.txt columns (1-indexed):
      frame, track_id, left, top, width, height, mark, class_id, visibility

    mark=0 → ignore region, mark=1 → active annotation
    class_id=1 → pedestrian (the class we care about)
    """
    frames: dict[int, dict[str, Any]] = {}
    if not gt_path.exists():
        return frames

    with gt_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 9:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            left = float(parts[2])
            top = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])
            mark = int(parts[6])
            class_id = int(parts[7])
            visibility = float(parts[8])

            x1, y1, x2, y2 = left, top, left + width, top + height

            if frame_id not in frames:
                frames[frame_id] = {"boxes": [], "track_ids": [], "visibilities": [], "ignore_boxes": []}

            is_ignore = mark == 0 or class_id != person_class_id
            if is_ignore:
                if include_ignore_regions:
                    frames[frame_id]["ignore_boxes"].append([x1, y1, x2, y2])
            else:
                if visibility >= visibility_threshold:
                    frames[frame_id]["boxes"].append([x1, y1, x2, y2])
                    frames[frame_id]["track_ids"].append(track_id)
                    frames[frame_id]["visibilities"].append(visibility)

    return frames


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MOT16Dataset(Dataset):
    """Frame-level dataset for MOT16 sequences.

    Each item is one frame. Supports:
    - Mosaic augmentation (4 random frames merged into one)
    - Copy-paste occlusion augmentation
    - Teacher cache paths for distillation

    Global identity mapping: (sequence_name, local_track_id) → global_label (0-indexed).
    """

    def __init__(
        self,
        seq_cfg: Mot16SequenceConfig,
        train: bool,
        transforms: MotTransforms,
    ) -> None:
        self.seq_cfg = seq_cfg
        self.train = train
        self.transforms = transforms

        # Build index: list of (sequence_name, frame_id, image_path, gt_entry)
        self._samples: list[dict[str, Any]] = []
        self._seq_sample_counts: dict[str, int] = {}

        # Build global identity map  (sequence_name, local_track_id) → global_label
        self._identity_map: dict[tuple[str, int], int] = {}
        identity_counter = 0

        for seq_name in seq_cfg.sequences:
            seq_dir = seq_cfg.root / "train" / seq_name
            if not seq_dir.exists():
                seq_dir = seq_cfg.root / "test" / seq_name
            img_dir = seq_dir / "img1"
            gt_path = seq_dir / "gt" / "gt.txt"

            # Parse sequence info for frame dims
            seqinfo_path = seq_dir / "seqinfo.ini"
            orig_size = self._read_orig_size(seqinfo_path)

            gt_by_frame = _parse_gt(
                gt_path,
                seq_cfg.person_class_id,
                seq_cfg.visibility_threshold,
                seq_cfg.include_ignore_regions,
            )

            image_paths = sorted(img_dir.glob("*.jpg"))
            count = 0
            for img_path in image_paths:
                frame_id = int(img_path.stem)
                gt = gt_by_frame.get(frame_id, {"boxes": [], "track_ids": [], "visibilities": [], "ignore_boxes": []})

                # Build global identity labels
                global_labels = []
                for tid in gt["track_ids"]:
                    key = (seq_name, tid)
                    if key not in self._identity_map:
                        self._identity_map[key] = identity_counter
                        identity_counter += 1
                    global_labels.append(self._identity_map[key])

                cache_path = None
                if seq_cfg.teacher_cache_root is not None:
                    cache_path = str(seq_cfg.teacher_cache_root / seq_name / f"{frame_id:06d}.pt")

                self._samples.append({
                    "image_path": str(img_path),
                    "sequence_name": seq_name,
                    "frame_idx": frame_id,
                    "orig_size": orig_size,   # (H, W) numpy array
                    "boxes": np.array(gt["boxes"], dtype=np.float32).reshape(-1, 4),
                    "track_ids": np.array(gt["track_ids"], dtype=np.int64),
                    "track_labels": np.array(global_labels, dtype=np.int64),
                    "visibilities": np.array(gt["visibilities"], dtype=np.float32),
                    "ignore_boxes": np.array(gt["ignore_boxes"], dtype=np.float32).reshape(-1, 4),
                    "teacher_cache_path": cache_path,
                })
                count += 1
            self._seq_sample_counts[seq_name] = count

        self._num_identities = identity_counter

    @staticmethod
    def _read_orig_size(seqinfo_path: Path) -> np.ndarray:
        if seqinfo_path.exists():
            cfg = configparser.ConfigParser()
            cfg.read(str(seqinfo_path))
            try:
                h = int(cfg["Sequence"]["imHeight"])
                w = int(cfg["Sequence"]["imWidth"])
                return np.array([h, w], dtype=np.int64)
            except Exception:
                pass
        return np.array([1080, 1920], dtype=np.int64)  # MOT16 default

    @property
    def num_identities(self) -> int:
        return self._num_identities

    def sequence_sample_weights(self) -> list[float]:
        """Per-sample weight inversely proportional to sequence size (for balanced sampling)."""
        weights: list[float] = []
        for sample in self._samples:
            count = self._seq_sample_counts[sample["sequence_name"]]
            weights.append(1.0 / max(1, count))
        return weights

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.train and random.random() < self.transforms.cfg.mosaic_prob:
            # Mosaic: pick 3 additional random frames
            extra_indices = [random.randint(0, len(self) - 1) for _ in range(3)]
            raw_samples = [self._load_raw(i) for i in [idx] + extra_indices]
            item = self.transforms.apply_mosaic(raw_samples)

            # Optional copy-paste after mosaic
            if random.random() < self.transforms.cfg.copy_paste_prob:
                donor_raws = [self._load_raw(random.randint(0, len(self) - 1)) for _ in range(3)]
                from datasets.transforms import _copy_paste
                item_raw = {
                    "image": Image.fromarray((item["images"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)),
                    "boxes": item["boxes"].numpy(),
                    "track_ids": item["track_ids"].numpy(),
                    "track_labels": item["track_labels"].numpy(),
                    "visibilities": item["visibilities"].numpy(),
                    "orig_size": item["orig_size"].numpy(),
                    "image_path": item["image_path"],
                    "sequence_name": item["sequence_name"],
                    "frame_idx": item["frame_idx"],
                }
                item_raw = _copy_paste(item_raw, donor_raws, self.transforms.cfg.input_size)
                from datasets.transforms import _to_tensor
                item["images"] = _to_tensor(item_raw["image"])
                item["boxes"] = torch.as_tensor(item_raw["boxes"], dtype=torch.float32)
                item["track_ids"] = torch.as_tensor(item_raw["track_ids"], dtype=torch.long)
                item["track_labels"] = torch.as_tensor(item_raw["track_labels"], dtype=torch.long)
                item["visibilities"] = torch.as_tensor(item_raw["visibilities"], dtype=torch.float32)
        else:
            raw = self._load_raw(idx)
            item = self.transforms.apply_single(raw)

        return item

    def _load_raw(self, idx: int) -> dict[str, Any]:
        meta = self._samples[idx]
        image = Image.open(meta["image_path"]).convert("RGB")
        return {**meta, "image": image}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_mot16_datasets(
    root: str,
    input_size: tuple[int, int],
    train_sequences: list[str],
    val_sequences: list[str],
    augmentation_config: dict,
    person_class_id: int = 1,
    visibility_threshold: float = 0.0,
    include_ignore_regions: bool = True,
    teacher_cache_root: str | None = None,
) -> tuple[MOT16Dataset, MOT16Dataset]:
    """Build train and validation MOT16Dataset instances."""
    root_path = Path(root)
    cache_path = Path(teacher_cache_root) if teacher_cache_root else None

    aug = augmentation_config
    train_cfg = TransformConfig(
        input_size=tuple(input_size),
        letterbox=aug.get("letterbox", True),
        hsv_h=aug.get("hue", 0.015),
        hsv_s=aug.get("saturation", 0.7),
        hsv_v=aug.get("brightness", 0.4),
        horizontal_flip_prob=aug.get("horizontal_flip_prob", 0.5),
        affine_degrees=aug.get("affine_degrees", 3.0),
        affine_translate=aug.get("affine_translate", 0.05),
        affine_scale_min=aug.get("affine_scale", [0.5, 2.0])[0],
        affine_scale_max=aug.get("affine_scale", [0.5, 2.0])[1],
        mosaic_prob=aug.get("mosaic_prob", 0.8),
        copy_paste_prob=aug.get("copy_paste_prob", 0.3),
        motion_blur_prob=aug.get("motion_blur_prob", 0.1),
        blur_kernel=aug.get("blur_kernel", 5),
        erase_prob=aug.get("erase_prob", 0.2),
    )
    val_cfg = TransformConfig(
        input_size=tuple(input_size),
        letterbox=True,
        mosaic_prob=0.0,
        copy_paste_prob=0.0,
        motion_blur_prob=0.0,
        erase_prob=0.0,
        horizontal_flip_prob=0.0,
        affine_degrees=0.0,
        affine_translate=0.0,
        affine_scale_min=1.0,
        affine_scale_max=1.0,
    )

    train_seq_cfg = Mot16SequenceConfig(
        root=root_path,
        sequences=train_sequences,
        input_size=tuple(input_size),
        person_class_id=person_class_id,
        visibility_threshold=visibility_threshold,
        include_ignore_regions=include_ignore_regions,
        teacher_cache_root=cache_path,
    )
    val_seq_cfg = Mot16SequenceConfig(
        root=root_path,
        sequences=val_sequences,
        input_size=tuple(input_size),
        person_class_id=person_class_id,
        visibility_threshold=visibility_threshold,
        include_ignore_regions=include_ignore_regions,
        teacher_cache_root=cache_path,
    )

    train_dataset = MOT16Dataset(train_seq_cfg, train=True, transforms=MotTransforms(train_cfg, train=True))
    val_dataset = MOT16Dataset(val_seq_cfg, train=False, transforms=MotTransforms(val_cfg, train=False))
    return train_dataset, val_dataset
