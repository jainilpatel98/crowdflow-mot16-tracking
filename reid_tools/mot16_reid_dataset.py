from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
from torch.utils.data import Dataset


PERSON_LIKE_CLASS_IDS = {1, 2}


@dataclass(frozen=True)
class ReIDSample:
    image_path: Path
    sequence: str
    frame_id: int
    track_id: int
    class_id: int
    visibility: float
    xyxy: tuple[float, float, float, float]
    identity_key: tuple[str, int]
    label: int


def resolve_sequence_dir(root: Path, sequence: str) -> Path:
    for split in ("train", "test"):
        candidate = root / split / sequence
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find sequence {sequence!r} under {root}")


def read_mot16_reid_samples(
    *,
    root: str | Path,
    sequences: Iterable[str],
    class_ids: set[int] | None = None,
    min_visibility: float = 0.25,
    min_box_size: int = 4,
) -> list[ReIDSample]:
    """Read MOT16 ReID samples from GT boxes.

    MOTChallenge GT columns:
      frame, id, left, top, width, height, mark, class, visibility

    This intentionally keeps only person-like classes by default:
      1 = pedestrian, 2 = person on vehicle/rider.
    """
    root_path = Path(root)
    allowed_classes = PERSON_LIKE_CLASS_IDS if class_ids is None else set(class_ids)
    samples: list[ReIDSample] = []
    identity_to_label: dict[tuple[str, int], int] = {}

    for sequence in sequences:
        seq_dir = resolve_sequence_dir(root_path, sequence)
        gt_path = seq_dir / "gt" / "gt.txt"
        img_dir = seq_dir / "img1"
        if not gt_path.exists():
            continue

        with gt_path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if len(row) < 9:
                    continue
                frame_id = int(float(row[0]))
                track_id = int(float(row[1]))
                left = float(row[2])
                top = float(row[3])
                width = float(row[4])
                height = float(row[5])
                mark = int(float(row[6]))
                class_id = int(float(row[7]))
                visibility = float(row[8])

                if mark != 1 or track_id <= 0:
                    continue
                if class_id not in allowed_classes:
                    continue
                if visibility < min_visibility:
                    continue
                if width < min_box_size or height < min_box_size:
                    continue

                image_path = img_dir / f"{frame_id:06d}.jpg"
                if not image_path.exists():
                    continue

                identity_key = (sequence, track_id)
                if identity_key not in identity_to_label:
                    identity_to_label[identity_key] = len(identity_to_label)
                label = identity_to_label[identity_key]
                samples.append(
                    ReIDSample(
                        image_path=image_path,
                        sequence=sequence,
                        frame_id=frame_id,
                        track_id=track_id,
                        class_id=class_id,
                        visibility=visibility,
                        xyxy=(left, top, left + width, top + height),
                        identity_key=identity_key,
                        label=label,
                    )
                )

    return samples


class MOT16ReIDDataset(Dataset):
    def __init__(self, samples: list[ReIDSample], transform=None) -> None:
        self.samples = samples
        self.transform = transform
        self.num_classes = len({sample.label for sample in samples})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            x1, y1, x2, y2 = sample.xyxy
            x1 = max(0, min(width - 1, int(round(x1))))
            y1 = max(0, min(height - 1, int(round(y1))))
            x2 = max(x1 + 1, min(width, int(round(x2))))
            y2 = max(y1 + 1, min(height, int(round(y2))))
            crop = image.crop((x1, y1, x2, y2))

        if self.transform is not None:
            crop = self.transform(crop)

        return {
            "image": crop,
            "label": sample.label,
            "sequence": sample.sequence,
            "track_id": sample.track_id,
            "frame_id": sample.frame_id,
            "class_id": sample.class_id,
            "identity_key": sample.identity_key,
        }

