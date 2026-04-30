from __future__ import annotations

from typing import Any

import torch


def mot_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a list of sample dicts into a batched dict.

    Images are stacked into a single tensor.
    Variable-length per-image tensors (boxes, track_ids, etc.) are kept as lists.
    """
    images = torch.stack([item["images"] for item in batch], dim=0)

    targets = []
    for item in batch:
        targets.append({
            "boxes": item["boxes"],
            "track_ids": item["track_ids"],
            "track_labels": item["track_labels"],
            "visibilities": item["visibilities"],
            "ignore_boxes": item["ignore_boxes"],
            "orig_size": item["orig_size"],
            "image_size": item["image_size"],
            "resize_scale": item.get("resize_scale", 1.0),
            "pad": item.get("pad", (0, 0)),
            "image_path": item.get("image_path", ""),
            "sequence_name": item.get("sequence_name", ""),
            "frame_idx": item.get("frame_idx", 0),
            "teacher_cache_path": item.get("teacher_cache_path"),
        })

    cache_paths = [t["teacher_cache_path"] for t in targets]

    return {
        "images": images,
        "targets": targets,
        "teacher_cache_paths": cache_paths,
    }
