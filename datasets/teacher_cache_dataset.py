from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class TeacherCacheDataset(Dataset):
    def __init__(self, base_dataset: Dataset, cache_root: str | Path) -> None:
        self.base_dataset = base_dataset
        self.cache_root = Path(cache_root)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base_dataset[index]
        target = dict(sample["target"])
        cache_path = target.get("teacher_cache_path")
        teacher_cache = None
        if cache_path and Path(cache_path).exists():
            teacher_cache = torch.load(cache_path, map_location="cpu")
        return {
            "image": sample["image"],
            "target": target,
            "teacher_cache": teacher_cache,
        }
