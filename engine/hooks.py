from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(state: dict, output_dir: str | Path, name: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    torch.save(state, path)
    return path
