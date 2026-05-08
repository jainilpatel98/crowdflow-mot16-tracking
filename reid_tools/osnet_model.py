from __future__ import annotations

from pathlib import Path

import torch
from boxmot.reid.backbones.osnet import osnet_x0_25
from boxmot.reid.core.registry import ReIDModelRegistry


def build_osnet_x0_25(num_classes: int, pretrained_weights: str | Path | None = None) -> torch.nn.Module:
    model = osnet_x0_25(num_classes=num_classes, pretrained=False, loss="triplet")
    if pretrained_weights:
        ReIDModelRegistry.load_pretrained_weights(model, Path(pretrained_weights))
    return model


def save_boxmot_compatible_checkpoint(model: torch.nn.Module, output_path: str | Path, metadata: dict | None = None) -> None:
    """Save a .pt checkpoint that BoxMOT can partially load for inference.

    BoxMOT discards unmatched classifier layers when loading custom weights. The
    feature extractor weights are what StrongSORT uses at eval/inference time.
    """
    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, output_path)

