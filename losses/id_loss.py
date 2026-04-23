from __future__ import annotations

import torch
from torch.nn import functional as F


def id_supervision_loss(logits: torch.Tensor | None, labels: torch.Tensor) -> torch.Tensor:
    if logits is None:
        return labels.new_tensor(0.0, dtype=torch.float32)
    if logits.numel() == 0 or labels.numel() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits, labels)
