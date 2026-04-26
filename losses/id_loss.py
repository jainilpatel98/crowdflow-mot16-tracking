from __future__ import annotations

import torch
from torch.nn import functional as F


def id_supervision_loss(
    logits: torch.Tensor | None,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Identity cross-entropy loss with optional label smoothing.

    Args:
        logits: ``(N, num_id_classes)`` ID classifier output.
        labels: ``(N,)`` integer track-ID targets.
        label_smoothing: Smoothing factor in [0, 1).  0 = standard CE.
            Set to 0.1 to prevent the model from becoming overconfident on
            training IDs, which improves val id_loss stability.
    """
    if logits is None:
        return labels.new_tensor(0.0, dtype=torch.float32)
    if logits.numel() == 0 or labels.numel() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
