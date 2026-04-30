from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvTower(nn.Module):
    """Stacked Conv → GroupNorm → SiLU blocks.

    Fix 5: GroupNorm replaces BatchNorm for training stability on small MOT
    batches.  The number of groups is capped to min(32, out_channels) and
    auto-adjusted to guarantee divisibility.

    num_layers is configurable (default 2 → pass 3 for deeper heads).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        groups: int = 32,
    ) -> None:
        super().__init__()
        layers = []
        current_channels = in_channels
        for _ in range(num_layers):
            num_groups = min(groups, hidden_channels)
            while hidden_channels % num_groups != 0:
                num_groups //= 2
            layers.extend(
                [
                    nn.Conv2d(current_channels, hidden_channels,
                              kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(num_groups, hidden_channels),
                    nn.SiLU(inplace=True),
                ]
            )
            current_channels = hidden_channels
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DetectionEmbeddingHead(nn.Module):
    """Joint detection + embedding head with per-level box scale (Fix 2).

    Fix 2 — Per-level learnable scale:
        A single shared box tower must produce LTRB pixel distances that span
        a 16× magnitude range across p3/p4/p5 (e.g., 4–64px at stride-8 vs
        64–256px at stride-32).  Without per-level scale, the regressor
        receives conflicting gradient signals that cause slow convergence and
        poor box accuracy at extreme scales.

        Each level has one learnable scalar s_i (init=0, so exp(s_i)=1 at
        init → identical to the old behaviour).  The box prediction becomes:
            box_distances = softplus(box_raw) * exp(s_i)
        The model learns the appropriate output scale per FPN level.

    Fix 5 — Deeper towers + GroupNorm:
        tower_layers is now a constructor argument (default 2, set to 3 in
        config).  GroupNorm replaces BatchNorm throughout.

    Fix 6 — Dense embedding supervision hook:
        emb_pred outputs a (B, emb_dim, H, W) map.  Previously this received
        zero gradient because the ROI-projector path (which IS trained) is
        separate.  The dense map is aligned to the ROI projector during
        training via dense_emb_alignment_loss (losses/kd_loss.py).
        No architectural change is needed here — the emb map is already in
        outputs["emb"].  The alignment loss reads it from there.
    """

    # Level names used to register per-level scale parameters.
    _LEVELS = ("p3", "p4", "p5")

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        emb_dim: int = 128,
        tower_layers: int = 2,
    ) -> None:
        super().__init__()
        self.cls_tower = ConvTower(in_channels, in_channels, num_layers=tower_layers)
        self.box_tower = ConvTower(in_channels, in_channels, num_layers=tower_layers)
        self.emb_tower = ConvTower(in_channels, in_channels, num_layers=tower_layers)
        # obj_tower kept for training centerness target (Fix 4: removed from
        # inference score, but centerness supervision still trains the head
        # which can be used for future quality-aware scoring).
        self.obj_tower = ConvTower(in_channels, in_channels, num_layers=tower_layers)

        self.cls_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.obj_pred = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.box_pred = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.emb_pred = nn.Conv2d(in_channels, emb_dim, kernel_size=1)

        # Fix 2: one learnable log-scale per FPN level, initialised to 0
        # so exp(0)=1 → no change at start of training.
        self.box_log_scales = nn.ParameterDict(
            {level: nn.Parameter(torch.zeros(1)) for level in self._LEVELS}
        )

    def forward_single(
        self,
        x: torch.Tensor,
        level_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_logits = self.cls_pred(self.cls_tower(x))
        obj_logits = self.obj_pred(self.obj_tower(x))
        # Fix 2: apply per-level exp scale after softplus
        box_raw = self.box_pred(self.box_tower(x))
        scale = self.box_log_scales[level_name].exp()        # scalar ≥ 0, init=1
        box_distances = F.softplus(box_raw) * scale
        embeddings = F.normalize(self.emb_pred(self.emb_tower(x)), dim=1)
        return cls_logits, obj_logits, box_distances, embeddings

    def forward(
        self,
        pyramid_features: dict[str, torch.Tensor],
    ) -> dict[str, dict[str, torch.Tensor]]:
        cls_outputs: dict[str, torch.Tensor] = {}
        obj_outputs: dict[str, torch.Tensor] = {}
        box_outputs: dict[str, torch.Tensor] = {}
        emb_outputs: dict[str, torch.Tensor] = {}
        for level_name, feature in pyramid_features.items():
            cls_logits, obj_logits, box_distances, embeddings = self.forward_single(
                feature, level_name
            )
            cls_outputs[level_name] = cls_logits
            obj_outputs[level_name] = obj_logits
            box_outputs[level_name] = box_distances
            emb_outputs[level_name] = embeddings
        return {"cls": cls_outputs, "obj": obj_outputs, "box": box_outputs, "emb": emb_outputs}
