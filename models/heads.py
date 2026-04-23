from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvTower(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 2) -> None:
        super().__init__()
        layers = []
        current_channels = in_channels
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Conv2d(current_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(hidden_channels),
                    nn.SiLU(inplace=True),
                ]
            )
            current_channels = hidden_channels
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DetectionEmbeddingHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 1, emb_dim: int = 128, tower_layers: int = 2) -> None:
        super().__init__()
        self.cls_tower = ConvTower(in_channels, in_channels, num_layers=tower_layers)
        self.obj_tower = ConvTower(in_channels, in_channels, num_layers=tower_layers)
        self.box_tower = ConvTower(in_channels, in_channels, num_layers=tower_layers)
        self.emb_tower = ConvTower(in_channels, in_channels, num_layers=tower_layers)

        self.cls_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.obj_pred = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.box_pred = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.emb_pred = nn.Conv2d(in_channels, emb_dim, kernel_size=1)

    def forward_single(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_logits = self.cls_pred(self.cls_tower(x))
        obj_logits = self.obj_pred(self.obj_tower(x))
        box_distances = F.softplus(self.box_pred(self.box_tower(x)))
        embeddings = F.normalize(self.emb_pred(self.emb_tower(x)), dim=1)
        return cls_logits, obj_logits, box_distances, embeddings

    def forward(self, pyramid_features: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
        cls_outputs = {}
        obj_outputs = {}
        box_outputs = {}
        emb_outputs = {}
        for level_name, feature in pyramid_features.items():
            cls_logits, obj_logits, box_distances, embeddings = self.forward_single(feature)
            cls_outputs[level_name] = cls_logits
            obj_outputs[level_name] = obj_logits
            box_outputs[level_name] = box_distances
            emb_outputs[level_name] = embeddings
        return {"cls": cls_outputs, "obj": obj_outputs, "box": box_outputs, "emb": emb_outputs}
