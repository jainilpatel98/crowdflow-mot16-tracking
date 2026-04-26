from __future__ import annotations

from torch import nn


class FeatureAdapter(nn.Module):
    """Projects student FPN features to teacher feature space for feature-KD.

    Uses Conv→BN→ReLU→Conv→BN (two-layer) to give the adapter enough
    expressiveness to learn a non-trivial alignment.  A single linear
    Conv→BN was too weak — the adapter output was dominated by the much
    larger cls_kd / box_kd loss terms and barely received any gradient.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        mid = max(in_channels, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class MultiScaleFeatureAdapters(nn.Module):
    def __init__(self, student_channels: dict[str, int], teacher_channels: dict[str, int]) -> None:
        super().__init__()
        self.adapters = nn.ModuleDict(
            {
                level: FeatureAdapter(student_channels[level], teacher_channels[level])
                for level in student_channels
            }
        )

    def forward(self, features: dict[str, object]) -> dict[str, object]:
        return {level: self.adapters[level](feature) for level, feature in features.items()}
