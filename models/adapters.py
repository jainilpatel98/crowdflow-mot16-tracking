from __future__ import annotations

from torch import nn


class FeatureAdapter(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
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
