from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyFPN(nn.Module):
    def __init__(self, in_channels: dict[str, int], out_channels: int = 128) -> None:
        super().__init__()
        self.lateral_c3 = ConvBNAct(in_channels["c3"], out_channels, kernel_size=1)
        self.lateral_c4 = ConvBNAct(in_channels["c4"], out_channels, kernel_size=1)
        self.lateral_c5 = ConvBNAct(in_channels["c5"], out_channels, kernel_size=1)

        self.out_p3 = ConvBNAct(out_channels, out_channels, kernel_size=3)
        self.out_p4 = ConvBNAct(out_channels, out_channels, kernel_size=3)
        self.out_p5 = ConvBNAct(out_channels, out_channels, kernel_size=3)

        self.down_p3 = ConvBNAct(out_channels, out_channels, kernel_size=3, stride=2)
        self.down_p4 = ConvBNAct(out_channels, out_channels, kernel_size=3, stride=2)

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        c3, c4, c5 = features["c3"], features["c4"], features["c5"]

        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")

        p3 = self.out_p3(p3)
        p4 = self.out_p4(p4)
        p5 = self.out_p5(p5)

        p4 = self.out_p4(p4 + self.down_p3(p3))
        p5 = self.out_p5(p5 + self.down_p4(p4))
        return {"p3": p3, "p4": p4, "p5": p5}
