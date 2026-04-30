from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBNAct(nn.Module):
    """Conv → GroupNorm → SiLU block.

    GroupNorm replaces BatchNorm throughout the FPN.  With small MOT16 batch
    sizes (12 images, often containing only a few persons each), BN running
    statistics are noisy.  GN(32) is stride- and batch-size-independent and
    significantly more stable for dense detection heads.

    groups=32 requires out_channels to be divisible by 32; with
    fpn_channels=384 that gives 12 channels/group which is healthy.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 32,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        num_groups = min(groups, out_channels)
        # Ensure divisibility — fall back to out_channels (i.e. LayerNorm-style)
        while out_channels % num_groups != 0:
            num_groups //= 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyFPN(nn.Module):
    """PANet-style FPN with two smoothing convolutions per level.

    Fix 5: Added a second 3×3 conv after the first smoothing conv at each
    level (standard PANet/PAFPN practice for large backbones).  The first
    conv fuses multi-scale features; the second refines them.  This doubles
    the neck depth without changing the output channel count or the
    interface with the detection head.

    GroupNorm is used throughout (see ConvBNAct) for training stability on
    small batches.
    """

    def __init__(self, in_channels: dict[str, int], out_channels: int = 128) -> None:
        super().__init__()
        self.lateral_c3 = ConvBNAct(in_channels["c3"], out_channels, kernel_size=1)
        self.lateral_c4 = ConvBNAct(in_channels["c4"], out_channels, kernel_size=1)
        self.lateral_c5 = ConvBNAct(in_channels["c5"], out_channels, kernel_size=1)

        # Top-down path: two 3×3 smoothing convs per level (Fix 5)
        self.smooth_p3_1 = ConvBNAct(out_channels, out_channels, kernel_size=3)
        self.smooth_p3_2 = ConvBNAct(out_channels, out_channels, kernel_size=3)
        self.smooth_p4_1 = ConvBNAct(out_channels, out_channels, kernel_size=3)
        self.smooth_p4_2 = ConvBNAct(out_channels, out_channels, kernel_size=3)
        self.smooth_p5_1 = ConvBNAct(out_channels, out_channels, kernel_size=3)
        self.smooth_p5_2 = ConvBNAct(out_channels, out_channels, kernel_size=3)

        # Bottom-up path: two convs per level as well
        self.down_p3 = ConvBNAct(out_channels, out_channels, kernel_size=3, stride=2)
        self.down_p4 = ConvBNAct(out_channels, out_channels, kernel_size=3, stride=2)
        self.bu_p4_1 = ConvBNAct(out_channels, out_channels, kernel_size=3)
        self.bu_p4_2 = ConvBNAct(out_channels, out_channels, kernel_size=3)
        self.bu_p5_1 = ConvBNAct(out_channels, out_channels, kernel_size=3)
        self.bu_p5_2 = ConvBNAct(out_channels, out_channels, kernel_size=3)

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        c3, c4, c5 = features["c3"], features["c4"], features["c5"]

        # Lateral projections
        l5 = self.lateral_c5(c5)
        l4 = self.lateral_c4(c4) + F.interpolate(l5, size=c4.shape[-2:], mode="nearest")
        l3 = self.lateral_c3(c3) + F.interpolate(l4, size=c3.shape[-2:], mode="nearest")

        # Top-down double smoothing (Fix 5)
        p3 = self.smooth_p3_2(self.smooth_p3_1(l3))
        p4 = self.smooth_p4_2(self.smooth_p4_1(l4))
        p5 = self.smooth_p5_2(self.smooth_p5_1(l5))

        # Bottom-up double smoothing (Fix 5)
        p4 = self.bu_p4_2(self.bu_p4_1(p4 + self.down_p3(p3)))
        p5 = self.bu_p5_2(self.bu_p5_1(p5 + self.down_p4(p4)))

        return {"p3": p3, "p4": p4, "p5": p5}
