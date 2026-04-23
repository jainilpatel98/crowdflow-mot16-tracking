from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    MobileNet_V3_Large_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
)
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large, resnet18, resnet50


class MobileNetV3SmallBackbone(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = MobileNet_V3_Small_Weights.DEFAULT
            except Exception:
                weights = None
        try:
            backbone = mobilenet_v3_small(weights=weights)
        except Exception:
            backbone = mobilenet_v3_small(weights=None)
        self.features = backbone.features
        self.out_channels = {"c3": 24, "c4": 48, "c5": 576}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        c3 = c4 = c5 = None
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index == 3:
                c3 = x
            elif index == 8:
                c4 = x
            elif index == 12:
                c5 = x
        assert c3 is not None and c4 is not None and c5 is not None
        return {"c3": c3, "c4": c4, "c5": c5}


class MobileNetV3LargeBackbone(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = MobileNet_V3_Large_Weights.DEFAULT
            except Exception:
                weights = None
        try:
            backbone = mobilenet_v3_large(weights=weights)
        except Exception:
            backbone = mobilenet_v3_large(weights=None)
        self.features = backbone.features
        self.out_channels = {"c3": 40, "c4": 112, "c5": 960}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        c3 = c4 = c5 = None
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index == 5:
                c3 = x
            elif index == 11:
                c4 = x
            elif index == 15:
                c5 = x
        assert c3 is not None and c4 is not None and c5 is not None
        return {"c3": c3, "c4": c4, "c5": c5}


class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = ResNet18_Weights.DEFAULT
            except Exception:
                weights = None
        try:
            backbone = resnet18(weights=weights)
        except Exception:
            backbone = resnet18(weights=None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.out_channels = {"c3": 128, "c4": 256, "c5": 512}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return {"c3": c3, "c4": c4, "c5": c5}


class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = ResNet50_Weights.DEFAULT
            except Exception:
                weights = None
        try:
            backbone = resnet50(weights=weights)
        except Exception:
            backbone = resnet50(weights=None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.out_channels = {"c3": 512, "c4": 1024, "c5": 2048}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return {"c3": c3, "c4": c4, "c5": c5}


def build_backbone(backbone_name: str, pretrained: bool = False) -> nn.Module:
    name = backbone_name.lower()
    if name == "mobilenetv3_small":
        return MobileNetV3SmallBackbone(pretrained=pretrained)
    if name == "mobilenetv3_large":
        return MobileNetV3LargeBackbone(pretrained=pretrained)
    if name == "resnet18":
        return ResNet18Backbone(pretrained=pretrained)
    if name == "resnet50":
        return ResNet50Backbone(pretrained=pretrained)
    raise ValueError(f"Unsupported backbone: {backbone_name}")
