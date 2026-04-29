from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    MobileNet_V3_Large_Weights,
    ResNet18_Weights,
    ResNet101_Weights,
    ResNet50_Weights,
    ResNeXt50_32X4D_Weights,
    ResNeXt101_32X8D_Weights,
)
from torchvision.models import (
    mobilenet_v3_small,
    mobilenet_v3_large,
    resnet18,
    resnet50,
    resnet101,
    resnext50_32x4d,
    resnext101_32x8d,
)
from torchvision.models.resnet import Bottleneck, ResNet


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


def _load_torchvision_backbone(factory, weights_enum, pretrained: bool):
    weights = None
    if pretrained and weights_enum is not None:
        try:
            weights = weights_enum.DEFAULT
        except Exception:
            weights = None
    try:
        return factory(weights=weights)
    except Exception:
        return factory(weights=None)


def _load_partial_pretrained_weights(model: nn.Module, factory, weights_enum) -> None:
    source = _load_torchvision_backbone(factory, weights_enum, pretrained=True)
    model.load_state_dict(source.state_dict(), strict=False)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True)
        self.expand = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)
        self.activation = nn.ReLU(inplace=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(x)
        scale = self.reduce(scale)
        scale = self.activation(scale)
        scale = self.expand(scale)
        scale = self.gate(scale)
        return x * scale


class SEBottleneck(Bottleneck):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer=None,
    ) -> None:
        super().__init__(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample,
            groups=groups,
            base_width=base_width,
            dilation=dilation,
            norm_layer=norm_layer,
        )
        self.se = SqueezeExcitation(planes * self.expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


def se_resnet50(weights=None, progress: bool = True, **kwargs) -> ResNet:
    del progress
    if weights is not None:
        raise ValueError("SE-ResNet50 does not have a dedicated torchvision weights enum.")
    return ResNet(SEBottleneck, [3, 4, 6, 3], **kwargs)


def se_resnet101(weights=None, progress: bool = True, **kwargs) -> ResNet:
    del progress
    if weights is not None:
        raise ValueError("SE-ResNet101 does not have a dedicated torchvision weights enum.")
    return ResNet(SEBottleneck, [3, 4, 23, 3], **kwargs)


class TorchvisionResNetBackbone(nn.Module):
    def __init__(self, backbone: nn.Module, out_channels: dict[str, int]) -> None:
        super().__init__()
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return {"c3": c3, "c4": c4, "c5": c5}


class ResNet18Backbone(TorchvisionResNetBackbone):
    def __init__(self, pretrained: bool = False) -> None:
        backbone = _load_torchvision_backbone(resnet18, ResNet18_Weights, pretrained)
        super().__init__(
            backbone=backbone,
            out_channels={"c3": 128, "c4": 256, "c5": 512},
        )


class ResNet50Backbone(TorchvisionResNetBackbone):
    def __init__(self, pretrained: bool = False) -> None:
        backbone = _load_torchvision_backbone(resnet50, ResNet50_Weights, pretrained)
        super().__init__(
            backbone=backbone,
            out_channels={"c3": 512, "c4": 1024, "c5": 2048},
        )


class ResNet101Backbone(TorchvisionResNetBackbone):
    def __init__(self, pretrained: bool = False) -> None:
        backbone = _load_torchvision_backbone(resnet101, ResNet101_Weights, pretrained)
        super().__init__(
            backbone=backbone,
            out_channels={"c3": 512, "c4": 1024, "c5": 2048},
        )


class ResNeXt50Backbone(TorchvisionResNetBackbone):
    def __init__(self, pretrained: bool = False) -> None:
        backbone = _load_torchvision_backbone(resnext50_32x4d, ResNeXt50_32X4D_Weights, pretrained)
        super().__init__(
            backbone=backbone,
            out_channels={"c3": 512, "c4": 1024, "c5": 2048},
        )


class ResNeXt101Backbone(TorchvisionResNetBackbone):
    def __init__(self, pretrained: bool = False) -> None:
        backbone = _load_torchvision_backbone(resnext101_32x8d, ResNeXt101_32X8D_Weights, pretrained)
        super().__init__(
            backbone=backbone,
            out_channels={"c3": 512, "c4": 1024, "c5": 2048},
        )


class SEResNet50Backbone(TorchvisionResNetBackbone):
    def __init__(self, pretrained: bool = False) -> None:
        backbone = _load_torchvision_backbone(se_resnet50, None, pretrained=False)
        if pretrained:
            _load_partial_pretrained_weights(backbone, resnet50, ResNet50_Weights)
        super().__init__(
            backbone=backbone,
            out_channels={"c3": 512, "c4": 1024, "c5": 2048},
        )


class SEResNet101Backbone(TorchvisionResNetBackbone):
    def __init__(self, pretrained: bool = False) -> None:
        backbone = _load_torchvision_backbone(se_resnet101, None, pretrained=False)
        if pretrained:
            _load_partial_pretrained_weights(backbone, resnet101, ResNet101_Weights)
        super().__init__(
            backbone=backbone,
            out_channels={"c3": 512, "c4": 1024, "c5": 2048},
        )


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
    if name == "resnet101":
        return ResNet101Backbone(pretrained=pretrained)
    if name in {"se_resnet50", "se-resnet50", "seresnet50"}:
        return SEResNet50Backbone(pretrained=pretrained)
    if name in {"se_resnet101", "se-resnet101", "seresnet101"}:
        return SEResNet101Backbone(pretrained=pretrained)
    if name in {"resnext50", "resnext-50", "resnext50_32x4d", "resnext50-32x4d"}:
        return ResNeXt50Backbone(pretrained=pretrained)
    if name in {"resnext101", "resnext-101", "resnext101_32x8d", "resnext101-32x8d"}:
        return ResNeXt101Backbone(pretrained=pretrained)
    raise ValueError(f"Unsupported backbone: {backbone_name}")
