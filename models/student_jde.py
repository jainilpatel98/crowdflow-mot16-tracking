from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torchvision.ops import roi_align

from models.backbone_mobilenetv3 import build_backbone
from models.heads import DetectionEmbeddingHead
from models.neck_fpn import TinyFPN


class ROIProjector(nn.Module):
    def __init__(self, in_channels: int, emb_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 7 * 7, emb_dim * 2),
            nn.BatchNorm1d(emb_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(emb_dim * 2, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.proj(x), dim=-1)


class StudentJDE(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        emb_dim: int = 128,
        num_classes: int = 1,
        fpn_channels: int = 256,
        pretrained_backbone: bool = False,
        num_id_classes: int = 0,
        roi_dropout: float = 0.0,
        tower_layers: int = 2,
        tower_dropout: float = 0.0,   # P5: head regularization; 0.1 for large backbones
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(backbone_name, pretrained=pretrained_backbone)
        self.neck = TinyFPN(self.backbone.out_channels, out_channels=fpn_channels)
        self.head = DetectionEmbeddingHead(
            fpn_channels, num_classes=num_classes, emb_dim=emb_dim,
            tower_layers=tower_layers,
            tower_dropout=tower_dropout,
        )
        self.roi_projector = ROIProjector(fpn_channels, emb_dim, dropout=roi_dropout)
        self.id_classifier = nn.Linear(emb_dim, num_id_classes) if num_id_classes > 0 else None
        self.feature_channels = {"p3": fpn_channels, "p4": fpn_channels, "p5": fpn_channels}
        self.emb_dim = emb_dim

    def forward(
        self,
        images: torch.Tensor,
        roi_boxes_per_image: Iterable[torch.Tensor] | None = None,
        roi_image_size: tuple[int, int] | None = None,
        id_boxes_per_image: Iterable[torch.Tensor] | None = None,
        detach_features_for_roi: bool = False,
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        backbone_features = self.backbone(images)
        pyramid_features = self.neck(backbone_features)
        predictions = self.head(pyramid_features)
        predictions["features"] = pyramid_features
        if (roi_boxes_per_image is not None or id_boxes_per_image is not None) and roi_image_size is not None:
            roi_feature_source = (
                {level: feature.detach() for level, feature in pyramid_features.items()}
                if detach_features_for_roi
                else pyramid_features
            )
            roi_outputs = {"features": roi_feature_source}
            if roi_boxes_per_image is not None:
                predictions["roi_embeddings"] = self.extract_roi_embeddings(
                    roi_outputs,
                    boxes_per_image=roi_boxes_per_image,
                    image_size=roi_image_size,
                )
            if id_boxes_per_image is not None:
                if roi_boxes_per_image is id_boxes_per_image and "roi_embeddings" in predictions:
                    id_embeddings = predictions["roi_embeddings"]
                else:
                    id_embeddings = self.extract_roi_embeddings(
                        roi_outputs,
                        boxes_per_image=id_boxes_per_image,
                        image_size=roi_image_size,
                    )
                predictions["id_embeddings"] = id_embeddings
                if self.id_classifier is not None:
                    predictions["id_logits"] = self.id_classifier(id_embeddings)
        return predictions

    def extract_roi_embeddings(
        self,
        outputs: dict[str, dict[str, torch.Tensor]],
        boxes_per_image: Iterable[torch.Tensor],
        image_size: tuple[int, int],
        feature_level: str = "p3",
    ) -> torch.Tensor:
        feature_map = outputs["features"][feature_level]
        image_h, image_w = image_size
        scale = feature_map.shape[-1] / float(image_w)
        rois = []
        for batch_index, boxes in enumerate(boxes_per_image):
            if boxes.numel() == 0:
                continue
            batch_column = torch.full((boxes.shape[0], 1), batch_index, device=boxes.device, dtype=boxes.dtype)
            rois.append(torch.cat((batch_column, boxes), dim=1))
        if not rois:
            return feature_map.new_zeros((0, self.emb_dim))
        rois_tensor = torch.cat(rois, dim=0)
        pooled = roi_align(feature_map, rois_tensor, output_size=(7, 7), spatial_scale=scale, aligned=True)
        return self.roi_projector(pooled)

    def classify_ids(self, embeddings: torch.Tensor) -> torch.Tensor | None:
        if self.id_classifier is None or embeddings.numel() == 0:
            return None
        return self.id_classifier(embeddings)
