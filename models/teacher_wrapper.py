from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align
from ultralytics import YOLO

def _split_flattened(flat_tensor: torch.Tensor, feature_maps: list[torch.Tensor]) -> list[torch.Tensor]:
    splits = [feature.shape[-2] * feature.shape[-1] for feature in feature_maps]
    chunks = torch.split(flat_tensor, splits, dim=-1)
    reshaped = []
    for chunk, feature in zip(chunks, feature_maps):
        batch, channels = chunk.shape[:2]
        height, width = feature.shape[-2:]
        reshaped.append(chunk.view(batch, channels, height, width))
    return reshaped


class TeacherWrapper(nn.Module):
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        person_class: int = 0,
        emb_dim: int = 128,
    ) -> None:
        super().__init__()
        self.yolo = YOLO(ckpt_path)
        self.model = self.yolo.model.to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.person_class = person_class
        self.emb_dim = emb_dim

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> dict[str, object]:
        predictions, aux = self.model(images)
        teacher_many = aux["one2many"]
        feature_maps = teacher_many["feats"]

        logits_levels = _split_flattened(
            teacher_many["scores"][:, self.person_class : self.person_class + 1, :],
            feature_maps,
        )
        raw_box_levels = _split_flattened(teacher_many["boxes"], feature_maps)

        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []
        for batch_index in range(predictions.shape[0]):
            pred = predictions[batch_index]
            if pred.numel() == 0:
                filtered_boxes.append(pred.new_zeros((0, 4)))
                filtered_scores.append(pred.new_zeros((0,)))
                filtered_labels.append(pred.new_zeros((0,), dtype=torch.long))
                continue
            mask = pred[:, 5].long() == self.person_class
            filtered = pred[mask]
            filtered_boxes.append(filtered[:, :4])
            filtered_scores.append(filtered[:, 4])
            filtered_labels.append(filtered[:, 5].long())

        p3, p4, p5 = feature_maps
        return {
            "features": {"p3": p3, "p4": p4, "p5": p5},
            "logits": {"p3": logits_levels[0], "p4": logits_levels[1], "p5": logits_levels[2]},
            "raw_boxes": {"p3": raw_box_levels[0], "p4": raw_box_levels[1], "p5": raw_box_levels[2]},
            "boxes": filtered_boxes,
            "scores": filtered_scores,
            "labels": filtered_labels,
            "spatial_feat": p5,
        }

    @torch.no_grad()
    def extract_roi_embeddings(
        self,
        spatial_feat: torch.Tensor,
        boxes_per_image: Iterable[torch.Tensor],
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        image_h, image_w = image_size
        scale = spatial_feat.shape[-1] / float(image_w)
        rois = []
        for batch_index, boxes in enumerate(boxes_per_image):
            if boxes.numel() == 0:
                continue
            batch_column = torch.full((boxes.shape[0], 1), batch_index, device=boxes.device, dtype=boxes.dtype)
            rois.append(torch.cat((batch_column, boxes), dim=1))
        if not rois:
            return spatial_feat.new_zeros((0, self.emb_dim))
        rois_tensor = torch.cat(rois, dim=0)
        pooled = roi_align(spatial_feat, rois_tensor, output_size=(7, 7), spatial_scale=scale, aligned=True)
        pooled = pooled.mean(dim=(-1, -2))
        if pooled.shape[1] != self.emb_dim:
            pooled = F.adaptive_avg_pool1d(pooled.unsqueeze(1), self.emb_dim).squeeze(1)
        return F.normalize(pooled, dim=-1)
