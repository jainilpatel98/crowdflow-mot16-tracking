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


# ---------------------------------------------------------------------------
# Teacher ROI projector  (Issue 2 fix)
# ---------------------------------------------------------------------------

class TeacherROIProjector(nn.Module):
    """A trainable MLP projector that extracts identity embeddings from the
    teacher's p3 feature map via ROI-Align.

    Architecture mirrors the student's ROIProjector so both operate in the
    same embedding space, enabling meaningful cosine alignment.

    This module is trained (not frozen) alongside the student, supervised by
    the ID cross-entropy loss. Its outputs serve as soft embedding targets for
    the student's embedding KD cosine loss.
    """

    def __init__(self, in_channels: int, emb_dim: int) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        hidden = emb_dim * 4
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 7 * 7, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: pooled ROI crops (N, C, 7, 7)."""
        return F.normalize(self.proj(x), dim=-1)

    def extract_embeddings(
        self,
        p3_feature: torch.Tensor,
        boxes_per_image: Iterable[torch.Tensor],
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        """ROI-Align on p3 then project to emb_dim.

        Args:
            p3_feature:     (B, C, H, W) teacher p3 feature map
            boxes_per_image: list[Tensor (N_i, 4)] xyxy boxes in image space
            image_size:     (H, W) of the input image

        Returns:
            (total_boxes, emb_dim) L2-normalised embeddings
        """
        image_h, image_w = image_size
        scale = p3_feature.shape[-1] / float(image_w)
        rois = []
        for batch_index, boxes in enumerate(boxes_per_image):
            if boxes.numel() == 0:
                continue
            batch_col = torch.full((boxes.shape[0], 1), batch_index,
                                   device=boxes.device, dtype=boxes.dtype)
            rois.append(torch.cat((batch_col, boxes), dim=1))
        if not rois:
            return p3_feature.new_zeros((0, self.emb_dim))
        rois_tensor = torch.cat(rois, dim=0)
        pooled = roi_align(p3_feature, rois_tensor, output_size=(7, 7),
                           spatial_scale=scale, aligned=True)
        return self.forward(pooled)


# ---------------------------------------------------------------------------
# Teacher YOLO wrapper
# ---------------------------------------------------------------------------

class TeacherWrapper(nn.Module):
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        person_class: int = 0,
        emb_dim: int = 128,
    ) -> None:
        super().__init__()
        yolo = YOLO(ckpt_path)
        self.model = yolo.model.to(device)
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
            teacher_many["scores"][:, self.person_class: self.person_class + 1, :],
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
            "spatial_feat": p3,   # use p3 (finest) for ROI projector input
        }
