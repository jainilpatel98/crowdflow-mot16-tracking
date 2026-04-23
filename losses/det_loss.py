from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from utils.assigners import LevelAssignment
from utils.box_ops import aligned_iou, distance2bbox


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    probs = logits.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probs * targets + (1 - probs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()


def detection_loss(
    outputs: dict[str, dict[str, torch.Tensor]],
    assignments: dict[str, LevelAssignment],
    use_focal: bool = True,
) -> dict[str, torch.Tensor]:
    cls_total = outputs["cls"]["p3"].new_tensor(0.0)
    obj_total = outputs["cls"]["p3"].new_tensor(0.0)
    box_total = outputs["cls"]["p3"].new_tensor(0.0)

    for level_name, level_assignment in assignments.items():
        cls_logits = outputs["cls"][level_name]
        obj_logits = outputs["obj"][level_name]
        box_pred = outputs["box"][level_name]

        if use_focal:
            cls_loss = sigmoid_focal_loss(cls_logits, level_assignment.cls_targets)
        else:
            cls_loss = F.binary_cross_entropy_with_logits(cls_logits, level_assignment.cls_targets)
        obj_loss = F.binary_cross_entropy_with_logits(obj_logits, level_assignment.obj_targets)

        pos_mask = level_assignment.pos_mask.squeeze(1)
        if pos_mask.any():
            pred_ltrb = box_pred.permute(0, 2, 3, 1)[pos_mask]
            target_ltrb = level_assignment.box_targets.permute(0, 2, 3, 1)[pos_mask]
            points = level_assignment.points.unsqueeze(0).expand(box_pred.shape[0], -1, -1, -1)[pos_mask]
            pred_boxes = distance2bbox(points, pred_ltrb)
            target_boxes = level_assignment.box_xyxy.permute(0, 2, 3, 1)[pos_mask]

            l1_loss = F.smooth_l1_loss(pred_ltrb, target_ltrb)
            iou_loss = 1.0 - aligned_iou(pred_boxes, target_boxes).mean()
            box_loss = l1_loss + iou_loss
        else:
            box_loss = box_pred.sum() * 0.0

        cls_total = cls_total + cls_loss
        obj_total = obj_total + obj_loss
        box_total = box_total + box_loss

    total = cls_total + obj_total + box_total
    return {
        "total": total,
        "cls": cls_total,
        "obj": obj_total,
        "box": box_total,
    }
