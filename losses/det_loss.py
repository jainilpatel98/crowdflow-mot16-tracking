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
    centerness_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sigmoid focal loss with optional per-cell centerness quality weight.

    Fix 4: When centerness_weight is provided (shape B,1,H,W), positive cells
    are weighted by their centerness score.  This replaces the old separate
    obj_loss (BCE on centerness) which was leaking centerness into inference
    through the score = cls * obj formulation, suppressing off-center TPs.

    Now centerness is a *training-only* quality signal, not an inference gate.
    """
    probs = logits.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probs * targets + (1 - probs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if centerness_weight is not None:
        # centerness_weight: (B,1,H,W) in [0,1]; 0 at negatives, c ∈(0,1] at positives.
        # Downweight easy negatives (already near-zero in focal) and emphasise
        # well-centred positive cells — improves box quality gradient signal.
        # Add 1 to background cells so they're not zeroed out (targets==0 → weight=1).
        quality = centerness_weight + (1.0 - targets)   # 1 at negatives, c+1 at positives
        loss = loss * quality.clamp(min=0.0)
    return loss.mean()


def detection_loss(
    outputs: dict[str, dict[str, torch.Tensor]],
    assignments: dict[str, LevelAssignment],
    use_focal: bool = True,
) -> dict[str, torch.Tensor]:
    """Joint classification + objectness + box regression loss.

    Fix 3: box_targets are stride-normalized (LTRB / stride).  Before calling
    distance2bbox we multiply back by stride to recover pixel-space LTRB.
    The student head (softplus * exp_scale) also outputs stride-normalized
    distances, so the comparison is valid.

    Fix 4: Centerness is used as a *quality weight* on the cls focal loss
    instead of being a separate BCE head that leaks into inference scoring.
    The obj_loss (BCE on centerness) is still computed so the obj_pred head
    remains supervised and can be used for future quality-aware scoring, but
    its contribution is now separate from the inference path.
    """
    cls_total = outputs["cls"]["p3"].new_tensor(0.0)
    obj_total = outputs["cls"]["p3"].new_tensor(0.0)
    box_total = outputs["cls"]["p3"].new_tensor(0.0)

    for level_name, level_assignment in assignments.items():
        stride = level_assignment.stride
        cls_logits = outputs["cls"][level_name]
        obj_logits = outputs["obj"][level_name]
        box_pred   = outputs["box"][level_name]   # stride-normalized LTRB (Fix 3)

        # Fix 4: pass centerness as quality weight to focal loss
        if use_focal:
            cls_loss = sigmoid_focal_loss(
                cls_logits,
                level_assignment.cls_targets,
                centerness_weight=level_assignment.obj_targets,   # Fix 4
            )
        else:
            cls_loss = F.binary_cross_entropy_with_logits(
                cls_logits, level_assignment.cls_targets
            )

        # obj_pred still supervised by centerness BCE — kept for potential
        # future use (quality-aware scoring, IoU head, etc.)
        obj_loss = F.binary_cross_entropy_with_logits(
            obj_logits, level_assignment.obj_targets
        )

        pos_mask = level_assignment.pos_mask.squeeze(1)
        if pos_mask.any():
            # box_pred and box_targets are both stride-normalized (Fix 3)
            pred_ltrb_norm   = box_pred.permute(0, 2, 3, 1)[pos_mask]       # (N, 4)
            target_ltrb_norm = level_assignment.box_targets.permute(0, 2, 3, 1)[pos_mask]

            # Decode to pixel space for IoU computation: multiply by stride (Fix 3)
            pred_ltrb_px   = pred_ltrb_norm   * stride
            target_ltrb_px = target_ltrb_norm * stride

            points = level_assignment.points.unsqueeze(0).expand(
                box_pred.shape[0], -1, -1, -1
            )[pos_mask]
            pred_boxes   = distance2bbox(points, pred_ltrb_px)
            target_boxes = level_assignment.box_xyxy.permute(0, 2, 3, 1)[pos_mask]

            # Smooth-L1 on normalized targets (Fix 3: gradient scale O(1))
            l1_loss  = F.smooth_l1_loss(pred_ltrb_norm, target_ltrb_norm)
            iou_loss = 1.0 - aligned_iou(pred_boxes, target_boxes).mean()
            box_loss = l1_loss + iou_loss
        else:
            box_loss = box_pred.sum() * 0.0

        cls_total = cls_total + cls_loss
        obj_total = obj_total + obj_loss
        box_total = box_total + box_loss

    total = cls_total + obj_total + box_total
    return {
        "total":  total,
        "cls":    cls_total,
        "obj":    obj_total,
        "box":    box_total,
    }
