from __future__ import annotations

import torch
from torch.nn import functional as F

from utils.assigners import LevelAssignment
from utils.box_ops import aligned_iou, distance2bbox


def _binary_logits_to_two_class(logits: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(logits)
    return torch.cat((zeros, logits), dim=1)


def classification_kd_loss(
    student_logits: dict[str, torch.Tensor],
    teacher_logits: dict[str, torch.Tensor],
    temperature: float = 2.0,
    fg_threshold: float = 0.1,
) -> torch.Tensor:
    """Foreground-weighted classification knowledge distillation.

    Instead of computing KL-divergence over all spatial cells (batchmean),
    which produces ~115K active terms per batch and drowns the sparse
    detection gradient signal, this version restricts KD to cells where the
    teacher is confident (teacher_sigmoid > fg_threshold).

    This reduces the active term count by ~99% while focusing the student
    precisely where the teacher detects persons. Falls back to global mean
    when no foreground cells are found in a batch.

    Args:
        student_logits: Per-level student classification logits {level: (B, C, H, W)}.
        teacher_logits: Per-level teacher classification logits {level: (B, 1, H, W)}.
        temperature:    Distillation temperature (scales soft targets and log-probs).
        fg_threshold:   Teacher sigmoid confidence threshold for foreground selection.
    """
    total = next(iter(student_logits.values())).new_tensor(0.0)
    for level_name, stu in student_logits.items():
        tea = teacher_logits[level_name]
        if stu.shape[-2:] != tea.shape[-2:]:
            tea = F.interpolate(tea, size=stu.shape[-2:], mode="bilinear", align_corners=False)
        stu_two = _binary_logits_to_two_class(stu)
        tea_two = _binary_logits_to_two_class(tea)
        log_probs = F.log_softmax(stu_two / temperature, dim=1)      # (B, 2, H, W)
        soft_targets = F.softmax(tea_two / temperature, dim=1)        # (B, 2, H, W)

        # Teacher confidence mask: (B, H, W) — True at foreground cells
        fg_mask = (tea.sigmoid() > fg_threshold).squeeze(1)           # (B, H, W)

        if fg_mask.any():
            # Select foreground positions: (N_fg, 2)
            # Permute to (B, H, W, 2) then index with fg_mask
            log_fg = log_probs.permute(0, 2, 3, 1)[fg_mask]           # (N_fg, 2)
            tgt_fg = soft_targets.permute(0, 2, 3, 1)[fg_mask]        # (N_fg, 2)
            # Per-element KL-div, then mean over selected positions
            kl = F.kl_div(log_fg, tgt_fg, reduction="batchmean") * (temperature ** 2)
        else:
            # No teacher foreground in this batch — fall back to global mean
            # (very rare in crowded MOT16; contributes negligible gradient)
            kl = F.kl_div(log_probs, soft_targets, reduction="batchmean") * (temperature ** 2)

        total = total + kl
    return total


def box_kd_loss(
    student_boxes: dict[str, torch.Tensor],
    teacher_boxes: dict[str, torch.Tensor],
    assignments: dict[str, LevelAssignment],
) -> torch.Tensor:
    total = next(iter(student_boxes.values())).new_tensor(0.0)
    for level_name, stu in student_boxes.items():
        tea = teacher_boxes[level_name]
        pos_mask = assignments[level_name].pos_mask.squeeze(1)
        if not pos_mask.any():
            total = total + stu.sum() * 0.0
            continue
        pred_ltrb = stu.permute(0, 2, 3, 1)[pos_mask]
        teacher_ltrb = tea.permute(0, 2, 3, 1)[pos_mask]
        points = assignments[level_name].points.unsqueeze(0).expand(stu.shape[0], -1, -1, -1)[pos_mask]
        pred_boxes = distance2bbox(points, pred_ltrb)
        teacher_boxes_decoded = distance2bbox(points, teacher_ltrb)
        smooth_l1 = F.smooth_l1_loss(pred_ltrb, teacher_ltrb)
        iou_term = 1.0 - aligned_iou(pred_boxes, teacher_boxes_decoded).mean()
        total = total + 0.5 * smooth_l1 + 0.5 * iou_term
    return total


def feature_distill_loss(
    student_features: dict[str, torch.Tensor],
    teacher_features: dict[str, torch.Tensor],
    adapters=None,
    normalize: bool = True,
) -> torch.Tensor:
    total = next(iter(student_features.values())).new_tensor(0.0)
    adapted = adapters(student_features) if adapters is not None else student_features
    for level_name, stu in adapted.items():
        tea = teacher_features[level_name]
        if stu.shape[-2:] != tea.shape[-2:]:
            stu = F.interpolate(stu, size=tea.shape[-2:], mode="bilinear", align_corners=False)
        if normalize:
            stu = F.normalize(stu.flatten(2), dim=1).view_as(stu)
            tea = F.normalize(tea.flatten(2), dim=1).view_as(tea)
        total = total + F.mse_loss(stu, tea.detach())
    return total


def embedding_cosine_loss(student_embeddings: torch.Tensor, teacher_embeddings: torch.Tensor) -> torch.Tensor:
    if student_embeddings.numel() == 0 or teacher_embeddings.numel() == 0:
        return student_embeddings.new_tensor(0.0)
    cosine = F.cosine_similarity(student_embeddings, teacher_embeddings.detach(), dim=-1)
    return (1.0 - cosine).mean()
