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
    fg_threshold: float = 0.02,
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
    teacher_logits: dict[str, torch.Tensor] | None = None,
    teacher_conf_threshold: float = 0.05,
) -> torch.Tensor:
    """Box knowledge distillation loss — intersection strategy.

    Computes smooth-L1 + (1-IoU) at the INTERSECTION of:
      (a) student-assigner positives  (cells where det_loss trains the box head)
      (b) teacher-confident cells     (cells where teacher has a valid box target)

    **Why intersection is critical:**
    Previous attempts used either (a) alone or (b) alone:
    - Student-assigner-only (train3): box_kd frozen at 77 because teacher/student
      assignment cells never spatially aligned — IoU ≈ 0 always.
    - Teacher-confident-only (train4/5): gradient CONFLICT. At teacher-positive /
      student-NEGATIVE cells, det_loss pushes classification → background, but
      box_kd pushes box head → teacher LTRB. The box head receives opposing
      gradients from two separate loss surfaces and plateaus at 24-25.

    At the intersection:
    - det_loss is ALREADY training the box head to regress GT LTRB at those cells.
    - box_kd adds the teacher's LTRB as a complementary soft target.
    - Both signals push the box head in the SAME direction (teacher ≈ GT for MOT16).
    - The loss decreases from the very first intersection-containing batch.

    Falls back to student-assigner positives if the intersection is empty (rare,
    mainly in the first 1-2 epochs before the teacher fires confidently on
    training images).

    Args:
        student_boxes:           {level: (B, 4, H, W)} student LTRB predictions.
        teacher_boxes:           {level: (B, 4, H, W)} teacher LTRB predictions.
        assignments:             Student assigner output (box targets + pos_mask).
        teacher_logits:          {level: (B, 1, H, W)} teacher cls logits for mask.
        teacher_conf_threshold:  Min teacher sigmoid to count as teacher-positive.
    """
    total = next(iter(student_boxes.values())).new_tensor(0.0)
    for level_name, stu in student_boxes.items():
        tea = teacher_boxes[level_name]
        asgn = assignments[level_name]

        # (a) Student-assigner positives: cells where det_loss trains the box head
        stu_pos = asgn.pos_mask.squeeze(1)  # (B, H, W) bool

        if not stu_pos.any():
            total = total + stu.sum() * 0.0
            continue

        # (b) Teacher-confident: cells where teacher provides a reliable LTRB target
        if teacher_logits is not None:
            tea_conf = teacher_logits[level_name].sigmoid().squeeze(1)  # (B, H, W)
            tea_pos  = tea_conf > teacher_conf_threshold                 # (B, H, W)

            # Intersection: both student assigner positive AND teacher confident
            pos_mask = stu_pos & tea_pos
            if not pos_mask.any():
                # Fallback: student assigner positives only (teacher not yet confident)
                pos_mask = stu_pos
        else:
            pos_mask = stu_pos

        # Anchor points from student assigner — always in correct student-stride space
        B = stu.shape[0]
        pts_batch = asgn.points.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
        points = pts_batch[pos_mask]                                  # (N_pos, 2)

        pred_ltrb    = stu.permute(0, 2, 3, 1)[pos_mask]  # (N_pos, 4) — student predictions
        teacher_ltrb = tea.permute(0, 2, 3, 1)[pos_mask]  # (N_pos, 4) — teacher soft target

        # Clamp teacher LTRB to non-negative (teacher LTRB should be positive pixel
        # distances; small negatives from floating-point rounding cause IoU = 0)
        teacher_ltrb = teacher_ltrb.clamp(min=0.0)

        pred_boxes_dec    = distance2bbox(points, pred_ltrb)
        teacher_boxes_dec = distance2bbox(points, teacher_ltrb)

        smooth_l1 = F.smooth_l1_loss(pred_ltrb, teacher_ltrb)
        iou_term  = 1.0 - aligned_iou(pred_boxes_dec, teacher_boxes_dec).mean()
        total = total + 0.5 * smooth_l1 + 0.5 * iou_term
    return total



def feature_distill_loss(
    student_features: dict[str, torch.Tensor],
    teacher_features: dict[str, torch.Tensor],
    adapters=None,
) -> torch.Tensor:
    """Attention-weighted feature distillation loss.

    Computes MSE between adapted student features and (detached) teacher
    features, weighted by teacher activation magnitude so that regions where
    the teacher FPN is active (near persons) contribute more gradient than
    background regions.

    This replaces the previous L2-normalized MSE, which collapsed the loss
    to a constant ~0.003-0.010 regardless of feature alignment quality —
    making it impossible for the adapter to receive a useful training signal.

    Attention weight: mean(|teacher_features|, dim=C) / (global_mean + eps),
    normalised so the average weight is 1.0.  This preserves the effective
    loss magnitude while focusing gradients on person-containing regions.
    """
    total = next(iter(student_features.values())).new_tensor(0.0)
    adapted = adapters(student_features) if adapters is not None else student_features
    for level_name, stu in adapted.items():
        tea = teacher_features[level_name].detach()
        if stu.shape[-2:] != tea.shape[-2:]:
            stu = F.interpolate(stu, size=tea.shape[-2:], mode="bilinear", align_corners=False)
        # Spatial attention: (B, 1, H, W), mean activation magnitude per cell
        attention = tea.abs().mean(dim=1, keepdim=True)          # (B, 1, H, W)
        attention = attention / (attention.mean() + 1e-6)         # normalize → mean=1
        loss = (attention * (stu - tea).pow(2)).mean()
        total = total + loss
    return total


def embedding_cosine_loss(student_embeddings: torch.Tensor, teacher_embeddings: torch.Tensor) -> torch.Tensor:
    if student_embeddings.numel() == 0 or teacher_embeddings.numel() == 0:
        return student_embeddings.new_tensor(0.0)
    cosine = F.cosine_similarity(student_embeddings, teacher_embeddings.detach(), dim=-1)
    return (1.0 - cosine).mean()
