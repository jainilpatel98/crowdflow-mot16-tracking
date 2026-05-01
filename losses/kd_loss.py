from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align

from utils.assigners import LevelAssignment
from utils.box_ops import aligned_ciou, aligned_iou, bbox2distance, box_iou, distance2bbox


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
        log_probs    = F.log_softmax(stu_two / temperature, dim=1)   # (B, 2, H, W)
        soft_targets = F.softmax(tea_two / temperature, dim=1)        # (B, 2, H, W)

        # Teacher confidence mask: (B, H, W) — True at foreground cells
        fg_mask = (tea.sigmoid() > fg_threshold).squeeze(1)           # (B, H, W)

        if fg_mask.any():
            log_fg = log_probs.permute(0, 2, 3, 1)[fg_mask]           # (N_fg, 2)
            tgt_fg = soft_targets.permute(0, 2, 3, 1)[fg_mask]        # (N_fg, 2)
            kl = F.kl_div(log_fg, tgt_fg, reduction="batchmean") * (temperature ** 2)
        else:
            kl = F.kl_div(log_probs, soft_targets, reduction="batchmean") * (temperature ** 2)

        total = total + kl
    return total


def box_kd_loss(
    student_boxes: dict[str, torch.Tensor],
    teacher_det_boxes: list[torch.Tensor],
    assignments: dict[str, LevelAssignment],
    iou_match_threshold: float = 0.3,
    teacher_weight: float = 0.25,
) -> torch.Tensor:
    """GT-anchored CIoU box distillation (P4 fix for train10).

    **Problem with the old approach:**
    The previous loss computed smooth_l1 + IoU against the *teacher's decoded
    box* exclusively.  The teacher box is its own regression output — not
    guaranteed to coincide with the GT annotation.  As a result the student
    memorised the teacher's *box style* (possibly systematically offset or
    differently scaled) rather than improving its GT IoU accuracy.  This
    caused mAP@0.5:0.95 to *regress* from 0.228 (train8) to 0.222 (train9)
    despite box_kd being active.

    **New loss:**
    For each assigner-positive cell:
      1. Compute CIoU(student_pred, GT_box) — primary signal, weight=0.75.
         This directly measures and optimises GT localization quality.
      2. If the GT at that cell has a high-IoU teacher match
         (iou ≥ iou_match_threshold), add CIoU(student_pred, teacher_box)
         as a soft regularizer, weight = teacher_weight (default 0.25).
         This preserves the semantic of "match teacher when it's reliable".

    Both CIoU terms operate in decoded pixel-space xyxy.

    Args:
        student_boxes:       {level: (B, 4, H, W)} stride-normalised LTRB.
        teacher_det_boxes:   list[Tensor(N_i, 4)] per-image xyxy teacher dets.
        assignments:         Student assigner output (pos_mask, box_xyxy, points, stride).
        iou_match_threshold: Min IoU(GT, teacher) to activate teacher CIoU term.
        teacher_weight:      Weight for the teacher soft-target CIoU term (default 0.25).
    """
    gt_weight = 1.0 - teacher_weight
    total = next(iter(student_boxes.values())).new_tensor(0.0)

    for level_name, stu in student_boxes.items():
        asgn   = assignments[level_name]
        stride = asgn.stride
        B, _, H, W = stu.shape

        stu_pos = asgn.pos_mask.squeeze(1)   # (B, H, W) bool
        if not stu_pos.any():
            total = total + stu.sum() * 0.0
            continue

        # --- Collect all positive-cell data across the batch ---
        pred_xyxy_list  = []   # decoded student xyxy
        gt_xyxy_list    = []   # GT xyxy (primary CIoU target)
        tea_xyxy_list   = []   # matched teacher xyxy (secondary CIoU target)
        tea_mask_list   = []   # bool: whether a teacher match was found per cell

        pts_batch = asgn.points.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)

        for bi in range(B):
            pos_i = stu_pos[bi]               # (H, W) bool
            if not pos_i.any():
                continue

            # GT xyxy at positive cells — (N_pos, 4) pixel space
            gt_xyxy = asgn.box_xyxy.permute(0, 2, 3, 1)[bi][pos_i]   # (N_pos, 4)

            # Decode student LTRB → xyxy (pixel space)
            matched_points = pts_batch[bi][pos_i]                              # (N_pos, 2)
            pred_ltrb      = stu.permute(0, 2, 3, 1)[bi][pos_i]               # (N_pos, 4) stride-norm
            pred_xyxy      = distance2bbox(matched_points, pred_ltrb * stride) # (N_pos, 4) pixel

            pred_xyxy_list.append(pred_xyxy)
            gt_xyxy_list.append(gt_xyxy)

            # --- Teacher soft target (optional per-cell) ---
            tea_boxes = teacher_det_boxes[bi]   # (T, 4)
            N_pos = gt_xyxy.shape[0]
            has_teacher = torch.zeros(N_pos, dtype=torch.bool, device=stu.device)
            tea_matched  = torch.zeros_like(gt_xyxy)  # (N_pos, 4) default zeros

            if tea_boxes.numel() > 0:
                iou_mat  = box_iou(gt_xyxy, tea_boxes)      # (N_pos, T)
                best_iou, best_t = iou_mat.max(dim=1)       # (N_pos,)
                matched  = best_iou >= iou_match_threshold
                tea_matched[matched] = tea_boxes[best_t[matched]]
                has_teacher = matched

            tea_xyxy_list.append(tea_matched)
            tea_mask_list.append(has_teacher)

        if not pred_xyxy_list:
            total = total + stu.sum() * 0.0
            continue

        pred_xyxy_all = torch.cat(pred_xyxy_list, dim=0)   # (N_total, 4)
        gt_xyxy_all   = torch.cat(gt_xyxy_list,   dim=0)   # (N_total, 4)
        tea_xyxy_all  = torch.cat(tea_xyxy_list,  dim=0)   # (N_total, 4)
        tea_mask_all  = torch.cat(tea_mask_list,   dim=0)   # (N_total,) bool

        # --- Primary: CIoU(pred, GT) for ALL positive cells ---
        gt_ciou  = aligned_ciou(pred_xyxy_all, gt_xyxy_all).mean()

        # --- Secondary: CIoU(pred, teacher) only where teacher matched ---
        if tea_mask_all.any():
            tea_ciou = aligned_ciou(
                pred_xyxy_all[tea_mask_all],
                tea_xyxy_all[tea_mask_all],
            ).mean()
        else:
            tea_ciou = pred_xyxy_all.sum() * 0.0

        total = total + gt_weight * gt_ciou + teacher_weight * tea_ciou

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

    Attention weight: mean(|teacher_features|, dim=C) / (global_mean + eps),
    normalised so the average weight is 1.0.
    """
    total = next(iter(student_features.values())).new_tensor(0.0)
    adapted = adapters(student_features) if adapters is not None else student_features
    for level_name, stu in adapted.items():
        tea = teacher_features[level_name].detach()
        if stu.shape[-2:] != tea.shape[-2:]:
            stu = F.interpolate(stu, size=tea.shape[-2:], mode="bilinear", align_corners=False)
        attention = tea.abs().mean(dim=1, keepdim=True)       # (B, 1, H, W)
        attention = attention / (attention.mean() + 1e-6)      # normalize → mean=1
        loss = (attention * (stu - tea).pow(2)).mean()
        total = total + loss
    return total


def dense_emb_alignment_loss(
    student_emb_outputs: dict[str, torch.Tensor],
    student_fpn_features: dict[str, torch.Tensor],
    roi_projector: nn.Module,
    assignments: dict[str, LevelAssignment],
    image_size: tuple[int, int],
    feature_level: str = "p3",
) -> torch.Tensor:
    """Align the dense emb_pred map to the trained ROI projector (Fix 6).

    **Problem:** The emb_tower + emb_pred in DetectionEmbeddingHead produces a
    dense per-cell embedding map that is used at inference for tracking.
    Previously it received ZERO gradient — the ROI projector is trained by
    emb_kd/id_loss but those gradients do not reach emb_pred.  Result: the
    inference embedding path outputs random noise.

    **Fix (Option B — proper):**
    At each student-assigner-positive cell:
    1. Read the GT box xyxy at that cell.
    2. ROI-pool from the NON-DETACHED FPN features (so gradients flow through
       emb_tower → neck → backbone).
    3. Project via roi_projector.detach() → embedding target (detached so
       gradients don't flow back through roi_projector from this path; it's
       already trained by emb_kd/id_loss).
    4. Cosine loss between emb_pred at the positive cell and the target.

    We use only p3 (finest level) for ROI pooling, matching the student's
    extract_roi_embeddings() which also uses p3 exclusively.

    Args:
        student_emb_outputs:  outputs["emb"] {level: (B, emb_dim, H, W)}
        student_fpn_features: pyramid_features, NOT detached
        roi_projector:        student.roi_projector (trained module)
        assignments:          assigner output (pos_mask, box_xyxy, stride)
        image_size:           (H, W) of input image
        feature_level:        FPN level for ROI pooling (default "p3")
    """
    # Use a single feature level for ROI pooling (p3, finest)
    p3_feat = student_fpn_features[feature_level]   # (B, C, H3, W3) — NOT detached
    image_h, image_w = image_size
    scale = p3_feat.shape[-1] / float(image_w)

    total       = p3_feat.new_tensor(0.0)
    num_matched = 0

    # Collect (positive cell emb_pred, GT box) pairs from all levels,
    # but pool from p3 only (consistent with inference path)
    all_rois   = []   # (batch_col, x1, y1, x2, y2) for roi_align
    all_emb_targets_idx = []   # indices into flat roi list for loss

    cell_embs = []   # dense emb_pred at positive cells — will align to roi_projector output

    for level_name, asgn in assignments.items():
        emb_map = student_emb_outputs[level_name]   # (B, emb_dim, H, W)
        B, emb_dim, H, W = emb_map.shape
        pos_mask = asgn.pos_mask.squeeze(1)          # (B, H, W)
        if not pos_mask.any():
            continue

        for bi in range(B):
            pos_i = pos_mask[bi]   # (H, W)
            if not pos_i.any():
                continue

            # emb_pred at positive cells: (N_pos, emb_dim)
            emb_at_pos = emb_map[bi].permute(1, 2, 0)[pos_i]   # (N_pos, emb_dim)

            # GT xyxy at positive cells: (N_pos, 4) pixel space
            gt_xyxy = asgn.box_xyxy.permute(0, 2, 3, 1)[bi][pos_i]   # (N_pos, 4)

            # Build ROI entries: (N_pos, 5) with batch column
            batch_col = torch.full(
                (gt_xyxy.shape[0], 1), float(bi),
                device=gt_xyxy.device, dtype=gt_xyxy.dtype,
            )
            rois = torch.cat([batch_col, gt_xyxy], dim=1)   # (N_pos, 5)

            all_rois.append(rois)
            cell_embs.append(emb_at_pos)

    if not all_rois:
        # No positive cells — return zero loss with gradient hook
        return student_emb_outputs[feature_level].sum() * 0.0

    rois_tensor = torch.cat(all_rois,   dim=0)   # (N_total, 5)
    cell_embs_t = torch.cat(cell_embs,  dim=0)   # (N_total, emb_dim)

    # ROI-pool from non-detached p3 — gradients flow through emb_tower
    pooled = roi_align(
        p3_feat, rois_tensor,
        output_size=(7, 7),
        spatial_scale=scale,
        aligned=True,
    )   # (N_total, C, 7, 7)

    # Project via roi_projector — DETACHED so gradients don't interfere with
    # emb_kd/id_loss training of the projector
    with torch.no_grad():
        targets = roi_projector(pooled)   # (N_total, emb_dim) — L2-normalised

    # Cosine loss: align dense emb_pred to roi_projector output
    cosine = F.cosine_similarity(cell_embs_t, targets, dim=-1)   # (N_total,)
    return (1.0 - cosine).mean()


def embedding_cosine_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
) -> torch.Tensor:
    if student_embeddings.numel() == 0 or teacher_embeddings.numel() == 0:
        return student_embeddings.new_tensor(0.0)
    cosine = F.cosine_similarity(student_embeddings, teacher_embeddings.detach(), dim=-1)
    return (1.0 - cosine).mean()
