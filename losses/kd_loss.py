from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align

from utils.assigners import LevelAssignment
from utils.box_ops import aligned_iou, bbox2distance, box_iou, distance2bbox


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
) -> torch.Tensor:
    """Box knowledge distillation — matched-detection strategy (Fix 1).

    Uses the teacher's DECODED xyxy detections (`teacher_outputs["boxes"]`,
    a list of per-image xyxy tensors in pixel space) rather than the raw
    internal `teacher_outputs["raw_boxes"]` DFL outputs.

    Previous attempts were broken because:
    - raw_boxes (train3–6): teacher DFL/model-space values in [-0.8, 13],
      NOT pixel-space LTRB. Passing these into distance2bbox gave nonsense
      boxes; IoU=0 always → loss frozen at 24-31 forever.
    - Intersection strategy (train6 attempt): still used raw_boxes. The
      coordinate fix was not the root cause — the FORMAT was wrong.

    **Matched-detection algorithm:**
    1. For each FPN level and each image in the batch, take the
       student-assigner positive cells (where det_loss trains the box head).
    2. At each positive cell the assigner stores the GT box xyxy.  Find the
       best-matching teacher detection by computing IoU between the GT xyxy
       and every teacher-detected box for that image.
    3. If the best match IoU > iou_match_threshold, the teacher detection is
       a valid soft target for this cell.
    4. Convert the matched teacher xyxy box → LTRB relative to the student
       anchor point via bbox2distance().  This is identical coordinate system
       to the student's box_pred (stride-normalised LTRB after Fix 3).
    5. Compute smooth_l1 + (1-IoU) between student pred and teacher target.

    **Why this works:**
    - Teacher decoded xyxy ARE in pixel space → compatible with student coords.
    - We only supervise at cells where det_loss also trains → no gradient
      conflict (both losses push box head in same direction).
    - IoU matching ensures we only distil when teacher actually detects the
      same GT object (avoids distiling wrong person in crowds).

    **Config:** box_kd weight is 0.0 by default; set to a non-zero value
    only after verifying teacher detections are reasonable.

    Args:
        student_boxes:       {level: (B, 4, H, W)} stride-normalised LTRB.
        teacher_det_boxes:   list[Tensor(N_i, 4)] per-image xyxy teacher dets.
        assignments:         Student assigner output (pos_mask, box_xyxy, points, stride).
        iou_match_threshold: Min IoU between GT xyxy and teacher detection to
                             count as a valid distillation target.
    """
    total = next(iter(student_boxes.values())).new_tensor(0.0)

    for level_name, stu in student_boxes.items():
        asgn   = assignments[level_name]
        stride = asgn.stride
        B, _, H, W = stu.shape

        stu_pos = asgn.pos_mask.squeeze(1)   # (B, H, W) bool
        if not stu_pos.any():
            total = total + stu.sum() * 0.0
            continue

        # Collect matched pairs across the batch
        pred_ltrb_list   = []
        target_ltrb_list = []
        points_list      = []

        pts_batch = asgn.points.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)

        for bi in range(B):
            pos_i = stu_pos[bi]               # (H, W) bool
            if not pos_i.any():
                continue

            # GT xyxy at positive cells — (N_pos, 4) pixel space
            gt_xyxy = asgn.box_xyxy.permute(0, 2, 3, 1)[bi][pos_i]   # (N_pos, 4)

            # Teacher detections for this image — (T, 4) pixel space
            tea_boxes = teacher_det_boxes[bi]   # (T, 4)
            if tea_boxes.numel() == 0:
                continue

            # IoU between each GT cell position and each teacher detection
            # gt_xyxy: (N_pos, 4), tea_boxes: (T, 4)
            iou_mat = box_iou(gt_xyxy, tea_boxes)       # (N_pos, T)
            best_iou, best_t = iou_mat.max(dim=1)       # (N_pos,), (N_pos,)

            matched = best_iou >= iou_match_threshold    # (N_pos,) bool
            if not matched.any():
                continue

            # Matched teacher xyxy → LTRB relative to student anchor point
            matched_tea_xyxy  = tea_boxes[best_t[matched]]             # (N_match, 4)
            matched_points    = pts_batch[bi][pos_i][matched]          # (N_match, 2)
            matched_pred_ltrb = stu.permute(0, 2, 3, 1)[bi][pos_i][matched]  # (N_match, 4)

            # bbox2distance: xyxy → LTRB in pixel space, then normalize by stride (Fix 3)
            tea_ltrb_px   = bbox2distance(matched_points, matched_tea_xyxy)  # pixel LTRB
            tea_ltrb_norm = tea_ltrb_px / stride                             # stride-norm

            pred_ltrb_list.append(matched_pred_ltrb)
            target_ltrb_list.append(tea_ltrb_norm)
            points_list.append(matched_points)

        if not pred_ltrb_list:
            total = total + stu.sum() * 0.0
            continue

        pred_ltrb_all   = torch.cat(pred_ltrb_list,   dim=0)   # (N_total, 4) stride-norm
        target_ltrb_all = torch.cat(target_ltrb_list, dim=0)   # (N_total, 4) stride-norm
        points_all      = torch.cat(points_list,       dim=0)   # (N_total, 2)

        # Clamp teacher targets to non-negative (GT box may be partially OOB)
        target_ltrb_all = target_ltrb_all.clamp(min=0.0)

        # Smooth-L1 on stride-normalised LTRB (O(1) scale)
        smooth_l1 = F.smooth_l1_loss(pred_ltrb_all, target_ltrb_all)

        # IoU on pixel-space decoded boxes
        pred_boxes_px   = distance2bbox(points_all, pred_ltrb_all   * stride)
        target_boxes_px = distance2bbox(points_all, target_ltrb_all * stride)
        iou_term = 1.0 - aligned_iou(pred_boxes_px, target_boxes_px).mean()

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
