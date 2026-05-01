from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from utils.box_ops import make_grid


@dataclass
class LevelAssignment:
    cls_targets: torch.Tensor
    obj_targets: torch.Tensor
    box_targets: torch.Tensor
    box_xyxy: torch.Tensor
    pos_mask: torch.Tensor
    track_ids: torch.Tensor
    points: torch.Tensor
    stride: int


class PyramidAssigner:
    """Fully-vectorised anchor-free assigner.

    For each FPN level:
    1. Filter GT boxes by area range (level-specific).
    2. Compute centre-radius mask: candidate grid cells within `center_radius`
       of each GT box centre (in grid-cell units).
    3. Compute point-in-box mask: grid point must lie inside the GT box.
       For small GT boxes (area < small_obj_area_threshold) the inside-box
       check is SKIPPED — only the centre-radius mask is used.  This prevents
       the recall ceiling for pedestrians smaller than ~32×32px, where so few
       grid cells fall inside the box that center_radius adjustments have no
       effect.
    4. Resolve conflicts: assign each grid cell to the smallest-area GT that
       covers it (tie-breaking by area encourages correct scale assignment).

    All operations are batched tensor ops — no Python loops over GT boxes or
    grid cells.
    """

    def __init__(
        self,
        strides: dict[str, int],
        area_ranges: dict[str, tuple[float, float]],
        center_radius: float | dict[str, float] = 1.5,
        small_obj_area_threshold: float = 1024.0,  # 32×32 px²
    ) -> None:
        self.strides = strides
        self.area_ranges = area_ranges
        # Allow per-level radius: {p3: 2.5, p4: 2.0, p5: 1.5}
        # or a single scalar applied to all levels.
        self.center_radius = center_radius
        # GT boxes with area below this threshold skip the point-in-box check.
        # Set to 0.0 to disable (revert to pure inside-check for all sizes).
        self.small_obj_area_threshold = small_obj_area_threshold

    def assign(
        self,
        outputs: dict[str, dict[str, torch.Tensor]],
        targets: list[dict[str, Any]],
    ) -> dict[str, LevelAssignment]:
        assignments: dict[str, LevelAssignment] = {}

        for level_name, cls_logits in outputs["cls"].items():
            device = cls_logits.device
            B, _, H, W = cls_logits.shape
            stride = self.strides[level_name]
            area_min, area_max = self.area_ranges[level_name]

            # Grid point centres in image space: (H, W, 2)
            points = make_grid(H, W, stride, device=device)
            # Flat grid: (H*W, 2)
            px = points[..., 0].reshape(-1)   # (H*W,) x-coords
            py = points[..., 1].reshape(-1)   # (H*W,) y-coords
            # Flat grid indices for the grid-cell axes
            gx = torch.arange(W, device=device, dtype=torch.float32)  # (W,)
            gy = torch.arange(H, device=device, dtype=torch.float32)  # (H,)

            cls_targets  = torch.zeros(B, 1, H, W, device=device)
            obj_targets  = torch.zeros(B, 1, H, W, device=device)
            box_targets  = torch.zeros(B, 4, H, W, device=device)
            box_xyxy_out = torch.zeros(B, 4, H, W, device=device)
            pos_mask     = torch.zeros(B, 1, H, W, dtype=torch.bool, device=device)
            track_ids    = torch.full((B, H, W), -1, dtype=torch.long, device=device)
            # Track smallest assigned area for conflict resolution
            assigned_area = torch.full((B, H, W), float("inf"), device=device)

            for bi, target in enumerate(targets):
                boxes = target["boxes"].to(device)          # (N, 4) xyxy
                ids   = target["track_labels"].to(device)   # (N,)
                if boxes.numel() == 0:
                    continue

                # ---- 1. Area filter ----
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # (N,)
                area_valid = (areas >= area_min) & (areas < area_max)
                if not area_valid.any():
                    continue
                boxes = boxes[area_valid]          # (M, 4)
                areas = areas[area_valid]          # (M,)
                ids   = ids[area_valid]            # (M,)
                M = boxes.shape[0]

                # ---- 2. Centre-radius mask  (M, H, W) ----
                # GT box centres in grid-cell fractional coordinates
                cx = (0.5 * (boxes[:, 0] + boxes[:, 2]) / stride - 0.5)  # (M,)
                cy = (0.5 * (boxes[:, 1] + boxes[:, 3]) / stride - 0.5)  # (M,)

                # Per-level radius (dict) or global scalar
                if isinstance(self.center_radius, dict):
                    radius = float(self.center_radius[level_name])
                else:
                    radius = float(self.center_radius)

                # Distance from each grid cell to each GT centre
                # gx: (W,)  cx: (M,)  → diff_x: (M, W)
                diff_x = (gx.unsqueeze(0) - cx.unsqueeze(1)).abs()   # (M, W)
                diff_y = (gy.unsqueeze(0) - cy.unsqueeze(1)).abs()   # (M, H)

                within_x = diff_x <= radius   # (M, W)
                within_y = diff_y <= radius   # (M, H)

                # (M, H, W) via broadcasting
                center_mask = within_y.unsqueeze(2) & within_x.unsqueeze(1)

                # ---- 3. Point-in-box mask  (M, H*W) → (M, H, W) ----
                # px, py: (H*W,)  boxes: (M, 4)
                left   = px.unsqueeze(0) - boxes[:, 0].unsqueeze(1)   # (M, H*W)
                top    = py.unsqueeze(0) - boxes[:, 1].unsqueeze(1)
                right  = boxes[:, 2].unsqueeze(1) - px.unsqueeze(0)
                bottom = boxes[:, 3].unsqueeze(1) - py.unsqueeze(0)
                inside = (left > 0) & (top > 0) & (right > 0) & (bottom > 0)
                inside = inside.view(M, H, W)                           # (M, H, W)

                # LTRB distances (M, H*W, 4) → (M, H, W, 4)
                ltrb = torch.stack([left, top, right, bottom], dim=-1).view(M, H, W, 4)

                # ---- 4. Combined validity & conflict resolution ----
                # For small GT boxes the point-in-box check is skipped:
                # a 16×24px box at stride=8 has only 2×3=6 cells strictly
                # inside, making center_radius expansions ineffective.  For
                # these objects we use center-only assignment so that the full
                # center_radius zone becomes active.
                is_small = (areas < self.small_obj_area_threshold)  # (M,)
                use_center_only = is_small.view(M, 1, 1)  # (M, 1, 1) broadcastable
                valid = torch.where(use_center_only, center_mask, center_mask & inside)

                # Replace invalid cells with inf area so they lose conflicts
                areas_exp = areas.view(M, 1, 1).expand(M, H, W)
                masked_areas = torch.where(valid, areas_exp,
                                           torch.full_like(areas_exp, float("inf")))

                # Minimum area per grid cell and index of best GT
                min_areas, best_gt = masked_areas.min(dim=0)  # (H, W), (H, W)
                has_valid = min_areas < float("inf")           # (H, W)

                # Only update cells where this batch item wins the conflict
                update = has_valid & (min_areas < assigned_area[bi])   # (H, W)
                if not update.any():
                    continue

                assigned_area[bi] = torch.where(update, min_areas, assigned_area[bi])

                # ---- 5. Write targets ----
                # pos_mask, cls_targets
                pos_mask[bi, 0]    |= update
                cls_targets[bi, 0]  = cls_targets[bi, 0].masked_fill(update, 1.0)

                # track_ids
                new_ids = ids[best_gt]         # (H, W)
                track_ids[bi] = torch.where(update, new_ids, track_ids[bi])

                # box_targets (LTRB / stride): gather best GT per cell.
                # Fix 3: Normalize by stride so the shared box regressor sees
                # O(1) targets regardless of FPN level.  Decode = pred * stride → pixel.
                best_gt_4 = best_gt.unsqueeze(0).unsqueeze(-1).expand(1, H, W, 4)
                best_ltrb = ltrb.gather(0, best_gt_4).squeeze(0)   # (H, W, 4) pixel LTRB
                best_ltrb_norm = (best_ltrb / stride).permute(2, 0, 1)  # (4, H, W) normalized
                for c in range(4):
                    box_targets[bi, c] = torch.where(update, best_ltrb_norm[c], box_targets[bi, c])

                # box_xyxy: gather best GT box coords
                best_boxes = boxes[best_gt.reshape(-1)].reshape(H, W, 4).permute(2, 0, 1)  # (4,H,W)
                for c in range(4):
                    box_xyxy_out[bi, c] = torch.where(update, best_boxes[c], box_xyxy_out[bi, c])

                # obj_targets (centerness) — computed from pixel LTRB (H,W,4)
                l_v, t_v, r_v, b_v = best_ltrb.unbind(-1)   # each (H, W)
                lr_min = torch.minimum(l_v, r_v)
                lr_max = torch.maximum(l_v, r_v).clamp(min=1e-6)
                tb_min = torch.minimum(t_v, b_v)
                tb_max = torch.maximum(t_v, b_v).clamp(min=1e-6)
                centerness = ((lr_min / lr_max) * (tb_min / tb_max)).clamp(min=0).sqrt()
                obj_targets[bi, 0] = torch.where(update, centerness, obj_targets[bi, 0])


            assignments[level_name] = LevelAssignment(
                cls_targets=cls_targets,
                obj_targets=obj_targets,
                box_targets=box_targets,
                box_xyxy=box_xyxy_out,
                pos_mask=pos_mask,
                track_ids=track_ids,
                points=points,
                stride=stride,
            )

        return assignments
