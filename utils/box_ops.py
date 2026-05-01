from __future__ import annotations

import torch
from torchvision.ops import nms


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(-1)
    return torch.stack((x, y, x + w, y + h), dim=-1)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack((x1, y1, x2 - x1, y2 - y1), dim=-1)


def clip_boxes_xyxy(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    boxes = boxes.clone()
    boxes[..., 0::2] = boxes[..., 0::2].clamp(0, width)
    boxes[..., 1::2] = boxes[..., 1::2].clamp(0, height)
    return boxes


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def aligned_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    rb = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    union = area1 + area2 - inter
    return inter / union.clamp(min=1e-6)


def aligned_ciou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Element-wise Complete IoU (CIoU) between aligned box pairs.

    CIoU = 1 - IoU + d²/c² + α·v
    where:
      d²/c² = centre-distance² / enclosing-diagonal²
      v      = (4/π²) * (arctan(tw/th) - arctan(pw/ph))²
      α      = v / (1 - IoU + v + eps)

    Args:
        pred:   (N, 4) predicted boxes in xyxy format.
        target: (N, 4) target boxes in xyxy format.

    Returns:
        (N,) CIoU loss values in [0, 2].
    """
    eps = 1e-7

    # IoU component
    lt   = torch.maximum(pred[..., :2], target[..., :2])
    rb   = torch.minimum(pred[..., 2:], target[..., 2:])
    wh   = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_p = box_area(pred)
    area_t = box_area(target)
    union  = area_p + area_t - inter
    iou    = inter / union.clamp(min=eps)

    # Centre-distance penalty: d² / c²
    pred_cx   = (pred[..., 0]   + pred[..., 2])   * 0.5
    pred_cy   = (pred[..., 1]   + pred[..., 3])   * 0.5
    target_cx = (target[..., 0] + target[..., 2]) * 0.5
    target_cy = (target[..., 1] + target[..., 3]) * 0.5
    d2 = (pred_cx - target_cx).pow(2) + (pred_cy - target_cy).pow(2)

    # Smallest enclosing box diagonal²
    enc_lt = torch.minimum(pred[..., :2], target[..., :2])
    enc_rb = torch.maximum(pred[..., 2:], target[..., 2:])
    enc_wh = (enc_rb - enc_lt).clamp(min=0)
    c2 = enc_wh[..., 0].pow(2) + enc_wh[..., 1].pow(2) + eps

    # Aspect-ratio penalty
    pw = (pred[..., 2]   - pred[..., 0]).clamp(min=eps)
    ph = (pred[..., 3]   - pred[..., 1]).clamp(min=eps)
    tw = (target[..., 2] - target[..., 0]).clamp(min=eps)
    th = (target[..., 3] - target[..., 1]).clamp(min=eps)
    v  = (4.0 / (torch.pi ** 2)) * (torch.atan(tw / th) - torch.atan(pw / ph)).pow(2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)

    return 1.0 - iou + d2 / c2 + alpha * v


def distance2bbox(points: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
    x1 = points[..., 0] - distances[..., 0]
    y1 = points[..., 1] - distances[..., 1]
    x2 = points[..., 0] + distances[..., 2]
    y2 = points[..., 1] + distances[..., 3]
    return torch.stack((x1, y1, x2, y2), dim=-1)


def bbox2distance(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    left = points[..., 0] - boxes[..., 0]
    top = points[..., 1] - boxes[..., 1]
    right = boxes[..., 2] - points[..., 0]
    bottom = boxes[..., 3] - points[..., 1]
    return torch.stack((left, top, right, bottom), dim=-1)


def make_grid(height: int, width: int, stride: int, device: torch.device) -> torch.Tensor:
    y_coords = torch.arange(height, device=device, dtype=torch.float32)
    x_coords = torch.arange(width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    return torch.stack(((xx + 0.5) * stride, (yy + 0.5) * stride), dim=-1)


def batched_nms_xyxy(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
    max_detections: int,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), device=boxes.device, dtype=torch.long)
    keep = nms(boxes, scores, iou_threshold)
    return keep[:max_detections]
