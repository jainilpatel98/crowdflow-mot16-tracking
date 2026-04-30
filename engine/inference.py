from __future__ import annotations

from typing import Any

import torch

from utils.box_ops import batched_nms_xyxy, distance2bbox, make_grid


def decode_student_candidates(
    outputs: dict[str, dict[str, torch.Tensor]],
    strides: dict[str, int],
) -> list[dict[str, torch.Tensor]]:
    """Decode raw head outputs into per-image candidate detections.

    Fix 3: box_pred outputs are stride-normalised LTRB (Fix 3 in assigners.py
    and det_loss.py).  Multiply by stride before distance2bbox to recover
    pixel-space LTRB.

    Fix 4: Score is sigmoid(cls) ONLY — centerness is NOT multiplied in.
    Previously score = sigmoid(cls) * sigmoid(obj/centerness) suppressed valid
    off-centre detections below the 0.05 threshold because centerness → 0 at
    box edges.  Removing it lifts the structural recall ceiling.
    """
    batch_size = next(iter(outputs["cls"].values())).shape[0]
    device     = next(iter(outputs["cls"].values())).device
    results: list[dict[str, torch.Tensor]] = []

    for batch_index in range(batch_size):
        all_boxes      = []
        all_scores     = []
        all_embeddings = []
        for level_name, cls_logits in outputs["cls"].items():
            stride   = strides[level_name]
            cls_map  = cls_logits[batch_index, 0]                   # (H, W)
            box_ltrb = outputs["box"][level_name][batch_index]       # (4, H, W) stride-norm
            emb_map  = outputs["emb"][level_name][batch_index]       # (emb_dim, H, W)

            height, width = cls_map.shape

            # Fix 4: score = sigmoid(cls) only — no centerness gate
            score_map  = cls_map.sigmoid()                           # (H, W)
            score_flat = score_map.flatten()                         # (H*W,)

            points = make_grid(height, width, stride, device=device).view(-1, 2)

            # Fix 3: decode stride-normalised LTRB → pixel LTRB → xyxy
            box_ltrb_px = box_ltrb * stride                          # pixel LTRB
            boxes = distance2bbox(points, box_ltrb_px.permute(1, 2, 0).reshape(-1, 4))

            embeddings = emb_map.permute(1, 2, 0).reshape(-1, emb_map.shape[0])

            all_boxes.append(boxes)
            all_scores.append(score_flat)
            all_embeddings.append(embeddings)

        if not all_boxes:
            results.append(
                {
                    "boxes":      torch.zeros((0, 4), device=device),
                    "scores":     torch.zeros((0,),   device=device),
                    "embeddings": torch.zeros((0, outputs["emb"]["p3"].shape[1]), device=device),
                }
            )
            continue

        results.append(
            {
                "boxes":      torch.cat(all_boxes,      dim=0),
                "scores":     torch.cat(all_scores,     dim=0),
                "embeddings": torch.cat(all_embeddings, dim=0),
            }
        )

    return results


def decode_student_outputs(
    outputs: dict[str, dict[str, torch.Tensor]],
    strides: dict[str, int],
    score_threshold: float = 0.25,
    nms_iou_threshold: float = 0.5,
    max_detections: int = 300,
) -> list[list[dict[str, Any]]]:
    results: list[list[dict[str, Any]]] = []

    for candidate in decode_student_candidates(outputs, strides):
        keep = candidate["scores"] > score_threshold
        if keep.sum() == 0:
            results.append([])
            continue

        boxes      = candidate["boxes"][keep]
        scores     = candidate["scores"][keep]
        embeddings = candidate["embeddings"][keep]
        keep = batched_nms_xyxy(
            boxes, scores,
            iou_threshold=nms_iou_threshold,
            max_detections=max_detections,
        )

        detections = []
        for index in keep.tolist():
            detections.append(
                {
                    "bbox_xyxy": boxes[index].detach().cpu(),
                    "score":     float(scores[index].item()),
                    "embedding": embeddings[index].detach().cpu(),
                }
            )
        results.append(detections)

    return results
