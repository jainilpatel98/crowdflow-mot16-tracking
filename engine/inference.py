from __future__ import annotations

from typing import Any

import torch

from utils.box_ops import batched_nms_xyxy, distance2bbox, make_grid


def decode_student_outputs(
    outputs: dict[str, dict[str, torch.Tensor]],
    strides: dict[str, int],
    score_threshold: float = 0.25,
    nms_iou_threshold: float = 0.5,
    max_detections: int = 300,
) -> list[list[dict[str, Any]]]:
    batch_size = next(iter(outputs["cls"].values())).shape[0]
    device = next(iter(outputs["cls"].values())).device
    results: list[list[dict[str, Any]]] = []

    for batch_index in range(batch_size):
        all_boxes = []
        all_scores = []
        all_embeddings = []
        for level_name, cls_logits in outputs["cls"].items():
            cls_map = cls_logits[batch_index, 0]
            obj_map = outputs["obj"][level_name][batch_index, 0]
            box_ltrb = outputs["box"][level_name][batch_index]
            emb_map = outputs["emb"][level_name][batch_index]

            height, width = cls_map.shape
            score_map = cls_map.sigmoid() * obj_map.sigmoid()
            score_flat = score_map.flatten()
            keep = score_flat > score_threshold
            if keep.sum() == 0:
                continue

            points = make_grid(height, width, strides[level_name], device=device).view(-1, 2)
            boxes = distance2bbox(points, box_ltrb.permute(1, 2, 0).reshape(-1, 4))
            embeddings = emb_map.permute(1, 2, 0).reshape(-1, emb_map.shape[0])

            all_boxes.append(boxes[keep])
            all_scores.append(score_flat[keep])
            all_embeddings.append(embeddings[keep])

        if not all_boxes:
            results.append([])
            continue

        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)
        embeddings = torch.cat(all_embeddings, dim=0)
        keep = batched_nms_xyxy(boxes, scores, iou_threshold=nms_iou_threshold, max_detections=max_detections)

        detections = []
        for index in keep.tolist():
            detections.append(
                {
                    "bbox_xyxy": boxes[index].detach().cpu(),
                    "score": float(scores[index].item()),
                    "embedding": embeddings[index].detach().cpu(),
                }
            )
        results.append(detections)

    return results
