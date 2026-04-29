from __future__ import annotations

from typing import Any

import torch
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal envs
    def tqdm(iterable, **kwargs):
        return iterable

from engine.inference import decode_student_candidates, decode_student_outputs
from utils.box_ops import batched_nms_xyxy, box_iou
from utils.metrics import DetectionMetricSummary


def _decode_detection_outputs(
    outputs,
    *,
    strides: dict[str, int],
    score_threshold: float,
    nms_iou_threshold: float,
    model_type: str = "student",
):
    if model_type == "teacher":
        detections = []
        for boxes, scores in zip(outputs["boxes"], outputs["scores"]):
            sample_dets = [
                {
                    "bbox_xyxy": box.detach().cpu().float(),
                    "score": float(score),
                }
                for box, score in zip(boxes, scores)
                if float(score) >= score_threshold
            ]
            detections.append(sample_dets)
        return detections
    return decode_student_outputs(
        outputs,
        strides=strides,
        score_threshold=score_threshold,
        nms_iou_threshold=nms_iou_threshold,
    )


def _match_detections(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int, list[float]]:
    if pred_boxes.numel() == 0 and gt_boxes.numel() == 0:
        return 0, 0, 0, []
    if pred_boxes.numel() == 0:
        return 0, 0, gt_boxes.shape[0], []
    if gt_boxes.numel() == 0:
        return 0, pred_boxes.shape[0], 0, []

    ious = box_iou(pred_boxes, gt_boxes)
    matched_gt = set()
    matched_pred = set()
    matched_ious = []

    flat_indices = torch.argsort(ious.reshape(-1), descending=True)
    num_gt = gt_boxes.shape[0]
    for flat_index in flat_indices.tolist():
        pred_idx = flat_index // num_gt
        gt_idx = flat_index % num_gt
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        iou_value = float(ious[pred_idx, gt_idx].item())
        if iou_value < iou_threshold:
            break
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
        matched_ious.append(iou_value)

    tp = len(matched_pred)
    fp = pred_boxes.shape[0] - tp
    fn = gt_boxes.shape[0] - tp
    return tp, fp, fn, matched_ious


def _compute_average_precision(
    predictions: list[dict[str, Any]],
    gt_by_image: dict[int, torch.Tensor],
    num_targets: int,
    iou_threshold: float,
) -> float:
    if num_targets == 0:
        return 0.0
    if not predictions:
        return 0.0

    predictions = sorted(predictions, key=lambda item: item["score"], reverse=True)
    matched_gt = {
        image_id: torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
        for image_id, gt_boxes in gt_by_image.items()
    }

    true_positives = torch.zeros(len(predictions), dtype=torch.float32)
    false_positives = torch.zeros(len(predictions), dtype=torch.float32)

    for pred_index, prediction in enumerate(predictions):
        image_id = prediction["image_id"]
        pred_box = prediction["box"].unsqueeze(0)
        gt_boxes = gt_by_image.get(image_id)
        if gt_boxes is None or gt_boxes.numel() == 0:
            false_positives[pred_index] = 1.0
            continue

        ious = box_iou(pred_box, gt_boxes).squeeze(0)
        best_iou, best_idx = torch.max(ious, dim=0)
        if best_iou.item() >= iou_threshold and not matched_gt[image_id][best_idx]:
            true_positives[pred_index] = 1.0
            matched_gt[image_id][best_idx] = True
        else:
            false_positives[pred_index] = 1.0

    cum_tp = torch.cumsum(true_positives, dim=0)
    cum_fp = torch.cumsum(false_positives, dim=0)
    recalls = cum_tp / max(1, num_targets)
    precisions = cum_tp / torch.clamp(cum_tp + cum_fp, min=1e-9)

    recalls = torch.cat((torch.tensor([0.0]), recalls, torch.tensor([1.0])))
    precisions = torch.cat((torch.tensor([0.0]), precisions, torch.tensor([0.0])))
    for index in range(precisions.shape[0] - 1, 0, -1):
        precisions[index - 1] = torch.maximum(precisions[index - 1], precisions[index])
    delta = recalls[1:] - recalls[:-1]
    return float(torch.sum(delta * precisions[1:]).item())


@torch.no_grad()
def collect_raw_predictions(
    model,
    data_loader,
    device: torch.device,
    strides: dict[str, int],
    nms_iou_threshold: float = 0.5,
    model_type: str = "student",
) -> tuple[dict[int, dict[str, Any]], dict[int, torch.Tensor], int]:
    """Run inference once and return raw per-image candidates for threshold sweep.

    Returns:
        raw_predictions: {image_id: {"boxes", "scores", "needs_nms", ...}}.
            For student models, this is the pre-threshold, pre-NMS candidate set.
            For teacher models, boxes/scores are already final detections.
        gt_by_image: {image_id: gt_boxes_xyxy} for mAP computation.
        gt_count: total number of ground-truth boxes.

    This is the backbone of the threshold sweep: the model runs exactly once,
    and each threshold is evaluated analytically with the same threshold->NMS
    order used during normal student inference.
    """
    model.eval()
    raw_predictions: dict[int, dict[str, Any]] = {}
    gt_by_image: dict[int, torch.Tensor] = {}
    gt_count = 0
    image_id = 0

    for batch in tqdm(data_loader, desc="Collecting predictions", leave=False):
        images = batch["images"].to(device, non_blocking=True)
        outputs = model(images)
        if model_type == "student":
            batch_candidates = decode_student_candidates(outputs, strides)
        else:
            batch_candidates = []
            for boxes, scores in zip(outputs["boxes"], outputs["scores"]):
                batch_candidates.append(
                    {
                        "boxes": boxes.detach().cpu().float(),
                        "scores": scores.detach().cpu().float(),
                        "needs_nms": False,
                    }
                )

        for candidate, target in zip(batch_candidates, batch["targets"]):
            gt_boxes = target["boxes"].cpu()
            gt_by_image[image_id] = gt_boxes
            gt_count += gt_boxes.shape[0]
            raw_predictions[image_id] = {
                "boxes": candidate["boxes"].detach().cpu().float(),
                "scores": candidate["scores"].detach().cpu().float(),
                "needs_nms": bool(candidate.get("needs_nms", model_type == "student")),
                "nms_iou_threshold": float(nms_iou_threshold),
            }
            image_id += 1

    return raw_predictions, gt_by_image, gt_count


def compute_metrics_at_threshold(
    raw_predictions: dict[int, dict[str, Any]],
    gt_by_image: dict[int, torch.Tensor],
    gt_count: int,
    score_threshold: float,
) -> "DetectionMetricSummary":
    """Compute detection metrics from pre-collected raw predictions at a given threshold.

    Filters ``raw_predictions`` by ``score_threshold`` and recomputes all
    metrics analytically — no model inference required.
    """
    tp_total = fp_total = fn_total = 0
    matched_ious_list: list[float] = []
    pred_count = 0
    filtered: list[dict[str, Any]] = []

    all_image_ids = set(gt_by_image.keys()) | set(raw_predictions.keys())
    for img_id in all_image_ids:
        raw = raw_predictions.get(img_id)
        if raw is None:
            img_preds = []
        else:
            boxes = raw["boxes"]
            scores = raw["scores"]
            keep = scores >= score_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            if raw.get("needs_nms", False) and boxes.numel() > 0:
                nms_keep = batched_nms_xyxy(
                    boxes,
                    scores,
                    iou_threshold=float(raw["nms_iou_threshold"]),
                    max_detections=300,
                )
                boxes = boxes[nms_keep]
                scores = scores[nms_keep]
            img_preds = []
            for box, score in zip(boxes, scores):
                prediction = {
                    "image_id": img_id,
                    "score": float(score.item()),
                    "box": box,
                }
                img_preds.append(prediction)
                filtered.append(prediction)
            pred_count += len(img_preds)

        gt_boxes = gt_by_image.get(img_id, torch.zeros((0, 4)))
        pred_boxes = (
            torch.stack([p["box"] for p in img_preds], dim=0)
            if img_preds else torch.zeros((0, 4))
        )
        tp, fp, fn, ious = _match_detections(pred_boxes, gt_boxes)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        matched_ious_list.extend(ious)

    precision = tp_total / max(1, tp_total + fp_total)
    recall = tp_total / max(1, tp_total + fn_total)
    mean_iou = sum(matched_ious_list) / max(1, len(matched_ious_list))

    ap_thresholds = [0.5 + 0.05 * i for i in range(10)]
    ap_values = [
        _compute_average_precision(filtered, gt_by_image, gt_count, t)
        for t in ap_thresholds
    ]
    return DetectionMetricSummary(
        precision=precision,
        recall=recall,
        mean_iou=mean_iou,
        map50=ap_values[0],
        map50_95=sum(ap_values) / len(ap_values),
        num_predictions=pred_count,
        num_targets=gt_count,
    )



@torch.no_grad()
def evaluate_detection(
    model,
    data_loader,
    device: torch.device,
    strides: dict[str, int],
    score_threshold: float = 0.25,
    nms_iou_threshold: float = 0.5,
    model_type: str = "student",
) -> DetectionMetricSummary:
    model.eval()
    tp_total = 0
    fp_total = 0
    fn_total = 0
    matched_ious: list[float] = []
    gt_by_image: dict[int, torch.Tensor] = {}
    predictions: list[dict[str, Any]] = []
    pred_count = 0
    gt_count = 0
    image_id = 0

    for batch in tqdm(data_loader, desc="Eval", leave=False):
        images = batch["images"].to(device, non_blocking=True)
        outputs = model(images)
        detections = _decode_detection_outputs(
            outputs,
            strides=strides,
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou_threshold,
            model_type=model_type,
        )
        for dets, target in zip(detections, batch["targets"]):
            pred_boxes = (
                torch.stack([det["bbox_xyxy"] for det in dets], dim=0)
                if dets
                else torch.zeros((0, 4), dtype=torch.float32)
            )
            gt_boxes = target["boxes"].cpu()
            gt_by_image[image_id] = gt_boxes
            tp, fp, fn, ious = _match_detections(pred_boxes, gt_boxes)
            tp_total += tp
            fp_total += fp
            fn_total += fn
            matched_ious.extend(ious)
            pred_count += pred_boxes.shape[0]
            gt_count += gt_boxes.shape[0]

            for det in dets:
                predictions.append(
                    {
                        "image_id": image_id,
                        "score": float(det["score"]),
                        "box": det["bbox_xyxy"].cpu().float(),
                    }
                )
            image_id += 1

    precision = tp_total / max(1, tp_total + fp_total)
    recall = tp_total / max(1, tp_total + fn_total)
    mean_iou = sum(matched_ious) / max(1, len(matched_ious))
    ap_thresholds = [0.5 + 0.05 * idx for idx in range(10)]
    ap_values = [
        _compute_average_precision(predictions, gt_by_image, gt_count, threshold)
        for threshold in ap_thresholds
    ]
    return DetectionMetricSummary(
        precision=precision,
        recall=recall,
        mean_iou=mean_iou,
        map50=ap_values[0],
        map50_95=sum(ap_values) / len(ap_values),
        num_predictions=pred_count,
        num_targets=gt_count,
    )
