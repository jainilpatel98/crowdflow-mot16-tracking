from __future__ import annotations

from typing import Any

import torch
from torchvision.ops import nms as torchvision_nms

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

from engine.inference import decode_student_candidates, decode_student_outputs
from utils.box_ops import batched_nms_xyxy, box_iou
from utils.metrics import DetectionMetricSummary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
        gt_idx   = flat_index % num_gt
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


def _match_with_iou_matrix(
    iou_matrix: torch.Tensor,   # (P, G)
    iou_threshold: float = 0.5,
) -> tuple[int, int, int, list[float]]:
    """Fast P/R/F1 matching using a precomputed IoU matrix (no box_iou call)."""
    P, G = iou_matrix.shape
    if P == 0 and G == 0:
        return 0, 0, 0, []
    if P == 0:
        return 0, 0, G, []
    if G == 0:
        return 0, P, 0, []

    matched_gt   = set()
    matched_pred = set()
    matched_ious = []

    flat = iou_matrix.reshape(-1)
    flat_indices = torch.argsort(flat, descending=True)
    for idx in flat_indices.tolist():
        pi = idx // G
        gi = idx % G
        if pi in matched_pred or gi in matched_gt:
            continue
        v = float(flat[idx].item())
        if v < iou_threshold:
            break
        matched_pred.add(pi)
        matched_gt.add(gi)
        matched_ious.append(v)

    tp = len(matched_pred)
    return tp, P - tp, G - tp, matched_ious


def _compute_average_precision(
    predictions: list[dict[str, Any]],
    gt_by_image: dict[int, torch.Tensor],
    num_targets: int,
    iou_threshold: float,
) -> float:
    """Standard AP: sort all predictions by score, build cumulative P-R curve."""
    if num_targets == 0 or not predictions:
        return 0.0

    predictions_sorted = sorted(predictions, key=lambda x: x["score"], reverse=True)
    matched_gt: dict[int, torch.Tensor] = {
        image_id: torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
        for image_id, gt_boxes in gt_by_image.items()
    }

    n = len(predictions_sorted)
    tp_arr = torch.zeros(n, dtype=torch.float32)
    fp_arr = torch.zeros(n, dtype=torch.float32)

    for i, pred in enumerate(predictions_sorted):
        image_id = pred["image_id"]
        pred_box = pred["box"].unsqueeze(0)
        gt_boxes = gt_by_image.get(image_id)
        if gt_boxes is None or gt_boxes.numel() == 0:
            fp_arr[i] = 1.0
            continue
        ious = box_iou(pred_box, gt_boxes).squeeze(0)
        best_iou, best_idx = ious.max(dim=0)
        if best_iou.item() >= iou_threshold and not matched_gt[image_id][best_idx]:
            tp_arr[i] = 1.0
            matched_gt[image_id][best_idx] = True
        else:
            fp_arr[i] = 1.0

    cum_tp = torch.cumsum(tp_arr, dim=0)
    cum_fp = torch.cumsum(fp_arr, dim=0)
    recalls    = cum_tp / max(1, num_targets)
    precisions = cum_tp / (cum_tp + cum_fp).clamp(min=1e-9)
    recalls    = torch.cat([torch.tensor([0.0]), recalls,    torch.tensor([1.0])])
    precisions = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])
    for idx in range(precisions.shape[0] - 1, 0, -1):
        precisions[idx - 1] = torch.maximum(precisions[idx - 1], precisions[idx])
    delta = recalls[1:] - recalls[:-1]
    return float((delta * precisions[1:]).sum().item())


# ---------------------------------------------------------------------------
# Fast AP computation using cached IoU matrices (for sweep mode)
# ---------------------------------------------------------------------------

def _compute_ap_all_thresholds_cached(
    nms_data: dict[int, dict[str, Any]],
    gt_count: int,
) -> list[float]:
    """Compute AP at 10 IoU thresholds using precomputed IoU matrices.

    This runs O(10 × N) where N = total post-NMS predictions, with no box_iou
    calls inside the loop.  IoU values are looked up from precomputed matrices
    cached in RAM.  Intended to be called ONCE before the threshold sweep.

    nms_data[img_id]:
        "scores":     (P,) float32 CPU tensor — post-NMS scores
        "iou_matrix": (P, G) float32 CPU tensor — IoU vs each GT box
    """
    # Build globally sorted (score, img_id, pred_idx) list — once
    entries: list[tuple[float, int, int]] = []
    for img_id, data in nms_data.items():
        scores = data["scores"]
        for i in range(scores.shape[0]):
            entries.append((float(scores[i].item()), img_id, i))
    entries.sort(key=lambda x: -x[0])   # descending score

    n = len(entries)
    iou_thresholds = [0.5 + 0.05 * k for k in range(10)]
    ap_values: list[float] = []

    for iou_thresh in iou_thresholds:
        # Per-image matched-GT boolean arrays
        matched: dict[int, list[bool]] = {
            img_id: [False] * (data["iou_matrix"].shape[1] if data["iou_matrix"].numel() > 0 else 0)
            for img_id, data in nms_data.items()
        }

        tp_arr = torch.zeros(n, dtype=torch.float32)
        fp_arr = torch.zeros(n, dtype=torch.float32)

        for i, (score, img_id, pred_idx) in enumerate(entries):
            iou_mat = nms_data[img_id]["iou_matrix"]   # (P, G)
            if iou_mat.numel() == 0 or iou_mat.shape[1] == 0:
                fp_arr[i] = 1.0
                continue
            ious = iou_mat[pred_idx]           # (G,)  — O(1) index, no new computation
            best_iou, best_gt = ious.max(dim=0)
            gi = best_gt.item()
            if best_iou.item() >= iou_thresh and not matched[img_id][gi]:
                tp_arr[i] = 1.0
                matched[img_id][gi] = True
            else:
                fp_arr[i] = 1.0

        cum_tp = tp_arr.cumsum(0)
        cum_fp = fp_arr.cumsum(0)
        rec  = cum_tp / max(1, gt_count)
        prec = cum_tp / (cum_tp + cum_fp).clamp(min=1e-9)
        rec  = torch.cat([torch.tensor([0.0]), rec,  torch.tensor([1.0])])
        prec = torch.cat([torch.tensor([0.0]), prec, torch.tensor([0.0])])
        for j in range(prec.shape[0] - 1, 0, -1):
            prec[j - 1] = torch.maximum(prec[j - 1], prec[j])
        ap_values.append(float(((rec[1:] - rec[:-1]) * prec[1:]).sum().item()))

    return ap_values


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_raw_predictions(
    model,
    data_loader,
    device: torch.device,
    strides: dict[str, int],
    nms_iou_threshold: float = 0.5,
    model_type: str = "student",
) -> tuple[dict[int, dict[str, Any]], dict[int, torch.Tensor], int]:
    """Run inference once and return NMS-filtered per-image data for sweep.

    For each image we store:
        "boxes":      (P, 4)  post-NMS boxes (CPU)
        "scores":     (P,)    post-NMS scores (CPU)
        "iou_matrix": (P, G)  IoU vs GT boxes (CPU) — precomputed on ``device``
        "gt_count":   int     number of GT boxes

    NMS and IoU matrix computation are done on ``device`` (GPU if available)
    and transferred to CPU for storage.  This is done ONCE during collection,
    so the sweep loop only does cheap score-threshold filtering.
    """
    model.eval()
    raw_predictions: dict[int, dict[str, Any]] = {}
    gt_by_image:     dict[int, torch.Tensor]    = {}
    gt_count = 0
    image_id = 0
    max_det  = 300

    for batch in tqdm(data_loader, desc="Collecting predictions", leave=False):
        images = batch["images"].to(device, non_blocking=True)
        outputs = model(images)

        if model_type == "student":
            # decode_student_candidates returns CPU tensors; move to device for NMS
            batch_candidates = decode_student_candidates(outputs, strides)
        else:
            batch_candidates = []
            for boxes, scores in zip(outputs["boxes"], outputs["scores"]):
                batch_candidates.append({
                    "boxes":     boxes.detach().cpu().float(),
                    "scores":    scores.detach().cpu().float(),
                    "needs_nms": False,
                })

        for candidate, target in zip(batch_candidates, batch["targets"]):
            gt_boxes_cpu = target["boxes"].cpu().float()
            gt_by_image[image_id] = gt_boxes_cpu
            gt_count += gt_boxes_cpu.shape[0]

            boxes_cpu  = candidate["boxes"].float()
            scores_cpu = candidate["scores"].float()

            if boxes_cpu.numel() > 0 and candidate.get("needs_nms", model_type == "student"):
                # NMS on device (GPU if available) — boxes come from decode which gives CPU;
                # move to device just for NMS, then back to CPU for storage
                b = boxes_cpu.to(device)
                s = scores_cpu.to(device)
                keep = torchvision_nms(b, s, iou_threshold=nms_iou_threshold)
                if keep.shape[0] > max_det:
                    keep = keep[:max_det]
                boxes_nms  = b[keep].cpu()
                scores_nms = s[keep].cpu()
            else:
                boxes_nms  = boxes_cpu
                scores_nms = scores_cpu
                if boxes_nms.shape[0] > max_det:
                    # Teacher detections: already sorted; just truncate
                    boxes_nms  = boxes_nms[:max_det]
                    scores_nms = scores_nms[:max_det]

            # Precompute IoU matrix on device — done ONCE per image
            if boxes_nms.numel() > 0 and gt_boxes_cpu.numel() > 0:
                b_dev  = boxes_nms.to(device)
                gt_dev = gt_boxes_cpu.to(device)
                iou_matrix = box_iou(b_dev, gt_dev).cpu()   # (P, G) — cheap on GPU
            else:
                iou_matrix = torch.zeros(boxes_nms.shape[0], gt_boxes_cpu.shape[0])

            raw_predictions[image_id] = {
                "boxes":      boxes_nms,
                "scores":     scores_nms,
                "iou_matrix": iou_matrix,
            }
            image_id += 1

    return raw_predictions, gt_by_image, gt_count


# ---------------------------------------------------------------------------
# Per-threshold metric computation  (sweep loop calls this for each threshold)
# ---------------------------------------------------------------------------

def compute_metrics_at_threshold(
    raw_predictions: dict[int, dict[str, Any]],
    gt_by_image: dict[int, torch.Tensor],
    gt_count: int,
    score_threshold: float,
    precomputed_ap_values: list[float] | None = None,
) -> "DetectionMetricSummary":
    """Compute P/R/F1/MeanIoU at ``score_threshold`` and mAP.

    P/R/F1 use the precomputed ``iou_matrix`` stored during collection — no
    new box_iou calls.  mAP uses ``precomputed_ap_values`` (computed once
    before the sweep loop).  If not provided, falls back to the slower path.
    """
    tp_total = fp_total = fn_total = 0
    matched_ious_list: list[float] = []
    pred_count = 0

    for img_id in sorted(gt_by_image.keys()):
        raw      = raw_predictions.get(img_id)
        gt_boxes = gt_by_image[img_id]
        G        = gt_boxes.shape[0]

        if raw is None or raw["boxes"].numel() == 0:
            fn_total  += G
            continue

        scores = raw["scores"]
        keep   = scores >= score_threshold

        if not keep.any():
            fn_total += G
            continue

        iou_sub = raw["iou_matrix"][keep]   # (P_kept, G) — free lookup

        pred_count += int(keep.sum().item())

        # Fast matching using cached IoU sub-matrix
        if G == 0:
            fp_total += int(keep.sum().item())
        else:
            tp, fp, fn, ious = _match_with_iou_matrix(iou_sub, iou_threshold=0.5)
            tp_total += tp
            fp_total += fp
            fn_total += fn
            matched_ious_list.extend(ious)

    precision = tp_total / max(1, tp_total + fp_total)
    recall    = tp_total / max(1, tp_total + fn_total)
    mean_iou  = sum(matched_ious_list) / max(1, len(matched_ious_list))

    if precomputed_ap_values is not None:
        ap_values = precomputed_ap_values
    else:
        # Slow fallback (not used in sweep; used when called standalone)
        all_preds: list[dict[str, Any]] = []
        for img_id, raw in raw_predictions.items():
            for box, score in zip(raw["boxes"], raw["scores"]):
                all_preds.append({"image_id": img_id, "score": float(score.item()), "box": box})
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]
        ap_values = [
            _compute_average_precision(all_preds, gt_by_image, gt_count, t)
            for t in iou_thresholds
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


# ---------------------------------------------------------------------------
# Inline training evaluation  (unchanged API, unchanged behaviour)
# ---------------------------------------------------------------------------

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
    """Used during training for checkpoint selection.  No sweep, no cached IoU."""
    model.eval()
    tp_total = 0
    fp_total = 0
    fn_total = 0
    matched_ious: list[float] = []
    gt_by_image: dict[int, torch.Tensor] = {}
    predictions: list[dict[str, Any]] = []
    pred_count = 0
    gt_count   = 0
    image_id   = 0

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
            gt_count   += gt_boxes.shape[0]

            for det in dets:
                predictions.append({
                    "image_id": image_id,
                    "score":    float(det["score"]),
                    "box":      det["bbox_xyxy"].cpu().float(),
                })
            image_id += 1

    precision = tp_total / max(1, tp_total + fp_total)
    recall    = tp_total / max(1, tp_total + fn_total)
    mean_iou  = sum(matched_ious)   / max(1, len(matched_ious))
    ap_thresholds = [0.5 + 0.05 * idx for idx in range(10)]
    ap_values = [
        _compute_average_precision(predictions, gt_by_image, gt_count, t)
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
