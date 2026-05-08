#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

if not hasattr(np, "asfarray"):
    np.asfarray = lambda values, dtype=float: np.asarray(values, dtype=dtype)

import motmetrics as mm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxmot.trackers.strongsort.strongsort import StrongSort
from datasets.transforms import letterbox_image, unletterbox_boxes_xyxy
from engine.inference import decode_student_outputs
from models.student_jde import StudentJDE
from utils.checkpoint import validate_checkpoint_shapes
from utils.config import load_yaml


VAL_SEQUENCES = ["MOT16-05", "MOT16-10"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate StrongSORT using BoxMOT OSNet crop embeddings.")
    parser.add_argument("--student-config", default="configs/student_distill.yaml")
    parser.add_argument("--tracker-config", default="configs/tracker.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--data-root", default="MOT16")
    parser.add_argument("--sequences", default=",".join(VAL_SEQUENCES))
    parser.add_argument("--reid-weights", required=True)
    parser.add_argument("--output-dir", default="runs/reid_osnet_x0_25_mot16/strongsort_eval")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def load_gt_by_frame(gt_path: Path) -> dict[int, list[dict[str, Any]]]:
    frame_map: dict[int, list[dict[str, Any]]] = {}
    with gt_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 9:
                continue
            frame = int(float(row[0]))
            track_id = int(float(row[1]))
            mark = int(float(row[6]))
            cls = int(float(row[7]))
            if mark != 1 or cls not in {1, 2} or track_id <= 0:
                continue
            x, y, w, h = map(float, row[2:6])
            frame_map.setdefault(frame, []).append({"id": track_id, "xywh": [x, y, w, h]})
    return frame_map


def evaluate_sequence(gt_by_frame: dict[int, list[dict[str, Any]]], pred_by_frame: dict[int, list[dict[str, Any]]]):
    acc = mm.MOTAccumulator(auto_id=False)
    for frame_id in sorted(set(gt_by_frame) | set(pred_by_frame)):
        gt_items = gt_by_frame.get(frame_id, [])
        pred_items = pred_by_frame.get(frame_id, [])
        gt_ids = [item["id"] for item in gt_items]
        pred_ids = [item["id"] for item in pred_items]
        gt_boxes = [item["xywh"] for item in gt_items]
        pred_boxes = [item["xywh"] for item in pred_items]
        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, distances, frameid=frame_id)
    return acc


def build_student(config: dict, checkpoint_path: Path, device: torch.device) -> StudentJDE:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    id_classes = checkpoint["student"]["id_classifier.weight"].shape[0] if "id_classifier.weight" in checkpoint["student"] else 0
    model = StudentJDE(
        backbone_name=config["student"]["backbone_name"],
        emb_dim=config["student"]["emb_dim"],
        num_classes=1,
        fpn_channels=config["student"]["fpn_channels"],
        pretrained_backbone=False,
        num_id_classes=id_classes,
        tower_layers=int(config["student"].get("tower_layers", 2)),
        tower_dropout=float(config["student"].get("tower_dropout", 0.0)),
    ).to(device)
    validate_checkpoint_shapes(checkpoint, model, feature_adapters=None)
    model.load_state_dict(checkpoint["student"])
    model.eval()
    return model


def run_sequence(
    *,
    model: StudentJDE,
    sequence_dir: Path,
    output_path: Path,
    config: dict,
    tracker_cfg: dict,
    reid_weights: Path,
    device: torch.device,
    max_frames: int,
) -> dict[int, list[dict[str, Any]]]:
    tracker = StrongSort(
        reid_weights=reid_weights,
        device=device,
        half=device.type != "cpu",
        min_conf=tracker_cfg.get("min_conf", 0.1),
        max_iou_dist=tracker_cfg.get("iou_threshold", 0.5),
        max_cos_dist=tracker_cfg.get("cosine_threshold", 0.25),
        max_age=tracker_cfg.get("max_age", 30),
        n_init=tracker_cfg.get("min_hits", 3),
        nn_budget=tracker_cfg.get("nn_budget", 100),
        mc_lambda=tracker_cfg.get("mc_lambda", 0.98),
        ema_alpha=tracker_cfg.get("ema_alpha", 0.9),
    )
    image_paths = sorted((sequence_dir / "img1").glob("*.jpg"))
    if max_frames > 0:
        image_paths = image_paths[:max_frames]
    input_h, input_w = tuple(load_yaml(config["dataset"]["config"])["dataset"]["input_size"])
    pred_by_frame: dict[int, list[dict[str, Any]]] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for image_path in image_paths:
            image_bgr = cv2.imread(str(image_path))
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            letterboxed, resize_scale, pad = letterbox_image(Image.fromarray(image_rgb), (input_h, input_w))
            image_tensor = torch.from_numpy(np.asarray(letterboxed).copy()).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(image_tensor)
                detections = decode_student_outputs(
                    outputs,
                    strides={level: int(value) for level, value in config["assigner"]["strides"].items()},
                    score_threshold=float(config["inference"]["score_threshold"]),
                    nms_iou_threshold=float(config["inference"]["nms_iou_threshold"]),
                )[0]
            det_rows = []
            if detections:
                det_boxes = torch.stack([det["bbox_xyxy"] for det in detections], dim=0)
                det_boxes = unletterbox_boxes_xyxy(det_boxes, scale=resize_scale, pad=pad, orig_size=image_rgb.shape[:2])
                for det, box in zip(detections, det_boxes):
                    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                    det_rows.append([x1, y1, x2, y2, float(det["score"]), 0])

            det_array = np.asarray(det_rows, dtype=np.float32).reshape(-1, 6)
            outputs = tracker.update(det_array, image_bgr, None)
            frame_id = int(image_path.stem)
            tracks: list[dict[str, Any]] = []
            for row in np.asarray(outputs):
                x1, y1, x2, y2, track_id, conf = row[:6]
                tracks.append({"id": int(track_id), "xywh": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]})
                handle.write(f"{frame_id},{int(track_id)},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{float(conf):.6f},-1,-1,-1\n")
            pred_by_frame[frame_id] = tracks
    return pred_by_frame


def main() -> int:
    args = parse_args()
    config = load_yaml(args.student_config)
    tracker_cfg = load_yaml(args.tracker_config)["tracker"]
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint or Path(config["training"]["output_dir"]) / "best.pt")
    model = build_student(config, checkpoint_path, device)
    data_root = Path(args.data_root)
    sequences = [item.strip() for item in args.sequences.split(",") if item.strip()]
    output_dir = Path(args.output_dir)

    accumulators = []
    names = []
    for sequence in sequences:
        sequence_dir = data_root / "train" / sequence
        pred = run_sequence(
            model=model,
            sequence_dir=sequence_dir,
            output_path=output_dir / "tracks" / f"{sequence}.txt",
            config=config,
            tracker_cfg=tracker_cfg,
            reid_weights=Path(args.reid_weights),
            device=device,
            max_frames=args.max_frames,
        )
        gt = load_gt_by_frame(sequence_dir / "gt" / "gt.txt")
        accumulators.append(evaluate_sequence(gt, pred))
        names.append(sequence)

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accumulators,
        names=names,
        metrics=["mota", "motp", "idf1", "idp", "idr", "precision", "recall", "num_switches", "num_false_positives", "num_misses"],
        generate_overall=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(summary.to_json(indent=2), encoding="utf-8")
    print(json.dumps(json.loads(summary.to_json()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
