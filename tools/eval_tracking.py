#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.transforms import letterbox_image, unletterbox_boxes_xyxy
from engine.inference import decode_student_outputs
from models.student_jde import StudentJDE
from trackers.deepsort_adapter import DeepSortAdapter
from trackers.strongsort_adapter import StrongSortAdapter
from utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tracking with the distilled student and export MOT-format results.")
    parser.add_argument("--config", default="configs/student_distill.yaml")
    parser.add_argument("--tracker-config", default="configs/tracker.yaml")
    parser.add_argument("--checkpoint", default="runs/student_distill/best.pt")
    parser.add_argument("--sequence-dir", required=True, help="Path to a sequence directory that contains img1/.")
    parser.add_argument("--output", default="outputs/student_tracking.txt")
    return parser.parse_args()


def build_tracker(name: str, tracker_cfg: dict):
    name = name.lower()
    if name == "deepsort":
        return DeepSortAdapter(
            max_age=tracker_cfg.get("max_age", 30),
            n_init=tracker_cfg.get("min_hits", 3),
            max_cosine_distance=tracker_cfg.get("cosine_threshold", 0.25),
            nn_budget=tracker_cfg.get("nn_budget", 100),
        )
    if name == "strongsort":
        return StrongSortAdapter(
            max_age=tracker_cfg.get("max_age", 30),
            max_iou_dist=tracker_cfg.get("iou_threshold", 0.5),
            max_cos_dist=tracker_cfg.get("cosine_threshold", 0.25),
            nn_budget=tracker_cfg.get("nn_budget", 100),
        )
    raise ValueError(f"Unsupported tracker: {name}")


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    tracker_cfg = load_yaml(args.tracker_config)["tracker"]
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    id_classes = checkpoint["student"]["id_classifier.weight"].shape[0] if "id_classifier.weight" in checkpoint["student"] else 0
    model = StudentJDE(
        backbone_name=config["student"]["backbone_name"],
        emb_dim=config["student"]["emb_dim"],
        num_classes=1,
        fpn_channels=config["student"]["fpn_channels"],
        pretrained_backbone=False,
        num_id_classes=id_classes,
    ).to(device)
    model.load_state_dict(checkpoint["student"])
    model.eval()

    tracker = build_tracker(tracker_cfg["name"], tracker_cfg)
    sequence_dir = Path(args.sequence_dir)
    image_paths = sorted((sequence_dir / "img1").glob("*.jpg"))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_h, input_w = tuple(load_yaml(config["dataset"]["config"])["dataset"]["input_size"])

    with output_path.open("w", encoding="utf-8") as handle:
        for image_path in image_paths:
            image_bgr = cv2.imread(str(image_path))
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            letterboxed, resize_scale, pad = letterbox_image(pil_image, (input_h, input_w))
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
            if detections:
                det_boxes = torch.stack([det["bbox_xyxy"] for det in detections], dim=0)
                det_boxes = unletterbox_boxes_xyxy(
                    det_boxes,
                    scale=resize_scale,
                    pad=pad,
                    orig_size=image_rgb.shape[:2],
                )
                for det, box in zip(detections, det_boxes):
                    det["bbox_xyxy"] = box
            tracks = tracker.update(detections, image_bgr)
            frame_id = int(image_path.stem)
            for track in tracks:
                x1, y1, x2, y2 = track["bbox_xyxy"]
                handle.write(
                    f"{frame_id},{track['track_id']},{x1:.2f},{y1:.2f},{(x2 - x1):.2f},{(y2 - y1):.2f},{track['score']:.4f},-1,-1,-1\n"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
