#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
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
from utils.checkpoint import validate_checkpoint_shapes
from utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tracking with the distilled student and export MOT-format results.")
    parser.add_argument("--config", default="configs/student_distill.yaml")
    parser.add_argument("--tracker-config", default="configs/tracker.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--sequence-dir", required=True, help="Path to a sequence directory that contains img1/.")
    parser.add_argument("--output", default="outputs/student_tracking.txt")
    parser.add_argument("--output-video", default="", help="Optional path for the annotated tracking video.")
    return parser.parse_args()


def build_tracker(name: str, tracker_cfg: dict, device: torch.device):
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
            device=device,
            reid_weights=tracker_cfg.get("reid_weights"),
            half=tracker_cfg.get("half"),
            max_age=tracker_cfg.get("max_age", 30),
            min_conf=tracker_cfg.get("min_conf", 0.1),
            max_iou_dist=tracker_cfg.get("iou_threshold", 0.5),
            max_cos_dist=tracker_cfg.get("cosine_threshold", 0.25),
            n_init=tracker_cfg.get("min_hits", 3),
            nn_budget=tracker_cfg.get("nn_budget", 100),
            mc_lambda=tracker_cfg.get("mc_lambda", 0.98),
            ema_alpha=tracker_cfg.get("ema_alpha", 0.9),
        )
    raise ValueError(f"Unsupported tracker: {name}")


def load_sequence_fps(sequence_dir: Path) -> float:
    parser = configparser.ConfigParser()
    parser.read(sequence_dir / "seqinfo.ini")
    return float(parser.getint("Sequence", "frameRate", fallback=30))


def create_video_writer(*, output_video_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))


def track_color(track_id: int) -> tuple[int, int, int]:
    if track_id < 0:
        return (160, 160, 160)
    return (
        64 + ((track_id * 53) % 192),
        64 + ((track_id * 97) % 192),
        64 + ((track_id * 193) % 192),
    )


def draw_tracks(frame_bgr: np.ndarray, tracks: list[dict], frame_id: int) -> np.ndarray:
    annotated = frame_bgr.copy()
    cv2.putText(
        annotated,
        f"Frame {frame_id}",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for track in tracks:
        x1, y1, x2, y2 = [int(round(v)) for v in track["bbox_xyxy"]]
        color = track_color(int(track["track_id"]))
        label = f"ID {int(track['track_id'])} {float(track['score']):.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    tracker_cfg = load_yaml(args.tracker_config)["tracker"]
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

    checkpoint_path = args.checkpoint or str(Path(config["training"]["output_dir"]) / "best.pt")
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
    ).to(device)
    validate_checkpoint_shapes(checkpoint, model, feature_adapters=None)
    model.load_state_dict(checkpoint["student"])
    model.eval()

    tracker = build_tracker(tracker_cfg["name"], tracker_cfg, device)
    sequence_dir = Path(args.sequence_dir)
    image_paths = sorted((sequence_dir / "img1").glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {sequence_dir / 'img1'}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_video_path = Path(args.output_video) if args.output_video else output_path.with_suffix(".mp4")
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    fps = load_sequence_fps(sequence_dir)
    input_h, input_w = tuple(load_yaml(config["dataset"]["config"])["dataset"]["input_size"])
    video_writer: cv2.VideoWriter | None = None

    try:
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

                annotated_frame = draw_tracks(image_bgr, tracks, frame_id)
                if video_writer is None:
                    height, width = annotated_frame.shape[:2]
                    video_writer = create_video_writer(
                        output_video_path=output_video_path,
                        fps=fps,
                        width=width,
                        height=height,
                    )
                video_writer.write(annotated_frame)

                for track in tracks:
                    x1, y1, x2, y2 = track["bbox_xyxy"]
                    handle.write(
                        f"{frame_id},{track['track_id']},{x1:.2f},{y1:.2f},{(x2 - x1):.2f},{(y2 - y1):.2f},{track['score']:.4f},-1,-1,-1\n"
                    )
    finally:
        if video_writer is not None:
            video_writer.release()

    print(f"Track results saved to: {output_path}")
    print(f"Tracking video saved to: {output_video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
