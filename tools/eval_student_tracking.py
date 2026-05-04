#!/usr/bin/env python3
"""
Evaluate student tracker on MOT16 sequences with metric computation.

Computes: MOTA, MOTP, IDF1, precision, recall, num_switches, etc.
Generates: per-sequence metrics CSV, overall metrics JSON, annotated videos, report.md
"""
from __future__ import annotations

import argparse
import configparser
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

if not hasattr(np, "asfarray"):
    np.asfarray = lambda values, dtype=float: np.asarray(values, dtype=dtype)

import motmetrics as mm

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

METRIC_KEYS = [
    "mota",
    "motp",
    "idf1",
    "idp",
    "idr",
    "precision",
    "recall",
    "num_switches",
    "num_false_positives",
    "num_misses",
    "num_objects",
]

DEFAULT_VAL_SEQUENCES = ["MOT16-02", "MOT16-04", "MOT16-05", "MOT16-09", "MOT16-10", "MOT16-11", "MOT16-13"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate student tracker on MOT16 sequences with metric computation."
    )
    parser.add_argument("--config", type=str, default="configs/student_distill.yaml")
    parser.add_argument("--tracker-config", type=str, default="configs/tracker.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--sequences", type=str, default=",".join(DEFAULT_VAL_SEQUENCES))
    parser.add_argument("--run-name", type=str, default="student_tracker_eval")
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-previews", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_gt_by_frame(gt_path: Path) -> dict[int, list[dict[str, Any]]]:
    """Load ground truth from MOT format file."""
    frame_map: dict[int, list[dict[str, Any]]] = {}
    if not gt_path.exists():
        return frame_map

    with gt_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            frame = int(float(row[0]))
            track_id = int(float(row[1]))
            x = float(row[2])
            y = float(row[3])
            w = float(row[4])
            h = float(row[5])
            mark = int(float(row[6]))
            cls = int(float(row[7]))

            # Filter: only valid annotations
            if mark != 1 or cls != 1 or track_id <= 0:
                continue

            frame_map.setdefault(frame, []).append(
                {
                    "id": track_id,
                    "xywh": [x, y, w, h],
                }
            )
    return frame_map


def build_tracker(name: str, tracker_cfg: dict, device: torch.device):
    """Build tracker from config."""
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
    """Load frame rate from seqinfo.ini."""
    parser = configparser.ConfigParser()
    parser.read(sequence_dir / "seqinfo.ini")
    return float(parser.getint("Sequence", "frameRate", fallback=30))


def create_video_writer(
    *, output_video_path: Path, fps: float, width: int, height: int
) -> cv2.VideoWriter:
    """Create MP4 video writer."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))


def track_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic color for track ID."""
    if track_id < 0:
        return (160, 160, 160)
    return (
        64 + ((track_id * 53) % 192),
        64 + ((track_id * 97) % 192),
        64 + ((track_id * 193) % 192),
    )


def draw_tracks(
    frame_bgr: np.ndarray, tracks: list[dict], frame_id: int, gt_boxes: list[dict] | None = None
) -> np.ndarray:
    """Draw tracking results and optional GT on frame."""
    annotated = frame_bgr.copy()

    # Draw GT boxes (green) if provided
    if gt_boxes:
        for gt_box in gt_boxes:
            x, y, w, h = gt_box["xywh"]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green

    # Draw predicted tracks (colored by ID)
    for track in tracks:
        if track["track_id"] < 0:
            continue
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

    # Frame ID in corner
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

    return annotated


def run_sequence_tracking(
    model: StudentJDE,
    tracker,
    config: dict,
    sequence_dir: Path,
    tracker_cfg: dict,
    output_tracks_dir: Path,
    preview_dir: Path,
    device: torch.device,
    *,
    max_frames: int | None,
    save_preview: bool,
) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]]]:
    """Run tracking on a single sequence. Returns (metadata, predictions by frame)."""
    image_paths = sorted((sequence_dir / "img1").glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {sequence_dir / 'img1'}")

    if max_frames is not None:
        image_paths = image_paths[:max_frames]

    fps = load_sequence_fps(sequence_dir)
    sequence_name = sequence_dir.name
    track_file_path = output_tracks_dir / f"{sequence_name}.txt"
    preview_path = preview_dir / f"{sequence_name}.mp4"

    # Load GT for visualization
    gt_path = sequence_dir / "gt" / "gt.txt"
    gt_by_frame = load_gt_by_frame(gt_path)

    input_h, input_w = tuple(load_yaml(config["dataset"]["config"])["dataset"]["input_size"])
    video_writer: cv2.VideoWriter | None = None
    mot_lines: list[str] = []
    frame_predictions: dict[int, list[dict[str, Any]]] = {}
    unique_track_ids: set[int] = set()

    try:
        for image_path in image_paths:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            letterboxed, resize_scale, pad = letterbox_image(pil_image, (input_h, input_w))
            image_tensor = torch.from_numpy(np.asarray(letterboxed).copy()).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(device)

            # Forward pass
            with torch.no_grad():
                outputs = model(image_tensor)
                detections = decode_student_outputs(
                    outputs,
                    strides={level: int(value) for level, value in config["assigner"]["strides"].items()},
                    score_threshold=float(config["inference"]["score_threshold"]),
                    nms_iou_threshold=float(config["inference"]["nms_iou_threshold"]),
                )[0]

            # Unletterbox detections
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

            # Update tracker
            tracks = tracker.update(detections, image_bgr)
            frame_id = int(image_path.stem)

            # Store predictions
            frame_predictions[frame_id] = [
                {
                    "id": int(track["track_id"]),
                    "xywh": [
                        track["bbox_xyxy"][0],
                        track["bbox_xyxy"][1],
                        track["bbox_xyxy"][2] - track["bbox_xyxy"][0],
                        track["bbox_xyxy"][3] - track["bbox_xyxy"][1],
                    ],
                    "conf": float(track["score"]),
                }
                for track in tracks
                if track["track_id"] >= 0
            ]

            for track in tracks:
                if track["track_id"] < 0:
                    continue
                unique_track_ids.add(int(track["track_id"]))
                x1, y1, x2, y2 = track["bbox_xyxy"]
                w = x2 - x1
                h = y2 - y1
                mot_lines.append(
                    f"{frame_id},{int(track['track_id'])},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{float(track['score']):.4f},-1,-1,-1"
                )

            # Draw and write video frame
            gt_boxes = gt_by_frame.get(frame_id, [])
            annotated_frame = draw_tracks(image_bgr, tracks, frame_id, gt_boxes)

            if save_preview:
                if video_writer is None:
                    height, width = annotated_frame.shape[:2]
                    video_writer = create_video_writer(
                        output_video_path=preview_path,
                        fps=fps,
                        width=width,
                        height=height,
                    )
                video_writer.write(annotated_frame)

    finally:
        if video_writer is not None:
            video_writer.release()

    # Write MOT format results
    track_file_path.write_text("\n".join(mot_lines), encoding="utf-8")

    metadata = {
        "sequence": sequence_name,
        "fps": fps,
        "frames": len(image_paths),
        "unique_track_ids": len(unique_track_ids),
        "track_file": str(track_file_path),
        "preview_file": str(preview_path) if save_preview else "",
    }

    return metadata, frame_predictions


def evaluate_sequence(
    gt_by_frame: dict[int, list[dict[str, Any]]],
    pred_by_frame: dict[int, list[dict[str, Any]]],
    sequence_name: str,
):
    """Compute tracking metrics for a single sequence."""
    acc = mm.MOTAccumulator(auto_id=False)

    for frame_id in sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys())):
        gt_items = gt_by_frame.get(frame_id, [])
        pred_items = pred_by_frame.get(frame_id, [])

        gt_ids = [item["id"] for item in gt_items]
        pred_ids = [item["id"] for item in pred_items]
        gt_boxes = [item["xywh"] for item in gt_items]
        pred_boxes = [item["xywh"] for item in pred_items]

        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, distances, frameid=frame_id)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=METRIC_KEYS, name=sequence_name)
    return summary


def format_float(value: Any) -> str:
    """Format float for display."""
    if isinstance(value, (float, int)):
        return f"{float(value):.4f}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown table."""
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(format_float(row[col]) for col in columns) + " |")
    return "\n".join([header, separator] + rows)


def write_report(
    run_dir: Path,
    config: dict[str, Any],
    sequence_runs: list[dict[str, Any]],
    sequence_metrics_df: pd.DataFrame,
    overall_metrics_row: dict[str, Any],
) -> None:
    """Write markdown report."""
    lines: list[str] = []
    lines.append("# Student Tracker MOT16 Evaluation Report")
    lines.append("")
    lines.append(f"Run directory: `{run_dir}`")
    lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- `student_config`: `{config['student_config']}`")
    lines.append(f"- `checkpoint`: `{config['checkpoint']}`")
    lines.append(f"- `tracker_config`: `{config['tracker_config']}`")
    lines.append(f"- `tracker_name`: `{config['tracker_name']}`")
    lines.append(f"- `device`: `{config['device']}`")
    lines.append(f"- `split`: `{config['split']}`")
    lines.append(f"- `sequences`: {config['sequences']}")

    lines.append("")
    lines.append("## Sequences")
    lines.append("")
    for item in sequence_runs:
        lines.append(f"- `{item['sequence']}`: frames={item['frames']}, unique_ids={item['unique_track_ids']}")

    lines.append("")
    lines.append("## Per-Sequence Metrics")
    lines.append("")
    lines.append(dataframe_to_markdown(sequence_metrics_df))

    lines.append("")
    lines.append("## Overall Metrics")
    lines.append("")
    for key, value in overall_metrics_row.items():
        lines.append(f"- `{key}`: `{format_float(value)}`")

    lines.append("")
    lines.append("## Artifact Paths")
    lines.append("")
    lines.append(f"- Tracks: `{run_dir / 'tracks'}`")
    lines.append(f"- Previews: `{run_dir / 'previews'}`")
    lines.append(f"- Per-sequence metrics: `{run_dir / 'per_sequence_metrics.csv'}`")
    lines.append(f"- Overall metrics: `{run_dir / 'overall_metrics.json'}`")

    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    project_root = Path.cwd()

    # Setup run directory
    run_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.run_name).strip("_") or "eval"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else Path("tracking_eval") / "runs"
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"{timestamp}_{run_name}"
    tracks_dir = run_dir / "tracks"
    previews_dir = run_dir / "previews"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    # Load configs
    config = load_yaml(args.config)
    tracker_cfg = load_yaml(args.tracker_config)["tracker"]
    checkpoint_path = args.checkpoint or str(Path(config["training"]["output_dir"]) / "best.pt")

    # Resolve device
    device = torch.device(resolve_device(args.device))

    # Build model
    print(f"[eval] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    id_classes = (
        checkpoint["student"]["id_classifier.weight"].shape[0]
        if "id_classifier.weight" in checkpoint["student"]
        else 0
    )
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

    # Build tracker
    tracker = build_tracker(tracker_cfg["name"], tracker_cfg, device)

    # Parse sequences
    sequences = [s.strip() for s in args.sequences.split(",") if s.strip()]
    split_dir = project_root / "MOT16" / args.split
    if not split_dir.exists():
        print(f"[error] Split directory not found: {split_dir}")
        return 1

    # Save config
    eval_config = {
        "student_config": args.config,
        "checkpoint": checkpoint_path,
        "tracker_config": args.tracker_config,
        "tracker_name": tracker_cfg["name"],
        "device": str(device),
        "split": args.split,
        "sequences": sequences,
        "max_frames": args.max_frames,
    }
    (run_dir / "config.json").write_text(json.dumps(eval_config, indent=2), encoding="utf-8")

    print(f"[eval] Run directory: {run_dir}")
    print(f"[eval] Model: {checkpoint_path}")
    print(f"[eval] Tracker: {tracker_cfg['name']}")
    print(f"[eval] Device: {device}")
    print(f"[eval] Sequences: {sequences}")
    print()

    # Run tracking on each sequence
    sequence_runs: list[dict[str, Any]] = []
    sequence_summaries: list[pd.DataFrame] = []

    for seq_name in sequences:
        sequence_dir = split_dir / seq_name
        if not sequence_dir.exists():
            print(f"[warn] Sequence not found: {sequence_dir}")
            continue

        print(f"[tracking] {seq_name}...")
        metadata, frame_predictions = run_sequence_tracking(
            model,
            tracker,
            config,
            sequence_dir,
            tracker_cfg,
            tracks_dir,
            previews_dir,
            device,
            max_frames=args.max_frames,
            save_preview=args.save_previews,
        )
        sequence_runs.append(metadata)

        # Load GT and compute metrics
        print(f"[metrics] {seq_name}...")
        gt_path = sequence_dir / "gt" / "gt.txt"
        gt_by_frame = load_gt_by_frame(gt_path)
        summary = evaluate_sequence(gt_by_frame, frame_predictions, seq_name)
        sequence_summaries.append(summary)
        print(f"  MOTA={summary.loc[seq_name, 'mota']:.4f}, " f"IDF1={summary.loc[seq_name, 'idf1']:.4f}")

    print()

    # Compute overall metrics
    if sequence_summaries:
        metrics_df = pd.concat(sequence_summaries, axis=0)
        overall_metrics = metrics_df.mean()
        metrics_df.to_csv(run_dir / "per_sequence_metrics.csv")
        (run_dir / "overall_metrics.json").write_text(
            json.dumps(overall_metrics.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        write_report(run_dir, eval_config, sequence_runs, metrics_df, overall_metrics.to_dict())

        print("[summary] Per-sequence metrics:")
        print(metrics_df.to_string())
        print()
        print("[summary] Overall metrics:")
        for key in METRIC_KEYS:
            if key in overall_metrics:
                print(f"  {key}: {overall_metrics[key]:.4f}")
    else:
        print("[error] No sequences processed successfully")
        return 1

    print()
    print(f"[done] Results saved to: {run_dir}")
    print(f"[done] Report: {run_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
