from __future__ import annotations

import argparse
import configparser
import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

if not hasattr(np, "asfarray"):
    np.asfarray = lambda values, dtype=float: np.asarray(values, dtype=dtype)

import motmetrics as mm

GT_COLUMNS = [
    "frame",
    "id",
    "bb_left",
    "bb_top",
    "bb_width",
    "bb_height",
    "mark",
    "class",
    "visibility",
]

DEFAULT_VAL_SEQUENCES = ["MOT16-11", "MOT16-13"]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MOT16 tracking and generate report-ready metrics.")
    parser.add_argument("--project-root", type=str, default=str(Path.cwd()))
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--tracker", type=str, default="")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--sequences", type=str, default=",".join(DEFAULT_VAL_SEQUENCES))
    parser.add_argument("--run-name", type=str, default="yolo26n_botsort_report")
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--persist", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--person-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-previews", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def resolve_project_root(project_root_arg: str) -> Path:
    project_root = Path(project_root_arg).resolve()
    if not (project_root / "MOT16").exists():
        raise FileNotFoundError(f"MOT16 folder not found under {project_root}")
    return project_root


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_gt_by_frame(gt_path: Path) -> dict[int, list[dict[str, Any]]]:
    frame_map: dict[int, list[dict[str, Any]]] = {}
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
            if mark != 1 or cls != 1 or track_id <= 0:
                continue
            frame_map.setdefault(frame, []).append(
                {
                    "id": track_id,
                    "xywh": [x, y, w, h],
                }
            )
    return frame_map


def draw_annotations(frame_bgr, boxes) -> tuple[Any, list[dict[str, Any]]]:
    annotated = frame_bgr.copy()
    tracks: list[dict[str, Any]] = []
    if boxes is None or boxes.xyxy is None:
        return annotated, tracks

    xyxy_list = boxes.xyxy.int().tolist()
    ids = boxes.id.int().tolist() if boxes.id is not None else [-1] * len(xyxy_list)
    confs = boxes.conf.tolist() if boxes.conf is not None else [0.0] * len(xyxy_list)

    for coords, track_id, conf in zip(xyxy_list, ids, confs):
        x1, y1, x2, y2 = coords
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        label = f"ID {track_id}" if track_id >= 0 else "ID ?"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 80, 0), 3)
        cv2.putText(
            annotated,
            label,
            (x1, max(24, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        tracks.append(
            {
                "id": int(track_id),
                "xywh": [float(x1), float(y1), float(width), float(height)],
                "conf": float(conf),
            }
        )

    return annotated, tracks


def load_seq_fps(seq_dir: Path) -> float:
    parser = configparser.ConfigParser()
    parser.read(seq_dir / "seqinfo.ini")
    return float(parser.getint("Sequence", "frameRate", fallback=30))


def run_sequence_tracking(
    model: YOLO,
    sequence_dir: Path,
    tracker_path: Path,
    output_tracks_dir: Path,
    preview_dir: Path,
    *,
    conf: float,
    iou: float,
    imgsz: int,
    persist: bool,
    person_only: bool,
    device: str,
    max_frames: int | None,
    save_preview: bool,
) -> dict[str, Any]:
    image_paths = sorted((sequence_dir / "img1").glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {sequence_dir / 'img1'}")

    fps = load_seq_fps(sequence_dir)
    sequence_name = sequence_dir.name
    track_file_path = output_tracks_dir / f"{sequence_name}.txt"
    preview_path = preview_dir / f"{sequence_name}.mp4"

    if max_frames is not None:
        selected_image_paths = image_paths[:max_frames]
        source: str | list[str] = [str(p) for p in selected_image_paths]
    else:
        selected_image_paths = image_paths
        source = str(sequence_dir / "img1")

    stream = model.track(
        source=source,
        tracker=str(tracker_path),
        classes=[0] if person_only else None,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        persist=persist,
        stream=True,
        verbose=False,
        device=device,
    )

    writer = None
    mot_lines: list[str] = []
    frame_track_counts: list[int] = []
    unique_track_ids: set[int] = set()
    frame_predictions: dict[int, list[dict[str, Any]]] = {}
    processed_frames = 0

    for idx, result in enumerate(stream):
        if max_frames is not None and idx >= max_frames:
            break
        if idx >= len(selected_image_paths):
            break

        frame_stem = Path(result.path).stem
        if frame_stem.isdigit():
            frame_id = int(frame_stem)
        else:
            frame_id = int(selected_image_paths[idx].stem)
        annotated, tracks = draw_annotations(result.orig_img, result.boxes)
        frame_predictions[frame_id] = tracks
        frame_track_counts.append(len([t for t in tracks if t["id"] >= 0]))
        processed_frames += 1

        if save_preview:
            if writer is None:
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    str(preview_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
            writer.write(annotated)

        for track in tracks:
            if track["id"] < 0:
                continue
            unique_track_ids.add(track["id"])
            x, y, w, h = track["xywh"]
            mot_lines.append(
                f"{frame_id},{track['id']},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{track['conf']:.6f},-1,-1,-1"
            )

    if writer is not None:
        writer.release()

    track_file_path.write_text("\n".join(mot_lines), encoding="utf-8")

    return {
        "sequence": sequence_name,
        "fps": fps,
        "frames": processed_frames,
        "track_file": str(track_file_path),
        "preview_file": str(preview_path) if save_preview else "",
        "unique_track_ids": len(unique_track_ids),
        "avg_tracks_per_frame": (sum(frame_track_counts) / len(frame_track_counts)) if frame_track_counts else 0.0,
        "frame_predictions": frame_predictions,
    }


def evaluate_sequence(gt_by_frame: dict[int, list[dict[str, Any]]], pred_by_frame: dict[int, list[dict[str, Any]]], sequence_name: str):
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
    if isinstance(value, (float, int)):
        return f"{float(value):.4f}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
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
    lines: list[str] = []
    lines.append("# MOT16 Tracking Report")
    lines.append("")
    lines.append(f"Run directory: `{run_dir}`")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    for key, value in config.items():
        lines.append(f"- `{key}`: `{value}`")

    lines.append("")
    lines.append("## Sequences")
    lines.append("")
    for item in sequence_runs:
        lines.append(
            f"- `{item['sequence']}`: frames={item['frames']}, unique_track_ids={item['unique_track_ids']}, "
            f"avg_tracks_per_frame={item['avg_tracks_per_frame']:.2f}"
        )

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
    lines.append(f"- Tracks folder: `{run_dir / 'tracks'}`")
    lines.append(f"- Preview folder: `{run_dir / 'previews'}`")
    lines.append(f"- CSV metrics: `{run_dir / 'per_sequence_metrics.csv'}`")
    lines.append(f"- Overall JSON: `{run_dir / 'overall_metrics.json'}`")

    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = resolve_project_root(args.project_root)
    run_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.run_name).strip("_") or "report"
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    model_path = Path(args.model).resolve() if args.model else (project_root / "yolo26n.pt")
    tracker_path = Path(args.tracker).resolve() if args.tracker else (project_root / "code" / "botsort_mot16_person.yaml")
    output_root = Path(args.output_root).resolve() if args.output_root else (project_root / "tracking_eval" / "runs")
    run_dir = output_root / f"{timestamp}_{run_name}"
    tracks_dir = run_dir / "tracks"
    previews_dir = run_dir / "previews"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    sequences = [s.strip() for s in args.sequences.split(",") if s.strip()]
    device = resolve_device(args.device)

    config = {
        "project_root": str(project_root),
        "model": str(model_path),
        "tracker": str(tracker_path),
        "split": args.split,
        "sequences": sequences,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "persist": args.persist,
        "person_only": args.person_only,
        "max_frames": args.max_frames,
        "device": device,
        "save_previews": args.save_previews,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"[run] model={model_path}")
    print(f"[run] tracker={tracker_path}")
    print(f"[run] device={device}")
    print(f"[run] sequences={sequences}")
    print(f"[run] run_dir={run_dir}")

    model = YOLO(str(model_path))

    sequence_runs: list[dict[str, Any]] = []
    sequence_summaries: list[pd.DataFrame] = []

    for sequence in sequences:
        seq_dir = project_root / "MOT16" / args.split / sequence
        gt_path = seq_dir / "gt" / "gt.txt"
        if not seq_dir.exists():
            raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")
        if args.split == "train" and not gt_path.exists():
            raise FileNotFoundError(f"GT file not found for evaluation: {gt_path}")

        print(f"[sequence] running {sequence}")
        run_info = run_sequence_tracking(
            model=model,
            sequence_dir=seq_dir,
            tracker_path=tracker_path,
            output_tracks_dir=tracks_dir,
            preview_dir=previews_dir,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            persist=args.persist,
            person_only=args.person_only,
            device=device,
            max_frames=args.max_frames,
            save_preview=args.save_previews,
        )
        sequence_runs.append(run_info)

        gt_by_frame = load_gt_by_frame(gt_path) if args.split == "train" else {}
        if gt_by_frame:
            evaluated_frames = set(run_info["frame_predictions"].keys())
            gt_by_frame = {frame_id: items for frame_id, items in gt_by_frame.items() if frame_id in evaluated_frames}
            summary = evaluate_sequence(gt_by_frame, run_info["frame_predictions"], sequence)
            sequence_summaries.append(summary)

    if not sequence_summaries:
        raise RuntimeError("No sequence summaries were generated. Evaluation requires GT-backed train sequences.")

    all_summary = pd.concat(sequence_summaries)
    per_sequence_df = all_summary.reset_index(names="sequence")
    per_sequence_df.to_csv(run_dir / "per_sequence_metrics.csv", index=False)

    numeric_cols = [c for c in per_sequence_df.columns if c != "sequence"]
    overall_row = {"sequence": "OVERALL_MEAN"}
    for col in numeric_cols:
        overall_row[col] = float(per_sequence_df[col].mean())

    (run_dir / "overall_metrics.json").write_text(json.dumps(overall_row, indent=2), encoding="utf-8")
    write_report(run_dir, config, sequence_runs, per_sequence_df, overall_row)

    print("[done] report generated")
    print(f"[done] report: {run_dir / 'report.md'}")
    print(f"[done] per-sequence metrics: {run_dir / 'per_sequence_metrics.csv'}")
    print(f"[done] overall metrics: {run_dir / 'overall_metrics.json'}")


if __name__ == "__main__":
    main()
