#!/usr/bin/env python3
"""Run YOLO person tracking on MOT16 with one worker per GPU.

The script parallelizes across sequences, which preserves tracker state inside
each sequence while still using multiple GPUs across the dataset.
"""

from __future__ import annotations

import argparse
import configparser
import json
import multiprocessing as mp
import queue
import sys
import traceback
from pathlib import Path

try:
    import cv2
    from PIL import Image
    from ultralytics import YOLO
except ModuleNotFoundError as exc:
    missing = exc.name or "required package"
    raise SystemExit(
        f"Missing dependency: {missing}. Install the project requirements in the Python environment "
        f"you use for tracking, for example: pip install -r requirements.txt"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("MOT16"),
        help="Path to the MOT16 dataset root.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="Dataset split to process.",
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Optional list of sequence names. Default: all sequences in the split.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="YOLO model weights. Default: largest local YOLO weights in code/.",
    )
    parser.add_argument(
        "--tracker",
        type=Path,
        default=Path("code/botsort_mot16_person.yaml"),
        help="Ultralytics tracker YAML.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/yolo_mot16_multigpu"),
        help="Directory for tracking text files, GIFs, and run metadata.",
    )
    parser.add_argument(
        "--gpus",
        default="0,1,2,3",
        help="Comma-separated GPU ids to use.",
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--vid-stride", type=int, default=1, help="Process every Nth input frame.")
    parser.add_argument("--gif-stride", type=int, default=4, help="Keep every Nth annotated frame in the GIF.")
    parser.add_argument(
        "--gif-max-frames",
        type=int,
        default=240,
        help="Maximum number of frames stored in each GIF.",
    )
    parser.add_argument(
        "--gif-width",
        type=int,
        default=960,
        help="Resize GIF frames to this width while preserving aspect ratio.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Bounding-box line width for annotations.",
    )
    parser.add_argument(
        "--save-empty",
        action="store_true",
        help="Write MOT rows with track id -1 when detections have no assigned ID yet.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Run the model in FP16 mode.",
    )
    return parser.parse_args()


def pick_default_model(code_dir: Path) -> Path:
    model_path = code_dir / "yolo26x.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Expected YOLO weights not found: {model_path}")
    return model_path


def discover_sequences(dataset_root: Path, split: str, requested: list[str] | None) -> list[Path]:
    split_root = dataset_root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Split directory not found: {split_root}")

    all_sequences = sorted(path for path in split_root.iterdir() if (path / "img1").is_dir())
    if requested is None:
        return all_sequences

    requested_set = set(requested)
    selected = [path for path in all_sequences if path.name in requested_set]
    missing = sorted(requested_set - {path.name for path in selected})
    if missing:
        raise FileNotFoundError(f"Unknown sequence(s): {', '.join(missing)}")
    return selected


def read_seqinfo(sequence_dir: Path) -> dict[str, int | str]:
    config = configparser.ConfigParser()
    config.read(sequence_dir / "seqinfo.ini")
    section = config["Sequence"]
    return {
        "name": section.get("name", sequence_dir.name),
        "frame_rate": section.getint("frameRate", fallback=30),
        "seq_length": section.getint("seqLength", fallback=0),
        "im_width": section.getint("imWidth", fallback=0),
        "im_height": section.getint("imHeight", fallback=0),
        "im_ext": section.get("imExt", fallback=".jpg"),
    }


def distribute_sequences(sequence_dirs: list[Path], gpus: list[str]) -> list[tuple[str, list[Path]]]:
    buckets = {gpu: [] for gpu in gpus}
    for index, sequence_dir in enumerate(sequence_dirs):
        gpu = gpus[index % len(gpus)]
        buckets[gpu].append(sequence_dir)
    return [(gpu, buckets[gpu]) for gpu in gpus if buckets[gpu]]


def draw_overlay(frame_bgr, sequence_name: str, gpu_id: str, frame_index: int) -> None:
    label = f"{sequence_name} | GPU {gpu_id} | frame {frame_index:06d}"
    cv2.rectangle(frame_bgr, (12, 12), (520, 52), (0, 0, 0), thickness=-1)
    cv2.putText(
        frame_bgr,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )


def resize_for_gif(frame_bgr, gif_width: int) -> Image.Image:
    height, width = frame_bgr.shape[:2]
    if gif_width and width > gif_width:
        scale = gif_width / float(width)
        resized = cv2.resize(frame_bgr, (gif_width, int(height * scale)), interpolation=cv2.INTER_AREA)
    else:
        resized = frame_bgr
    return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))


def write_gif(frames: list[Image.Image], output_path: Path, fps: int, gif_stride: int, vid_stride: int) -> None:
    if not frames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    effective_fps = max(1, int(round(fps / max(1, gif_stride * vid_stride))))
    duration_ms = int(round(1000 / effective_fps))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def write_mot_results(rows: list[list[float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            frame_id, track_id, x, y, w, h, score, world_x, world_y, world_z = row
            handle.write(
                f"{int(frame_id)},{int(track_id)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.4f},{world_x:.0f},{world_y:.0f},{world_z:.0f}\n"
            )


def process_sequence(
    sequence_dir: Path,
    model_path: Path,
    tracker_path: Path,
    output_root: Path,
    gpu_id: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    seqinfo = read_seqinfo(sequence_dir)
    sequence_name = str(seqinfo["name"])
    image_dir = sequence_dir / "img1"

    model = YOLO(str(model_path))
    stream = model.track(
        source=str(image_dir),
        stream=True,
        device=gpu_id,
        classes=[0],
        tracker=str(tracker_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        persist=True,
        half=args.half,
        verbose=False,
        save=False,
        show=False,
    )

    rows: list[list[float]] = []
    gif_frames: list[Image.Image] = []
    processed_frames = 0
    kept_gif_frames = 0

    for frame_index, result in enumerate(stream, start=1):
        processed_frames += 1
        result_path = Path(result.path) if getattr(result, "path", None) else None
        source_frame_id = frame_index
        if result_path is not None and result_path.stem.isdigit():
            source_frame_id = int(result_path.stem)
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().tolist()
            confs = boxes.conf.cpu().tolist() if boxes.conf is not None else [1.0] * len(xyxy)
            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [-1] * len(xyxy)
            classes = boxes.cls.int().cpu().tolist() if boxes.cls is not None else [0] * len(xyxy)
            for box, score, track_id, cls_id in zip(xyxy, confs, ids, classes):
                if cls_id != 0:
                    continue
                if track_id < 0 and not args.save_empty:
                    continue
                x1, y1, x2, y2 = box
                rows.append(
                    [
                        source_frame_id,
                        track_id,
                        x1,
                        y1,
                        max(0.0, x2 - x1),
                        max(0.0, y2 - y1),
                        score,
                        -1,
                        -1,
                        -1,
                    ]
                )

        if args.gif_max_frames > 0 and source_frame_id % args.gif_stride == 0 and kept_gif_frames < args.gif_max_frames:
            annotated = result.plot(conf=True, labels=True, line_width=args.line_width)
            draw_overlay(annotated, sequence_name=sequence_name, gpu_id=gpu_id, frame_index=source_frame_id)
            gif_frames.append(resize_for_gif(annotated, gif_width=args.gif_width))
            kept_gif_frames += 1

    seq_output_dir = output_root / sequence_name
    txt_path = seq_output_dir / f"{sequence_name}.txt"
    gif_path = seq_output_dir / f"{sequence_name}.gif"
    meta_path = seq_output_dir / "run_summary.json"

    write_mot_results(rows, txt_path)
    write_gif(
        gif_frames,
        gif_path,
        fps=int(seqinfo["frame_rate"]),
        gif_stride=args.gif_stride,
        vid_stride=args.vid_stride,
    )

    summary = {
        "sequence": sequence_name,
        "gpu": gpu_id,
        "model": str(model_path),
        "tracker": str(tracker_path),
        "processed_frames": processed_frames,
        "gif_frames": kept_gif_frames,
        "detections_written": len(rows),
        "mot_txt": str(txt_path),
        "gif": str(gif_path),
    }
    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def worker_main(
    gpu_id: str,
    sequence_dirs: list[str],
    model_path: str,
    tracker_path: str,
    output_root: str,
    args_dict: dict[str, object],
    result_queue: mp.Queue,
) -> None:
    class Args:
        pass

    args = Args()
    for key, value in args_dict.items():
        setattr(args, key, value)

    try:
        summaries = []
        for sequence_dir in sequence_dirs:
            summaries.append(
                process_sequence(
                    sequence_dir=Path(sequence_dir),
                    model_path=Path(model_path),
                    tracker_path=Path(tracker_path),
                    output_root=Path(output_root),
                    gpu_id=gpu_id,
                    args=args,
                )
            )
        result_queue.put({"gpu": gpu_id, "ok": True, "summaries": summaries})
    except Exception as exc:
        result_queue.put(
            {
                "gpu": gpu_id,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def main() -> int:
    args = parse_args()
    if args.gif_stride < 1 or args.vid_stride < 1:
        raise ValueError("--gif-stride and --vid-stride must be >= 1")

    code_dir = Path(__file__).resolve().parent
    dataset_root = args.dataset_root.resolve()
    model_path = args.model.resolve() if args.model else pick_default_model(code_dir).resolve()
    tracker_path = args.tracker.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not tracker_path.exists():
        raise FileNotFoundError(f"Tracker config not found: {tracker_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("No GPU ids provided.")

    sequence_dirs = discover_sequences(dataset_root=dataset_root, split=args.split, requested=args.sequences)
    assignments = distribute_sequences(sequence_dirs, gpus)
    if not assignments:
        raise RuntimeError("No sequences selected.")

    run_manifest = {
        "dataset_root": str(dataset_root),
        "split": args.split,
        "model": str(model_path),
        "tracker": str(tracker_path),
        "gpus": gpus,
        "assignments": {gpu: [path.name for path in seqs] for gpu, seqs in assignments},
    }
    (output_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    processes: list[mp.Process] = []
    args_dict = vars(args).copy()
    args_dict["dataset_root"] = str(args.dataset_root)
    args_dict["model"] = str(args.model) if args.model else None
    args_dict["tracker"] = str(args.tracker)
    args_dict["output_root"] = str(args.output_root)

    for gpu_id, assigned_sequences in assignments:
        process = ctx.Process(
            target=worker_main,
            args=(
                gpu_id,
                [str(path) for path in assigned_sequences],
                str(model_path),
                str(tracker_path),
                str(output_root),
                args_dict,
                result_queue,
            ),
        )
        process.start()
        processes.append(process)

    results = []
    remaining = len(processes)
    while remaining > 0:
        try:
            item = result_queue.get(timeout=5)
            results.append(item)
            remaining -= 1
            status = "ok" if item["ok"] else "failed"
            print(f"[GPU {item['gpu']}] {status}")
        except queue.Empty:
            alive = sum(1 for process in processes if process.is_alive())
            print(f"Waiting for workers... {alive} still running", flush=True)

    exit_code = 0
    for process in processes:
        process.join()
        if process.exitcode not in (0, None):
            exit_code = process.exitcode or 1

    failures = [item for item in results if not item["ok"]]
    (output_root / "worker_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    if failures:
        for failure in failures:
            print(f"\nWorker on GPU {failure['gpu']} failed:", file=sys.stderr)
            print(failure["error"], file=sys.stderr)
            print(failure["traceback"], file=sys.stderr)
        return exit_code or 1

    print("\nCompleted sequences:")
    for item in sorted(results, key=lambda result: result["gpu"]):
        for summary in item["summaries"]:
            print(
                f"  {summary['sequence']} | GPU {summary['gpu']} | "
                f"frames={summary['processed_frames']} | gif={summary['gif']}"
            )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
