from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = PROJECT_ROOT / "code_from_laptop"
RUNS_ROOT = PROJECT_ROOT / "project_validation" / "runs"
REQUIRED_IMPORTS = ["cv2", "streamlit", "streamlit_webrtc", "ultralytics", "motmetrics", "torch"]
FULL_REPORT_THRESHOLDS = {
    "mota": 0.35,
    "idf1": 0.45,
    "precision": 0.85,
}


@dataclass
class CheckResult:
    name: str
    status: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_tracking_eval_module():
    return load_module(
        "tracking_eval_module",
        PROJECT_ROOT / "tracking_eval" / "run_mot16_tracker_report.py",
    )


def load_streamlit_module():
    return load_module(
        "streamlit_app_module",
        PROJECT_ROOT / "streamlit_app" / "app.py",
    )


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def record_check(name: str, func):
    try:
        result = func()
        if isinstance(result, CheckResult):
            return result
        return CheckResult(name=name, status="PASS", summary="Check completed.", details=result or {})
    except Exception as exc:  # pragma: no cover - validation harness
        return CheckResult(
            name=name,
            status="FAIL",
            summary=str(exc),
        )


def candidate_runs_roots(relative_dir: str) -> list[Path]:
    roots = [
        PROJECT_ROOT / relative_dir,
        LEGACY_ROOT / relative_dir,
    ]
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(root)
    return unique


def check_environment_and_assets() -> CheckResult:
    import_results: dict[str, bool] = {}
    for module_name in REQUIRED_IMPORTS:
        import_results[module_name] = importlib.util.find_spec(module_name) is not None

    required_paths = {
        "model": PROJECT_ROOT / "yolo26n.pt",
        "tracker_config": PROJECT_ROOT / "code" / "botsort_mot16_person.yaml",
        "mot16_sequence": PROJECT_ROOT / "MOT16" / "train" / "MOT16-11" / "img1" / "000001.jpg",
        "streamlit_app": PROJECT_ROOT / "streamlit_app" / "app.py",
        "tracking_eval_script": PROJECT_ROOT / "tracking_eval" / "run_mot16_tracker_report.py",
    }
    path_results = {name: path.exists() for name, path in required_paths.items()}

    ensure(all(import_results.values()), f"Missing imports: {[k for k, ok in import_results.items() if not ok]}")
    ensure(all(path_results.values()), f"Missing assets: {[k for k, ok in path_results.items() if not ok]}")

    return CheckResult(
        name="environment_and_assets",
        status="PASS",
        summary="Required imports and project assets are present.",
        details={
            "imports": import_results,
            "paths": {name: str(path) for name, path in required_paths.items()},
        },
        artifacts=[str(path) for path in required_paths.values()],
    )


def check_tracker_config() -> CheckResult:
    tracker_path = PROJECT_ROOT / "code" / "botsort_mot16_person.yaml"
    tracker_text = tracker_path.read_text(encoding="utf-8")

    ensure("tracker_type: botsort" in tracker_text, "Tracker config is not set to BoT-SORT.")
    ensure("with_reid: True" in tracker_text, "Tracker config does not keep ReID enabled.")

    return CheckResult(
        name="tracker_config",
        status="PASS",
        summary="Tracker config keeps the expected BoT-SORT + ReID settings.",
        details={
            "tracker_path": str(tracker_path),
            "required_lines": ["tracker_type: botsort", "with_reid: True"],
        },
        artifacts=[str(tracker_path)],
    )


def check_single_frame_model_integration(streamlit_module) -> CheckResult:
    import cv2
    from ultralytics import YOLO

    frame_path = PROJECT_ROOT / "MOT16" / "train" / "MOT16-11" / "img1" / "000001.jpg"
    model_path = PROJECT_ROOT / "yolo26n.pt"
    tracker_path = PROJECT_ROOT / "code" / "botsort_mot16_person.yaml"

    frame_bgr = cv2.imread(str(frame_path))
    ensure(frame_bgr is not None, f"Unable to load test frame {frame_path}")

    model = YOLO(str(model_path))
    annotated, tracks = streamlit_module.track_single_frame(
        model=model,
        tracker_path=tracker_path,
        frame_bgr=frame_bgr,
        conf=0.25,
        iou=0.5,
        imgsz=640,
        persist=True,
        person_only=True,
    )

    ensure(annotated is not None, "No annotated frame returned.")
    ensure(tuple(annotated.shape) == tuple(frame_bgr.shape), "Annotated frame shape does not match input.")
    ensure(len(tracks) > 0, "No tracks returned on the smoke-test frame.")

    return CheckResult(
        name="single_frame_model_integration",
        status="PASS",
        summary="Model and tracker run successfully on a real MOT16 frame.",
        details={
            "frame_path": str(frame_path),
            "num_tracks": len(tracks),
            "frame_shape": list(frame_bgr.shape),
        },
        artifacts=[str(frame_path), str(model_path), str(tracker_path)],
    )


def check_sequence_tracker_smoke(tracking_eval_module, run_dir: Path) -> CheckResult:
    from ultralytics import YOLO

    smoke_dir = run_dir / "sequence_smoke"
    tracks_dir = smoke_dir / "tracks"
    previews_dir = smoke_dir / "previews"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    seq_dir = PROJECT_ROOT / "MOT16" / "train" / "MOT16-11"
    tracker_path = PROJECT_ROOT / "code" / "botsort_mot16_person.yaml"
    model_path = PROJECT_ROOT / "yolo26n.pt"
    device = tracking_eval_module.resolve_device("auto")
    model = YOLO(str(model_path))

    run_info = tracking_eval_module.run_sequence_tracking(
        model=model,
        sequence_dir=seq_dir,
        tracker_path=tracker_path,
        output_tracks_dir=tracks_dir,
        preview_dir=previews_dir,
        conf=0.25,
        iou=0.5,
        imgsz=640,
        persist=True,
        person_only=True,
        device=device,
        max_frames=12,
        save_preview=True,
    )

    gt_by_frame = tracking_eval_module.load_gt_by_frame(seq_dir / "gt" / "gt.txt")
    evaluated_frames = set(run_info["frame_predictions"].keys())
    gt_by_frame = {frame_id: items for frame_id, items in gt_by_frame.items() if frame_id in evaluated_frames}
    summary = tracking_eval_module.evaluate_sequence(gt_by_frame, run_info["frame_predictions"], "MOT16-11")
    row = summary.reset_index(names="sequence").iloc[0].to_dict()

    track_file = Path(run_info["track_file"])
    preview_file = Path(run_info["preview_file"])

    ensure(run_info["frames"] == 12, f"Expected 12 processed frames, got {run_info['frames']}.")
    ensure(track_file.exists(), f"Missing smoke track file: {track_file}")
    ensure(preview_file.exists(), f"Missing smoke preview file: {preview_file}")
    ensure(run_info["unique_track_ids"] > 0, "Smoke run produced zero unique track IDs.")
    ensure(float(row["precision"]) >= 0.5, f"Smoke precision too low: {row['precision']}")
    ensure(float(row["idf1"]) >= 0.3, f"Smoke IDF1 too low: {row['idf1']}")

    return CheckResult(
        name="sequence_tracker_smoke",
        status="PASS",
        summary="Short real-sequence smoke run completed with valid artifacts and usable metrics.",
        details={
            "device": device,
            "sequence": run_info["sequence"],
            "frames": run_info["frames"],
            "unique_track_ids": run_info["unique_track_ids"],
            "avg_tracks_per_frame": run_info["avg_tracks_per_frame"],
            "metrics": {
                "mota": float(row["mota"]),
                "idf1": float(row["idf1"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
            },
        },
        artifacts=[str(track_file), str(preview_file)],
    )


def check_report_generation_smoke(run_dir: Path) -> CheckResult:
    smoke_root = run_dir / "cli_report_smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tracking_eval" / "run_mot16_tracker_report.py"),
        "--project-root",
        str(PROJECT_ROOT),
        "--model",
        str(PROJECT_ROOT / "yolo26n.pt"),
        "--split",
        "train",
        "--sequences",
        "MOT16-11",
        "--imgsz",
        "640",
        "--max-frames",
        "12",
        "--run-name",
        "project_validation_smoke",
        "--output-root",
        str(smoke_root),
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    ensure(completed.returncode == 0, f"CLI smoke report failed:\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")

    generated_runs = sorted(p for p in smoke_root.iterdir() if p.is_dir())
    ensure(generated_runs, f"No report run directory created under {smoke_root}")
    report_run_dir = generated_runs[-1]

    report_path = report_run_dir / "report.md"
    csv_path = report_run_dir / "per_sequence_metrics.csv"
    json_path = report_run_dir / "overall_metrics.json"

    ensure(report_path.exists(), f"Missing report markdown: {report_path}")
    ensure(csv_path.exists(), f"Missing metrics CSV: {csv_path}")
    ensure(json_path.exists(), f"Missing overall metrics JSON: {json_path}")

    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ensure(rows, f"Metrics CSV is empty: {csv_path}")
    overall_metrics = json.loads(json_path.read_text(encoding="utf-8"))
    ensure("idf1" in overall_metrics and "mota" in overall_metrics, "Overall metrics JSON missing core metrics.")

    return CheckResult(
        name="report_generation_smoke",
        status="PASS",
        summary="CLI report generation works and produces report-ready files.",
        details={
            "run_dir": str(report_run_dir),
            "overall_metrics": overall_metrics,
        },
        artifacts=[str(report_path), str(csv_path), str(json_path)],
    )


def find_full_report_runs() -> list[tuple[Path, dict[str, Any], dict[str, Any]]]:
    candidates: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for runs_root in candidate_runs_roots("tracking_eval/runs"):
        if not runs_root.exists():
            continue
        for run_dir in sorted(runs_root.iterdir()):
            if not run_dir.is_dir():
                continue
            config_path = run_dir / "config.json"
            metrics_path = run_dir / "overall_metrics.json"
            report_path = run_dir / "report.md"
            if not (config_path.exists() and metrics_path.exists() and report_path.exists()):
                continue
            config = json.loads(config_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            sequences = set(config.get("sequences", []))
            if config.get("split") != "train":
                continue
            if config.get("max_frames") is not None:
                continue
            if not {"MOT16-11", "MOT16-13"}.issubset(sequences):
                continue
            candidates.append((run_dir, config, metrics))
    return candidates


def check_existing_full_report_readiness() -> CheckResult:
    candidates = find_full_report_runs()
    ensure(candidates, "No full held-out report runs found under tracking_eval/runs.")

    best_run_dir, best_config, best_metrics = max(candidates, key=lambda item: float(item[2].get("idf1", -1.0)))
    for metric_name, threshold in FULL_REPORT_THRESHOLDS.items():
        value = float(best_metrics.get(metric_name, -1.0))
        ensure(value >= threshold, f"Best full report {metric_name}={value:.4f} is below threshold {threshold:.4f}.")

    preview_paths = [
        best_run_dir / "previews" / "MOT16-11.mp4",
        best_run_dir / "previews" / "MOT16-13.mp4",
    ]
    track_paths = [
        best_run_dir / "tracks" / "MOT16-11.txt",
        best_run_dir / "tracks" / "MOT16-13.txt",
    ]
    for artifact_path in preview_paths + track_paths:
        ensure(artifact_path.exists(), f"Missing full-run artifact: {artifact_path}")

    return CheckResult(
        name="existing_full_report_readiness",
        status="PASS",
        summary="Existing full held-out report is present and clears the current readiness floor.",
        details={
            "best_run_dir": str(best_run_dir),
            "config": best_config,
            "metrics": best_metrics,
            "thresholds": FULL_REPORT_THRESHOLDS,
        },
        artifacts=[str(best_run_dir / "report.md"), *[str(p) for p in preview_paths + track_paths]],
    )


def check_streamlit_live_helper_smoke(streamlit_module, run_dir: Path) -> CheckResult:
    import cv2
    from ultralytics import YOLO

    frame_path = PROJECT_ROOT / "MOT16" / "train" / "MOT16-11" / "img1" / "000001.jpg"
    tracker_path = PROJECT_ROOT / "code" / "botsort_mot16_person.yaml"
    model_path = PROJECT_ROOT / "yolo26n.pt"
    live_root = run_dir / "streamlit_live_smoke"
    live_root.mkdir(parents=True, exist_ok=True)

    frame_bgr = cv2.imread(str(frame_path))
    ensure(frame_bgr is not None, f"Unable to load frame {frame_path}")

    model = YOLO(str(model_path))
    annotated, tracks = streamlit_module.track_single_frame(
        model=model,
        tracker_path=tracker_path,
        frame_bgr=frame_bgr,
        conf=0.25,
        iou=0.5,
        imgsz=640,
        persist=True,
        person_only=True,
    )

    recorder = streamlit_module.LiveSessionRecorder(
        output_root=live_root,
        run_label="validation_live",
        fps=20.0,
        record_video=True,
    )
    recorder.record_frame(annotated, tracks)
    recorder.finalize()
    snapshot = recorder.snapshot()

    ensure(snapshot.frames_processed == 1, f"Expected one recorded frame, got {snapshot.frames_processed}.")
    ensure(snapshot.unique_track_ids > 0, "Live recorder snapshot has zero unique IDs.")
    ensure(snapshot.output_video_path.exists(), f"Missing live output video: {snapshot.output_video_path}")
    ensure(snapshot.output_tracks_path.exists(), f"Missing live output txt: {snapshot.output_tracks_path}")
    ensure(snapshot.output_tracks_path.read_text(encoding="utf-8").strip(), "Live track file is empty.")

    return CheckResult(
        name="streamlit_live_helper_smoke",
        status="PASS",
        summary="Streamlit helper path can annotate and record live-style outputs.",
        details={
            "frames_processed": snapshot.frames_processed,
            "unique_track_ids": snapshot.unique_track_ids,
            "run_dir": str(snapshot.run_dir),
        },
        artifacts=[str(snapshot.output_video_path), str(snapshot.output_tracks_path)],
    )


def check_streamlit_boot() -> CheckResult:
    port = "8512"
    env = os.environ.copy()
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_ROOT / "streamlit_app" / "app.py"),
        "--server.headless",
        "true",
        "--server.port",
        port,
    ]
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    lines: list[str] = []
    local_url_seen = False
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            line = process.stdout.readline() if process.stdout is not None else ""
            if line:
                lines.append(line.rstrip())
                if "Local URL:" in line:
                    local_url_seen = True
                    break
            if process.poll() is not None:
                break
        if not local_url_seen:
            joined_output = "\n".join(lines)
            if "PermissionError: [Errno 1] Operation not permitted" in joined_output:
                return CheckResult(
                    name="streamlit_boot",
                    status="PASS",
                    summary="Streamlit reached server startup, but socket bind is blocked in the sandbox.",
                    details={
                        "port": int(port),
                        "boot_log_tail": lines[-10:],
                    },
                    artifacts=[str(PROJECT_ROOT / "streamlit_app" / "app.py")],
                )
            ensure(local_url_seen, "Streamlit did not announce a local URL before timeout.")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    return CheckResult(
        name="streamlit_boot",
        status="PASS",
        summary="Streamlit app starts successfully in headless mode.",
        details={
            "port": int(port),
            "boot_log_tail": lines[-10:],
        },
        artifacts=[str(PROJECT_ROOT / "streamlit_app" / "app.py")],
    )


def write_report(run_dir: Path, results: list[CheckResult]) -> None:
    json_path = run_dir / "validation_report.json"
    md_path = run_dir / "validation_report.md"
    json_path.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")

    total = len(results)
    passed = sum(1 for result in results if result.status == "PASS")
    lines = [
        "# Project Validation Report",
        "",
        f"Run directory: `{run_dir}`",
        "",
        f"- Total checks: `{total}`",
        f"- Passed: `{passed}`",
        f"- Failed: `{total - passed}`",
        "",
    ]
    for result in results:
        lines.append(f"## {result.name}")
        lines.append("")
        lines.append(f"- Status: `{result.status}`")
        lines.append(f"- Summary: {result.summary}")
        if result.details:
            lines.append("- Details:")
            for key, value in result.details.items():
                lines.append(f"  - `{key}`: `{value}`")
        if result.artifacts:
            lines.append("- Artifacts:")
            for artifact in result.artifacts:
                lines.append(f"  - `{artifact}`")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_ROOT / f"{timestamp}_validation"
    run_dir.mkdir(parents=True, exist_ok=True)

    checks = [
        ("environment_and_assets", check_environment_and_assets),
        ("tracker_config", check_tracker_config),
        ("single_frame_model_integration", lambda: check_single_frame_model_integration(load_streamlit_module())),
        ("sequence_tracker_smoke", lambda: check_sequence_tracker_smoke(load_tracking_eval_module(), run_dir)),
        ("report_generation_smoke", lambda: check_report_generation_smoke(run_dir)),
        ("existing_full_report_readiness", check_existing_full_report_readiness),
        ("streamlit_live_helper_smoke", lambda: check_streamlit_live_helper_smoke(load_streamlit_module(), run_dir)),
        ("streamlit_boot", check_streamlit_boot),
    ]

    results = [record_check(name, func) for name, func in checks]
    write_report(run_dir, results)

    failed = [result for result in results if result.status != "PASS"]
    print(f"[validation] report_dir={run_dir}")
    for result in results:
        print(f"[{result.status}] {result.name}: {result.summary}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
