from __future__ import annotations

import configparser
import json
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import streamlit as st
from ultralytics import YOLO

try:
    import av
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
except ImportError:
    av = None
    VideoProcessorBase = object  # type: ignore[assignment]
    WebRtcMode = None
    webrtc_streamer = None


@dataclass(frozen=True)
class TrackingConfig:
    model_path: Path
    tracker_path: Path
    conf: float
    iou: float
    imgsz: int
    persist: bool
    person_only: bool


@dataclass
class RunResult:
    output_video_path: Path
    output_tracks_path: Path
    frames_processed: int
    unique_track_ids: int
    avg_tracks_per_frame: float
    fps_used: float
    run_dir: Path


@dataclass
class LiveSnapshot:
    output_video_path: Path
    output_tracks_path: Path
    frames_processed: int
    unique_track_ids: int
    avg_tracks_per_frame: float
    fps_used: float
    run_dir: Path
    record_video: bool
    finalized: bool
    last_error: str | None


SHOWCASE_SEQUENCES = {"MOT16-11", "MOT16-13"}


class LiveSessionRecorder:
    def __init__(self, *, output_root: Path, run_label: str, fps: float, record_video: bool) -> None:
        run_id = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = output_root / f"{run_label}_{run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.output_video_path = self.run_dir / "tracked_live_output.mp4"
        self.output_tracks_path = self.run_dir / "tracks_live_mot.txt"
        self.fps = fps
        self.record_video = record_video

        self._lock = threading.Lock()
        self._writer: cv2.VideoWriter | None = None
        self._track_file = self.output_tracks_path.open("w", encoding="utf-8")
        self._frames_processed = 0
        self._track_count_sum = 0
        self._unique_ids: set[int] = set()
        self._finalized = False
        self._last_error: str | None = None

    def record_frame(self, annotated_frame, tracks: list[tuple[int, list[int], float]]) -> None:
        with self._lock:
            if self._finalized:
                return

            if self.record_video and self._writer is None:
                height, width = annotated_frame.shape[:2]
                self._writer = create_video_writer(
                    output_video_path=self.output_video_path,
                    fps=self.fps,
                    width=width,
                    height=height,
                )

            if self._writer is not None:
                self._writer.write(annotated_frame)

            frame_idx_1_based = self._frames_processed + 1
            frame_track_count = 0
            for track_id, xyxy, conf_score in tracks:
                if track_id < 0:
                    continue
                frame_track_count += 1
                self._unique_ids.add(track_id)
                self._track_file.write(write_mot_line(frame_idx_1_based, track_id, xyxy, conf_score) + "\n")

            self._track_file.flush()
            self._frames_processed += 1
            self._track_count_sum += frame_track_count
            self._last_error = None

    def set_error(self, message: str) -> None:
        with self._lock:
            self._last_error = message

    def finalize(self) -> None:
        with self._lock:
            if self._finalized:
                return

            if self._writer is not None:
                self._writer.release()
                self._writer = None

            if not self._track_file.closed:
                self._track_file.close()

            self._finalized = True

    def snapshot(self) -> LiveSnapshot:
        with self._lock:
            avg_tracks_per_frame = (
                float(self._track_count_sum / self._frames_processed) if self._frames_processed else 0.0
            )
            return LiveSnapshot(
                output_video_path=self.output_video_path,
                output_tracks_path=self.output_tracks_path,
                frames_processed=self._frames_processed,
                unique_track_ids=len(self._unique_ids),
                avg_tracks_per_frame=avg_tracks_per_frame,
                fps_used=self.fps,
                run_dir=self.run_dir,
                record_video=self.record_video,
                finalized=self._finalized,
                last_error=self._last_error,
            )


class LiveTrackingProcessor(VideoProcessorBase):
    def __init__(self, tracking_config: TrackingConfig, session_recorder: LiveSessionRecorder) -> None:
        self.tracking_config = tracking_config
        self.session_recorder = session_recorder
        self.model = YOLO(str(tracking_config.model_path))

    def recv(self, frame):
        frame_bgr = frame.to_ndarray(format="bgr24")
        try:
            annotated_frame, tracks = track_single_frame(
                model=self.model,
                tracker_path=self.tracking_config.tracker_path,
                frame_bgr=frame_bgr,
                conf=self.tracking_config.conf,
                iou=self.tracking_config.iou,
                imgsz=self.tracking_config.imgsz,
                persist=self.tracking_config.persist,
                person_only=self.tracking_config.person_only,
            )
            self.session_recorder.record_frame(annotated_frame, tracks)
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
        except Exception as exc:  # pragma: no cover - runtime safeguard
            self.session_recorder.set_error(str(exc))
            error_frame = frame_bgr.copy()
            cv2.putText(
                error_frame,
                "Live tracking error",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                error_frame,
                str(exc)[:100],
                (20, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            return av.VideoFrame.from_ndarray(error_frame, format="bgr24")


def detect_project_root(start: Path) -> Path:
    if (start / "MOT16").exists():
        return start
    if (start.parent / "MOT16").exists():
        return start.parent
    return start


def list_available_models(project_root: Path) -> list[Path]:
    model_candidates = sorted(project_root.glob("*.pt")) + sorted((project_root / "code").glob("*.pt"))
    seen: set[Path] = set()
    unique_models: list[Path] = []
    for path in model_candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_models.append(resolved)
    return unique_models


def mot16_sequence_fps(sequence_dir: Path) -> float:
    seqinfo_path = sequence_dir / "seqinfo.ini"
    parser = configparser.ConfigParser()
    parser.read(seqinfo_path)
    return float(parser.getint("Sequence", "frameRate", fallback=30))


def draw_tracks(frame_bgr, boxes) -> tuple[Any, list[tuple[int, list[int], float]]]:
    frame = frame_bgr.copy()
    drawn_tracks: list[tuple[int, list[int], float]] = []

    if boxes is None or boxes.xyxy is None:
        return frame, drawn_tracks

    xyxy_list = boxes.xyxy.int().tolist()
    ids = boxes.id.int().tolist() if boxes.id is not None else [-1] * len(xyxy_list)
    confs = boxes.conf.tolist() if boxes.conf is not None else [0.0] * len(xyxy_list)

    for coords, track_id, conf in zip(xyxy_list, ids, confs):
        x1, y1, x2, y2 = coords
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 80, 0), 2)
        label = f"ID {track_id}" if track_id >= 0 else "ID ?"
        cv2.putText(
            frame,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        drawn_tracks.append((track_id, [x1, y1, x2, y2], float(conf)))

    return frame, drawn_tracks


def create_video_writer(*, output_video_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))


def first_tracking_result(results: Any):
    if results is None:
        return None
    if isinstance(results, list):
        return results[0] if results else None
    try:
        iterator = iter(results)
    except TypeError:
        return results
    return next(iterator, None)


def track_single_frame(
    *,
    model: YOLO,
    tracker_path: Path,
    frame_bgr,
    conf: float,
    iou: float,
    imgsz: int,
    persist: bool,
    person_only: bool,
) -> tuple[Any, list[tuple[int, list[int], float]]]:
    results = model.track(
        source=[frame_bgr],
        tracker=str(tracker_path),
        classes=[0] if person_only else None,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        persist=persist,
        verbose=False,
    )
    first_result = first_tracking_result(results)
    return draw_tracks(frame_bgr, None if first_result is None else first_result.boxes)


def write_mot_line(frame_idx_1_based: int, track_id: int, xyxy: list[int], conf: float) -> str:
    x1, y1, x2, y2 = xyxy
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    confidence = f"{conf:.6f}"
    return f"{frame_idx_1_based},{track_id},{x1},{y1},{width},{height},{confidence},-1,-1,-1"


def run_tracking(
    *,
    model_path: Path,
    tracker_path: Path,
    source: str | list[str],
    fps: float,
    conf: float,
    iou: float,
    imgsz: int,
    persist: bool,
    person_only: bool,
    max_frames: int | None,
    output_root: Path,
    run_label: str,
    progress_callback,
) -> RunResult:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{run_label}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    output_video_path = run_dir / "tracked_output.mp4"
    output_tracks_path = run_dir / "tracks_mot.txt"

    model = YOLO(str(model_path))

    tracking_results = model.track(
        source=source,
        tracker=str(tracker_path),
        classes=[0] if person_only else None,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        persist=persist,
        stream=True,
        verbose=False,
    )

    total_frames = estimate_source_frame_budget(source, max_frames)
    writer = None
    frame_count = 0
    track_count_sum = 0
    unique_ids: set[int] = set()
    mot_lines: list[str] = []

    for idx, result in enumerate(tracking_results, start=1):
        if max_frames is not None and idx > max_frames:
            break

        frame_bgr = result.orig_img
        annotated_frame, tracks = draw_tracks(frame_bgr, result.boxes)

        if writer is None:
            height, width = annotated_frame.shape[:2]
            writer = create_video_writer(
                output_video_path=output_video_path,
                fps=fps,
                width=width,
                height=height,
            )

        writer.write(annotated_frame)

        frame_track_count = 0
        for track_id, xyxy, conf_score in tracks:
            if track_id < 0:
                continue
            unique_ids.add(track_id)
            frame_track_count += 1
            mot_lines.append(write_mot_line(idx, track_id, xyxy, conf_score))

        track_count_sum += frame_track_count
        frame_count += 1
        progress_callback(frame_count, total_frames)

    if writer is not None:
        writer.release()

    output_tracks_path.write_text("\n".join(mot_lines), encoding="utf-8")

    avg_tracks_per_frame = float(track_count_sum / frame_count) if frame_count else 0.0
    return RunResult(
        output_video_path=output_video_path,
        output_tracks_path=output_tracks_path,
        frames_processed=frame_count,
        unique_track_ids=len(unique_ids),
        avg_tracks_per_frame=avg_tracks_per_frame,
        fps_used=fps,
        run_dir=run_dir,
    )


def mot16_source_paths(sequence_dir: Path) -> list[str]:
    return [str(path) for path in sorted((sequence_dir / "img1").glob("*.jpg"))]


def video_source_fps(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    capture.release()
    return fps if fps > 0 else 30.0


def estimate_source_frame_budget(source: str | list[str], max_frames: int | None) -> int | None:
    if isinstance(source, list):
        total_frames = len(source)
    else:
        capture = cv2.VideoCapture(str(source))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        capture.release()
        if total_frames <= 0:
            return max_frames if max_frames is not None else None

    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    return total_frames if total_frames > 0 else None


def safe_name(raw: str) -> str:
    allowed = [c if c.isalnum() or c in "-_" else "_" for c in raw.strip()]
    name = "".join(allowed).strip("_")
    return name or "run"


def load_json_file(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def candidate_run_roots(project_root: Path, relative_dir: str) -> list[Path]:
    roots = [
        project_root / relative_dir,
        project_root / "code_from_laptop" / relative_dir,
    ]
    unique_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_roots.append(root)
    return unique_roots


def find_best_tracking_report(project_root: Path) -> dict[str, Any] | None:
    best_run: dict[str, Any] | None = None
    for runs_root in candidate_run_roots(project_root, "tracking_eval/runs"):
        if not runs_root.exists():
            continue

        for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
            config_path = run_dir / "config.json"
            metrics_path = run_dir / "overall_metrics.json"
            report_path = run_dir / "report.md"
            metrics_csv_path = run_dir / "per_sequence_metrics.csv"
            if not (config_path.exists() and metrics_path.exists() and report_path.exists() and metrics_csv_path.exists()):
                continue

            config = load_json_file(config_path)
            metrics = load_json_file(metrics_path)
            if not isinstance(config, dict) or not isinstance(metrics, dict):
                continue

            sequences = {str(sequence) for sequence in config.get("sequences", [])}
            if config.get("split") != "train" or config.get("max_frames") is not None:
                continue
            if not SHOWCASE_SEQUENCES.issubset(sequences):
                continue

            try:
                idf1_score = float(metrics.get("idf1", -1.0))
            except (TypeError, ValueError):
                continue

            preview_paths = {
                sequence: run_dir / "previews" / f"{sequence}.mp4"
                for sequence in sorted(SHOWCASE_SEQUENCES)
            }
            candidate = {
                "run_dir": run_dir,
                "metrics": metrics,
                "report_path": report_path,
                "metrics_csv_path": metrics_csv_path,
                "metrics_json_path": metrics_path,
                "preview_paths": preview_paths,
                "idf1_score": idf1_score,
            }
            if best_run is None or candidate["idf1_score"] > best_run["idf1_score"]:
                best_run = candidate

    return best_run


def find_latest_validation_summary(project_root: Path) -> dict[str, Any] | None:
    validation_dirs: list[Path] = []
    for runs_root in candidate_run_roots(project_root, "project_validation/runs"):
        if not runs_root.exists():
            continue
        validation_dirs.extend(
            path for path in runs_root.iterdir() if path.is_dir() and path.name.endswith("_validation")
        )

    if not validation_dirs:
        return None

    run_dir = sorted(validation_dirs, key=lambda path: path.name)[-1]
    report_path = run_dir / "validation_report.json"
    results = load_json_file(report_path)
    if not isinstance(results, list):
        return None

    passed = sum(1 for item in results if isinstance(item, dict) and item.get("status") == "PASS")
    failed_items = [
        item.get("name", "unknown")
        for item in results
        if isinstance(item, dict) and item.get("status") != "PASS"
    ]
    return {
        "run_dir": run_dir,
        "report_path": report_path,
        "total": len(results),
        "passed": passed,
        "failed": len(failed_items),
        "failed_items": failed_items,
    }


def render_project_readiness(project_root: Path) -> None:
    st.subheader("Class Demo Readiness")

    latest_validation = find_latest_validation_summary(project_root)
    if latest_validation is None:
        st.info("No validation run has been recorded yet.")
    elif latest_validation["failed"] == 0:
        st.success(
            f"Latest validation is green: {latest_validation['passed']} / {latest_validation['total']} checks passed."
        )
    else:
        st.warning(
            f"Latest validation has {latest_validation['failed']} failing checks: "
            f"{', '.join(latest_validation['failed_items'])}"
        )

    best_report = find_best_tracking_report(project_root)
    if best_report is None:
        st.warning("No full held-out MOT16 report is available yet.")
        return

    metrics = best_report["metrics"]
    metric_cols = st.columns(4)
    metric_cols[0].metric("MOTA", f"{float(metrics.get('mota', 0.0)):.4f}")
    metric_cols[1].metric("IDF1", f"{float(metrics.get('idf1', 0.0)):.4f}")
    metric_cols[2].metric("Precision", f"{float(metrics.get('precision', 0.0)):.4f}")
    metric_cols[3].metric("Recall", f"{float(metrics.get('recall', 0.0)):.4f}")

    st.caption(f"Best held-out report found at `{best_report['run_dir']}`.")
    st.write(f"Report Markdown: `{best_report['report_path']}`")
    st.write(f"Per-sequence metrics CSV: `{best_report['metrics_csv_path']}`")
    if latest_validation is not None:
        st.write(f"Latest validation report: `{latest_validation['report_path']}`")

    with st.expander("Preview current showcase artifacts", expanded=False):
        preview_cols = st.columns(2)
        for idx, sequence in enumerate(sorted(best_report["preview_paths"])):
            preview_path = best_report["preview_paths"][sequence]
            with preview_cols[idx % 2]:
                st.markdown(f"**{sequence}**")
                if preview_path.exists():
                    st.video(str(preview_path))
                else:
                    st.caption(f"Preview not found: `{preview_path}`")

    with st.expander("Download report files", expanded=False):
        st.download_button(
            label="Download best report (.md)",
            data=best_report["report_path"].read_bytes(),
            file_name=best_report["report_path"].name,
            mime="text/markdown",
        )
        st.download_button(
            label="Download per-sequence metrics (.csv)",
            data=best_report["metrics_csv_path"].read_bytes(),
            file_name=best_report["metrics_csv_path"].name,
            mime="text/csv",
        )
        st.download_button(
            label="Download overall metrics (.json)",
            data=best_report["metrics_json_path"].read_bytes(),
            file_name=best_report["metrics_json_path"].name,
            mime="application/json",
        )


def render_run_summary(result: RunResult) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Frames Processed", result.frames_processed)
    col2.metric("Unique Track IDs", result.unique_track_ids)
    col3.metric("Avg Tracks / Frame", f"{result.avg_tracks_per_frame:.2f}")

    st.write(f"Run directory: `{result.run_dir}`")
    st.write(f"FPS used for output: `{result.fps_used:.2f}`")

    st.subheader("Tracked Video")
    if result.output_video_path.exists():
        st.video(str(result.output_video_path))
    else:
        st.warning("Output video file was not created.")

    st.subheader("Download Outputs")
    if result.output_video_path.exists():
        st.download_button(
            label="Download tracked video (.mp4)",
            data=result.output_video_path.read_bytes(),
            file_name=result.output_video_path.name,
            mime="video/mp4",
        )
    if result.output_tracks_path.exists():
        st.download_button(
            label="Download track IDs (.txt, MOT format)",
            data=result.output_tracks_path.read_bytes(),
            file_name=result.output_tracks_path.name,
            mime="text/plain",
        )

    st.subheader("Track File Preview")
    if result.output_tracks_path.exists():
        preview_lines = result.output_tracks_path.read_text(encoding="utf-8").splitlines()[:20]
        st.code("\n".join(preview_lines) if preview_lines else "(no tracks written)", language="text")


def render_live_summary(snapshot: LiveSnapshot) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Frames Processed", snapshot.frames_processed)
    col2.metric("Unique Track IDs", snapshot.unique_track_ids)
    col3.metric("Avg Tracks / Frame", f"{snapshot.avg_tracks_per_frame:.2f}")

    st.write(f"Live run directory: `{snapshot.run_dir}`")
    st.write(f"FPS used for live recording: `{snapshot.fps_used:.2f}`")
    if snapshot.last_error:
        st.warning(f"Latest live error: {snapshot.last_error}")

    if snapshot.frames_processed == 0:
        st.caption("Start the live stream and allow camera access to begin tracking.")
        return

    st.subheader("Live Output Files")
    if snapshot.record_video:
        if snapshot.output_video_path.exists() and snapshot.finalized:
            st.video(str(snapshot.output_video_path))
            st.download_button(
                label="Download annotated live video (.mp4)",
                data=snapshot.output_video_path.read_bytes(),
                file_name=snapshot.output_video_path.name,
                mime="video/mp4",
            )
        else:
            st.caption("Annotated live video becomes downloadable after you stop the stream.")
    else:
        st.caption("Annotated MP4 recording is disabled for this live run.")

    if snapshot.output_tracks_path.exists():
        st.download_button(
            label="Download live track IDs (.txt, MOT format)",
            data=snapshot.output_tracks_path.read_bytes(),
            file_name=snapshot.output_tracks_path.name,
            mime="text/plain",
        )
        preview_lines = snapshot.output_tracks_path.read_text(encoding="utf-8").splitlines()[:20]
        st.code("\n".join(preview_lines) if preview_lines else "(no tracks written)", language="text")


def prepare_live_session(
    *,
    output_root: Path,
    run_label: str,
    fps: float,
    record_video: bool,
    tracking_config: TrackingConfig,
) -> tuple[LiveSessionRecorder, int]:
    session_recorder = LiveSessionRecorder(
        output_root=output_root,
        run_label=run_label,
        fps=fps,
        record_video=record_video,
    )
    st.session_state["live_session_recorder"] = session_recorder
    st.session_state["live_tracking_config"] = tracking_config
    st.session_state["live_webrtc_key"] = st.session_state.get("live_webrtc_key", 0) + 1
    return session_recorder, st.session_state["live_webrtc_key"]


def app() -> None:
    st.set_page_config(page_title="MOT16 People Tracking", layout="wide")
    st.title("MOT16 People Tracking - Streamlit App")
    st.caption(
        "Pedestrian tracking for MOT16-style workflows: live webcam, uploaded video, or MOT16 sequences."
    )

    script_dir = Path(__file__).resolve().parent
    project_root = detect_project_root(script_dir.parent)
    output_root = script_dir / "runs"
    output_root.mkdir(parents=True, exist_ok=True)
    render_project_readiness(project_root)
    st.divider()

    st.sidebar.header("Configuration")

    available_models = list_available_models(project_root)
    if not available_models:
        st.error("No `.pt` model files found in project root or `code/`.")
        st.stop()

    default_tracker = project_root / "code" / "botsort_mot16_person.yaml"
    default_model_idx = 0
    for idx, model_path in enumerate(available_models):
        if model_path.name == "yolo26n.pt":
            default_model_idx = idx
            break

    model_path = st.sidebar.selectbox(
        "Model (.pt)",
        options=available_models,
        index=default_model_idx,
        format_func=lambda p: str(Path(p).relative_to(project_root)),
    )

    tracker_path_text = st.sidebar.text_input(
        "Tracker config path",
        value=str(default_tracker),
    )
    tracker_path = Path(tracker_path_text).expanduser().resolve()

    source_mode = st.sidebar.radio(
        "Input source",
        options=["Live webcam", "Upload video", "MOT16 sequence"],
        index=0,
    )
    person_only = st.sidebar.checkbox("Person class only (class=0)", value=True)
    conf = st.sidebar.slider("Confidence threshold", min_value=0.05, max_value=0.9, value=0.25, step=0.05)
    iou = st.sidebar.slider("IoU threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
    imgsz = st.sidebar.select_slider("Image size", options=[640, 960, 1280, 1536], value=960)
    persist = st.sidebar.checkbox("Persist tracks across frames", value=True)
    run_name = st.sidebar.text_input("Run label", value="tracking")

    if not tracker_path.exists():
        st.error(f"Tracker config not found: {tracker_path}")
        st.stop()
    if not Path(model_path).exists():
        st.error(f"Model not found: {model_path}")
        st.stop()

    tracking_config = TrackingConfig(
        model_path=Path(model_path),
        tracker_path=tracker_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        persist=persist,
        person_only=person_only,
    )

    if source_mode == "Live webcam":
        if webrtc_streamer is None or WebRtcMode is None or av is None:
            st.error(
                "Live webcam mode requires `streamlit-webrtc`. Install app dependencies with "
                "`python -m pip install -r streamlit_app/requirements.txt`."
            )
            st.stop()

        live_record_video = st.sidebar.checkbox("Record annotated live video", value=True)
        live_output_fps = st.sidebar.slider("Live recording FPS", min_value=10, max_value=30, value=20, step=1)

        st.info(
            "This live mode is for person tracking only. After changing live settings, click "
            "`Prepare New Live Session` so the webcam stream uses the updated configuration."
        )

        live_run_label = safe_name(f"{run_name}_live")
        reset_live_session = st.button("Prepare New Live Session", type="primary")

        if "live_session_recorder" not in st.session_state or "live_tracking_config" not in st.session_state:
            live_session_recorder, live_webrtc_key = prepare_live_session(
                output_root=output_root,
                run_label=live_run_label,
                fps=float(live_output_fps),
                record_video=live_record_video,
                tracking_config=tracking_config,
            )
        elif reset_live_session:
            previous_live_session = st.session_state["live_session_recorder"]
            previous_live_session.finalize()
            live_session_recorder, live_webrtc_key = prepare_live_session(
                output_root=output_root,
                run_label=live_run_label,
                fps=float(live_output_fps),
                record_video=live_record_video,
                tracking_config=tracking_config,
            )
        else:
            live_session_recorder = st.session_state["live_session_recorder"]
            live_webrtc_key = st.session_state.get("live_webrtc_key", 1)

        active_live_config = st.session_state.get("live_tracking_config")
        if active_live_config != tracking_config:
            st.warning("Live settings changed. Click `Prepare New Live Session` to apply them to the webcam.")

        st.caption(
            "Camera access in Streamlit works on `localhost` or over HTTPS. Stop the stream to finalize the MP4."
        )

        webrtc_ctx = webrtc_streamer(
            key=f"live-tracker-{live_webrtc_key}",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=lambda: LiveTrackingProcessor(
                tracking_config=st.session_state["live_tracking_config"],
                session_recorder=st.session_state["live_session_recorder"],
            ),
            async_processing=True,
        )

        live_status = st.empty()

        if webrtc_ctx.state.playing:
            while webrtc_ctx.state.playing:
                live_status.container()
                with live_status.container():
                    render_live_summary(st.session_state["live_session_recorder"].snapshot())
                time.sleep(1.0)
        else:
            snapshot = st.session_state["live_session_recorder"].snapshot()
            if snapshot.frames_processed > 0 and not snapshot.finalized:
                st.session_state["live_session_recorder"].finalize()
                snapshot = st.session_state["live_session_recorder"].snapshot()
            with live_status.container():
                render_live_summary(snapshot)
        return

    source: str | list[str]
    source_fps: float
    run_label = safe_name(run_name)

    max_frames_enabled = st.sidebar.checkbox("Limit number of frames", value=True)
    max_frames = (
        st.sidebar.number_input("Max frames", min_value=10, max_value=10000, value=300, step=10)
        if max_frames_enabled
        else None
    )

    if source_mode == "MOT16 sequence":
        mot_root = project_root / "MOT16"
        if not mot_root.exists():
            st.error(f"MOT16 folder not found at {mot_root}")
            st.stop()

        split = st.selectbox("Split", options=["train", "test"], index=1)
        split_dir = mot_root / split
        sequences = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])
        sequence = st.selectbox("Sequence", options=sequences, index=0)
        sequence_dir = split_dir / sequence

        source = mot16_source_paths(sequence_dir)
        if not source:
            st.error(f"No frames found in {sequence_dir / 'img1'}")
            st.stop()
        source_fps = mot16_sequence_fps(sequence_dir)
        st.info(f"Selected `{split}/{sequence}` with {len(source)} frames at {source_fps:.2f} FPS.")
        run_label = safe_name(f"{run_label}_{split}_{sequence}")
    else:
        uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded is None:
            st.warning("Upload a video file to run tracking.")
            st.stop()

        temp_video = Path(tempfile.gettempdir()) / f"streamlit_track_{int(time.time())}_{uploaded.name}"
        temp_video.write_bytes(uploaded.getbuffer())
        source = str(temp_video)
        source_fps = video_source_fps(temp_video)
        st.info(f"Uploaded video: `{uploaded.name}` at detected {source_fps:.2f} FPS.")
        run_label = safe_name(f"{run_label}_{Path(uploaded.name).stem}")

    run_button = st.button("Run Tracking", type="primary")

    if run_button:
        progress = st.progress(0)
        progress_status = st.empty()
        frame_budget = estimate_source_frame_budget(source, int(max_frames) if max_frames is not None else None)

        def on_progress(current: int, total_frames: int | None) -> None:
            if total_frames:
                percent_complete = min(99, int((current / total_frames) * 100))
                progress.progress(percent_complete)
                progress_status.caption(f"Processed {current} / {total_frames} frames")
            else:
                progress.progress(min(95, max(1, current)))
                progress_status.caption(f"Processed {current} frames")

        with st.spinner("Running tracking..."):
            result = run_tracking(
                model_path=tracking_config.model_path,
                tracker_path=tracking_config.tracker_path,
                source=source,
                fps=source_fps,
                conf=tracking_config.conf,
                iou=tracking_config.iou,
                imgsz=tracking_config.imgsz,
                persist=tracking_config.persist,
                person_only=tracking_config.person_only,
                max_frames=int(max_frames) if max_frames is not None else None,
                output_root=output_root,
                run_label=run_label,
                progress_callback=on_progress,
            )

        progress.progress(100)
        if frame_budget:
            progress_status.caption(f"Processed {result.frames_processed} / {frame_budget} frames")
        else:
            progress_status.caption(f"Processed {result.frames_processed} frames")
        st.success("Tracking run completed.")
        render_run_summary(result)


if __name__ == "__main__":
    app()
