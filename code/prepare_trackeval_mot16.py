#!/usr/bin/env python3
"""Prepare MOT16 train outputs for TrackEval.

This script does not compute metrics itself. It builds the folder structure
expected by TrackEval so you can run standard MOT metrics including HOTA,
CLEAR, and Identity metrics on the train split.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


TRAIN_SEQUENCES = [
    "MOT16-02",
    "MOT16-04",
    "MOT16-05",
    "MOT16-09",
    "MOT16-10",
    "MOT16-11",
    "MOT16-13",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("MOT16"))
    parser.add_argument(
        "--tracker-output-root",
        type=Path,
        default=Path("outputs/yolo_mot16_multigpu"),
        help="Folder that contains one subdirectory per sequence with <seq>.txt inside.",
    )
    parser.add_argument(
        "--tracker-name",
        default="yolo26x_botsort_person",
        help="Tracker name to expose to TrackEval.",
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path("outputs/trackeval_bundle"),
        help="Where to create the TrackEval-compatible directory structure.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks.",
    )
    parser.add_argument(
        "--trackeval-root",
        type=Path,
        default=None,
        help="Optional local TrackEval repository root. If provided, the script prints a ready-to-run command.",
    )
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    tracker_output_root = args.tracker_output_root.resolve()
    bundle_root = args.bundle_root.resolve()

    gt_root = bundle_root / "data" / "gt" / "mot_challenge"
    trackers_root = bundle_root / "data" / "trackers" / "mot_challenge"
    benchmark_root = gt_root / "MOT16-train"
    tracker_root = trackers_root / "MOT16-train" / args.tracker_name / "data"
    seqmaps_root = gt_root / "seqmaps"

    missing = []
    for sequence in TRAIN_SEQUENCES:
        gt_txt = dataset_root / "train" / sequence / "gt" / "gt.txt"
        seqinfo = dataset_root / "train" / sequence / "seqinfo.ini"
        pred_txt = tracker_output_root / sequence / f"{sequence}.txt"
        if not gt_txt.exists():
            missing.append(str(gt_txt))
        if not seqinfo.exists():
            missing.append(str(seqinfo))
        if not pred_txt.exists():
            missing.append(str(pred_txt))

    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    for sequence in TRAIN_SEQUENCES:
        gt_txt = dataset_root / "train" / sequence / "gt" / "gt.txt"
        seqinfo = dataset_root / "train" / sequence / "seqinfo.ini"
        pred_txt = tracker_output_root / sequence / f"{sequence}.txt"

        link_or_copy(gt_txt, benchmark_root / sequence / "gt" / "gt.txt", args.copy)
        link_or_copy(seqinfo, benchmark_root / sequence / "seqinfo.ini", args.copy)
        link_or_copy(pred_txt, tracker_root / f"{sequence}.txt", args.copy)

    seqmaps_root.mkdir(parents=True, exist_ok=True)
    seqmap_path = seqmaps_root / "MOT16-train.txt"
    seqmap_path.write_text("name\n" + "\n".join(TRAIN_SEQUENCES) + "\n", encoding="utf-8")

    print("Prepared TrackEval bundle:")
    print(f"  GT root: {gt_root}")
    print(f"  Tracker root: {trackers_root}")
    print(f"  Seqmap: {seqmap_path}")

    if args.trackeval_root:
        trackeval_root = args.trackeval_root.resolve()
        command = (
            f"python {trackeval_root / 'scripts' / 'run_mot_challenge.py'} "
            f"--GT_FOLDER {gt_root} "
            f"--TRACKERS_FOLDER {trackers_root} "
            f"--BENCHMARK MOT16 "
            f"--SPLIT_TO_EVAL train "
            f"--TRACKERS_TO_EVAL {args.tracker_name} "
            f"--METRICS HOTA CLEAR Identity"
        )
        print("\nRun TrackEval with:")
        print(command)
    else:
        print("\nNext step:")
        print("  Run TrackEval's run_mot_challenge.py against the generated bundle.")
        print("  Metrics to request: HOTA CLEAR Identity")

    print("\nLocal evaluation is only possible for MOT16 train sequences.")
    print("For MOT16 test sequences, submit the tracker outputs to the MOTChallenge server.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
