# MOT16 Tracking Evaluation

This folder contains a report-ready tracking evaluation pipeline for the strongest practical model family currently supported by this repository:

- detector family: local YOLO checkpoints such as `yolo26n.pt` and `yolo26m.pt`
- tracker: `BoT-SORT`
- tracker config: `code/botsort_mot16_person.yaml`

## Why this setup

From the collected research, BoT-SORT is the strongest practical tracker already aligned with the current repo.
It uses:

- motion
- appearance / ReID
- camera motion compensation

This makes it much more realistic to evaluate here than attempting a full reimplementation of FastTracker or FeatureSORT from scratch.

For detector choice, do not assume the larger checkpoint is always better.
On a local 30-frame smoke slice of `MOT16-11`, `yolo26n.pt` beat `yolo26m.pt` on both MOTA and IDF1.
Use the evaluation script to choose the final detector for the report.

## Main script

- `tracking_eval/run_mot16_tracker_report.py`

## What it produces

For each run:

- MOTChallenge-format track files
- annotated preview videos
- per-sequence metrics CSV
- overall metrics JSON
- Markdown report for the class report

## Default evaluation split

Held-out train sequences with GT:

- `MOT16-11`
- `MOT16-13`

## Run

```bash
source .venv/bin/activate
.venv/bin/python tracking_eval/run_mot16_tracker_report.py \
  --project-root /Users/jainil/PycharmProjects/deep_learning_project \
  --model /Users/jainil/PycharmProjects/deep_learning_project/yolo26n.pt \
  --split train \
  --sequences MOT16-11,MOT16-13 \
  --run-name yolo26n_botsort_val
```

## Faster smoke run

```bash
source .venv/bin/activate
.venv/bin/python tracking_eval/run_mot16_tracker_report.py \
  --project-root /Users/jainil/PycharmProjects/deep_learning_project \
  --model /Users/jainil/PycharmProjects/deep_learning_project/yolo26n.pt \
  --split train \
  --sequences MOT16-11 \
  --max-frames 60 \
  --imgsz 960 \
  --run-name smoke
```

## Report outputs

Runs are written to:

- `tracking_eval/runs/<timestamp>_<name>/`

Key files:

- `report.md`
- `per_sequence_metrics.csv`
- `overall_metrics.json`
- `tracks/*.txt`
- `previews/*.mp4`
