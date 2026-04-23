# MOT16 Tracking Evaluation

This folder contains a report-ready tracking evaluation pipeline for the strongest practical detector family currently supported by this repository:

- detector family: local YOLO checkpoints such as `yolo26n.pt`, `yolo26m.pt`, and `code/yolo26x.pt`
- tracker: `BoT-SORT`
- tracker config: `code/botsort_mot16_person.yaml`

## Why this setup

The active server repo already has working YOLO tracking assets and TrackEval-compatible exports. The laptop code adds a cleaner held-out evaluation layer on top of that so checkpoints can be compared on the same MOT16 validation sequences instead of by ad hoc notebook inspection.

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
.venv/bin/python tracking_eval/run_mot16_tracker_report.py \
  --project-root /home/research/tracking_crowded_people \
  --model /home/research/tracking_crowded_people/yolo26n.pt \
  --split train \
  --sequences MOT16-11,MOT16-13 \
  --run-name yolo26n_botsort_val
```

## Faster smoke run

```bash
.venv/bin/python tracking_eval/run_mot16_tracker_report.py \
  --project-root /home/research/tracking_crowded_people \
  --model /home/research/tracking_crowded_people/yolo26n.pt \
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
