# Tracking Evaluation Plan

Date: 2026-04-13

## Goal

Produce a strong, report-ready MOT16 tracking pipeline and evaluation package using the best feasible model already supported by this repository.

## Chosen model family

Research-backed practical choice:

- Detector family: local YOLO checkpoints (`yolo26n.pt`, `yolo26m.pt`)
- Tracker: `BoT-SORT`
- Config: `code/botsort_mot16_person.yaml`

Detector variant should be selected empirically on held-out MOT16 train sequences, not by size alone.
In a local 30-frame smoke comparison on `MOT16-11`, `yolo26n.pt` outperformed `yolo26m.pt`.

## Why this choice

- It is the closest strong method already supported by the current repo.
- BoT-SORT is directly backed by the collected research papers.
- The repo already contains:
  - Ultralytics dependency
  - local YOLO weights
  - BoT-SORT config with ReID and GMC enabled
- Full reimplementation of FastTracker or FeatureSORT is not realistic here without a large external code and weight setup.

## Deliverables

1. A script to run tracking on MOT16 sequences and export MOTChallenge-format `.txt` files.
2. A script path that computes report-ready tracking metrics from GT on held-out train sequences.
3. A Markdown report with:
   - run configuration
   - per-sequence metrics
   - overall metrics
   - artifact paths
4. Optional annotated preview videos for qualitative examples in the report.

## Evaluation split

Use held-out MOT16 train sequences for report numbers:

- `MOT16-11`
- `MOT16-13`

These have GT, so they are safe for internal evaluation and class reporting.

## Main metrics to report

- MOTA
- MOTP
- IDF1
- IDP
- IDR
- precision
- recall
- ID switches
- false positives
- false negatives

## Outputs

`tracking_eval/runs/<timestamp>_<name>/`

- `config.json`
- `report.md`
- `per_sequence_metrics.csv`
- `overall_metrics.json`
- `tracks/<sequence>.txt`
- `previews/<sequence>.mp4`

## Execution order

1. Run the tracker on validation sequences.
2. Export track files and preview videos.
3. Evaluate against GT.
4. Write summary tables for the report.
