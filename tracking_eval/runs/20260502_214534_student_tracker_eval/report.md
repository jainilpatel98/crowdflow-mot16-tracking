# Student Tracker MOT16 Evaluation Report

Run directory: `tracking_eval/runs/20260502_214534_student_tracker_eval`
Timestamp: 2026-05-02 21:46:58

## Configuration

- `student_config`: `configs/student_distill_resnet50.yaml`
- `checkpoint`: `runs/student_distill_resnet50/best.pt`
- `tracker_config`: `configs/tracker.yaml`
- `tracker_name`: `strongsort`
- `device`: `cuda`
- `split`: `train`
- `sequences`: ['MOT16-11', 'MOT16-13']

## Sequences

- `MOT16-11`: frames=900, unique_ids=239
- `MOT16-13`: frames=750, unique_ids=117

## Per-Sequence Metrics

| mota | motp | idf1 | idp | idr | precision | recall | num_switches | num_false_positives | num_misses | num_objects |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3958 | 0.2071 | 0.3995 | 0.4082 | 0.3911 | 0.7201 | 0.6900 | 239.0000 | 2460.0000 | 2844.0000 | 9174.0000 |
| 0.1301 | 0.2913 | 0.2799 | 0.5919 | 0.1833 | 0.7279 | 0.2254 | 126.0000 | 965.0000 | 8869.0000 | 11450.0000 |

## Overall Metrics

- `mota`: `0.2630`
- `motp`: `0.2492`
- `idf1`: `0.3397`
- `idp`: `0.5001`
- `idr`: `0.2872`
- `precision`: `0.7240`
- `recall`: `0.4577`
- `num_switches`: `182.5000`
- `num_false_positives`: `1712.5000`
- `num_misses`: `5856.5000`
- `num_objects`: `10312.0000`

## Artifact Paths

- Tracks: `tracking_eval/runs/20260502_214534_student_tracker_eval/tracks`
- Previews: `tracking_eval/runs/20260502_214534_student_tracker_eval/previews`
- Per-sequence metrics: `tracking_eval/runs/20260502_214534_student_tracker_eval/per_sequence_metrics.csv`
- Overall metrics: `tracking_eval/runs/20260502_214534_student_tracker_eval/overall_metrics.json`