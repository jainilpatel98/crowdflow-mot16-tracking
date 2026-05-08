# Student Tracker MOT16 Evaluation Report

Run directory: `tracking_eval/runs/20260502_215227_student_tracker_eval`
Timestamp: 2026-05-02 21:56:20

## Configuration

- `student_config`: `configs/student_distill_resnet50.yaml`
- `checkpoint`: `runs/student_distill_resnet50/best.pt`
- `tracker_config`: `configs/tracker.yaml`
- `tracker_name`: `strongsort`
- `device`: `cuda`
- `split`: `train`
- `sequences`: ['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09', 'MOT16-10', 'MOT16-11', 'MOT16-13']

## Sequences

- `MOT16-02`: frames=600, unique_ids=139
- `MOT16-04`: frames=1050, unique_ids=131
- `MOT16-05`: frames=837, unique_ids=242
- `MOT16-09`: frames=525, unique_ids=126
- `MOT16-10`: frames=654, unique_ids=130
- `MOT16-11`: frames=900, unique_ids=238
- `MOT16-13`: frames=750, unique_ids=117

## Per-Sequence Metrics

| mota | motp | idf1 | idp | idr | precision | recall | num_switches | num_false_positives | num_misses | num_objects |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0795 | 0.2786 | 0.2144 | 0.3817 | 0.1491 | 0.6210 | 0.2426 | 268.0000 | 2640.0000 | 13507.0000 | 17833.0000 |
| 0.3119 | 0.2120 | 0.4579 | 0.8324 | 0.3158 | 0.9152 | 0.3472 | 149.0000 | 1530.0000 | 31043.0000 | 47557.0000 |
| 0.3192 | 0.2772 | 0.4743 | 0.5132 | 0.4409 | 0.7115 | 0.6112 | 301.0000 | 1690.0000 | 2651.0000 | 6818.0000 |
| 0.4369 | 0.2549 | 0.3604 | 0.3761 | 0.3460 | 0.7725 | 0.7107 | 339.0000 | 1100.0000 | 1521.0000 | 5257.0000 |
| 0.2127 | 0.2738 | 0.3245 | 0.4705 | 0.2477 | 0.7233 | 0.3807 | 276.0000 | 1794.0000 | 7628.0000 | 12318.0000 |
| 0.3979 | 0.2071 | 0.4037 | 0.4124 | 0.3954 | 0.7199 | 0.6901 | 218.0000 | 2463.0000 | 2843.0000 | 9174.0000 |
| 0.1301 | 0.2913 | 0.2799 | 0.5919 | 0.1833 | 0.7279 | 0.2254 | 126.0000 | 965.0000 | 8869.0000 | 11450.0000 |

## Overall Metrics

- `mota`: `0.2697`
- `motp`: `0.2564`
- `idf1`: `0.3593`
- `idp`: `0.5112`
- `idr`: `0.2969`
- `precision`: `0.7416`
- `recall`: `0.4583`
- `num_switches`: `239.5714`
- `num_false_positives`: `1740.2857`
- `num_misses`: `9723.1429`
- `num_objects`: `15772.4286`

## Artifact Paths

- Tracks: `tracking_eval/runs/20260502_215227_student_tracker_eval/tracks`
- Previews: `tracking_eval/runs/20260502_215227_student_tracker_eval/previews`
- Per-sequence metrics: `tracking_eval/runs/20260502_215227_student_tracker_eval/per_sequence_metrics.csv`
- Overall metrics: `tracking_eval/runs/20260502_215227_student_tracker_eval/overall_metrics.json`