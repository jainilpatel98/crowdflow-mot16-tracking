# Model Readiness Status

This file answers two practical questions:

## 1. Which pipeline is most ready for demo right now?

The strongest fully validated demo path currently available in this server repo is:

- detector: `yolo26n.pt`
- tracker: `BoT-SORT`
- tracker config: `code/botsort_mot16_person.yaml`
- held-out evaluator: `tracking_eval/run_mot16_tracker_report.py`
- demo UI: `streamlit_app/app.py`

Why this is the current best ready path:

- the laptop copy already contained a full held-out report on `MOT16-11` and `MOT16-13`
- those artifacts are now readable from the active server app and validation layer
- the pipeline is operationally much more mature than the custom student tracker results

Best completed held-out report currently present:

- `code_from_laptop/tracking_eval/runs/20260413_005829_final_val_yolo26n_botsort_960/report.md`

Key metrics from that report:

- `MOTA = 0.4040`
- `IDF1 = 0.4971`
- `Precision = 0.8978`
- `Recall = 0.4656`

## 2. Is the custom student model already better?

Not based on the saved tracking artifacts currently in this repo.

Saved student tracking summary:

- `outputs/student_tracking_phase1/student_phase1/pedestrian_summary.txt`

Metrics there are substantially weaker:

- `MOTA = -25.324`
- `IDF1 = 16.190`

That means the student pipeline is still a research/training path, not the best current demo path.

## Important comparison note

The server repo also contains a stronger YOLO tracking result on MOT16 **train** sequences:

- `outputs/trackeval_bundle/data/trackers/mot_challenge/MOT16-train/yolo26x_botsort_person/pedestrian_summary.txt`

Those train-split metrics are:

- `MOTA = 45.113`
- `IDF1 = 55.699`

But that result is not directly comparable to the held-out `MOT16-11` / `MOT16-13` report above because the evaluation split is different.

## Recommendation

- Use the promoted `streamlit_app/`, `tracking_eval/`, and `project_validation/` code for the active server workflow.
- Use `yolo26n.pt + BoT-SORT` as the current fully validated demo baseline.
- A fresh 30-frame smoke comparison on `MOT16-11` now exists:
  - `tracking_eval/runs/20260423_124628_smoke_yolo26n_30/overall_metrics.json`
  - `tracking_eval/runs/20260423_124643_smoke_yolo26x_30/overall_metrics.json`
- In that smoke comparison, `code/yolo26x.pt` beat `yolo26n.pt` on `MOTA` (`0.3975` vs `0.3911`) and `IDF1` (`0.6332` vs `0.5740`), but with lower `precision` (`0.8092` vs `0.9557`).
- Because that comparison is only 30 frames of one sequence, keep `yolo26n` as the fully validated baseline until `code/yolo26x.pt` is rerun on the full held-out `MOT16-11` and `MOT16-13` split.
- Keep the custom distillation stack in place, but do not present it as the best current model until it closes the gap on held-out tracking metrics.
