# Class Project Runbook

This file is the single place to look when you want to:

- run the validation tests
- generate report-ready tracking results
- run the Streamlit demo app
- understand which model is ready now
- finish custom detector training before class

## 1. Recommended class-demo pipeline

Use this as the main project pipeline unless you fully finish the Faster R-CNN training notebook:

- detector: `yolo26n.pt`
- tracker: `BoT-SORT`
- tracker config: `code/botsort_mot16_person.yaml`
- evaluation script: `tracking_eval/run_mot16_tracker_report.py`
- demo app: `streamlit_app/app.py`

Why this is the safest option:

- it already has a completed held-out report
- it already has preview videos and MOT-format tracking files
- it already passes the automated validation suite
- it is ready for class demos right now

## 2. Environment setup

From the project root:

```bash
cd /Users/jainil/PycharmProjects/deep_learning_project
source .venv/bin/activate
python -m pip install -r streamlit_app/requirements.txt
python -m pip install -r tracking_eval/requirements.txt
```

If you want to run the Faster R-CNN notebook training path and Jupyter is missing:

```bash
python -m pip install jupyterlab notebook
```

## 3. Run the full validation suite

This is the first command to run before a demo, report refresh, or major code change.

```bash
cd /Users/jainil/PycharmProjects/deep_learning_project
source .venv/bin/activate
python project_validation/run_project_validation.py
```

What it checks:

- required imports and assets
- tracker config sanity
- single-frame detector + tracker smoke test
- short sequence tracking smoke test
- CLI report generation smoke test
- existing full held-out report readiness
- Streamlit live-output helper smoke test
- Streamlit app boot

Validation outputs are written to:

- `project_validation/runs/<timestamp>_validation/validation_report.md`
- `project_validation/runs/<timestamp>_validation/validation_report.json`

## 4. Generate report-ready tracking results

This is the main command for the class report and quantitative results.

```bash
cd /Users/jainil/PycharmProjects/deep_learning_project
source .venv/bin/activate
python tracking_eval/run_mot16_tracker_report.py \
  --project-root /Users/jainil/PycharmProjects/deep_learning_project \
  --model /Users/jainil/PycharmProjects/deep_learning_project/yolo26n.pt \
  --split train \
  --sequences MOT16-11,MOT16-13 \
  --imgsz 960 \
  --run-name final_val_yolo26n_botsort_960
```

Main outputs:

- `tracking_eval/runs/<timestamp>_<run_name>/report.md`
- `tracking_eval/runs/<timestamp>_<run_name>/per_sequence_metrics.csv`
- `tracking_eval/runs/<timestamp>_<run_name>/overall_metrics.json`
- `tracking_eval/runs/<timestamp>_<run_name>/tracks/*.txt`
- `tracking_eval/runs/<timestamp>_<run_name>/previews/*.mp4`

## 5. Run a faster smoke evaluation

Use this when you want a quick end-to-end check without waiting for the full held-out run.

```bash
cd /Users/jainil/PycharmProjects/deep_learning_project
source .venv/bin/activate
python tracking_eval/run_mot16_tracker_report.py \
  --project-root /Users/jainil/PycharmProjects/deep_learning_project \
  --model /Users/jainil/PycharmProjects/deep_learning_project/yolo26n.pt \
  --split train \
  --sequences MOT16-11 \
  --imgsz 640 \
  --max-frames 60 \
  --run-name smoke
```

## 6. Run the Streamlit app

```bash
cd /Users/jainil/PycharmProjects/deep_learning_project
source .venv/bin/activate
streamlit run streamlit_app/app.py
```

The app supports:

- `Live webcam`
- `Upload video`
- `MOT16 sequence`

Live webcam notes:

- camera access works on `localhost` or over `HTTPS`
- after changing live settings, click `Prepare New Live Session`
- stop the stream to finalize the recorded live `.mp4`

## 7. Current ready-to-show results

The strongest completed report currently in the repo is:

- `tracking_eval/runs/20260413_005829_final_val_yolo26n_botsort_960/report.md`

Current overall metrics from that run:

- `MOTA = 0.4040`
- `IDF1 = 0.4971`
- `Precision = 0.8978`
- `Recall = 0.4656`

This run is the safest one to use in the class report unless you produce a newer run that beats it.

## 8. How to train the custom Faster R-CNN baseline

Important:

- this is the custom-training path
- this path is not currently the main ready-for-class pipeline
- do not present it as finished unless the checkpoint and outputs are actually created

Notebook:

- `code/mot16-fasterrcnn-train-eval.ipynb`

Notebook training setup:

- training sequences: `MOT16-02`, `MOT16-04`, `MOT16-05`, `MOT16-09`, `MOT16-10`
- validation sequences: `MOT16-11`, `MOT16-13`
- epochs: `10`
- checkpoint target: `output/fasterrcnn/fasterrcnn_heads_e10_best.pth`

Run the notebook:

```bash
cd /Users/jainil/PycharmProjects/deep_learning_project
source .venv/bin/activate
jupyter lab
```

Then open:

- `code/mot16-fasterrcnn-train-eval.ipynb`

And run the notebook cells in order.

Required outputs before calling this model trained:

- `output/fasterrcnn/fasterrcnn_heads_e10_best.pth`
- `output/fasterrcnn_videos/train_MOT16-11_fasterrcnn_track.gif` or equivalent exported sequence output
- `output/fasterrcnn_tracking_results/MOT16-11_fasterrcnn.txt`
- validation metrics recorded from the notebook evaluation flow

Important evaluation note:

- the notebook evaluation cell uses `MOTMetrics`
- it may require MATLAB Engine for Python and the devkit setup
- if MATLAB Engine is missing, the notebook can still train, but the final MOTMetrics cell will not complete

## 9. What to say in class

Safe wording if you present the current ready pipeline:

- we evaluated a pedestrian tracking system on MOT16
- we used a finished YOLO detector checkpoint with a BoT-SORT tracker
- we validated the pipeline end to end with automated smoke checks
- we generated report-ready MOT metrics on held-out train sequences
- we also prepared a Faster R-CNN notebook baseline for custom training, but that should only be presented as complete if its checkpoint and outputs exist

Unsafe wording to avoid unless you finish the notebook training path:

- do not say the Faster R-CNN detector is fully trained if `output/fasterrcnn/fasterrcnn_heads_e10_best.pth` does not exist
- do not say the class demo model was trained from scratch in this repo if you are showing `yolo26n.pt`

## 10. Definition of ready before class

Before class day, rerun this checklist:

1. `python project_validation/run_project_validation.py` passes with zero failures.
2. The full report run exists and has the metrics you want to present.
3. The Streamlit app opens and runs at least one uploaded-video test.
4. The Streamlit live webcam mode works on the exact machine you will use in class.
5. If you want to claim custom detector training, the Faster R-CNN checkpoint and outputs exist.
