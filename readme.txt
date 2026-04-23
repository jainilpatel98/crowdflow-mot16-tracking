Project pipeline for YOLO-to-CNN distillation on MOT16

Environment
- Use the project virtualenv:
  `.venv/bin/python`
- Install or refresh deps:
  `.venv/bin/pip install -r requirements.txt`

1. Fine-tune the YOLO teacher on MOT16 person-only
```bash
.venv/bin/python tools/train_teacher.py --config configs/teacher_finetune.yaml
```

2. Cache teacher outputs for student distillation
```bash
.venv/bin/python tools/cache_teacher_outputs.py --config configs/student_distill.yaml --split train
.venv/bin/python tools/cache_teacher_outputs.py --config configs/student_distill.yaml --split val
```

3. Train the student detector + embedding model
```bash
.venv/bin/python tools/train_student.py --config configs/student_distill.yaml
```

4. Evaluate student detection on MOT16 val sequences
```bash
.venv/bin/python tools/eval_detection.py \
  --config configs/student_distill.yaml \
  --checkpoint runs/student_distill/best.pt
```

5. Run student tracking on one sequence and export MOT-format output
```bash
.venv/bin/python tools/eval_tracking.py \
  --config configs/student_distill.yaml \
  --tracker-config configs/tracker.yaml \
  --checkpoint runs/student_distill/best.pt \
  --sequence-dir MOT16/train/MOT16-10 \
  --output outputs/student_tracking/MOT16-10.txt
```

6. Export the student to ONNX
```bash
.venv/bin/python tools/export_onnx.py \
  --config configs/student_distill.yaml \
  --checkpoint runs/student_distill/best.pt \
  --output runs/student_distill/student.onnx
```

Notes
- `configs/dataset_mot16.yaml` controls sequence splits and augmentation defaults.
- `configs/student_distill.yaml` controls teacher/student architecture, phase weights, optimizer, and inference thresholds.
- The training defaults are now data-aware from the MOT16 notebook analysis:
  mixed `1920x1080` and `640x480` inputs use 640-letterbox instead of aspect-ratio distortion,
  embedding/ID supervision ignores boxes below `0.25` visibility,
  and train sampling is balanced by sequence so `MOT16-04` does not dominate.
- Teacher caching is deterministic resize-only. If you want random train-time augmentation to match teacher and student exactly, run live teacher distillation instead of cache mode.
- `tools/eval_tracking.py` expects `deep_sort_realtime` for DeepSORT or `boxmot` for StrongSORT if you enable those trackers.

7. Generate a held-out MOT16 tracking report for a YOLO checkpoint
```bash
.venv/bin/python tracking_eval/run_mot16_tracker_report.py \
  --project-root /home/research/tracking_crowded_people \
  --model /home/research/tracking_crowded_people/yolo26n.pt \
  --split train \
  --sequences MOT16-11,MOT16-13 \
  --run-name yolo26n_botsort_val
```

8. Launch the Streamlit demo app
```bash
.venv/bin/python -m pip install -r streamlit_app/requirements.txt
.venv/bin/python -m streamlit run streamlit_app/app.py
```

9. Run the end-to-end validation suite
```bash
.venv/bin/python project_validation/run_project_validation.py
```

TrackEval for MOT16 train metrics
```bash
.venv/bin/python code/prepare_trackeval_mot16.py \
  --dataset-root MOT16 \
  --tracker-output-root outputs/yolo_mot16_multigpu \
  --bundle-root outputs/trackeval_bundle \
  --tracker-name yolo26x_botsort_person

.venv/bin/python TrackEval/scripts/run_mot_challenge.py \
  --GT_FOLDER /home/research/tracking_crowded_people/outputs/trackeval_bundle/data/gt/mot_challenge \
  --TRACKERS_FOLDER /home/research/tracking_crowded_people/outputs/trackeval_bundle/data/trackers/mot_challenge \
  --BENCHMARK MOT16 \
  --SPLIT_TO_EVAL train \
  --TRACKERS_TO_EVAL yolo26x_botsort_person \
  --METRICS HOTA CLEAR Identity
```

Current repo layout
- `tools/`, `models/`, `engine/`, `datasets/`: student distillation and custom training stack
- `code/`: YOLO/BoT-SORT tracking utilities and TrackEval bundle preparation
- `tracking_eval/`: held-out MOT16 evaluation reports for YOLO checkpoints
- `streamlit_app/`: demo UI for webcam, uploads, and MOT16 sequences
- `project_validation/`: smoke checks and readiness validation for the demo pipeline
