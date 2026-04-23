# YOLO -> Faster R-CNN Distillation

This folder contains a separate, reproducible distillation pipeline.

It does **not** modify your existing notebooks or tracker code.

## What it does

1. Builds a teacher pseudo-label cache from YOLO detections on MOT16 images.
2. Trains a Faster R-CNN student using mixed supervision:
   - ground-truth loss
   - weighted pseudo-label loss (`lambda_kd`)
3. Evaluates validation loss and detection precision/recall/F1.

## Script

- `distillation/yolo_to_frcnn_distill.py`

Subcommands:

- `build-cache`
- `train`
- `evaluate`

## Quick start

From project root:

```bash
source .venv/bin/activate

# 1) Build teacher cache from YOLO
.venv/bin/python distillation/yolo_to_frcnn_distill.py build-cache \
  --project-root /Users/jainil/PycharmProjects/deep_learning_project \
  --teacher-model /Users/jainil/PycharmProjects/deep_learning_project/yolo26n.pt \
  --split train \
  --output-jsonl /Users/jainil/PycharmProjects/deep_learning_project/distillation/artifacts/teacher_cache_train.jsonl

# 2) Train student
.venv/bin/python distillation/yolo_to_frcnn_distill.py train \
  --project-root /Users/jainil/PycharmProjects/deep_learning_project \
  --cache-jsonl /Users/jainil/PycharmProjects/deep_learning_project/distillation/artifacts/teacher_cache_train.jsonl \
  --epochs 3 \
  --batch-size 2 \
  --lambda-kd 0.5
```

## Useful debug options

For fast smoke tests:

```bash
.venv/bin/python distillation/yolo_to_frcnn_distill.py build-cache \
  --project-root /Users/jainil/PycharmProjects/deep_learning_project \
  --teacher-model /Users/jainil/PycharmProjects/deep_learning_project/yolo26n.pt \
  --split train \
  --max-images 50 \
  --output-jsonl /Users/jainil/PycharmProjects/deep_learning_project/distillation/artifacts/teacher_cache_debug.jsonl

.venv/bin/python distillation/yolo_to_frcnn_distill.py train \
  --project-root /Users/jainil/PycharmProjects/deep_learning_project \
  --cache-jsonl /Users/jainil/PycharmProjects/deep_learning_project/distillation/artifacts/teacher_cache_debug.jsonl \
  --epochs 1 \
  --max-train-images 100 \
  --max-val-images 50
```

## Output locations

- Teacher cache:
  - `distillation/artifacts/*.jsonl`
- Student runs:
  - `distillation/runs/distill_run_<timestamp>/`
  - `best_student.pt`
  - `last_student.pt`
  - `history.json`
  - `config.json`

## Notes

- Default class setup is pedestrian-only (`NUM_CLASSES = 2` including background).
- Validation metrics are quick detection metrics at IoU=0.5, not full MOT tracking metrics.
- Use your existing tracking pipeline to compare downstream ID behavior after detector distillation.
