# YOLO -> Faster R-CNN Distillation Plan

Date: 2026-03-24

## Goal
Distill knowledge from a YOLO teacher into a Faster R-CNN student for pedestrian detection on MOT16, then use the student for downstream tracking.

## Scope
- Keep existing notebooks/code unchanged.
- Implement distillation in a separate `distillation/` folder.
- Use MOT16 train sequences for training/tuning.
- Keep MOT16 test split for demo inference only.

## Distillation Strategy
1. Run YOLO teacher on MOT16 train images and save pseudo-label cache.
2. Train Faster R-CNN student with mixed supervision:
   - Ground-truth detection loss.
   - Additional weighted pseudo-label loss from teacher detections.
3. Evaluate student on held-out sequences with:
   - validation loss
   - detection precision/recall/F1 at IoU 0.5
4. Export model checkpoints and training history.

## Why this design
- Practical and stable with current project assets.
- Uses official Faster R-CNN training API in Torchvision.
- Uses Ultralytics result boxes (`xyxy/conf/cls`) directly.
- Avoids invasive changes to current notebooks.

## Outputs
- `distillation/artifacts/teacher_cache_*.jsonl`
- `distillation/runs/<timestamp>/best_student.pt`
- `distillation/runs/<timestamp>/last_student.pt`
- `distillation/runs/<timestamp>/history.json`
- `distillation/runs/<timestamp>/config.json`

## Execution order
1. Build teacher cache.
2. Train student with mixed supervision.
3. Evaluate and inspect run artifacts.

## Command skeleton
```bash
# 1) Teacher pseudo-label cache
.venv/bin/python distillation/yolo_to_frcnn_distill.py build-cache \
  --project-root /Users/jainil/PycharmProjects/deep_learning_project \
  --teacher-model /Users/jainil/PycharmProjects/deep_learning_project/yolo26n.pt \
  --output-jsonl /Users/jainil/PycharmProjects/deep_learning_project/distillation/artifacts/teacher_cache_train.jsonl

# 2) Student training
.venv/bin/python distillation/yolo_to_frcnn_distill.py train \
  --project-root /Users/jainil/PycharmProjects/deep_learning_project \
  --cache-jsonl /Users/jainil/PycharmProjects/deep_learning_project/distillation/artifacts/teacher_cache_train.jsonl \
  --epochs 3 \
  --batch-size 2 \
  --lambda-kd 0.5
```

## Success criteria
- Cache built successfully from teacher detections.
- Training loop runs end-to-end and saves checkpoints.
- Validation metrics produced and written to run history.
