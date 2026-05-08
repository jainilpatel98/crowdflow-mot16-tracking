# CrowdFlow — YOLO-to-CNN Knowledge Distillation on MOT16

Multi-object tracking pipeline that distils a frozen YOLOv6-X teacher into a
compact CNN student (JDE-style: detection + re-ID in one forward pass) using
multi-signal knowledge distillation on the MOT16 pedestrian dataset.

---

## Architecture Overview

### Teacher Model — YOLOv6-X (frozen)

The teacher is a pre-trained, person-fine-tuned YOLOv6-X YOLO model that
runs live alongside the student during training (no offline caching). It provides
four distillation signals:

- **Soft classification targets** — teacher confidence maps at every FPN cell
- **Box regression guidance** — teacher-matched LTRB predictions as a secondary target
- **FPN feature maps** — raw spatial features at p3/p4/p5 for dense alignment
- **Identity embeddings** — via a trainable `TeacherROIProjector` MLP (see below)

The teacher weights are fully frozen (`freeze: true`). Only its `TeacherROIProjector`
is co-trained with the student.

### Student Model — StudentJDE

```
Input image
    │
    ▼
Backbone (ResNeXt101-32x8d, pretrained on ImageNet)
  stem → layer1 → layer2 → layer3 → layer4
    │
    ▼
Neck — TinyFPN
  Produces p3 (stride 8), p4 (stride 16), p5 (stride 32)
  All levels unified to fpn_channels=384
    │
    ├──► Detection & Embedding Head (per level, shared weights)
    │      cls_tower  (3× Conv→GN→SiLU→Dropout2d)  → cls_pred  (1×1 → 1 ch)
    │      obj_tower  (3× Conv→GN→SiLU→Dropout2d)  → obj_pred  (1×1 → 1 ch, centerness)
    │      box_tower  (3× Conv→GN→SiLU→Dropout2d)  → box_pred  (1×1 → 4 ch, LTRB/stride)
    │      emb_tower  (3× Conv→GN→SiLU→Dropout2d)  → emb_pred  (1×1 → 128 ch, L2-norm)
    │
    └──► ROIProjector  [training only — not used at inference]
           ROI-Align(p3, GT boxes, 7×7) → Flatten → Linear → BN → ReLU → Linear → L2-norm
           → 128-d re-ID embedding used for emb_kd and id_loss
```

**At inference** only the backbone + neck + head are active. The `ROIProjector`
and `id_classifier` are training scaffolding that are skipped. Each surviving
NMS detection reads its tracking embedding directly from `emb_pred[cell]` —
no second forward pass needed.

### Key Design Decisions

| Design | Rationale |
|---|---|
| GroupNorm in all towers | Small MOT16 batch size (12) makes BatchNorm statistics unreliable |
| `Dropout2d(p=0.1)` in head towers | Prevents large student (ResNeXt101) from memorising distillation noise |
| Backbone frozen until epoch 25 | Prevents backbone drift corrupting the detection FPN during early distillation |
| Stride-normalised LTRB targets | Box regression targets are divided by stride so the same head sees O(1)-scale targets regardless of level |
| Score = sigmoid(cls) only | Removing centerness gate (score × centerness) lifted a structural recall ceiling where off-centre detections were suppressed below the threshold |

---

## Training Signals (7 Loss Terms)

Training uses a **piecewise-linear loss schedule** that ramps each signal in
at the right time to prevent interference:

```
total_loss = w_det      × det_loss          # 1. GT-supervised detection
           + w_cls_kd   × cls_kd            # 2. Soft classification KD
           + w_box_kd   × box_kd            # 3. GT-anchored box KD
           + w_feat     × feat_kd           # 4. FPN feature alignment
           + w_emb      × emb_kd            # 5. ROI embedding KD
           + w_id       × id_loss           # 6. Identity cross-entropy
           + w_emb×0.5  × dense_emb         # 7. Dense embedding alignment
```

| # | Loss | Signal | Active from |
|---|---|---|---|
| 1 | `det` | Focal cls + centerness + CIoU box vs GT | epoch 1 |
| 2 | `cls_kd` | KL-div between student/teacher soft cls logits | epoch 1 |
| 3 | `box_kd` | 0.75×CIoU(pred, GT) + 0.25×CIoU(pred, teacher box) | epoch 15 |
| 4 | `feat_kd` | MSE between student/teacher FPN features via learned adapters | epoch 8 |
| 5 | `emb_kd` | Cosine loss: student ROI embed ↔ teacher ROI projector embed | epoch 28 |
| 6 | `id_loss` | Cross-entropy over MOT16 track IDs (label smoothing 0.15) | epoch 35 |
| 7 | `dense_emb` | Cosine: emb_pred at positive cells ↔ student ROIProjector (detached) | epoch 28 |

**Why `box_kd` uses GT-anchored targets (Fix):** Pure teacher imitation for
box regression causes the student to inherit the teacher's localization errors.
The 0.75/0.25 blend treats the teacher as a soft regularizer while keeping GT
as the primary target.

**Why `dense_emb` is needed (Fix):** The `emb_kd` and `id_loss` gradients flow
through detached FPN features into the `ROIProjector` only — they never reach
the `emb_tower` + `emb_pred` path that is actually used at inference. Without
`dense_emb`, the inference embedding head receives zero identity-relevant gradient
and outputs near-random tracking descriptors.

---

## Assigner — PyramidAssigner

Anchor-free, fully vectorised assignment across three FPN levels:

1. **Area filter** — each level handles a specific person-size range (p3: <64px, p4: 48–144px, p5: 120px+)
2. **Centre-radius mask** — candidate cells within `center_radius` grid cells of the GT box centre
3. **Point-in-box mask** — cell centre pixel must lie strictly inside the GT box
4. **Conflict resolution** — when multiple GT boxes claim a cell, the smallest-area GT wins

**Small-object fix:** For GT boxes smaller than 32×32px (1024 px²), step 3 is
skipped. At stride=8 a 16×24px person spans only 2×3 = 6 grid cells. No matter
how large `center_radius` is set, the inside-box check caps positives at 6 cells
and makes radius expansions inert. Disabling the check for small objects allows
the full centre-radius zone to become active, providing 4× more training signal
per tiny pedestrian per frame and directly breaking the recall ceiling.

---

## Trainable Teacher ROI Projector

The YOLOv6-X teacher was trained for detection, not re-ID. Its raw FPN features
have no concept of person identity. The original design mean-pooled those features
to produce embedding targets — but the resulting cosine loss was flat at its
maximum (≈1.0 cosine distance) for the first 50 epochs, providing zero signal.

The fix: a small learnable MLP (`TeacherROIProjector`) is trained **jointly with
the student** using the same MOT16 identity labels. By the time the student's
embedding KD activates (epoch 28), the teacher projector has been identity-
supervised for 28 epochs and produces genuinely discriminative 128-d targets
calibrated to this dataset's appearance distribution.

---

## Evaluation — Threshold Sweep

`eval_detection.py` supports two modes:

- **Single threshold** — fast evaluation at one operating point
- **Threshold sweep** — model inference runs once; raw IoUs are cached, then
  mAP, Precision, Recall, F1, and MeanIoU are computed analytically across the
  full threshold range in under 2 minutes

The sweep identifies the optimal confidence threshold for each checkpoint.
Training defaults use `score_threshold: 0.50` (the sweep-optimal F1 point for
the ResNeXt101 train9 checkpoint). For tracking workloads where recall matters
more than precision, `0.40` is recommended.

Both the student and the teacher can be swept:
```
--model-type student --sweep-threshold
--model-type teacher --sweep-threshold
```

---

## Student Model Zoo

| Config | Backbone | FPN ch | Head depth | Status |
|---|---|---|---|---|
| `student_distill_resnext101.yaml` | ResNeXt101-32x8d | 384 | 3 layers | Primary (train10) |
| `student_distill_resnet101.yaml` | ResNet101 | 384 | 3 layers | Queued |
| `student_distill_resnet50.yaml` | ResNet50 | 256 | 2 layers | Queued |
| `student_distill_resnext50.yaml` | ResNeXt50-32x4d | 256 | 2 layers | Queued |
| `student_distill_se_resnet101.yaml` | SE-ResNet101 | 384 | 2 layers | Queued |
| `student_distill_se_resnet50.yaml` | SE-ResNet50 | 256 | 2 layers | Queued |
| `student_distill_mobilenetv3_large.yaml` | MobileNetV3-Large | 256 | 2 layers | Queued |

---

## Environment

```bash
# Use the project virtualenv
.venv/bin/python

# Install or refresh dependencies
.venv/bin/pip install -r requirements.txt
```

---

## Training Pipeline

### 1. Fine-tune the YOLO teacher on MOT16 (person-only)

```bash
.venv/bin/python tools/train_teacher.py --config configs/teacher_finetune.yaml
```

### 2. Train the student (live distillation — no offline caching required)

The teacher runs live alongside the student. No pre-caching step is needed.
Pick the config for the backbone you want to train:

```bash
# ResNeXt101 (primary, largest student)
torchrun --nproc_per_node=NUM_GPUS tools/train_student.py \
  --config configs/student_distill_resnext101.yaml

# ResNet50 (compact student)
torchrun --nproc_per_node=NUM_GPUS tools/train_student.py \
  --config configs/student_distill_resnet50.yaml
```

Single-GPU training (no torchrun):
```bash
.venv/bin/python tools/train_student.py --config configs/student_distill_resnext101.yaml
```

> **Note:** Teacher caching (`cache_teacher_outputs.py`) is still supported for
> development / debugging via `use_cache: true` in the config. Live distillation
> (`use_cache: false`) is the default and is recommended because it ensures the
> teacher and student always see the same augmented image.

### 3. Evaluate detection quality

Single threshold (uses `inference.score_threshold` from config):
```bash
.venv/bin/python tools/eval_detection.py \
  --config configs/student_distill_resnext101.yaml \
  --checkpoint runs/student_distill_resnext101/best.pt
```

Threshold sweep (find the optimal confidence cutoff):
```bash
.venv/bin/python tools/eval_detection.py \
  --config configs/student_distill_resnext101.yaml \
  --checkpoint runs/student_distill_resnext101/best.pt \
  --sweep-threshold

# Also sweep the teacher for comparison
.venv/bin/python tools/eval_detection.py \
  --config configs/student_distill_resnext101.yaml \
  --model-type teacher \
  --sweep-threshold
```

### 4. Run tracking on a MOT16 sequence

```bash
.venv/bin/python tools/eval_tracking.py \
  --config configs/student_distill_resnext101.yaml \
  --tracker-config configs/tracker.yaml \
  --checkpoint runs/student_distill_resnext101/best.pt \
  --sequence-dir MOT16/train/MOT16-10 \
  --output outputs/student_tracking/MOT16-10.txt
```

### 5. Export to ONNX

```bash
.venv/bin/python tools/export_onnx.py \
  --config configs/student_distill_resnext101.yaml \
  --checkpoint runs/student_distill_resnext101/best.pt \
  --output runs/student_distill_resnext101/student.onnx
```

---

## YOLO Teacher — MOT16 Tracking Report

Run held-out evaluation using the raw YOLO teacher with BoT-SORT tracking:

```bash
.venv/bin/python tracking_eval/run_mot16_tracker_report.py \
  --project-root /home/research/tracking_crowded_people \
  --model code/yolo26x.pt \
  --split train \
  --sequences MOT16-11,MOT16-13 \
  --run-name yolo26x_botsort_val
```

## TrackEval — Formal HOTA / CLEAR / Identity Metrics

```bash
.venv/bin/python code/prepare_trackeval_mot16.py \
  --dataset-root MOT16 \
  --tracker-output-root outputs/yolo_mot16_multigpu \
  --bundle-root outputs/trackeval_bundle \
  --tracker-name yolo26x_botsort_person

.venv/bin/python TrackEval/scripts/run_mot_challenge.py \
  --GT_FOLDER outputs/trackeval_bundle/data/gt/mot_challenge \
  --TRACKERS_FOLDER outputs/trackeval_bundle/data/trackers/mot_challenge \
  --BENCHMARK MOT16 \
  --SPLIT_TO_EVAL train \
  --TRACKERS_TO_EVAL yolo26x_botsort_person \
  --METRICS HOTA CLEAR Identity
```

---

## Streamlit Demo

```bash
.venv/bin/pip install -r streamlit_app/requirements.txt
.venv/bin/python -m streamlit run streamlit_app/app.py
```

Supports webcam input, video file uploads, and MOT16 sequence playback.

---

## Smoke Tests

```bash
# Quick end-to-end validation (small batch, short run)
.venv/bin/python tools/train_student.py \
  --config configs/student_distill_resnet50_smoke.yaml

# Full project validation suite
.venv/bin/python project_validation/run_project_validation.py
```

---

## Repository Layout

```
configs/          Student and teacher training configs (one per backbone)
datasets/         MOT16 dataset loader, transforms, collate
engine/
  trainer.py      Training loop, 7-signal loss, rolling F1 checkpointing
  evaluator.py    Detection metrics, threshold sweep infrastructure
  inference.py    Decode dense head outputs → tracked detections
losses/
  det_loss.py     Focal + centerness + CIoU detection loss
  kd_loss.py      cls_kd, box_kd (GT-anchored), feat_kd, emb_kd, dense_emb
models/
  backbone_mobilenetv3.py  Unified backbone builder (ResNet, ResNeXt, SE-*, MobileNetV3)
  neck_fpn.py     TinyFPN — 3-level feature pyramid
  heads.py        DetectionEmbeddingHead with configurable tower depth + GroupNorm
  adapters.py     Learnable FPN channel adapters (student ↔ teacher feature alignment)
  student_jde.py  StudentJDE — full model with ROIProjector and id_classifier
  teacher_wrapper.py  TeacherWrapper + TeacherROIProjector (co-trained)
utils/
  assigners.py    PyramidAssigner with small-object centre-only fix
  box_ops.py      CIoU, distance2bbox, make_grid, NMS utilities
  checkpoint.py   Channel-shape validation for adapter checkpoint loading
tools/
  train_student.py   DDP-compatible student training entry point
  eval_detection.py  Detection evaluation + threshold sweep (student and teacher)
  eval_tracking.py   MOT16 tracking evaluation
  export_onnx.py     Export student to ONNX
  train_teacher.py   YOLO teacher MOT16 fine-tuning
code/             YOLO / BoT-SORT tracking utilities, TrackEval bundle prep
tracking_eval/    Held-out MOT16 evaluation reports for YOLO checkpoints
streamlit_app/    Demo UI
project_validation/ Smoke checks and readiness validation
```

---

## Configuration Notes

- `configs/dataset_mot16.yaml` — sequence splits, augmentation, visibility thresholds.
  Train sequences: MOT16-02, MOT16-04. Val sequences: MOT16-09, MOT16-11.
  Sampling is balanced by sequence so longer sequences (e.g. MOT16-04) do not dominate.
- `configs/student_distill_resnext101.yaml` — the fully-tuned primary config.
  All loss schedule values, assigner radii, and optimizer LRs reflect the findings
  from the train4–train10 ablation series.
- Embedding / ID supervision ignores GT boxes with visibility < 0.25 (heavy occlusion).
- Input images are letterboxed to 640px (preserving aspect ratio) rather than
  stretched, because MOT16 mixes 1920×1080 and 640×480 sequences.
