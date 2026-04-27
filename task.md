# Implementation Task

## Changes

- [x] **1. Detach FPN for emb/id losses** — `engine/trainer.py`
  - Remove pre-computed roi_embeddings from student forward pass
  - Recompute student_roi from `{k: v.detach()}` features in `_compute_total_loss`
  - Same detached path for id_embeddings

- [x] **2. Periodic checkpoints + best.pt on val_det** — `engine/trainer.py`
  - Added `checkpoint_interval=5` param
  - Saves `epoch_{N:03d}.pt` every 5 epochs
  - Changed `best_val` → `best_val_det`; criterion `val_metrics["det"] < best_val_det`

- [x] **3. Label smoothing on id_loss** — `losses/id_loss.py` + all configs
  - Added `label_smoothing=0.0` param to `id_supervision_loss` (default 0 = no change)
  - Added `id_label_smoothing: 0.1` under `training:` in all 4 full-training configs
  - Wired through `train_student.py` → trainer → loss calls (student + teacher projector)

- [x] **4. Per-loss ramp schedule** — `engine/trainer.py` + `train_student.py` + resnet50 config
  - Added `LossRampSchedule` class (piecewise-linear per-loss weights)
  - Old `phases` list → `PhaseWeightSchedule` (backward compat, smoke/mobilenet/runtime/base configs)
  - New `loss_schedule` dict → `LossRampSchedule` (resnet50 config)
  - `train_student.py` auto-detects via `config.get("loss_schedule") or config["phases"]`

- [x] **5/6/7. Document** — `adaptive_training_design.md` already contains full detail

- [x] **Final review** — verified trainer.py, all configs, smoke backward compat
