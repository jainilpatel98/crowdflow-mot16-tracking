#!/usr/bin/env python3
"""
Quick reference for using the transferred ReID tools.
"""

# ======================== ReID Training ========================
#
# Train OSNet x0.25 ReID model on MOT16 data:
#
#   python reid_tools/train_osnet_reid.py \
#     --data-root MOT16 \
#     --output-dir runs/reid_osnet_x0_25_mot16
#
# Uses batch-hard triplet loss with:
#   - MOT16 train sequences: MOT16-02, MOT16-04, MOT16-09, MOT16-11, MOT16-13
#   - MOT16 val sequences: MOT16-05, MOT16-10
#   - Person-like classes only (class_id in {1, 2})
#   - min_visibility >= 0.25

# ======================== ReID Evaluation ========================
#
# Evaluate ReID retrieval metrics:
#
#   python reid_tools/eval_reid.py \
#     --weights runs/reid_osnet_x0_25_mot16/osnet_x0_25_mot16_best.pt
#
# Outputs: rank-1 accuracy, mean AP, etc.

# ======================== StrongSORT + OSNet ========================
#
# Evaluate StrongSORT tracking with OSNet ReID appearance model:
#
#   python reid_tools/eval_strongsort_osnet.py \
#     --reid-weights runs/reid_osnet_x0_25_mot16/osnet_x0_25_mot16_best.pt
#
# This uses OSNet crops for association instead of student embeddings.

# ======================== MOT16 ReID Dataset ========================
#
# Read MOT16 ReID samples from GT boxes:
#
#   from reid_tools.mot16_reid_dataset import read_mot16_reid_samples, MOT16ReIDDataset
#
#   samples = read_mot16_reid_samples(
#     root="MOT16",
#     sequences=["MOT16-02", "MOT16-04", "MOT16-05"],
#     class_ids={1, 2},           # person-like only
#     min_visibility=0.25,
#     min_box_size=4
#   )
#   dataset = MOT16ReIDDataset(samples, transform=my_transform)

# ======================== Available Weights ========================
#
# Pre-trained OSNet weights:
#   runs/reid_osnet_x0_25_mot16/osnet_x0_25_mot16_best.pt
#   runs/reid_osnet_x0_25_mot16/osnet_x0_25_mot16_last.pt
#
# These are trained on MOT16 train/val splits with batch-hard triplet loss.
