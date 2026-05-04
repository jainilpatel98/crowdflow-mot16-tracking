# StrongSORT OSNet ReID Tools

Standalone training and evaluation code for the BoxMOT StrongSORT appearance
model. This does not change the existing detector/tracker code path.

The MOT16 ReID dataset keeps only valid person-like classes:

- `mark == 1`
- `class_id in {1, 2}`
- `track_id > 0`
- `visibility >= 0.25`

`class_id == 1` is pedestrian and `class_id == 2` is person on vehicle/rider.

## Train

```bash
.venv/bin/python reid_tools/train_osnet_reid.py \
  --data-root MOT16 \
  --output-dir runs/reid_osnet_x0_25_mot16
```

The best checkpoint is written as:

```text
runs/reid_osnet_x0_25_mot16/osnet_x0_25_mot16_best.pt
```

## Evaluate ReID Retrieval

```bash
.venv/bin/python reid_tools/eval_reid.py \
  --weights runs/reid_osnet_x0_25_mot16/osnet_x0_25_mot16_best.pt
```

## Evaluate StrongSORT With OSNet Crop Embeddings

This path calls BoxMOT StrongSORT without student embeddings, so the OSNet crop
model is actually used for association.

```bash
.venv/bin/python reid_tools/eval_strongsort_osnet.py \
  --reid-weights runs/reid_osnet_x0_25_mot16/osnet_x0_25_mot16_best.pt
```

