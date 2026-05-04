#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reid_tools.mot16_reid_dataset import MOT16ReIDDataset, read_mot16_reid_samples
from reid_tools.osnet_model import build_osnet_x0_25
from reid_tools.reid_metrics import cosine_distance_summary, evaluate_reid
from reid_tools.train_osnet_reid import VAL_SEQUENCES, build_transforms, extract_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OSNet ReID weights on MOT16 validation crops.")
    parser.add_argument("--data-root", default="MOT16")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--sequences", default=",".join(VAL_SEQUENCES))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-visibility", type=float, default=0.25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sequences = [item.strip() for item in args.sequences.split(",") if item.strip()]
    samples = read_mot16_reid_samples(
        root=args.data_root,
        sequences=sequences,
        class_ids={1, 2},
        min_visibility=args.min_visibility,
    )
    dataset = MOT16ReIDDataset(samples, transform=build_transforms(train=False))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = build_osnet_x0_25(num_classes=max(1, dataset.num_classes), pretrained_weights=args.weights).to(args.device)
    features, labels, sample_sequences = extract_features(model, loader, torch.device(args.device))

    result = {"overall": {**evaluate_reid(features, labels), **cosine_distance_summary(features, labels)}}
    by_sequence: dict[str, list[int]] = defaultdict(list)
    for index, sequence in enumerate(sample_sequences):
        by_sequence[sequence].append(index)
    result["per_sequence"] = {}
    for sequence, indices in by_sequence.items():
        idx = np.asarray(indices, dtype=np.int64)
        result["per_sequence"][sequence] = {
            **evaluate_reid(features[idx], labels[idx]),
            **cosine_distance_summary(features[idx], labels[idx]),
        }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

