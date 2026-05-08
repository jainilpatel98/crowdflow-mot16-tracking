#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reid_tools.mot16_reid_dataset import MOT16ReIDDataset, read_mot16_reid_samples
from reid_tools.osnet_model import build_osnet_x0_25, save_boxmot_compatible_checkpoint
from reid_tools.reid_metrics import cosine_distance_summary, evaluate_reid


TRAIN_SEQUENCES = ["MOT16-02", "MOT16-04", "MOT16-09", "MOT16-11", "MOT16-13"]
VAL_SEQUENCES = ["MOT16-05", "MOT16-10"]
DEFAULT_PRETRAINED = ".venv/lib/python3.12/site-packages/models/osnet_x0_25_msmt17.pt"


class RandomIdentityBatchSampler(Sampler[list[int]]):
    def __init__(self, labels: list[int], identities_per_batch: int, instances_per_identity: int) -> None:
        self.labels = labels
        self.identities_per_batch = identities_per_batch
        self.instances_per_identity = instances_per_identity
        self.index_by_label: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            self.index_by_label[int(label)].append(index)
        self.identities = list(self.index_by_label)
        self.batch_size = identities_per_batch * instances_per_identity
        self.num_batches = max(1, len(labels) // self.batch_size)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            selected_ids = random.sample(self.identities, k=min(self.identities_per_batch, len(self.identities)))
            batch: list[int] = []
            for identity in selected_ids:
                indices = self.index_by_label[identity]
                replace = len(indices) < self.instances_per_identity
                chosen = np.random.choice(indices, size=self.instances_per_identity, replace=replace)
                batch.extend(int(i) for i in chosen.tolist())
            yield batch


def build_transforms(train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((256, 128)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.2, hue=0.02),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def batch_hard_triplet_loss(features: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    features = nn.functional.normalize(features, dim=1)
    distances = torch.cdist(features, features, p=2)
    same = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    different = ~same
    hardest_positive = (distances * same.float()).max(dim=1).values
    masked_negative = distances.masked_fill(~different, 1e6)
    hardest_negative = masked_negative.min(dim=1).values
    valid = same.sum(dim=1) > 1
    if not valid.any():
        return features.new_tensor(0.0)
    return nn.functional.relu(hardest_positive[valid] - hardest_negative[valid] + margin).mean()


@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sequences: list[str] = []
    for batch in loader:
        images = batch["image"].to(device)
        output = model(images)
        output = nn.functional.normalize(output, dim=1)
        features.append(output.cpu().numpy())
        labels.append(batch["label"].numpy())
        sequences.extend(batch["sequence"])
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0), sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BoxMOT-compatible OSNet ReID on MOT16 person-like classes.")
    parser.add_argument("--data-root", default="MOT16")
    parser.add_argument("--output-dir", default="runs/reid_osnet_x0_25_mot16")
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-identities", type=int, default=8)
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--triplet-margin", type=float, default=0.3)
    parser.add_argument("--triplet-weight", type=float, default=1.0)
    parser.add_argument("--min-visibility", type=float, default=0.25)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_samples = read_mot16_reid_samples(
        root=args.data_root,
        sequences=TRAIN_SEQUENCES,
        class_ids={1, 2},
        min_visibility=args.min_visibility,
    )
    val_samples = read_mot16_reid_samples(
        root=args.data_root,
        sequences=VAL_SEQUENCES,
        class_ids={1, 2},
        min_visibility=args.min_visibility,
    )
    if not train_samples:
        raise RuntimeError("No training ReID samples found. Check MOT16 path and filters.")
    if not val_samples:
        raise RuntimeError("No validation ReID samples found. Check MOT16 path and filters.")

    train_dataset = MOT16ReIDDataset(train_samples, transform=build_transforms(train=True))
    val_dataset = MOT16ReIDDataset(val_samples, transform=build_transforms(train=False))
    sampler = RandomIdentityBatchSampler(
        labels=[sample.label for sample in train_samples],
        identities_per_batch=args.batch_identities,
        instances_per_identity=args.instances,
    )
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_identities * args.instances, shuffle=False, num_workers=args.num_workers)

    model = build_osnet_x0_25(num_classes=train_dataset.num_classes, pretrained_weights=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce_loss = nn.CrossEntropyLoss()

    metadata = {
        "train_sequences": TRAIN_SEQUENCES,
        "val_sequences": VAL_SEQUENCES,
        "class_ids": [1, 2],
        "min_visibility": args.min_visibility,
        "num_train_ids": train_dataset.num_classes,
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    best_map = -1.0
    log_path = output_dir / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log_handle:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            running_ce = 0.0
            running_triplet = 0.0
            progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
            for batch in progress:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits, features = model(images)
                loss_ce = ce_loss(logits, labels)
                loss_triplet = batch_hard_triplet_loss(features, labels, margin=args.triplet_margin)
                loss = loss_ce + args.triplet_weight * loss_triplet
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item())
                running_ce += float(loss_ce.item())
                running_triplet += float(loss_triplet.item())
                progress.set_postfix(loss=f"{loss.item():.4f}", ce=f"{loss_ce.item():.4f}", tri=f"{loss_triplet.item():.4f}")

            scheduler.step()
            val_features, val_labels, _ = extract_features(model, val_loader, device)
            metrics = evaluate_reid(val_features, val_labels)
            metrics.update(cosine_distance_summary(val_features, val_labels))
            num_batches = max(1, len(train_loader))
            row = {
                "epoch": epoch,
                "train_loss": running_loss / num_batches,
                "train_ce": running_ce / num_batches,
                "train_triplet": running_triplet / num_batches,
                "lr": scheduler.get_last_lr()[0],
                **metrics,
            }
            log_handle.write(json.dumps(row) + "\n")
            log_handle.flush()
            print(json.dumps(row, indent=2))

            last_path = output_dir / "osnet_x0_25_mot16_last.pt"
            save_boxmot_compatible_checkpoint(model, last_path, {**metadata, "epoch": epoch, "metrics": metrics})
            if metrics["mAP"] > best_map:
                best_map = metrics["mAP"]
                best_path = output_dir / "osnet_x0_25_mot16_best.pt"
                save_boxmot_compatible_checkpoint(model, best_path, {**metadata, "epoch": epoch, "metrics": metrics})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
