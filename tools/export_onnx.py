#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.student_jde import StudentJDE
from utils.config import load_yaml


class OnnxExportWrapper(nn.Module):
    def __init__(self, model: StudentJDE) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        outputs = self.model(images)
        return (
            outputs["cls"]["p3"],
            outputs["cls"]["p4"],
            outputs["cls"]["p5"],
            outputs["obj"]["p3"],
            outputs["obj"]["p4"],
            outputs["obj"]["p5"],
            outputs["box"]["p3"],
            outputs["box"]["p4"],
            outputs["box"]["p5"],
            outputs["emb"]["p3"],
            outputs["emb"]["p4"],
            outputs["emb"]["p5"],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the student model to ONNX.")
    parser.add_argument("--config", default="configs/student_distill.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    output_dir = Path(config["training"]["output_dir"])
    checkpoint_path = args.checkpoint or str(output_dir / "best.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    id_classes = checkpoint["student"]["id_classifier.weight"].shape[0] if "id_classifier.weight" in checkpoint["student"] else 0

    model = StudentJDE(
        backbone_name=config["student"]["backbone_name"],
        emb_dim=config["student"]["emb_dim"],
        num_classes=1,
        fpn_channels=config["student"]["fpn_channels"],
        pretrained_backbone=False,
        num_id_classes=id_classes,
        tower_layers=int(config["student"].get("tower_layers", 2)),
    )
    model.load_state_dict(checkpoint["student"])
    model.eval()

    dummy = torch.randn(1, 3, *tuple(load_yaml(config["dataset"]["config"])["dataset"]["input_size"]))
    output_path = Path(args.output) if args.output else output_dir / "student.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_model = OnnxExportWrapper(model)
    torch.onnx.export(
        export_model,
        dummy,
        output_path,
        input_names=["images"],
        output_names=[
            "cls_p3",
            "cls_p4",
            "cls_p5",
            "obj_p3",
            "obj_p4",
            "obj_p5",
            "box_p3",
            "box_p4",
            "box_p5",
            "emb_p3",
            "emb_p4",
            "emb_p5",
        ],
        opset_version=17,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
