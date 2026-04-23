from __future__ import annotations

import unittest

import torch

from models.student_jde import StudentJDE
from utils.assigners import PyramidAssigner


class AssignerSmokeTest(unittest.TestCase):
    def test_assigner_marks_positive_locations(self) -> None:
        model = StudentJDE(pretrained_backbone=False, num_id_classes=4)
        outputs = model(torch.randn(1, 3, 640, 640))
        targets = [
            {
                "boxes": torch.tensor([[32.0, 48.0, 96.0, 160.0]]),
                "track_ids": torch.tensor([1]),
                "track_labels": torch.tensor([0]),
                "ignore_boxes": torch.zeros((0, 4)),
                "image_size": torch.tensor([640, 640]),
            }
        ]
        assigner = PyramidAssigner(
            strides={"p3": 8, "p4": 16, "p5": 32},
            area_ranges={"p3": (0, 4096), "p4": (4096, 16384), "p5": (16384, 1e9)},
            center_radius=1.5,
        )
        assignments = assigner.assign(outputs, targets)
        positive_total = sum(int(level.pos_mask.sum().item()) for level in assignments.values())
        self.assertGreater(positive_total, 0)


if __name__ == "__main__":
    unittest.main()
