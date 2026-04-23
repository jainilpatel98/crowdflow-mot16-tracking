from __future__ import annotations

import unittest

import torch

from models.student_jde import StudentJDE


class StudentForwardSmokeTest(unittest.TestCase):
    def test_output_shapes(self) -> None:
        model = StudentJDE(pretrained_backbone=False, num_id_classes=8)
        outputs = model(torch.randn(2, 3, 640, 640))
        self.assertEqual(outputs["cls"]["p3"].shape, (2, 1, 80, 80))
        self.assertEqual(outputs["obj"]["p4"].shape, (2, 1, 40, 40))
        self.assertEqual(outputs["box"]["p5"].shape, (2, 4, 20, 20))
        self.assertEqual(outputs["emb"]["p3"].shape[1], 128)


if __name__ == "__main__":
    unittest.main()
