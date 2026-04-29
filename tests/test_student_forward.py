from __future__ import annotations

import unittest

import torch

from models.student_jde import StudentJDE


class StudentForwardSmokeTest(unittest.TestCase):
    def test_output_shapes_resnet50(self) -> None:
        """Default (resnet50, fpn_channels=256) output shapes."""
        model = StudentJDE(pretrained_backbone=False, num_id_classes=8)
        # Default is resnet50 / fpn_channels=256
        outputs = model(torch.randn(2, 3, 640, 640))
        self.assertEqual(outputs["cls"]["p3"].shape, (2, 1, 80, 80))
        self.assertEqual(outputs["obj"]["p4"].shape, (2, 1, 40, 40))
        self.assertEqual(outputs["box"]["p5"].shape, (2, 4, 20, 20))
        self.assertEqual(outputs["emb"]["p3"].shape[1], 128)

    def test_output_shapes_mobilenet_small(self) -> None:
        """MobileNetV3-Small (fpn_channels=128) shapes."""
        model = StudentJDE(backbone_name="mobilenetv3_small", fpn_channels=128,
                           pretrained_backbone=False, num_id_classes=4)
        outputs = model(torch.randn(1, 3, 640, 640))
        self.assertEqual(outputs["cls"]["p3"].shape, (1, 1, 80, 80))
        self.assertEqual(outputs["emb"]["p3"].shape[1], 128)

    def test_output_shapes_resnext101_small_input(self) -> None:
        """ResNeXt-101 should wire through the same FPN/head contract."""
        model = StudentJDE(
            backbone_name="resnext101_32x8d",
            fpn_channels=384,
            pretrained_backbone=False,
            num_id_classes=4,
        )
        outputs = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(outputs["cls"]["p3"].shape, (1, 1, 8, 8))
        self.assertEqual(outputs["obj"]["p4"].shape, (1, 1, 4, 4))
        self.assertEqual(outputs["box"]["p5"].shape, (1, 4, 2, 2))
        self.assertEqual(outputs["emb"]["p3"].shape[1], 128)

    def test_output_shapes_se_resnet50_small_input(self) -> None:
        """SE-ResNet-50 should expose the same c3/c4/c5 contract as ResNet."""
        model = StudentJDE(
            backbone_name="se_resnet50",
            fpn_channels=384,
            pretrained_backbone=False,
            num_id_classes=4,
        )
        outputs = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(outputs["cls"]["p3"].shape, (1, 1, 8, 8))
        self.assertEqual(outputs["obj"]["p4"].shape, (1, 1, 4, 4))
        self.assertEqual(outputs["box"]["p5"].shape, (1, 4, 2, 2))
        self.assertEqual(outputs["emb"]["p3"].shape[1], 128)


if __name__ == "__main__":
    unittest.main()
