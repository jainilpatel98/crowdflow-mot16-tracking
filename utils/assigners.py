from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from utils.box_ops import make_grid


@dataclass
class LevelAssignment:
    cls_targets: torch.Tensor
    obj_targets: torch.Tensor
    box_targets: torch.Tensor
    box_xyxy: torch.Tensor
    pos_mask: torch.Tensor
    track_ids: torch.Tensor
    points: torch.Tensor
    stride: int


class PyramidAssigner:
    def __init__(
        self,
        strides: dict[str, int],
        area_ranges: dict[str, tuple[float, float]],
        center_radius: float = 1.5,
    ) -> None:
        self.strides = strides
        self.area_ranges = area_ranges
        self.center_radius = center_radius

    def assign(
        self,
        outputs: dict[str, dict[str, torch.Tensor]],
        targets: list[dict[str, Any]],
    ) -> dict[str, LevelAssignment]:
        assignments: dict[str, LevelAssignment] = {}

        for level_name, cls_logits in outputs["cls"].items():
            device = cls_logits.device
            batch_size, _, height, width = cls_logits.shape
            stride = self.strides[level_name]
            points = make_grid(height, width, stride, device=device)

            cls_targets = torch.zeros((batch_size, 1, height, width), device=device)
            obj_targets = torch.zeros((batch_size, 1, height, width), device=device)
            box_targets = torch.zeros((batch_size, 4, height, width), device=device)
            box_xyxy = torch.zeros((batch_size, 4, height, width), device=device)
            pos_mask = torch.zeros((batch_size, 1, height, width), device=device, dtype=torch.bool)
            track_ids = torch.full((batch_size, height, width), -1, device=device, dtype=torch.long)
            assigned_area = torch.full((batch_size, height, width), float("inf"), device=device)

            area_min, area_max = self.area_ranges[level_name]
            for batch_index, target in enumerate(targets):
                boxes = target["boxes"].to(device)
                ids = target["track_labels"].to(device)
                if boxes.numel() == 0:
                    continue

                box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                centers_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
                centers_y = 0.5 * (boxes[:, 1] + boxes[:, 3])

                for gt_index, box in enumerate(boxes):
                    area = box_areas[gt_index].item()
                    if not (area_min <= area < area_max):
                        continue

                    center_x = centers_x[gt_index] / stride - 0.5
                    center_y = centers_y[gt_index] / stride - 0.5

                    x_min = max(0, int(torch.floor(center_x - self.center_radius).item()))
                    x_max = min(width - 1, int(torch.ceil(center_x + self.center_radius).item()))
                    y_min = max(0, int(torch.floor(center_y - self.center_radius).item()))
                    y_max = min(height - 1, int(torch.ceil(center_y + self.center_radius).item()))

                    for yy in range(y_min, y_max + 1):
                        for xx in range(x_min, x_max + 1):
                            point = points[yy, xx]
                            left = point[0] - box[0]
                            top = point[1] - box[1]
                            right = box[2] - point[0]
                            bottom = box[3] - point[1]
                            if min(left, top, right, bottom) <= 0:
                                continue
                            if area >= assigned_area[batch_index, yy, xx]:
                                continue

                            assigned_area[batch_index, yy, xx] = area
                            cls_targets[batch_index, 0, yy, xx] = 1.0
                            pos_mask[batch_index, 0, yy, xx] = True
                            track_ids[batch_index, yy, xx] = ids[gt_index]
                            box_targets[batch_index, :, yy, xx] = torch.tensor(
                                [left, top, right, bottom],
                                device=device,
                            )
                            box_xyxy[batch_index, :, yy, xx] = box

                            lr_min = min(left, right)
                            lr_max = max(left, right)
                            tb_min = min(top, bottom)
                            tb_max = max(top, bottom)
                            centerness = ((lr_min / lr_max) * (tb_min / tb_max)) ** 0.5
                            obj_targets[batch_index, 0, yy, xx] = centerness

            assignments[level_name] = LevelAssignment(
                cls_targets=cls_targets,
                obj_targets=obj_targets,
                box_targets=box_targets,
                box_xyxy=box_xyxy,
                pos_mask=pos_mask,
                track_ids=track_ids,
                points=points,
                stride=stride,
            )

        return assignments
