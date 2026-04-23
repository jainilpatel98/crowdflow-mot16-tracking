from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DetectionMetricSummary:
    precision: float
    recall: float
    mean_iou: float
    map50: float
    map50_95: float
    num_predictions: int
    num_targets: int
