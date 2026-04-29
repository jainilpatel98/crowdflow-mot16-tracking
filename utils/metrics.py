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

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        if denom <= 0.0:
            return 0.0
        return 2.0 * self.precision * self.recall / denom
