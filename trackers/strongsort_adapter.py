from __future__ import annotations

import numpy as np


class StrongSortAdapter:
    def __init__(self, device: str = "cuda", **kwargs) -> None:
        try:
            from boxmot import StrongSort
        except ImportError as exc:
            raise ImportError(
                "StrongSORT support requires the `boxmot` package."
            ) from exc
        self.tracker = StrongSort(device=device, **kwargs)

    def update(self, detections: list[dict], image=None) -> list[dict]:
        if not detections:
            return []
        det_array = []
        embeds = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            det_array.append([float(x1), float(y1), float(x2), float(y2), float(det["score"]), 0.0])
            embeds.append(np.asarray(det["embedding"], dtype=np.float32))

        outputs = self.tracker.update(np.asarray(det_array, dtype=np.float32), image, np.asarray(embeds))
        results = []
        for row in outputs:
            x1, y1, x2, y2, track_id, score, *_ = row.tolist()
            results.append(
                {
                    "track_id": int(track_id),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "score": float(score),
                }
            )
        return results
