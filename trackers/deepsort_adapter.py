from __future__ import annotations

import numpy as np


class DeepSortAdapter:
    def __init__(self, **kwargs) -> None:
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError as exc:
            raise ImportError(
                "DeepSORT support requires the `deep_sort_realtime` package."
            ) from exc
        self.tracker = DeepSort(**kwargs)

    def update(self, detections: list[dict], frame=None) -> list[dict]:
        bbs = []
        embeds = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            bbox_ltwh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            bbs.append((bbox_ltwh, float(det["score"]), "person"))
            embeds.append(np.asarray(det["embedding"], dtype=np.float32))

        tracks = self.tracker.update_tracks(bbs, embeds=embeds, frame=frame)
        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            det_conf = getattr(track, "det_conf", None)
            results.append(
                {
                    "track_id": int(track.track_id),
                    "bbox_xyxy": [float(v) for v in ltrb],
                    "score": float(det_conf if det_conf is not None else 1.0),
                }
            )
        return results
