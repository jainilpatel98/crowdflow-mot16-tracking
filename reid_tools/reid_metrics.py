from __future__ import annotations

from collections import defaultdict

import numpy as np


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return features / norms


def query_gallery_split(labels: np.ndarray, sequences: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Use the first sample per identity as query and remaining samples as gallery."""
    by_label: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        by_label[int(label)].append(index)

    query_indices: list[int] = []
    gallery_indices: list[int] = []
    for indices in by_label.values():
        if len(indices) < 2:
            continue
        query_indices.append(indices[0])
        gallery_indices.extend(indices[1:])

    return np.asarray(query_indices, dtype=np.int64), np.asarray(gallery_indices, dtype=np.int64)


def evaluate_reid(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    topk: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    features = l2_normalize(features.astype(np.float32))
    labels = labels.astype(np.int64)
    query_idx, gallery_idx = query_gallery_split(labels)
    if len(query_idx) == 0 or len(gallery_idx) == 0:
        return {"mAP": 0.0, **{f"rank{k}": 0.0 for k in topk}, "queries": 0.0, "gallery": float(len(gallery_idx))}

    qf = features[query_idx]
    gf = features[gallery_idx]
    ql = labels[query_idx]
    gl = labels[gallery_idx]
    distances = 1.0 - np.matmul(qf, gf.T)

    aps: list[float] = []
    rank_hits = {k: 0 for k in topk}
    for row, query_label in zip(distances, ql):
        order = np.argsort(row)
        matches = (gl[order] == query_label).astype(np.float32)
        if matches.sum() == 0:
            aps.append(0.0)
            continue
        cumulative = np.cumsum(matches)
        precision = cumulative / (np.arange(len(matches), dtype=np.float32) + 1.0)
        aps.append(float((precision * matches).sum() / matches.sum()))
        for k in topk:
            rank_hits[k] += int(matches[:k].sum() > 0)

    result = {
        "mAP": float(np.mean(aps)) if aps else 0.0,
        "queries": float(len(query_idx)),
        "gallery": float(len(gallery_idx)),
    }
    for k in topk:
        result[f"rank{k}"] = rank_hits[k] / max(1, len(query_idx))
    return result


def cosine_distance_summary(features: np.ndarray, labels: np.ndarray, max_pairs: int = 200_000) -> dict[str, float]:
    features = l2_normalize(features.astype(np.float32))
    labels = labels.astype(np.int64)
    n = len(labels)
    if n < 2:
        return {"same_cosine": 0.0, "different_cosine": 0.0}

    same: list[float] = []
    different: list[float] = []
    pair_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            cosine = float(np.dot(features[i], features[j]))
            if labels[i] == labels[j]:
                same.append(cosine)
            else:
                different.append(cosine)
            pair_count += 1
            if pair_count >= max_pairs:
                break
        if pair_count >= max_pairs:
            break

    return {
        "same_cosine": float(np.mean(same)) if same else 0.0,
        "different_cosine": float(np.mean(different)) if different else 0.0,
    }

