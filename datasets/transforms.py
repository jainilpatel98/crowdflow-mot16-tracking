from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TransformConfig:
    input_size: tuple[int, int] = (640, 640)          # (H, W)
    letterbox: bool = True
    # Color jitter (HSV, YOLO-style)
    hsv_h: float = 0.015          # hue shift fraction
    hsv_s: float = 0.7            # saturation gain range: [1-s, 1+s]
    hsv_v: float = 0.4            # value gain range:      [1-v, 1+v]
    # Geometric
    horizontal_flip_prob: float = 0.5
    affine_degrees: float = 3.0
    affine_translate: float = 0.05
    affine_scale_min: float = 0.5      # YOLO-style large-scale jitter
    affine_scale_max: float = 2.0
    # Mosaic
    mosaic_prob: float = 0.8      # probability of applying mosaic per sample
    # Copy-paste occlusion augmentation
    copy_paste_prob: float = 0.3
    # Motion blur
    motion_blur_prob: float = 0.1
    blur_kernel: int = 5
    # Random erasing (simulate occlusion)
    erase_prob: float = 0.2
    erase_max_area: float = 0.15  # max fraction of box area to erase


# ---------------------------------------------------------------------------
# Letterbox helpers
# ---------------------------------------------------------------------------

def letterbox_image(
    image: Image.Image,
    target_size: tuple[int, int],
    fill: tuple[int, int, int] = (114, 114, 114),
) -> tuple[Image.Image, float, tuple[int, int]]:
    """Resize image preserving aspect ratio, pad to target_size.

    Returns:
        letterboxed image, scale factor, (pad_top, pad_left)
    """
    orig_w, orig_h = image.size
    target_h, target_w = target_size
    scale = min(target_h / orig_h, target_w / orig_w)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    canvas = Image.new("RGB", (target_w, target_h), fill)
    canvas.paste(resized, (pad_left, pad_top))
    return canvas, scale, (pad_top, pad_left)


def unletterbox_boxes_xyxy(
    boxes: torch.Tensor,
    scale: float,
    pad: tuple[int, int],
    orig_size: tuple[int, int],
) -> torch.Tensor:
    """Invert letterbox transform on predicted boxes (xyxy, in letterboxed space).

    Args:
        boxes:     (N, 4) xyxy in letterboxed image coordinates
        scale:     the scale used during letterbox_image
        pad:       (pad_top, pad_left)
        orig_size: (orig_H, orig_W) of the original image

    Returns:
        (N, 4) xyxy clipped to original image size
    """
    pad_top, pad_left = pad
    orig_h, orig_w = orig_size
    boxes = boxes.clone().float()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_h)
    return boxes


def _boxes_after_letterbox(
    boxes: np.ndarray,
    scale: float,
    pad_top: int,
    pad_left: int,
) -> np.ndarray:
    """Apply letterbox transform to xyxy numpy boxes."""
    if len(boxes) == 0:
        return boxes
    b = boxes.copy().astype(np.float32)
    b[:, [0, 2]] = b[:, [0, 2]] * scale + pad_left
    b[:, [1, 3]] = b[:, [1, 3]] * scale + pad_top
    return b


# ---------------------------------------------------------------------------
# Color augmentation
# ---------------------------------------------------------------------------

def _hsv_jitter(image: Image.Image, h: float, s: float, v: float) -> Image.Image:
    """Apply YOLO-style HSV jitter."""
    img = np.array(image, dtype=np.uint8)
    # Convert to HSV
    img_hsv = np.array(Image.fromarray(img).convert("HSV"), dtype=np.float32)
    hue_shift = random.uniform(-h * 180, h * 180)
    sat_gain = random.uniform(1 - s, 1 + s)
    val_gain = random.uniform(1 - v, 1 + v)
    img_hsv[..., 0] = (img_hsv[..., 0] + hue_shift) % 180
    img_hsv[..., 1] = (img_hsv[..., 1] * sat_gain).clip(0, 255)
    img_hsv[..., 2] = (img_hsv[..., 2] * val_gain).clip(0, 255)
    return Image.fromarray(img_hsv.astype(np.uint8), mode="HSV").convert("RGB")


# ---------------------------------------------------------------------------
# Mosaic augmentation
# ---------------------------------------------------------------------------

def _mosaic_4(
    samples: list[dict[str, Any]],
    target_size: tuple[int, int],
) -> dict[str, Any]:
    """Combine 4 samples into a single mosaic image.

    Each sample must have keys: image (PIL), boxes (np [N,4] xyxy),
    track_ids, track_labels, visibilities, ignore_boxes.
    """
    th, tw = target_size
    cx = tw // 2
    cy = th // 2
    canvas = Image.new("RGB", (tw, th), (114, 114, 114))

    # quadrant placements: (x_start, y_start, x_end, y_end)
    placements = [
        (0, 0, cx, cy),
        (cx, 0, tw, cy),
        (0, cy, cx, th),
        (cx, cy, tw, th),
    ]

    all_boxes: list[np.ndarray] = []
    all_track_ids: list[np.ndarray] = []
    all_track_labels: list[np.ndarray] = []
    all_vis: list[np.ndarray] = []
    all_ignore: list[np.ndarray] = []

    for sample, (x1, y1, x2, y2) in zip(samples, placements):
        cell_h = y2 - y1
        cell_w = x2 - x1
        img_lb, scale, (pt, pl) = letterbox_image(sample["image"], (cell_h, cell_w))
        canvas.paste(img_lb, (x1, y1))

        boxes = sample["boxes"]  # (N, 4) xyxy in original space
        if len(boxes) > 0:
            b = _boxes_after_letterbox(boxes, scale, pt, pl)
            # shift to canvas position
            b[:, [0, 2]] += x1
            b[:, [1, 3]] += y1
            # clip to cell
            b[:, [0, 2]] = b[:, [0, 2]].clip(x1, x2)
            b[:, [1, 3]] = b[:, [1, 3]].clip(y1, y2)
            # filter degenerate boxes
            valid = ((b[:, 2] - b[:, 0]) > 2) & ((b[:, 3] - b[:, 1]) > 2)
            b = b[valid]
            all_boxes.append(b)
            all_track_ids.append(sample["track_ids"][valid])
            all_track_labels.append(sample["track_labels"][valid])
            all_vis.append(sample["visibilities"][valid])

        ign = sample.get("ignore_boxes", np.zeros((0, 4), dtype=np.float32))
        if len(ign) > 0:
            ign2 = _boxes_after_letterbox(ign, scale, pt, pl)
            ign2[:, [0, 2]] += x1
            ign2[:, [1, 3]] += y1
            all_ignore.append(ign2)

    merged_boxes = np.concatenate(all_boxes, axis=0) if all_boxes else np.zeros((0, 4), np.float32)
    merged_ids = np.concatenate(all_track_ids) if all_track_ids else np.zeros((0,), dtype=np.int64)
    merged_labels = np.concatenate(all_track_labels) if all_track_labels else np.zeros((0,), dtype=np.int64)
    merged_vis = np.concatenate(all_vis) if all_vis else np.zeros((0,), dtype=np.float32)
    merged_ign = np.concatenate(all_ignore, axis=0) if all_ignore else np.zeros((0, 4), np.float32)

    return {
        "image": canvas,
        "boxes": merged_boxes,
        "track_ids": merged_ids,
        "track_labels": merged_labels,
        "visibilities": merged_vis,
        "ignore_boxes": merged_ign,
        # metadata from first sample
        "orig_size": samples[0]["orig_size"],
        "image_path": samples[0]["image_path"],
        "sequence_name": samples[0]["sequence_name"],
        "frame_idx": samples[0]["frame_idx"],
        "teacher_cache_path": None,   # mosaic invalidates cache
    }


# ---------------------------------------------------------------------------
# Copy-paste (paste random person instances onto image)
# ---------------------------------------------------------------------------

def _copy_paste(
    target: dict[str, Any],
    donors: list[dict[str, Any]],
    target_size: tuple[int, int],
    max_instances: int = 5,
) -> dict[str, Any]:
    """Paste random foreground crops from donor samples onto target image."""
    if not donors or not random.random() < 1.0:
        return target
    th, tw = target_size
    img = target["image"].copy()
    boxes = target["boxes"].copy()
    track_ids = target["track_ids"].copy()
    track_labels = target["track_labels"].copy()
    vis = target["visibilities"].copy()

    num_paste = random.randint(1, max_instances)
    added_boxes: list[np.ndarray] = []
    added_ids: list[np.ndarray] = []
    added_labels: list[np.ndarray] = []
    added_vis: list[np.ndarray] = []

    for _ in range(num_paste):
        donor = random.choice(donors)
        d_boxes = donor["boxes"]
        if len(d_boxes) == 0:
            continue
        idx = random.randint(0, len(d_boxes) - 1)
        x1, y1, x2, y2 = d_boxes[idx].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(donor["image"].width, x2)
        y2 = min(donor["image"].height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = donor["image"].crop((x1, y1, x2, y2))
        w, h = crop.size
        # random placement
        px = random.randint(0, max(0, tw - w))
        py = random.randint(0, max(0, th - h))
        img.paste(crop, (px, py))
        added_boxes.append(np.array([[px, py, px + w, py + h]], dtype=np.float32))
        added_ids.append(np.array([donor["track_ids"][idx]], dtype=np.int64))
        added_labels.append(np.array([donor["track_labels"][idx]], dtype=np.int64))
        added_vis.append(np.array([donor["visibilities"][idx]], dtype=np.float32))

    if added_boxes:
        boxes = np.concatenate([boxes] + added_boxes, axis=0)
        track_ids = np.concatenate([track_ids] + added_ids)
        track_labels = np.concatenate([track_labels] + added_labels)
        vis = np.concatenate([vis] + added_vis)

    return {**target, "image": img, "boxes": boxes, "track_ids": track_ids,
            "track_labels": track_labels, "visibilities": vis}


# ---------------------------------------------------------------------------
# Single-sample transform (letterbox + optional augmentation)
# ---------------------------------------------------------------------------

def _apply_affine_and_flip(
    image: Image.Image,
    boxes: np.ndarray,
    cfg: TransformConfig,
    train: bool,
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Apply random affine transform + horizontal flip (training only).

    Returns:
        image, boxes, valid_mask — a boolean mask over the *original* box indices
        that survived the affine/flip filter.  Apply the mask to all parallel
        arrays (track_ids, track_labels, visibilities) in the caller.
        For val (train=False) the mask is all-True.
    """
    if not train:
        return image, boxes, np.ones(len(boxes), dtype=bool)

    w, h = image.size
    # Cumulative validity (starts all-True)
    valid = np.ones(len(boxes), dtype=bool)

    do_flip = random.random() < cfg.horizontal_flip_prob
    if do_flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if len(boxes) > 0:
            boxes = boxes.copy()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

    # Small affine: rotation + translation (scale handled at letterbox stage)
    if cfg.affine_degrees > 0 or cfg.affine_translate > 0:
        angle = random.uniform(-cfg.affine_degrees, cfg.affine_degrees)
        tx = random.uniform(-cfg.affine_translate, cfg.affine_translate) * w
        ty = random.uniform(-cfg.affine_translate, cfg.affine_translate) * h
        image = image.rotate(angle, translate=(tx, ty), resample=Image.BILINEAR, fillcolor=(114, 114, 114))
        if len(boxes) > 0:
            boxes = boxes.copy()
            boxes[:, [0, 2]] += tx
            boxes[:, [1, 3]] += ty
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
            affine_valid = ((boxes[:, 2] - boxes[:, 0]) > 2) & ((boxes[:, 3] - boxes[:, 1]) > 2)
            boxes = boxes[affine_valid]
            valid = valid & affine_valid  # accumulate into the overall mask

    return image, boxes, valid


def _to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)   # (3, H, W)


# ---------------------------------------------------------------------------
# MotTransforms
# ---------------------------------------------------------------------------

class MotTransforms:
    """Applies the full MOT augmentation pipeline.

    For training:  HSV jitter → mosaic (managed by dataset) → scale jitter
                   → letterbox → affine + flip → motion blur → random erase
    For validation: letterbox only (deterministic)
    """

    def __init__(self, cfg: TransformConfig, train: bool = True) -> None:
        self.cfg = cfg
        self.train = train

    # ------------------------------------------------------------------
    # Public interface called by the dataset
    # ------------------------------------------------------------------

    def apply_single(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Transform a single raw sample dict into a model-ready dict."""
        cfg = self.cfg
        image: Image.Image = sample["image"]
        boxes: np.ndarray  = sample["boxes"]
        track_ids:     np.ndarray = sample["track_ids"]
        track_labels:  np.ndarray = sample["track_labels"]
        visibilities:  np.ndarray = sample["visibilities"]

        if self.train:
            # HSV color jitter
            image = _hsv_jitter(image, cfg.hsv_h, cfg.hsv_s, cfg.hsv_v)
            # Affine + flip — returns valid mask to keep boxes/metadata in sync
            image, boxes, valid = _apply_affine_and_flip(image, boxes, cfg, train=True)
            if not valid.all():
                track_ids    = track_ids[valid]
                track_labels = track_labels[valid]
                visibilities = visibilities[valid]

        # Scale jitter (train) or fixed scale (val)
        if self.train:
            scale_factor = random.uniform(cfg.affine_scale_min, cfg.affine_scale_max)
            orig_w, orig_h = image.size
            jw = max(1, int(orig_w * scale_factor))
            jh = max(1, int(orig_h * scale_factor))
            image = image.resize((jw, jh), Image.BILINEAR)
            if len(boxes) > 0:
                boxes = boxes.copy().astype(np.float32)
                boxes *= scale_factor

        # Letterbox to input_size
        image_lb, scale, (pad_top, pad_left) = letterbox_image(image, cfg.input_size)
        lb_boxes = _boxes_after_letterbox(boxes, scale, pad_top, pad_left) if len(boxes) > 0 else boxes

        if self.train:
            # Motion blur
            if random.random() < cfg.motion_blur_prob:
                image_lb = image_lb.filter(ImageFilter.GaussianBlur(radius=1))

        # Convert to tensor
        image_tensor = _to_tensor(image_lb)

        return {
            "images":       image_tensor,
            "boxes":        torch.as_tensor(lb_boxes,     dtype=torch.float32),
            "track_ids":    torch.as_tensor(track_ids,    dtype=torch.long),
            "track_labels": torch.as_tensor(track_labels, dtype=torch.long),
            "visibilities": torch.as_tensor(visibilities, dtype=torch.float32),
            "ignore_boxes": torch.as_tensor(sample.get("ignore_boxes", np.zeros((0, 4), np.float32)), dtype=torch.float32),
            "orig_size":    torch.as_tensor(sample["orig_size"],   dtype=torch.long),
            "image_size":   torch.as_tensor(cfg.input_size,        dtype=torch.long),
            "resize_scale": scale,
            "pad":          (pad_top, pad_left),
            "image_path":      sample["image_path"],
            "sequence_name":   sample["sequence_name"],
            "frame_idx":       sample["frame_idx"],
            "teacher_cache_path": sample.get("teacher_cache_path"),
        }

    def apply_mosaic(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Mosaic-combine 4 raw samples and apply post-mosaic augmentation."""
        cfg = self.cfg
        assert len(samples) == 4

        # Apply per-image color jitter before merging
        for s in samples:
            s["image"] = _hsv_jitter(s["image"], cfg.hsv_h, cfg.hsv_s, cfg.hsv_v)

        merged = _mosaic_4(samples, cfg.input_size)

        # Post-mosaic affine + flip — apply valid mask to all parallel arrays
        image, boxes, valid = _apply_affine_and_flip(
            merged["image"], merged["boxes"], cfg, train=True
        )
        track_ids    = merged["track_ids"][valid]
        track_labels = merged["track_labels"][valid]
        visibilities = merged["visibilities"][valid]

        # Motion blur
        if random.random() < cfg.motion_blur_prob:
            image = image.filter(ImageFilter.GaussianBlur(radius=1))

        image_tensor = _to_tensor(image)

        return {
            "images": image_tensor,
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "track_ids":    torch.as_tensor(track_ids,    dtype=torch.long),
            "track_labels": torch.as_tensor(track_labels, dtype=torch.long),
            "visibilities": torch.as_tensor(visibilities, dtype=torch.float32),
            "ignore_boxes": torch.as_tensor(merged["ignore_boxes"], dtype=torch.float32),
            "orig_size": torch.as_tensor(merged["orig_size"], dtype=torch.long),
            "image_size": torch.as_tensor(cfg.input_size, dtype=torch.long),
            "resize_scale": 1.0,
            "pad": (0, 0),
            "image_path": merged["image_path"],
            "sequence_name": merged["sequence_name"],
            "frame_idx": merged["frame_idx"],
            "teacher_cache_path": None,   # mosaic always invalidates cache
        }

    def build_transform_config_from_dict(d: dict) -> TransformConfig:
        """Helper to build TransformConfig from a YAML augmentation dict."""
        return TransformConfig(
            hsv_h=d.get("hue", 0.015),
            hsv_s=d.get("saturation", 0.7),
            hsv_v=d.get("brightness", 0.4),
            horizontal_flip_prob=d.get("horizontal_flip_prob", 0.5),
            affine_degrees=d.get("affine_degrees", 3.0),
            affine_translate=d.get("affine_translate", 0.05),
            affine_scale_min=d.get("affine_scale", [0.5, 2.0])[0],
            affine_scale_max=d.get("affine_scale", [0.5, 2.0])[1],
            mosaic_prob=d.get("mosaic_prob", 0.8),
            copy_paste_prob=d.get("copy_paste_prob", 0.3),
            motion_blur_prob=d.get("motion_blur_prob", 0.1),
            blur_kernel=d.get("blur_kernel", 5),
            erase_prob=d.get("erase_prob", 0.2),
        )
