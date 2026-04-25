from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

from engine.hooks import save_checkpoint
from losses.det_loss import detection_loss
from losses.id_loss import id_supervision_loss
from losses.kd_loss import box_kd_loss, classification_kd_loss, embedding_cosine_loss, feature_distill_loss
from utils.distributed import is_main_process, reduce_scalar
from utils.logger import SmoothedValue


# ---------------------------------------------------------------------------
# Phase weight schedule
# ---------------------------------------------------------------------------

class PhaseWeightSchedule:
    def __init__(self, phases: list[dict[str, Any]]) -> None:
        self.phases = phases

    def for_epoch(self, epoch: int) -> dict[str, float]:
        for phase in self.phases:
            if phase["start_epoch"] <= epoch <= phase["end_epoch"]:
                return phase["weights"]
        return self.phases[-1]["weights"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _move_targets_to_device(targets: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    moved = []
    for target in targets:
        moved_target = dict(target)
        for key in ("boxes", "visibilities", "track_ids", "track_labels",
                    "ignore_boxes", "orig_size", "image_size", "resize_scale", "pad"):
            if key in moved_target and torch.is_tensor(moved_target[key]):
                moved_target[key] = moved_target[key].to(device)
        moved.append(moved_target)
    return moved


def _select_supervision_boxes(
    targets: list[dict[str, Any]],
    min_visibility: float,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    selected_boxes = []
    selected_labels = []
    for target in targets:
        vis = target["visibilities"]
        if vis.numel() == 0:
            selected_boxes.append(target["boxes"][:0])
            continue
        mask = vis >= min_visibility
        selected_boxes.append(target["boxes"][mask])
        if target["track_labels"].numel() > 0:
            selected_labels.append(target["track_labels"][mask])
    labels = (
        torch.cat(selected_labels, dim=0)
        if selected_labels
        else targets[0]["boxes"].new_zeros((0,), dtype=torch.long)
    )
    return selected_boxes, labels


def _load_teacher_cache_batch(
    cache_paths: list[str | None],
    device: torch.device,
) -> dict[str, Any] | None:
    """Load a pre-cached batch of teacher outputs.

    Returns None if any path is missing, signalling the caller to run the
    teacher live instead.

    NOTE: Cache was computed on deterministic (non-augmented) images. When
    the student is trained with heavy augmentation and use_cache=True, the
    feature-KD and box-KD losses will see a spatial mismatch between the
    student (augmented) and teacher (canonical) inputs. The detection and ID
    losses are unaffected. Default is use_cache=False (live teacher with
    shared augmented input).
    """
    if not cache_paths or any(p is None or not Path(p).exists() for p in cache_paths):
        return None

    caches = [torch.load(p, map_location=device) for p in cache_paths]
    levels = list(caches[0]["features"].keys())
    batch: dict[str, Any] = {
        "features":  {lv: torch.cat([c["features"][lv]  for c in caches], dim=0) for lv in levels},
        "logits":    {lv: torch.cat([c["logits"][lv]    for c in caches], dim=0) for lv in levels},
        "raw_boxes": {lv: torch.cat([c["raw_boxes"][lv] for c in caches], dim=0) for lv in levels},
        "boxes":     [c["boxes"]  for c in caches],
        "scores":    [c["scores"] for c in caches],
        "spatial_feat": torch.cat([c["spatial_feat"] for c in caches], dim=0),
    }
    if all("roi_embeddings" in c for c in caches):
        batch["roi_embeddings_per_image"] = [c["roi_embeddings"] for c in caches]
    if all("visibilities" in c for c in caches):
        batch["visibilities_per_image"] = [c["visibilities"] for c in caches]
    return batch


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DistillationTrainer:
    """Trains the student JDE model with multi-signal knowledge distillation.

    Distillation signals:
      - cls_kd : soft classification targets (KL divergence, teacher logits)
      - box_kd : box regression imitation (Smooth-L1 + IoU at positive cells)
      - feat    : feature-map MSE after adapter projection
      - emb     : ROI embedding cosine alignment (student ↔ teacher_roi_projector)
      - det     : hard detection loss on ground-truth labels (focal + IoU)
      - id      : identity cross-entropy on track IDs (student + teacher projector)

    The teacher_roi_projector is trained jointly with the student and
    supervised by ID cross-entropy. Its embeddings serve as soft targets for
    the student's embedding cosine loss.
    """

    def __init__(
        self,
        student_model,
        teacher_model,
        feature_adapters,
        teacher_roi_projector,          # TeacherROIProjector (trainable)
        teacher_id_classifier,          # nn.Linear for teacher projector ID loss
        assigner,
        optimizer,
        scheduler,
        device: torch.device,
        phases: list[dict[str, Any]],
        output_dir: str | Path,
        amp: bool = True,
        grad_clip: float = 1.0,
        temperature: float = 2.0,
        use_teacher_cache: bool = False,
        embedding_min_visibility: float = 0.25,
        id_min_visibility: float = 0.25,
        logger=None,
    ) -> None:
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.feature_adapters = feature_adapters
        self.teacher_roi_projector = teacher_roi_projector
        self.teacher_id_classifier = teacher_id_classifier
        self.assigner = assigner
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.amp = amp
        self.grad_clip = grad_clip
        self.temperature = temperature
        self.use_teacher_cache = use_teacher_cache
        self.logger = logger
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
        self.phase_schedule = PhaseWeightSchedule(phases)
        self.best_val = float("inf")
        self.embedding_min_visibility = embedding_min_visibility
        self.id_min_visibility = id_min_visibility

    # ------------------------------------------------------------------
    # Module accessors (handles DDP wrapping)
    # ------------------------------------------------------------------

    def _unwrap(self, module):
        return module.module if hasattr(module, "module") else module

    def _student(self):
        return self._unwrap(self.student_model)

    def _adapters(self):
        return self._unwrap(self.feature_adapters) if self.feature_adapters is not None else None

    def _tea_proj(self):
        return self._unwrap(self.teacher_roi_projector) if self.teacher_roi_projector is not None else None

    # ------------------------------------------------------------------
    # Teacher outputs
    # ------------------------------------------------------------------

    def _compute_teacher_outputs(
        self,
        images: torch.Tensor,
        cache_paths: list[str | None],
    ) -> dict[str, Any]:
        if self.use_teacher_cache:
            cached = _load_teacher_cache_batch(cache_paths, self.device)
            if cached is not None:
                return cached
        if self.teacher_model is None:
            raise RuntimeError(
                "Teacher outputs required but teacher_model is None and cache is unavailable. "
                "Either provide a teacher checkpoint or pre-compute the cache."
            )
        return self.teacher_model(images)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_total_loss(
        self,
        student_outputs: dict[str, Any],
        teacher_outputs: dict[str, Any],
        targets: list[dict[str, Any]],
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        weights = self.phase_schedule.for_epoch(epoch)
        assignments = self.assigner.assign(student_outputs, targets)

        # --- Detection loss ---
        det_losses = detection_loss(student_outputs, assignments)

        # --- Classification KD ---
        cls_kd = classification_kd_loss(
            student_outputs["cls"], teacher_outputs["logits"],
            temperature=self.temperature,
        )

        # --- Box KD ---
        box_kd = box_kd_loss(
            student_outputs["box"], teacher_outputs["raw_boxes"], assignments,
        )

        # --- Feature distillation ---
        feat_kd = feature_distill_loss(
            student_outputs["features"],
            teacher_outputs["features"],
            adapters=self._adapters(),
        )

        # --- Embedding KD (student ROI ↔ teacher ROI projector) ---
        image_size = tuple(targets[0]["image_size"].tolist())
        emb_boxes, _ = _select_supervision_boxes(targets, self.embedding_min_visibility)

        student_roi = student_outputs.get("roi_embeddings")
        if student_roi is None:
            student_roi = self._student().extract_roi_embeddings(
                student_outputs, boxes_per_image=emb_boxes, image_size=image_size,
            )

        # Teacher embedding: use trainable projector on p3 (live or cached)
        tea_proj = self._tea_proj()
        if "roi_embeddings_per_image" in teacher_outputs and tea_proj is None:
            # Legacy cache path: cached embeddings available
            vis_lists = teacher_outputs.get("visibilities_per_image")
            teacher_roi_list = []
            for i, embs in enumerate(teacher_outputs["roi_embeddings_per_image"]):
                embs = embs.to(self.device)
                if vis_lists is not None:
                    vis = vis_lists[i].to(self.device)
                    embs = embs[vis >= self.embedding_min_visibility]
                teacher_roi_list.append(embs)
            teacher_roi = (
                torch.cat(teacher_roi_list, dim=0)
                if teacher_roi_list
                else student_roi.new_zeros((0, student_roi.shape[-1]))
            )
        elif tea_proj is not None:
            # Use trainable teacher projector on p3 (same spatial_feat = p3)
            teacher_roi = tea_proj.extract_embeddings(
                teacher_outputs["spatial_feat"],
                boxes_per_image=emb_boxes,
                image_size=image_size,
            )
        else:
            teacher_roi = student_roi.new_zeros((0, student_roi.shape[-1]))

        emb_kd = embedding_cosine_loss(student_roi, teacher_roi)

        # --- ID loss (student) ---
        id_boxes, id_targets = _select_supervision_boxes(targets, self.id_min_visibility)
        if weights["id"] > 0.0 and id_targets.numel() > 0:
            id_logits = student_outputs.get("id_logits")
            if id_logits is None:
                if self.id_min_visibility == self.embedding_min_visibility:
                    id_embeddings = student_roi
                else:
                    id_embeddings = student_outputs.get("id_embeddings")
                    if id_embeddings is None:
                        id_embeddings = self._student().extract_roi_embeddings(
                            student_outputs, boxes_per_image=id_boxes, image_size=image_size,
                        )
                id_logits = self._student().classify_ids(id_embeddings)
            id_loss = id_supervision_loss(id_logits, id_targets)

            # Also supervise the teacher projector with ID loss (makes it discriminative)
            if tea_proj is not None and self.teacher_id_classifier is not None and teacher_roi.shape[0] > 0:
                tea_id_logits = self.teacher_id_classifier(teacher_roi)
                tea_id_loss = id_supervision_loss(tea_id_logits, id_targets[:teacher_roi.shape[0]])
                id_loss = id_loss + tea_id_loss
        else:
            id_loss = student_roi.sum() * 0.0

        # --- Total ---
        total_loss = (
            weights["det"]    * det_losses["total"]
            + weights["cls_kd"] * cls_kd
            + weights["box_kd"] * box_kd
            + weights["feat"]   * feat_kd
            + weights["emb"]    * emb_kd
            + weights["id"]     * id_loss
        )

        log_items = {
            "loss":    float(total_loss.detach()),
            "det":     float(det_losses["total"].detach()),
            "det_cls": float(det_losses["cls"].detach()),
            "det_obj": float(det_losses["obj"].detach()),
            "det_box": float(det_losses["box"].detach()),
            "cls_kd":  float(cls_kd.detach()),
            "box_kd":  float(box_kd.detach()),
            "feat_kd": float(feat_kd.detach()),
            "emb_kd":  float(emb_kd.detach()),
            "id_loss": float(id_loss.detach()),
        }
        return total_loss, log_items

    def _reduce_log_items(self, log_items: dict[str, float]) -> dict[str, float]:
        return {k: reduce_scalar(v, self.device) for k, v in log_items.items()}

    # ------------------------------------------------------------------
    # Training / validation loops
    # ------------------------------------------------------------------

    def _set_train(self) -> None:
        self.student_model.train()
        if self.feature_adapters is not None:
            self.feature_adapters.train()
        if self.teacher_roi_projector is not None:
            self.teacher_roi_projector.train()
        if self.teacher_id_classifier is not None:
            self.teacher_id_classifier.train()

    def _set_eval(self) -> None:
        self.student_model.eval()
        if self.feature_adapters is not None:
            self.feature_adapters.eval()
        if self.teacher_roi_projector is not None:
            self.teacher_roi_projector.eval()
        if self.teacher_id_classifier is not None:
            self.teacher_id_classifier.eval()

    def train_one_epoch(self, data_loader, epoch: int) -> dict[str, float]:
        self._set_train()
        meter_keys = ["loss", "det", "cls_kd", "box_kd", "feat_kd", "emb_kd", "id_loss"]
        meters = {k: SmoothedValue() for k in meter_keys}

        progress = tqdm(data_loader, desc=f"Train {epoch}", leave=False,
                        disable=not is_main_process())
        for batch in progress:
            images = batch["images"].to(self.device, non_blocking=True)
            targets = _move_targets_to_device(batch["targets"], self.device)
            teacher_outputs = self._compute_teacher_outputs(
                images, batch.get("teacher_cache_paths", [])
            )
            image_size = tuple(targets[0]["image_size"].tolist())
            phase_weights = self.phase_schedule.for_epoch(epoch)
            emb_boxes, _ = _select_supervision_boxes(targets, self.embedding_min_visibility)
            id_boxes = None
            if phase_weights["id"] > 0.0:
                id_boxes, _ = _select_supervision_boxes(targets, self.id_min_visibility)
                if self.id_min_visibility == self.embedding_min_visibility:
                    id_boxes = emb_boxes

            with torch.amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                student_outputs = self.student_model(
                    images,
                    roi_boxes_per_image=emb_boxes,
                    roi_image_size=image_size,
                    id_boxes_per_image=id_boxes,
                )
                total_loss, log_items = self._compute_total_loss(
                    student_outputs, teacher_outputs, targets, epoch
                )

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.student_model.parameters())
                + (list(self.feature_adapters.parameters()) if self.feature_adapters else [])
                + (list(self.teacher_roi_projector.parameters()) if self.teacher_roi_projector else [])
                + (list(self.teacher_id_classifier.parameters()) if self.teacher_id_classifier else []),
                self.grad_clip,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()

            reduced = self._reduce_log_items(log_items)
            for k, m in meters.items():
                m.update(reduced[k])
            if is_main_process() and hasattr(progress, "set_postfix"):
                progress.set_postfix({k: f"{m.avg:.4f}" for k, m in meters.items()})

        return {k: m.avg for k, m in meters.items()}

    @torch.no_grad()
    def validate(self, data_loader, epoch: int) -> dict[str, float]:
        self._set_eval()
        meter_keys = ["loss", "det", "cls_kd", "box_kd", "feat_kd", "emb_kd", "id_loss"]
        meters = {k: SmoothedValue() for k in meter_keys}

        progress = tqdm(data_loader, desc=f"Val {epoch}", leave=False,
                        disable=not is_main_process())
        for batch in progress:
            images = batch["images"].to(self.device, non_blocking=True)
            targets = _move_targets_to_device(batch["targets"], self.device)
            teacher_outputs = self._compute_teacher_outputs(
                images, batch.get("teacher_cache_paths", [])
            )
            image_size = tuple(targets[0]["image_size"].tolist())
            phase_weights = self.phase_schedule.for_epoch(epoch)
            emb_boxes, _ = _select_supervision_boxes(targets, self.embedding_min_visibility)
            id_boxes = None
            if phase_weights["id"] > 0.0:
                id_boxes, _ = _select_supervision_boxes(targets, self.id_min_visibility)
                if self.id_min_visibility == self.embedding_min_visibility:
                    id_boxes = emb_boxes
            student_outputs = self.student_model(
                images,
                roi_boxes_per_image=emb_boxes,
                roi_image_size=image_size,
                id_boxes_per_image=id_boxes,
            )
            _, log_items = self._compute_total_loss(
                student_outputs, teacher_outputs, targets, epoch
            )
            reduced = self._reduce_log_items(log_items)
            for k, m in meters.items():
                m.update(reduced[k])
            if is_main_process() and hasattr(progress, "set_postfix"):
                progress.set_postfix({k: f"{m.avg:.4f}" for k, m in meters.items()})

        return {k: m.avg for k, m in meters.items()}

    def fit(self, train_loader, val_loader, epochs: int) -> None:
        history = []
        for epoch in range(1, epochs + 1):
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            if hasattr(val_loader, "sampler") and hasattr(val_loader.sampler, "set_epoch"):
                val_loader.sampler.set_epoch(epoch)

            train_metrics = self.train_one_epoch(train_loader, epoch)
            val_metrics   = self.validate(val_loader, epoch)
            record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            history.append(record)

            if self.logger is not None and is_main_process():
                self.logger.info("Epoch %d train=%s val=%s", epoch, train_metrics, val_metrics)

            if not is_main_process():
                continue

            projector_state = (
                self._unwrap(self.teacher_roi_projector).state_dict()
                if self.teacher_roi_projector is not None else None
            )
            tea_id_state = (
                self._unwrap(self.teacher_id_classifier).state_dict()
                if self.teacher_id_classifier is not None else None
            )
            checkpoint = {
                "epoch": epoch,
                "student": self._student().state_dict(),
                "feature_adapters": (self._adapters().state_dict() if self._adapters() is not None else None),
                "teacher_roi_projector": projector_state,
                "teacher_id_classifier": tea_id_state,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
            save_checkpoint(checkpoint, self.output_dir, "last.pt")
            if val_metrics["loss"] < self.best_val:
                self.best_val = val_metrics["loss"]
                save_checkpoint(checkpoint, self.output_dir, "best.pt")

            (self.output_dir / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
