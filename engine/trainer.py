from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal envs
    def tqdm(iterable, **kwargs):
        return iterable

from engine.hooks import save_checkpoint
from losses.det_loss import detection_loss
from losses.id_loss import id_supervision_loss
from losses.kd_loss import box_kd_loss, classification_kd_loss, embedding_cosine_loss, feature_distill_loss
from utils.distributed import is_main_process, reduce_scalar
from utils.logger import SmoothedValue


class PhaseWeightSchedule:
    def __init__(self, phases: list[dict[str, Any]]) -> None:
        self.phases = phases

    def for_epoch(self, epoch: int) -> dict[str, float]:
        for phase in self.phases:
            if phase["start_epoch"] <= epoch <= phase["end_epoch"]:
                return phase["weights"]
        return self.phases[-1]["weights"]


def _move_targets_to_device(targets: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    moved = []
    for target in targets:
        moved_target = dict(target)
        for key in ("boxes", "visibilities", "track_ids", "track_labels", "ignore_boxes", "orig_size", "image_size", "resize_scale", "pad"):
            if key in moved_target and torch.is_tensor(moved_target[key]):
                moved_target[key] = moved_target[key].to(device)
        moved.append(moved_target)
    return moved


def _load_teacher_cache_batch(cache_paths: list[str | None], device: torch.device) -> dict[str, Any] | None:
    if not cache_paths or any(path is None or not Path(path).exists() for path in cache_paths):
        return None

    caches = [torch.load(path, map_location=device) for path in cache_paths]
    levels = caches[0]["features"].keys()
    batch = {
        "features": {level: torch.cat([cache["features"][level] for cache in caches], dim=0) for level in levels},
        "logits": {level: torch.cat([cache["logits"][level] for cache in caches], dim=0) for level in levels},
        "raw_boxes": {level: torch.cat([cache["raw_boxes"][level] for cache in caches], dim=0) for level in levels},
        "boxes": [cache["boxes"] for cache in caches],
        "scores": [cache["scores"] for cache in caches],
        "spatial_feat": torch.cat([cache["spatial_feat"] for cache in caches], dim=0),
    }
    if all("roi_embeddings" in cache for cache in caches):
        batch["roi_embeddings_per_image"] = [cache["roi_embeddings"] for cache in caches]
    if all("visibilities" in cache for cache in caches):
        batch["visibilities_per_image"] = [cache["visibilities"] for cache in caches]
    return batch


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
    labels = torch.cat(selected_labels, dim=0) if selected_labels else targets[0]["boxes"].new_zeros((0,), dtype=torch.long)
    return selected_boxes, labels


class DistillationTrainer:
    def __init__(
        self,
        student_model,
        teacher_model,
        feature_adapters,
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

    def _student_module(self):
        return self.student_model.module if hasattr(self.student_model, "module") else self.student_model

    def _feature_adapter_module(self):
        if self.feature_adapters is None:
            return None
        return self.feature_adapters.module if hasattr(self.feature_adapters, "module") else self.feature_adapters

    def _compute_teacher_outputs(self, images: torch.Tensor, cache_paths: list[str | None]) -> dict[str, Any]:
        if self.use_teacher_cache:
            cached = _load_teacher_cache_batch(cache_paths, self.device)
            if cached is not None:
                return cached
        if self.teacher_model is None:
            raise RuntimeError("Teacher outputs were requested but teacher_model is None and cache is unavailable.")
        return self.teacher_model(images)

    def _compute_total_loss(
        self,
        student_outputs: dict[str, Any],
        teacher_outputs: dict[str, Any],
        targets: list[dict[str, Any]],
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        weights = self.phase_schedule.for_epoch(epoch)
        assignments = self.assigner.assign(student_outputs, targets)

        det_losses = detection_loss(student_outputs, assignments)
        cls_kd = classification_kd_loss(student_outputs["cls"], teacher_outputs["logits"], temperature=self.temperature)
        box_kd = box_kd_loss(student_outputs["box"], teacher_outputs["raw_boxes"], assignments)
        feat_kd = feature_distill_loss(
            student_outputs["features"],
            teacher_outputs["features"],
            adapters=self._feature_adapter_module(),
        )

        embedding_boxes, _ = _select_supervision_boxes(targets, self.embedding_min_visibility)
        student_roi = student_outputs.get("roi_embeddings")
        if student_roi is None:
            student_roi = self._student_module().extract_roi_embeddings(
                student_outputs,
                boxes_per_image=embedding_boxes,
                image_size=tuple(targets[0]["image_size"].tolist()),
            )
        if "roi_embeddings_per_image" in teacher_outputs:
            teacher_roi = []
            visibility_lists = teacher_outputs.get("visibilities_per_image")
            for image_index, roi_embeddings in enumerate(teacher_outputs["roi_embeddings_per_image"]):
                roi_embeddings = roi_embeddings.to(self.device)
                if visibility_lists is not None:
                    vis = visibility_lists[image_index].to(self.device)
                    roi_embeddings = roi_embeddings[vis >= self.embedding_min_visibility]
                teacher_roi.append(roi_embeddings)
            teacher_roi = torch.cat(teacher_roi, dim=0) if teacher_roi else student_roi.new_zeros((0, student_roi.shape[-1]))
        else:
            teacher_roi = self.teacher_model.extract_roi_embeddings(
                teacher_outputs["spatial_feat"],
                boxes_per_image=embedding_boxes,
                image_size=tuple(targets[0]["image_size"].tolist()),
            )
        emb_kd = embedding_cosine_loss(student_roi, teacher_roi)

        id_boxes, id_targets = _select_supervision_boxes(targets, self.id_min_visibility)
        if weights["id"] > 0.0 and id_targets.numel() > 0:
            id_logits = student_outputs.get("id_logits")
            if id_logits is None:
                if self.id_min_visibility == self.embedding_min_visibility:
                    id_embeddings = student_roi
                else:
                    id_embeddings = student_outputs.get("id_embeddings")
                    if id_embeddings is None:
                        id_embeddings = self._student_module().extract_roi_embeddings(
                            student_outputs,
                            boxes_per_image=id_boxes,
                            image_size=tuple(targets[0]["image_size"].tolist()),
                        )
                id_logits = self._student_module().classify_ids(id_embeddings)
            id_loss = id_supervision_loss(id_logits, id_targets)
        else:
            id_loss = student_roi.sum() * 0.0

        total_loss = (
            weights["det"] * det_losses["total"]
            + weights["cls_kd"] * cls_kd
            + weights["box_kd"] * box_kd
            + weights["feat"] * feat_kd
            + weights["emb"] * emb_kd
            + weights["id"] * id_loss
        )
        log_items = {
            "loss": float(total_loss.detach().item()),
            "det": float(det_losses["total"].detach().item()),
            "det_cls": float(det_losses["cls"].detach().item()),
            "det_obj": float(det_losses["obj"].detach().item()),
            "det_box": float(det_losses["box"].detach().item()),
            "cls_kd": float(cls_kd.detach().item()),
            "box_kd": float(box_kd.detach().item()),
            "feat_kd": float(feat_kd.detach().item()),
            "emb_kd": float(emb_kd.detach().item()),
            "id_loss": float(id_loss.detach().item()),
        }
        return total_loss, log_items

    def _reduce_log_items(self, log_items: dict[str, float]) -> dict[str, float]:
        return {key: reduce_scalar(value, self.device) for key, value in log_items.items()}

    def train_one_epoch(self, data_loader, epoch: int) -> dict[str, float]:
        self.student_model.train()
        if self.feature_adapters is not None:
            self.feature_adapters.train()
        meters = {name: SmoothedValue() for name in ["loss", "det", "cls_kd", "box_kd", "feat_kd", "emb_kd", "id_loss"]}

        progress = tqdm(data_loader, desc=f"Train {epoch}", leave=False, disable=not is_main_process())
        for batch in progress:
            images = batch["images"].to(self.device, non_blocking=True)
            targets = _move_targets_to_device(batch["targets"], self.device)
            teacher_outputs = self._compute_teacher_outputs(images, batch.get("teacher_cache_paths", []))
            image_size = tuple(targets[0]["image_size"].tolist())
            phase_weights = self.phase_schedule.for_epoch(epoch)
            embedding_boxes, _ = _select_supervision_boxes(targets, self.embedding_min_visibility)
            id_boxes = None
            if phase_weights["id"] > 0.0:
                id_boxes, _ = _select_supervision_boxes(targets, self.id_min_visibility)
                if self.id_min_visibility == self.embedding_min_visibility:
                    id_boxes = embedding_boxes

            with torch.amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                student_outputs = self.student_model(
                    images,
                    roi_boxes_per_image=embedding_boxes,
                    roi_image_size=image_size,
                    id_boxes_per_image=id_boxes,
                )
                total_loss, log_items = self._compute_total_loss(student_outputs, teacher_outputs, targets, epoch)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.grad_clip)
            if self.feature_adapters is not None:
                torch.nn.utils.clip_grad_norm_(self.feature_adapters.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # Note: scheduler.step() is called per-iteration (correct for LambdaLR)
            # The warning from PyTorch about step order can be ignored for per-iteration schedulers
            if self.scheduler is not None:
                self.scheduler.step()

            reduced = self._reduce_log_items(log_items)
            for key, meter in meters.items():
                meter.update(reduced[key])
            if is_main_process() and hasattr(progress, "set_postfix"):
                progress.set_postfix({key: f"{meter.avg:.4f}" for key, meter in meters.items()})

        return {key: meter.avg for key, meter in meters.items()}

    @torch.no_grad()
    def validate(self, data_loader, epoch: int) -> dict[str, float]:
        self.student_model.eval()
        if self.feature_adapters is not None:
            self.feature_adapters.eval()
        meters = {name: SmoothedValue() for name in ["loss", "det", "cls_kd", "box_kd", "feat_kd", "emb_kd", "id_loss"]}

        progress = tqdm(data_loader, desc=f"Val {epoch}", leave=False, disable=not is_main_process())
        for batch in progress:
            images = batch["images"].to(self.device, non_blocking=True)
            targets = _move_targets_to_device(batch["targets"], self.device)
            teacher_outputs = self._compute_teacher_outputs(images, batch.get("teacher_cache_paths", []))
            image_size = tuple(targets[0]["image_size"].tolist())
            phase_weights = self.phase_schedule.for_epoch(epoch)
            embedding_boxes, _ = _select_supervision_boxes(targets, self.embedding_min_visibility)
            id_boxes = None
            if phase_weights["id"] > 0.0:
                id_boxes, _ = _select_supervision_boxes(targets, self.id_min_visibility)
                if self.id_min_visibility == self.embedding_min_visibility:
                    id_boxes = embedding_boxes
            student_outputs = self.student_model(
                images,
                roi_boxes_per_image=embedding_boxes,
                roi_image_size=image_size,
                id_boxes_per_image=id_boxes,
            )
            _, log_items = self._compute_total_loss(student_outputs, teacher_outputs, targets, epoch)
            reduced = self._reduce_log_items(log_items)
            for key, meter in meters.items():
                meter.update(reduced[key])
            if is_main_process() and hasattr(progress, "set_postfix"):
                progress.set_postfix({key: f"{meter.avg:.4f}" for key, meter in meters.items()})

        return {key: meter.avg for key, meter in meters.items()}

    def fit(self, train_loader, val_loader, epochs: int) -> None:
        history = []
        for epoch in range(1, epochs + 1):
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            if hasattr(val_loader, "sampler") and hasattr(val_loader.sampler, "set_epoch"):
                val_loader.sampler.set_epoch(epoch)

            train_metrics = self.train_one_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, epoch)
            record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            history.append(record)

            if self.logger is not None and is_main_process():
                self.logger.info("Epoch %d train=%s val=%s", epoch, train_metrics, val_metrics)

            if not is_main_process():
                continue

            checkpoint = {
                "epoch": epoch,
                "student": self._student_module().state_dict(),
                "feature_adapters": self._feature_adapter_module().state_dict() if self.feature_adapters is not None else None,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
            save_checkpoint(checkpoint, self.output_dir, "last.pt")
            if val_metrics["loss"] < self.best_val:
                self.best_val = val_metrics["loss"]
                save_checkpoint(checkpoint, self.output_dir, "best.pt")

            history_path = self.output_dir / "history.json"
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
