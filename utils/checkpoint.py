from __future__ import annotations

import sys
from typing import Any

import torch
from torch import nn


def validate_checkpoint_shapes(
    checkpoint: dict[str, Any],
    student: nn.Module,
    feature_adapters: nn.Module | None,
    teacher_roi_projector: nn.Module | None = None,
) -> None:
    """Validate that checkpoint weight shapes match the current model.

    Prints a detailed mismatch report and calls sys.exit(1) on failure,
    instead of letting PyTorch raise an opaque RuntimeError deep inside
    load_state_dict.

    Args:
        checkpoint:           The dict returned by torch.load(path).
        student:              The StudentJDE instance.
        feature_adapters:     The MultiScaleFeatureAdapters instance (or None).
        teacher_roi_projector: The TeacherROIProjector instance (or None).
    """
    errors: list[str] = []

    def _check_state(model: nn.Module, ckpt_state: dict[str, torch.Tensor], label: str) -> None:
        model_state = model.state_dict()
        for key, param in model_state.items():
            if key not in ckpt_state:
                errors.append(f"  [MISSING in checkpoint] {label}.{key}  expected {tuple(param.shape)}")
            elif ckpt_state[key].shape != param.shape:
                errors.append(
                    f"  [SHAPE MISMATCH] {label}.{key}:\n"
                    f"    checkpoint → {tuple(ckpt_state[key].shape)}\n"
                    f"    model      → {tuple(param.shape)}"
                )
        for key in ckpt_state:
            if key not in model_state:
                errors.append(f"  [EXTRA in checkpoint] {label}.{key}  (not in current model)")

    # --- student ---
    if "student" in checkpoint:
        _check_state(student, checkpoint["student"], "student")
    else:
        errors.append("  [MISSING] checkpoint has no 'student' key")

    # --- feature_adapters ---
    if feature_adapters is not None:
        if "feature_adapters" in checkpoint and checkpoint["feature_adapters"] is not None:
            _check_state(feature_adapters, checkpoint["feature_adapters"], "feature_adapters")
        else:
            errors.append("  [MISSING] checkpoint has no 'feature_adapters' key (model expects adapters)")

    # --- teacher_roi_projector ---
    if teacher_roi_projector is not None:
        if "teacher_roi_projector" in checkpoint and checkpoint["teacher_roi_projector"] is not None:
            _check_state(teacher_roi_projector, checkpoint["teacher_roi_projector"], "teacher_roi_projector")
        # Not an error if absent — projector may not have been saved in older checkpoints

    if errors:
        sep = "=" * 72
        print(f"\n{sep}", file=sys.stderr)
        print("CHECKPOINT SHAPE VALIDATION FAILED", file=sys.stderr)
        print(sep, file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        print(sep, file=sys.stderr)
        print("\nLikely causes:", file=sys.stderr)
        print("  • backbone_name, fpn_channels, or emb_dim was changed since the checkpoint was saved.", file=sys.stderr)
        print("  • The config file does not match the one used for training.", file=sys.stderr)
        print("  • Run with the same --config as the original training run.", file=sys.stderr)
        print(sep + "\n", file=sys.stderr)
        sys.exit(1)
