"""MazeHard task-level batch helpers.

Owns raw-to-canonical MazeHard field coercion and extraction of task
dataclasses from generic batch mappings so controller and script code remains
task-agnostic.  These helpers do not depend on any execution mode.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import numpy as np
import torch
from torch import Tensor

from ehc_sn.data.schema import CHANNEL_SOLUTION, O_ID, validate_npz
from ehc_sn.data.transforms import channels_to_grid
from ehc_sn.types import Batch

from .contracts import MazeHardTargets, MazeHardTaskInput

MAZE_HARD_BATCH_KEYS: Final[tuple[str, ...]] = ("input_ids", "labels")
"""Canonical generic batch keys required by MazeHard rollout consumers."""


# =============================================================================
def coerce_maze_hard_batch(
    raw: Mapping[str, Any],
) -> Batch:
    """Convert raw MazeHard channels into canonical token and label tensors.

    Supports both single-maze arrays ``(H, W)`` and aligned stacked arrays
    ``(B, H, W)``. Spatial dimensions are flattened while any leading batch
    dimensions are preserved.
    """
    channels = _coerce_numpy_channels(raw)
    _validate_channel_stack_shapes(channels)

    grid = channels_to_grid(channels)["grid"]
    input_ids = _flatten_spatial_to_tensor(grid, dtype=np.int64, name="grid")
    labels = input_ids.clone()

    if CHANNEL_SOLUTION in channels:
        solution_mask = _flatten_spatial_to_tensor(
            channels[CHANNEL_SOLUTION] > 0,
            dtype=np.bool_,
            name=CHANNEL_SOLUTION,
        ).to(dtype=torch.bool)
        labels = torch.where(solution_mask, torch.full_like(labels, O_ID), labels)

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


# =============================================================================
def extract_maze_hard_task_input(
    batch: Batch,
) -> MazeHardTaskInput:
    """Extract the model-facing MazeHard task input from one generic batch."""
    input_ids, _ = _validate_maze_hard_batch(batch)
    return MazeHardTaskInput(input_ids=input_ids)


# =============================================================================
def extract_maze_hard_targets(
    batch: Batch,
) -> MazeHardTargets:
    """Extract MazeHard supervision targets from one generic batch."""
    _, labels = _validate_maze_hard_batch(batch)
    return MazeHardTargets(labels=labels)


# =============================================================================
def _coerce_numpy_channels(
    raw: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Normalize one raw MazeHard channel mapping to numpy arrays."""
    channels: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        if isinstance(value, np.ndarray):
            channels[key] = value
            continue
        if isinstance(value, Tensor):
            channels[key] = value.detach().cpu().numpy()
            continue
        raise TypeError(f"Unsupported MazeHard channel type for key {key!r}: {type(value).__name__}.")

    validate_npz(channels)
    return channels


# =============================================================================
def _validate_channel_stack_shapes(
    channels: dict[str, np.ndarray],
) -> None:
    """Validate that all raw MazeHard channels share the same full shape."""
    reference_name, reference = next(iter(channels.items()))
    mismatched = {name: value.shape for name, value in channels.items() if value.shape != reference.shape}
    if mismatched:
        detail = ", ".join(f"{name}={shape}" for name, shape in mismatched.items())
        raise ValueError(
            "MazeHard batch requires aligned raw channel shapes; "
            f"expected all channels to match {reference_name}={reference.shape}, got {detail}."
        )


# =============================================================================
def _flatten_spatial_to_tensor(
    array: np.ndarray,
    *,
    dtype: Any,
    name: str,
) -> Tensor:
    """Flatten the spatial tail of a 2D or 3D MazeHard array to a torch tensor."""
    if array.ndim not in (2, 3):
        raise ValueError(f"MazeHard field {name!r} must have shape (H, W) or (B, H, W), got {array.shape}.")
    return torch.from_numpy(array.reshape(*array.shape[:-2], -1).astype(dtype, copy=False))


# =============================================================================
def _validate_maze_hard_batch(
    batch: Batch,
) -> tuple[Tensor, Tensor]:
    """Validate and normalize the canonical MazeHard batch mapping."""
    missing = [key for key in MAZE_HARD_BATCH_KEYS if key not in batch]
    if missing:
        raise KeyError("MazeHard batch is missing required keys: " + ", ".join(missing) + ".")

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    if input_ids.ndim != 2:
        raise ValueError(f"MazeHard input_ids must have shape (B, S), got {tuple(input_ids.shape)}.")
    if labels.ndim != 2:
        raise ValueError(f"MazeHard labels must have shape (B, S), got {tuple(labels.shape)}.")
    if tuple(labels.shape) != tuple(input_ids.shape):
        raise ValueError(f"MazeHard labels must match input_ids shape {tuple(input_ids.shape)}, got {tuple(labels.shape)}.")

    return input_ids.to(dtype=torch.int64), labels.to(dtype=torch.int64)


# =============================================================================
__all__ = [
    "MAZE_HARD_BATCH_KEYS",
    "coerce_maze_hard_batch",
    "extract_maze_hard_targets",
    "extract_maze_hard_task_input",
]
