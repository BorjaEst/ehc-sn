"""MazeHard task-level runtime helpers.

Owns typed extraction of task dataclasses from generic batch mappings and
raw channel-to-canonical-batch coercion so controller and script code
remains task-agnostic and adapter-free.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import numpy as np
import torch
from torch import Tensor

from ehc_sn.types import Batch

from .contracts import MazeHardTargets, MazeHardTaskInput

MAZE_HARD_BATCH_KEYS: Final[tuple[str, ...]] = ("input_ids", "labels")
"""Canonical generic batch keys required by MazeHard rollout consumers."""


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


# Private maze SEM vocabulary IDs used by batch coercion.
WALL_ID: int = 1
EMPTY_ID: int = 2
START_ID: int = 3
GOAL_ID: int = 4
PATH_ID: int = 5  # solution-overlay label token
SEM_VOCAB_SIZE: int = 5  # base semantic vocabulary (PAD..GOAL)
MAZE_HARD_VOCAB_SIZE: int = PATH_ID + 1  # full vocab including PATH

_MANDATORY_GRID2D_CHANNEL: str = "topology"
_CHANNEL_SOLUTION: str = "solution"


# =============================================================================
def coerce_maze_hard_batch(raw: Mapping[str, Any]) -> Batch:
    """Convert raw MazeHard channels into canonical token and label tensors.

    Task-owned tokenization: raw channel arrays -> ``{"input_ids", "labels"}``.
    Supports both single-maze arrays ``(H, W)`` and aligned stacked arrays
    ``(B, H, W)``.  Spatial dimensions are flattened while any leading batch
    dimensions are preserved.
    """
    channels = _coerce_numpy_channels(raw)
    _validate_channel_stack_shapes(channels)

    grid = _channels_to_grid(channels)
    input_ids = _flatten_spatial_to_tensor(grid, dtype=np.int64, name="grid")
    labels = input_ids.clone()

    if _CHANNEL_SOLUTION in channels:
        solution_mask = _flatten_spatial_to_tensor(
            channels[_CHANNEL_SOLUTION] > 0,
            dtype=np.bool_,
            name=_CHANNEL_SOLUTION,
        ).to(dtype=torch.bool)
        labels = torch.where(solution_mask, torch.full_like(labels, PATH_ID), labels)

    return {"input_ids": input_ids, "labels": labels}


def _channels_to_grid(channels: dict[str, np.ndarray]) -> np.ndarray:
    topology = channels[_MANDATORY_GRID2D_CHANNEL]
    grid = np.where(topology, EMPTY_ID, WALL_ID).astype(np.int32)
    if "start" in channels:
        grid = np.where(channels["start"], START_ID, grid)
    if "goals" in channels:
        grid = np.where(channels["goals"], GOAL_ID, grid)
    return grid


def _coerce_numpy_channels(raw: Mapping[str, Any]) -> dict[str, np.ndarray]:
    channels: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        if isinstance(value, np.ndarray):
            channels[key] = value
            continue
        if isinstance(value, Tensor):
            channels[key] = value.detach().cpu().numpy()
            continue
        raise TypeError(f"Unsupported MazeHard channel type for key {key!r}: {type(value).__name__}.")
    if _MANDATORY_GRID2D_CHANNEL not in channels:
        raise ValueError(f"MazeHard batch must contain the mandatory '{_MANDATORY_GRID2D_CHANNEL}' channel.")
    return channels


def _validate_channel_stack_shapes(channels: dict[str, np.ndarray]) -> None:
    reference_name, reference = next(iter(channels.items()))
    mismatched = {name: value.shape for name, value in channels.items() if value.shape != reference.shape}
    if mismatched:
        detail = ", ".join(f"{name}={shape}" for name, shape in mismatched.items())
        raise ValueError(
            "MazeHard batch requires aligned raw channel shapes; "
            f"expected all channels to match {reference_name}={reference.shape}, got {detail}."
        )


def _flatten_spatial_to_tensor(array: np.ndarray, *, dtype: Any, name: str) -> Tensor:
    if array.ndim not in (2, 3):
        raise ValueError(f"MazeHard field {name!r} must have shape (H, W) or (B, H, W), got {array.shape}.")
    return torch.from_numpy(array.reshape(*array.shape[:-2], -1).astype(dtype, copy=False))


# =============================================================================
__all__ = [
    "MAZE_HARD_BATCH_KEYS",
    "MAZE_HARD_VOCAB_SIZE",
    "SEM_VOCAB_SIZE",
    "WALL_ID",
    "EMPTY_ID",
    "START_ID",
    "GOAL_ID",
    "PATH_ID",
    "coerce_maze_hard_batch",
    "extract_maze_hard_targets",
    "extract_maze_hard_task_input",
]
