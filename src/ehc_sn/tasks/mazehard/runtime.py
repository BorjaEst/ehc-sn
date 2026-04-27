"""MazeHard task-level runtime helpers.

Owns typed extraction of task dataclasses from generic batch mappings so
controller and script code remains task-agnostic.  Raw channel-to-batch
coercion lives in :mod:`ehc_sn.data.mazehard` (data layer).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

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


# =============================================================================
__all__ = [
    "MAZE_HARD_BATCH_KEYS",
    "extract_maze_hard_targets",
    "extract_maze_hard_task_input",
]
