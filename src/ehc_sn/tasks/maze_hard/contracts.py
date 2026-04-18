"""MazeHard task-owned contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from torch import Tensor

from ehc_sn.envs.mazehard import IGNORE_LABEL_ID

MAZE_HARD_IGNORE_LABEL_ID: Final[int] = IGNORE_LABEL_ID
"""Canonical ignore label used for masked MazeHard supervision."""


# =============================================================================
@dataclass(frozen=True)
class MazeHardTaskInput:
    """Task-owned MazeHard token inputs."""

    input_ids: Tensor


# =============================================================================
@dataclass(frozen=True)
class MazeHardTargets:
    """Supervision targets for MazeHard token prediction."""

    labels: Tensor


# =============================================================================
@dataclass(frozen=True)
class MazeHardTaskOutput:
    """Task-owned MazeHard prediction payload."""

    task_logits: Tensor


# =============================================================================
__all__ = [
    "MAZE_HARD_IGNORE_LABEL_ID",
    "MazeHardTargets",
    "MazeHardTaskInput",
    "MazeHardTaskOutput",
]
