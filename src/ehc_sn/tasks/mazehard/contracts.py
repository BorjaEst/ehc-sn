"""MazeHard task-owned contracts and constants.

MazeHard is the batch token-prediction task family: full-maze token sequence
observations, masked supervision, and sequence-accuracy score.

Contracts owns: semantic task input, output, targets, and constants.
Score report lives in :mod:`ehc_sn.tasks.mazehard.evaluation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from torch import Tensor

MAZE_HARD_IGNORE_LABEL_ID: Final[int] = -100
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
