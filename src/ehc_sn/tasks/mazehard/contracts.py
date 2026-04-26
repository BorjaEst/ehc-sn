"""MazeHard task-owned contracts and constants.

MazeHard is the batch token-prediction task family: full-maze token sequence
observations, masked supervision, and sequence-accuracy score.

The benchmark-facing score type :class:`MazeHardAggregateReport` is defined
here so benchmarks can depend on the task contracts layer without pulling in
the full evaluation module.
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
@dataclass(frozen=True)
class MazeHardAggregateReport:
    """Typed benchmark-facing aggregate score report for a MazeHard batch.

    All tensors are scalar (0-d) float32.  This is the canonical task-owned
    benchmark surface; benchmarks consume these fields, not mode internals.

    Attributes:
        tokens_accuracy: Token-level accuracy over all supervised tokens.
        sequences_accuracy: Mean per-sequence token accuracy.
        sequences_exact: Fraction of fully correct sequences.
    """

    tokens_accuracy: Tensor
    sequences_accuracy: Tensor
    sequences_exact: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        """Return the canonical key-value dict for logging."""
        return {
            "tokens/accuracy": self.tokens_accuracy,
            "sequences/accuracy": self.sequences_accuracy,
            "sequences/exact": self.sequences_exact,
        }


# =============================================================================
__all__ = [
    "MAZE_HARD_IGNORE_LABEL_ID",
    "MazeHardAggregateReport",
    "MazeHardTargets",
    "MazeHardTaskInput",
    "MazeHardTaskOutput",
]
