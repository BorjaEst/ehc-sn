"""MazeHard task evaluation helpers.

Owns sequence-level correctness metrics and the benchmark-facing
:class:`MazeHardScoreReport` aggregate score surface.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .contracts import (
    MAZE_HARD_IGNORE_LABEL_ID,
    MazeHardTargets,
    MazeHardTaskOutput,
)

MAZEHARD_PRIMARY_METRIC_NAME: str = "token_accuracy"
"""Canonical primary benchmark metric for MazeHard: token-level accuracy."""


# =============================================================================
@dataclass(frozen=True)
class MazeHardScoreReport:
    """Typed benchmark-facing aggregate score report for a MazeHard batch.

    All tensors are scalar (0-d) float32.  Benchmarks consume these fields;
    mode internals are not part of this surface.

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
@dataclass(frozen=True)
class MazeHardStepScore:
    """Per-sequence correctness summary for MazeHard token prediction."""

    valid_mask: Tensor
    token_is_correct: Tensor

    @property
    def valid_token_count(self) -> Tensor:
        """Return the number of supervised tokens per sequence."""
        return self.valid_mask.sum(dim=-1)

    @property
    def sequence_accuracy(self) -> Tensor:
        """Return per-sequence token accuracy over non-ignored labels."""
        token_count = self.valid_token_count.clamp_min(1).to(
            dtype=torch.float32
        )
        correct = self.token_is_correct.to(dtype=torch.float32).sum(dim=-1)
        return correct / token_count

    @property
    def sequence_is_correct(self) -> Tensor:
        """Return whether each sequence is exactly correct on all supervised tokens."""
        correct = self.token_is_correct.to(dtype=torch.int64).sum(dim=-1)
        return correct == self.valid_token_count.to(dtype=torch.int64)


# =============================================================================
def build_maze_hard_step_score(  # --------------------------------------------
    output: MazeHardTaskOutput | Tensor,
    targets: MazeHardTargets | Tensor,
    *,
    ignore_label_id: int = MAZE_HARD_IGNORE_LABEL_ID,
) -> MazeHardStepScore:
    """Return masked token-correctness metrics for a MazeHard batch."""
    logits = (
        output.task_logits if isinstance(output, MazeHardTaskOutput) else output
    )
    labels = targets.labels if isinstance(targets, MazeHardTargets) else targets
    valid_mask = labels != ignore_label_id
    token_is_correct = valid_mask & logits.argmax(dim=-1).eq(labels)
    return MazeHardStepScore(
        valid_mask=valid_mask, token_is_correct=token_is_correct
    )


# =============================================================================
def is_maze_hard_sequence_correct(  # -----------------------------------------
    output: MazeHardTaskOutput | Tensor,
    targets: MazeHardTargets | Tensor,
    *,
    ignore_label_id: int = MAZE_HARD_IGNORE_LABEL_ID,
) -> Tensor:
    """Return the exact-correctness flag for each MazeHard sequence."""
    return build_maze_hard_step_score(
        output,
        targets,
        ignore_label_id=ignore_label_id,
    ).sequence_is_correct


# =============================================================================
def compute_maze_hard_sequence_accuracy(  # -----------------------------------
    output: MazeHardTaskOutput | Tensor,
    targets: MazeHardTargets | Tensor,
    *,
    ignore_label_id: int = MAZE_HARD_IGNORE_LABEL_ID,
) -> Tensor:
    """Return per-sequence token accuracy for one MazeHard batch."""
    return build_maze_hard_step_score(
        output,
        targets,
        ignore_label_id=ignore_label_id,
    ).sequence_accuracy.unsqueeze(-1)


# =============================================================================
def build_maze_hard_score_report(  # ------------------------------------------
    metrics: MazeHardStepScore,
) -> MazeHardScoreReport:
    """Return typed aggregate benchmark report for a MazeHard batch.

    Args:
        metrics: Per-sequence correctness summary from :func:`build_maze_hard_step_score`.

    Returns:
        :class:`MazeHardScoreReport` with scalar accuracy fields.
    """
    token_correct_sum = metrics.token_is_correct.to(dtype=torch.float32).sum()
    token_count_sum = (
        metrics.valid_mask.to(dtype=torch.float32).sum().clamp_min(1.0)
    )
    sequence_count = metrics.sequence_accuracy.new_tensor(
        float(metrics.sequence_accuracy.shape[0])
    ).clamp_min(1.0)
    return MazeHardScoreReport(
        tokens_accuracy=token_correct_sum / token_count_sum,
        sequences_accuracy=metrics.sequence_accuracy.sum() / sequence_count,
        sequences_exact=metrics.sequence_is_correct.to(
            dtype=torch.float32
        ).sum()
        / sequence_count,
    )


# =============================================================================
__all__ = [
    "MazeHardScoreReport",
    "MazeHardStepScore",
    "build_maze_hard_score_report",
    "build_maze_hard_step_score",
    "compute_maze_hard_sequence_accuracy",
    "is_maze_hard_sequence_correct",
]
