"""MazeHard task evaluation helpers.

Owns sequence-level correctness metrics and builds the typed
:class:`~ehc_sn.tasks.mazehard.contracts.MazeHardAggregateReport`
benchmark-facing score surface.  Import
:class:`~ehc_sn.tasks.mazehard.contracts.MazeHardAggregateReport` from
:mod:`ehc_sn.tasks.mazehard.contracts` or the mazehard task barrel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .contracts import MAZE_HARD_IGNORE_LABEL_ID, MazeHardAggregateReport as _MazeHardAggregateReport, MazeHardTargets, MazeHardTaskOutput


# =============================================================================
@dataclass(frozen=True)
class MazeHardSequenceMetrics:
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
        token_count = self.valid_token_count.clamp_min(1).to(dtype=torch.float32)
        correct = self.token_is_correct.to(dtype=torch.float32).sum(dim=-1)
        return correct / token_count

    @property
    def sequence_is_correct(self) -> Tensor:
        """Return whether each sequence is exactly correct on all supervised tokens."""
        correct = self.token_is_correct.to(dtype=torch.int64).sum(dim=-1)
        return correct == self.valid_token_count.to(dtype=torch.int64)


# =============================================================================
def evaluate_maze_hard_sequences(  # ------------------------------------------
    output: MazeHardTaskOutput | Tensor,
    targets: MazeHardTargets | Tensor,
    *,
    ignore_label_id: int = MAZE_HARD_IGNORE_LABEL_ID,
) -> MazeHardSequenceMetrics:
    """Return masked token-correctness metrics for a MazeHard batch."""
    logits = output.task_logits if isinstance(output, MazeHardTaskOutput) else output
    labels = targets.labels if isinstance(targets, MazeHardTargets) else targets
    valid_mask = labels != ignore_label_id
    token_is_correct = valid_mask & logits.argmax(dim=-1).eq(labels)
    return MazeHardSequenceMetrics(valid_mask=valid_mask, token_is_correct=token_is_correct)


# =============================================================================
def is_maze_hard_sequence_correct(  # -----------------------------------------
    output: MazeHardTaskOutput | Tensor,
    targets: MazeHardTargets | Tensor,
    *,
    ignore_label_id: int = MAZE_HARD_IGNORE_LABEL_ID,
) -> Tensor:
    """Return the exact-correctness flag for each MazeHard sequence."""
    return evaluate_maze_hard_sequences(
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
    return evaluate_maze_hard_sequences(
        output,
        targets,
        ignore_label_id=ignore_label_id,
    ).sequence_accuracy.unsqueeze(-1)


# =============================================================================
def build_maze_hard_report(  # ------------------------------------------------
    metrics: MazeHardSequenceMetrics,
) -> _MazeHardAggregateReport:
    """Return typed aggregate benchmark report for a MazeHard batch.

    Args:
        metrics: Per-sequence correctness summary from :func:`evaluate_maze_hard_sequences`.

    Returns:
        :class:`~ehc_sn.tasks.mazehard.contracts.MazeHardAggregateReport`
        with scalar accuracy fields.
    """
    token_correct_sum = metrics.token_is_correct.to(dtype=torch.float32).sum()
    token_count_sum = metrics.valid_mask.to(dtype=torch.float32).sum().clamp_min(1.0)
    sequence_count = metrics.sequence_accuracy.new_tensor(float(metrics.sequence_accuracy.shape[0])).clamp_min(1.0)
    return _MazeHardAggregateReport(
        tokens_accuracy=token_correct_sum / token_count_sum,
        sequences_accuracy=metrics.sequence_accuracy.sum() / sequence_count,
        sequences_exact=metrics.sequence_is_correct.to(dtype=torch.float32).sum() / sequence_count,
    )


# =============================================================================
__all__ = [
    "MazeHardSequenceMetrics",
    "build_maze_hard_report",
    "compute_maze_hard_sequence_accuracy",
    "evaluate_maze_hard_sequences",
    "is_maze_hard_sequence_correct",
]
