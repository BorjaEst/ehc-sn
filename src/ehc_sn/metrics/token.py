"""Token-supervision metrics — correctness statistics and metric builders.

This module owns the token-level accuracy computation and
:class:`StepMetrics` construction used by supervised objectives
(ACT, hybrid RL).  Loss primitives live in
:mod:`ehc_sn.objectives.supervised.token`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ehc_sn.metrics.step_metrics import (
    RatioStat,
    RolloutAgg,
    StepMetrics,
    TokenAgg,
    TransitionAgg,
)


# =============================================================================
@dataclass(frozen=True)
class AccuracyStats:
    """Token-level correctness statistics shared by token-supervised heads."""

    mask: Tensor
    is_correct: Tensor

    @property
    def loss_counts(self) -> Tensor:
        """Number of eligible tokens per sequence, shape ``(B,)``."""
        return self.mask.sum(-1)

    @property
    def loss_divisor(self) -> Tensor:
        """Safe divisor for per-sequence averages, shape ``(B, 1)``."""
        return self.loss_counts.clamp_min(1).unsqueeze(-1)

    @property
    def seq_is_correct(self) -> Tensor:
        """Whether every eligible token in a sequence was predicted correctly."""
        return self.is_correct.sum(-1) == self.loss_counts


# =============================================================================
def compute_accuracy_stats(  # ------------------------------------------------
    logits_token: Tensor,
    labels: Tensor,
    *,
    ignore_label_id: int = -100,
) -> AccuracyStats:
    """Compute masked token correctness statistics out of graph."""
    mask = labels != ignore_label_id
    is_correct = mask & (torch.argmax(logits_token, dim=-1) == labels)
    return AccuracyStats(mask=mask, is_correct=is_correct)


# =============================================================================
def build_token_step_metrics(  # ----------------------------------------------
    steps: Tensor,
    completed: Tensor,
    stats: AccuracyStats,
    extras: dict[str, RatioStat],
) -> StepMetrics:
    """Build generic per-step token-supervision metrics."""
    eligible_mask = stats.loss_counts > 0
    completed_mask = completed & eligible_mask

    eligible_weights = eligible_mask.to(torch.float32)
    completed_weights = completed_mask.to(torch.float32)

    token_correct_per_seq = stats.is_correct.to(torch.float32).sum(-1)
    token_count_per_seq = stats.loss_counts.clamp_min(1).to(torch.float32)
    seq_accuracy = token_correct_per_seq / token_count_per_seq
    seq_exact = stats.seq_is_correct.to(torch.float32)

    return StepMetrics(
        episode=RolloutAgg(
            completed_count=completed_weights.sum(),
            eligible_count=eligible_weights.sum(),
            accuracy_sum=(seq_accuracy * completed_weights).sum(),
            exact_sum=(seq_exact * completed_weights).sum(),
            steps_sum=(steps * completed_weights.to(steps.dtype)).sum(),
        ),
        episode_tokens=TokenAgg(
            token_correct_sum=(token_correct_per_seq * completed_weights).sum(),
            token_count_sum=(token_count_per_seq * completed_weights).sum(),
        ),
        step=TransitionAgg(
            evaluated_count=eligible_weights.sum(),
            eligible_count=eligible_weights.sum(),
            accuracy_sum=(seq_accuracy * eligible_weights).sum(),
            exact_sum=(seq_exact * eligible_weights).sum(),
            steps_sum=(steps * eligible_weights.to(steps.dtype)).sum(),
        ),
        step_tokens=TokenAgg(
            token_correct_sum=(token_correct_per_seq * eligible_weights).sum(),
            token_count_sum=(token_count_per_seq * eligible_weights).sum(),
        ),
        extras=extras,
    )


# =============================================================================
__all__ = [
    "AccuracyStats",
    "build_token_step_metrics",
    "compute_accuracy_stats",
]
