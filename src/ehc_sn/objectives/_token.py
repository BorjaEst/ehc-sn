"""Shared token-supervision helpers.

This module defines the :class:`TokenSupervisionBinding` protocol and reusable
token-level loss/metric utilities for objective implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor

from ehc_sn.metrics.step_metrics import (
    RatioStat,
    RolloutAgg,
    StepMetrics,
    TokenAgg,
    TransitionAgg,
)
from ehc_sn.rollouts.runtime import CarrySnapshot
from ehc_sn.types import Batch

IGNORE_LABEL_ID: int = -100


# =============================================================================
class TokenSupervisionBinding[TargetsT](Protocol):
    """Extraction seam for token-supervised rollout objectives.

    The ``executed_batch`` input is the executed-step payload (``record.batch``),
    which is authoritative for current-step supervision. The ``snapshot`` input
    is a frozen post-step snapshot intended only for continuity facts or
    lightweight projections, not step-truth tensors.
    """

    def extract_logits(  # ----------------------------------------------------
        self,
        executed_batch: Batch,
        snapshot: CarrySnapshot,
        step_output: Any,
    ) -> Tensor:
        """Return supervised logits for one executed step."""

    def extract_targets(  # ---------------------------------------------------
        self,
        executed_batch: Batch,
        snapshot: CarrySnapshot,
        step_output: Any,
    ) -> TargetsT:
        """Return task-owned supervision targets for one executed step."""

    def evaluate_sequences(  # ------------------------------------------------
        self,
        logits: Tensor,
        targets: TargetsT,
    ) -> "AccuracyStats":
        """Return sequence-level accuracy statistics for one executed step."""


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
    ignore_label_id: int = IGNORE_LABEL_ID,
) -> AccuracyStats:
    """Compute masked token correctness statistics out of graph."""
    mask = labels != ignore_label_id
    is_correct = mask & (torch.argmax(logits_token, dim=-1) == labels)
    return AccuracyStats(mask=mask, is_correct=is_correct)


# =============================================================================
def compute_token_loss_sum(  # ---------------------------------------------------
    loss_fn: Any,
    logits_token: Tensor,
    labels: Tensor,
    stats: AccuracyStats,
    *,
    ignore_label_id: int = IGNORE_LABEL_ID,
    token_weights: Tensor | None = None,
) -> Tensor:
    """Compute the summed supervised Token loss over the batch."""
    loss_per_token = compute_token_loss_unreduced(
        loss_fn,
        logits_token,
        labels,
        ignore_label_id=ignore_label_id,
        token_weights=token_weights,
    )
    loss_per_seq = loss_per_token.sum(-1) / stats.loss_counts.clamp_min(1)
    return loss_per_seq.sum()


# =============================================================================
def compute_token_loss_unreduced(  # ------------------------------------------
    loss_fn: Any,
    logits_token: Tensor,
    labels: Tensor,
    *,
    ignore_label_id: int = IGNORE_LABEL_ID,
    token_weights: Tensor | None = None,
) -> Tensor:
    """Return the per-token loss tensor before sequence reduction."""
    loss_per_token = loss_fn(logits_token, labels, ignore_index=ignore_label_id)
    if token_weights is not None:
        if token_weights.shape != labels.shape:
            raise ValueError(
                "token_weights must match labels shape for Token loss weighting.",
            )
        loss_per_token = loss_per_token * token_weights.to(loss_per_token.dtype)
    return loss_per_token


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
    "IGNORE_LABEL_ID",
    "AccuracyStats",
    "TokenSupervisionBinding",
    "build_token_step_metrics",
    "compute_accuracy_stats",
    "compute_token_loss_unreduced",
    "compute_token_loss_sum",
]
