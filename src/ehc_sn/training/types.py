"""Step-metrics types for HRM training paradigms.

These dataclasses carry the sufficient statistics — numerator/denominator pairs —
that :func:`~ehc_sn.metrics.update_metrics_from_step` routes into TorchMetrics
:class:`~ehc_sn.metrics.torchmetrics.RatioMetric` instances for correct
weighted epoch-level aggregation.

Each field is a scalar tensor produced by the loss head and accumulated over
batches via :class:`~ehc_sn.metrics.torchmetrics.RatioMetric`.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


# =================================================================================================
@dataclass(frozen=True)
class HaltedAgg:
    """Aggregates over sequences that halted on the current step."""

    halted_count: Tensor
    eligible_count: Tensor
    accuracy_sum: Tensor
    exact_sum: Tensor
    steps_sum: Tensor
    q_halt_correct_sum: Tensor
    q_continue_correct_sum: Tensor


# =================================================================================================
@dataclass(frozen=True)
class TokenAgg:
    """Token-level aggregates for halted sequences."""

    token_correct_sum: Tensor
    token_count_sum: Tensor


# =================================================================================================
@dataclass(frozen=True)
class LossAgg:
    """Per-step loss aggregates (batch-level sums)."""

    # ~~ General losses applicable to both ACT and RL heads ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    lm_loss_sum: Tensor
    q_halt_loss_sum: Tensor
    q_continue_loss_sum: Tensor
    batch_count: Tensor

    # ~~ RL-specific (zero for non-RL) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    actor_loss_sum: Tensor  # e.g. policy gradient
    critic_loss_sum: Tensor  # e.g. value function MSE
    entropy_loss_sum: Tensor  # e.g. action distribution entropy regularization
    q_value_loss_sum: Tensor  # e.g. auxiliary loss on vmPFC features


# =================================================================================================
@dataclass(frozen=True)
class StepMetrics:
    """Aggregated per-step metrics used for logging and control flow."""

    halted: HaltedAgg
    tokens: TokenAgg
    loss: LossAgg


# =================================================================================================
__all__ = ["HaltedAgg", "LossAgg", "StepMetrics", "TokenAgg"]
