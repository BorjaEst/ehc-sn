from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor


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


@dataclass(frozen=True)
class TokenAgg:
    """Token-level aggregates for halted sequences."""

    token_correct_sum: Tensor
    token_count_sum: Tensor


@dataclass(frozen=True)
class LossAgg:
    """Per-step loss aggregates (batch-level sums)."""

    lm_loss_sum: Tensor
    q_halt_loss_sum: Tensor
    q_continue_loss_sum: Tensor
    batch_count: Tensor


@dataclass(frozen=True)
class StepMetrics:
    """Aggregated per-step metrics used for logging and control flow."""

    halted: HaltedAgg
    tokens: TokenAgg
    loss: LossAgg
