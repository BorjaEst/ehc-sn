"""Shared step-metrics contracts for rollout objectives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from torch import Tensor


# =============================================================================
@dataclass(frozen=True)
class RatioStat:
    """A single aggregated ratio statistic.

    Both fields are sums over the current step / batch.
    """

    numerator_sum: Tensor
    denominator_sum: Tensor


# =============================================================================
@dataclass(frozen=True)
class RolloutAgg:
    """Lifecycle aggregates over completed sequences on the current step."""

    completed_count: Tensor
    eligible_count: Tensor
    accuracy_sum: Tensor
    exact_sum: Tensor
    steps_sum: Tensor


# =============================================================================
@dataclass(frozen=True)
class TokenAgg:
    """Token-level aggregates over a selected subset of sequences."""

    token_correct_sum: Tensor
    token_count_sum: Tensor


# =============================================================================
@dataclass(frozen=True)
class TransitionAgg:
    """Aggregates over all evaluated sequences on the current step."""

    evaluated_count: Tensor
    eligible_count: Tensor
    accuracy_sum: Tensor
    exact_sum: Tensor
    steps_sum: Tensor


# =============================================================================
@dataclass(frozen=True)
class StepMetrics:
    """Aggregated per-step metrics used for logging and control flow.

    ``extras`` stores algorithm-specific ratio metrics keyed by a stable internal
    name such as ``"loss_token"`` or ``"loss_actor"``.
    """

    episode: RolloutAgg
    episode_tokens: TokenAgg
    step: TransitionAgg
    step_tokens: TokenAgg
    extras: Mapping[str, RatioStat]


# =============================================================================
__all__ = [
    "RatioStat",
    "RolloutAgg",
    "StepMetrics",
    "TokenAgg",
    "TransitionAgg",
]
