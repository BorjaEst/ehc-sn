"""Generic step-metrics types for rollout-based objectives.

The shared layer exposes three pieces of structure:

* **Episode aggregates** for completed sequences on the current step.
* **Step aggregates** for all currently evaluated sequences on the current step.
* **Keyed ratio extras** for algorithm-specific losses and diagnostics.

This keeps the generic layer free of ACT- or RL-specific field names while
preserving the routing model used by :func:`~ehc_sn.metrics.update_metrics_from_step`.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from torch import Tensor


# =================================================================================================
@dataclass(frozen=True)
class RatioStat:
    """A single aggregated ratio statistic.

    Both fields are sums over the current step / batch.
    """

    numerator_sum: Tensor
    denominator_sum: Tensor


# =================================================================================================
@dataclass(frozen=True)
class RolloutAgg:
    """Lifecycle aggregates over completed sequences on the current step."""

    completed_count: Tensor
    eligible_count: Tensor
    accuracy_sum: Tensor
    exact_sum: Tensor
    steps_sum: Tensor


# =================================================================================================
@dataclass(frozen=True)
class TokenAgg:
    """Token-level aggregates over a selected subset of sequences."""

    token_correct_sum: Tensor
    token_count_sum: Tensor


# =================================================================================================
@dataclass(frozen=True)
class TransitionAgg:
    """Aggregates over all evaluated sequences on the current step."""

    evaluated_count: Tensor
    eligible_count: Tensor
    accuracy_sum: Tensor
    exact_sum: Tensor
    steps_sum: Tensor


# =================================================================================================
@dataclass(frozen=True)
class StepMetrics:
    """Aggregated per-step metrics used for logging and control flow.

    ``extras`` stores algorithm-specific ratio metrics keyed by a stable internal
    name such as ``"loss_lm"`` or ``"loss_actor"``.
    """

    episode: RolloutAgg
    episode_tokens: TokenAgg
    step: TransitionAgg
    step_tokens: TokenAgg
    extras: Mapping[str, RatioStat]


# =================================================================================================
__all__ = ["RatioStat", "RolloutAgg", "StepMetrics", "TokenAgg", "TransitionAgg"]
