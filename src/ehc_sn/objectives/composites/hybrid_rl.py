"""Hybrid RL objective: token-supervised Token loss plus value-control.

This is the *hybrid* batch-loss module for models that jointly train a
token-prediction head alongside a value-based ACT head (e.g. the
``maze_hard`` HRM v2 training path).

This module is consumed directly by the HRM v2 training surface via the
learner-owned TD(0) batch path — it is not a rollout-scoring objective.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor, nn

import ehc_sn.loss.cross_entropy as cross_entropy_module
import ehc_sn.metrics.signals as S
from ehc_sn.loss.cross_entropy import LossType
from ehc_sn.metrics.keys import LOSS_TOKEN, RL_LOSS_Q_VALUE, RL_LOSS_STATE_VALUE
from ehc_sn.metrics.step_metrics import RatioStat, StepMetrics
from ehc_sn.metrics.token import (
    AccuracyStats,
    build_token_step_metrics,
    compute_accuracy_stats,
)
from ehc_sn.objectives.control.q_value import (
    QValueObjective,
    SelectedQInput,
)
from ehc_sn.objectives.control.state_value import (
    StateValueObjective,
    ValueRegressionInput,
)
from ehc_sn.objectives.supervised.token import (
    TokenObjectiveInput,
    TokenPredictionObjective,
)
from ehc_sn.utils.detach import DetachMixin


# =============================================================================
class HybridRLLossConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`HybridRLObjective`.

    Attributes:
        token_loss: Name of the token-level supervised loss function.
        gamma: Discount factor for TD(0) return computation.
        c_state_value: Coefficient for the state-value regression loss.
        c_q_value: Coefficient for the auxiliary Q-value regression loss.
    """

    token_loss: LossType = Field(
        default="softmax_cross_entropy",
        description="The loss function to use for the modeling loss.",
    )
    gamma: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="TD(0) discount factor for return computation.",
    )
    c_state_value: float = Field(
        default=0.5,
        ge=0.0,
        description="State-value regression loss coefficient.",
    )
    c_q_value: float = Field(
        default=0.5,
        ge=0.0,
        description="vmPFC auxiliary Q-predictor loss coefficient.",
    )


# =============================================================================
@dataclass(frozen=True)
class HybridValueBatch:
    """Fully materialized value-control batch consumed by :class:`HybridRLObjective`.

    All TD(0) post-processing (returns, advantages) is performed by the learner
    before this batch is constructed. The objective is a pure function over
    the pre-computed fields — it does not instantiate distributions, compute
    log-probabilities, or regress values directly to rewards.

    Shapes assume ``B`` batch slots and ``S`` sequence length.
    """

    actions: Tensor  # (B,) sampled action indices
    q_values: Tensor  # (B, A) value scores over halt/continue
    rewards: Tensor  # (B,) immediate scalar reward
    done: Tensor  # (B,) combined done flag (terminated | truncated | max_steps)
    terminated: Tensor  # (B,) episode terminated flag
    truncated: Tensor  # (B,) episode truncated flag
    state_values: Tensor  # (B,) V(s_t) from the critic
    task_logits: Tensor  # (B, S, V) token-prediction logits for Token loss
    labels: Tensor  # (B, S) supervision targets for Token loss
    bootstrap_value: Tensor  # (B,) V(s_{t+1}) used for TD(0) target
    returns: Tensor  # (B,) TD(0) return: r + gamma * V(s_{t+1}) * (1 - done)
    steps: Tensor  # (B,) per-slot step counters after this step
    halted: Tensor  # (B,) per-slot done flags (used for episode metrics)
    token_weights: Tensor | None = None  # (B, S) optional Token loss weights


# =============================================================================
@dataclass(frozen=True)
class HybridRLLosses(DetachMixin):
    """Bundle of per-step loss terms for the hybrid RL objective (summed over batch)."""

    loss_token_sum: Tensor
    loss_q_value_sum: Tensor
    loss_state_value_sum: Tensor

    @property
    def total(self) -> Tensor:
        """Total scalar loss for the step."""
        return (
            self.loss_token_sum
            + self.loss_q_value_sum
            + self.loss_state_value_sum
        )


# =============================================================================
@dataclass(frozen=True)
class HybridRLObjectiveStep:
    """A single rollout/loss step produced by :class:`HybridRLObjective`."""

    losses: HybridRLLosses  # Combined losses for this step, kept live for backward()
    metrics: StepMetrics  # Aggregated metrics for this step, used for logging
    signals: dict[str, Tensor]  # Diagnostic signals for logging

    def __post_init__(self) -> None:
        if self.signals is None:
            object.__setattr__(self, "signals", {})

    @property
    def loss(self) -> Tensor:
        """Scalar loss for this step, used for back-propagation."""
        return self.losses.total


# =============================================================================
@dataclass(frozen=True)
class HybridValueTerms:
    """Per-example hybrid RL loss terms (unreduced)."""

    loss_token: Tensor
    loss_state_value: Tensor
    loss_q_value: Tensor


# =============================================================================
@dataclass
class HybridValueContext:
    """Shared context resolved once per hybrid RL step."""

    batch: HybridValueBatch
    stats: AccuracyStats
    is_warmup: bool

    terms: HybridValueTerms | None = field(default=None, init=False)
    losses: HybridRLLosses | None = field(default=None, init=False)
    metrics: StepMetrics | None = field(default=None, init=False)
    signals: dict[str, Tensor] = field(default_factory=dict, init=False)


# =============================================================================
class HybridRLObjective(nn.Module):
    """Hybrid RL batch-loss module: token-supervised Token loss plus value-control.

    The sole entry point is :meth:`compute_step`, which accepts a fully
    materialized :class:`HybridValueBatch` assembled by the learner
    (e.g. :class:`~ehp_sn.training.q_halting.TD0ActorCriticBatchBuilder`) and
    returns loss, metrics, and signals.  It is a pure function over its input.

    This class is **not** a rollout-scoring objective.  It does not implement
    ``evaluate_step`` or ``forward`` over rollout chunks.  Validation scoring
    is done by :class:`~ehp_sn.training.q_halting.ZeroBootstrapActorCriticValidationScorer`.

    Used by the ``maze_hard`` HRM v2 training path.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        config: HybridRLLossConfig,
    ) -> None:
        """Create the batch-loss module from its configuration."""
        super().__init__()
        self._config = config
        self._token_component = TokenPredictionObjective(
            loss_fn=self._config.token_loss,
        )
        self._q_component = QValueObjective()
        self._v_component = StateValueObjective()

    @property
    def config(self) -> HybridRLLossConfig:
        """Return the batch-loss configuration."""
        return self._config

    @property
    def token_loss_fn(self) -> Any:
        """Return the configured token-level loss function."""
        return getattr(cross_entropy_module, self._config.token_loss)

    def compute_step(  # ------------------------------------------------------
        self,
        batch: HybridValueBatch,
        *,
        is_warmup: bool = False,
    ) -> HybridRLObjectiveStep:
        """Score a pre-materialized value-control batch and return loss, metrics, and signals.

        This is the canonical entry point for the live HRM v2 training path.
        The caller (learner) owns TD(0) post-processing and provides a fully
        materialized batch; this method is a pure function over that input.
        """
        context = self.build_context(batch, is_warmup=is_warmup)
        context.terms = self.compute_terms(context)
        context.losses = self.compute_losses(context)
        context.metrics = self.evaluate_metrics(context)
        context.signals = self.compute_signals(context)
        return self.build_output(context)

    def build_context(  # -----------------------------------------------------
        self,
        batch: HybridValueBatch,
        *,
        is_warmup: bool = False,
        **_: Any,
    ) -> HybridValueContext:
        """Resolve token stats and other shared context for one batch step."""
        stats = compute_accuracy_stats(batch.task_logits, batch.labels)
        return HybridValueContext(batch=batch, stats=stats, is_warmup=is_warmup)

    def compute_terms(  # -----------------------------------------------------
        self,
        context: HybridValueContext,
        **_: Any,
    ) -> HybridValueTerms:
        """Compute unreduced loss terms for a pre-materialized batch."""
        batch = context.batch
        token_input = TokenObjectiveInput(
            logits=batch.task_logits,
            labels=batch.labels,
            weights=batch.token_weights,
        )
        token_result = self._token_component(token_input)
        loss_token = token_result.terms["token_per_element"]
        if context.is_warmup:
            zero = batch.task_logits.new_zeros(batch.rewards.shape[0])
            loss_state_value = zero
            loss_q_value = zero
        else:
            v_input = ValueRegressionInput(
                predictions=batch.state_values,
                targets=batch.returns,
            )
            loss_state_value = self._v_component(v_input).terms[
                "state_value_per_sample"
            ]

            q_input = SelectedQInput(
                q_values=batch.q_values,
                actions=batch.actions,
                targets=batch.returns.detach(),
            )
            loss_q_value = self._q_component(q_input).terms[
                "q_value_per_sample"
            ]
        return HybridValueTerms(
            loss_token=loss_token,
            loss_state_value=loss_state_value,
            loss_q_value=loss_q_value,
        )

    def compute_losses(  # ----------------------------------------------------
        self,
        context: HybridValueContext,
        **_: Any,
    ) -> HybridRLLosses:
        """Reduce per-example hybrid RL loss terms to batch sums."""
        if context.terms is None:
            raise RuntimeError(
                "HybridRLObjective.compute_losses requires terms."
            )
        loss_token_per_seq = context.terms.loss_token.sum(
            -1
        ) / context.stats.loss_counts.clamp_min(1)
        loss_token_sum = loss_token_per_seq.sum()
        return HybridRLLosses(
            loss_token_sum=loss_token_sum,
            loss_state_value_sum=self.config.c_state_value
            * context.terms.loss_state_value.sum(),
            loss_q_value_sum=self.config.c_q_value
            * context.terms.loss_q_value.sum(),
        )

    def evaluate_metrics(  # -------------------------------------------------
        self,
        context: HybridValueContext,
        **_: Any,
    ) -> StepMetrics:
        """Compute per-step metrics for the hybrid RL objective."""
        if context.losses is None:
            raise RuntimeError(
                "HybridRLObjective.evaluate_metrics requires losses."
            )
        batch = context.batch
        extras = self._build_metric_ratios(
            context.losses, batch_size=int(batch.rewards.shape[0])
        )
        steps = (
            batch.steps
            if batch.steps is not None
            else torch.zeros(
                batch.rewards.shape[0],
                dtype=torch.long,
                device=batch.rewards.device,
            )
        )
        return build_token_step_metrics(
            steps, batch.halted, context.stats, extras
        )

    def _build_metric_ratios(  # ----------------------------------------------
        self,
        losses: HybridRLLosses,
        *,
        batch_size: int,
    ) -> dict[str, RatioStat]:
        """Pack hybrid RL loss terms into detached generic ratio metrics."""
        batch_count = losses.loss_token_sum.new_tensor(
            batch_size, dtype=torch.float32
        )
        return {
            LOSS_TOKEN: RatioStat(
                numerator_sum=losses.loss_token_sum.detach(),
                denominator_sum=batch_count,
            ),
            RL_LOSS_STATE_VALUE: RatioStat(
                numerator_sum=losses.loss_state_value_sum.detach(),
                denominator_sum=batch_count,
            ),
            RL_LOSS_Q_VALUE: RatioStat(
                numerator_sum=losses.loss_q_value_sum.detach(),
                denominator_sum=batch_count,
            ),
        }

    def compute_signals(  # ---------------------------------------------------
        self,
        context: HybridValueContext,
        **_: Any,
    ) -> dict[str, Tensor]:
        """Compute lightweight diagnostic signals for the hybrid RL objective."""
        if context.losses is None:
            raise RuntimeError(
                "HybridRLObjective.compute_signals requires losses."
            )
        batch = context.batch
        losses = context.losses
        td_error = batch.returns - batch.state_values
        return {
            S.REWARD_MEAN: batch.rewards.mean().detach(),
            S.REWARD_STD: batch.rewards.std().detach(),
            S.Q_MEAN: batch.q_values.detach().mean(),
            S.Q_STD: batch.q_values.detach().std(),
            S.RPE_MAGNITUDE: td_error.abs().mean(),
            S.LOSS_STATE_VALUE: losses.loss_state_value_sum.detach(),
            S.LOSS_Q_VALUE: losses.loss_q_value_sum.detach(),
        }  # fmt: skip

    def build_output(  # -----------------------------------------------------
        self,
        context: HybridValueContext,
        **_: Any,
    ) -> HybridRLObjectiveStep:
        """Build the final step output from computed context fields."""
        if context.losses is None or context.metrics is None:
            raise RuntimeError(
                "HybridRLObjective.build_output requires losses and metrics."
            )
        return HybridRLObjectiveStep(
            losses=context.losses,
            metrics=context.metrics,
            signals=context.signals,
        )


# =============================================================================
__all__ = [
    "HybridValueBatch",
    "HybridValueContext",
    "HybridValueTerms",
    "HybridRLLossConfig",
    "HybridRLObjective",
    "HybridRLLosses",
    "HybridRLObjectiveStep",
]
