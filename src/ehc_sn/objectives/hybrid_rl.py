"""Hybrid RL objective: token-supervised LM loss plus actor-critic.

This is the *hybrid* batch-loss module for models that jointly train a
token-prediction head alongside an actor-critic RL head (e.g. the
``maze_hard`` HRM v2 training path).

This module is consumed directly by the HRM v2 training surface via the
learner-owned TD(0) batch path — it is not a rollout-scoring objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor, nn

import ehc_sn.loss.cross_entropy as cross_entropy_module
from ehc_sn.loss.cross_entropy import LossType
from ehc_sn.metrics import signals as S
from ehc_sn.metrics.keys import LOSS_LM, RL_LOSS_ACTOR, RL_LOSS_CRITIC, RL_LOSS_ENTROPY, RL_LOSS_Q_VALUE
from ehc_sn.objectives._token import AccuracyStats, build_token_step_metrics, compute_accuracy_stats, compute_lm_loss_sum
from ehc_sn.training.types import RatioStat, StepMetrics
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class HybridRLLossConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`HybridRLLossHead`.

    Attributes:
        function: Name of the token-level supervised loss function.
        gamma: Discount factor for TD(0) return computation.
        c_actor: Coefficient for the policy gradient (actor) loss.
        c_critic: Coefficient for the value regression (critic) loss.
        c_entropy: Coefficient for entropy regularization.
        c_q_value: Coefficient for the auxiliary Q-value regression loss.
    """

    function: LossType = Field(
        default="stablemax_cross_entropy",
        description="The loss function to use for the modeling loss.",
    )
    gamma: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="TD(0) discount factor for return computation.",
    )
    c_actor: float = Field(
        default=1.0,
        ge=0.0,
        description="Actor loss coefficient.",
    )
    c_critic: float = Field(
        default=0.5,
        ge=0.0,
        description="Critic loss coefficient.",
    )
    c_entropy: float = Field(
        default=0.01,
        ge=0.0,
        description="Entropy regularization coefficient.",
    )
    c_q_value: float = Field(
        default=0.5,
        ge=0.0,
        description="vmPFC auxiliary Q-predictor loss coefficient.",
    )


# =================================================================================================
@dataclass(frozen=True)
class HybridActorCriticBatch:
    """Fully materialized actor-critic batch consumed by :class:`HybridRLLossHead`.

    All TD(0) post-processing (returns, advantages) is performed by the learner
    before this batch is constructed. The objective is a pure function over
    the pre-computed fields — it does not instantiate distributions, compute
    log-probabilities, or regress values directly to rewards.

    Shapes assume ``B`` batch slots and ``S`` sequence length.
    """

    actions: Tensor  # (B,)      sampled action indices
    policy_logits: Tensor  # (B, A)    raw actor-head logits
    rewards: Tensor  # (B,)      immediate scalar reward
    done: Tensor  # (B,)      combined done flag (terminated | truncated | max_steps)
    terminated: Tensor  # (B,)      episode terminated flag
    truncated: Tensor  # (B,)      episode truncated flag
    value_estimates: Tensor  # (B,)      V(s_t) from the critic
    action_log_prob: Tensor  # (B,)      log pi(a_t | s_t)
    action_entropy: Tensor  # (B,)      H[pi(. | s_t)]
    task_logits: Tensor  # (B, S, V) token-prediction logits for LM loss
    labels: Tensor  # (B, S)    supervision targets for LM loss
    bootstrap_value: Tensor  # (B,)      V(s_{t+1}) used for TD(0) target
    returns: Tensor  # (B,)      TD(0) return: r + gamma * V(s_{t+1}) * (1 - done)
    advantages: Tensor  # (B,)      detached advantage: returns - V(s_t)
    steps: Tensor  # (B,)      per-slot step counters after this step
    halted: Tensor  # (B,)      per-slot done flags (used for episode metrics)


# =================================================================================================
@dataclass(frozen=True)
class HybridRLLosses(DetachMixin):
    """Bundle of per-step loss terms for the hybrid RL objective (summed over batch)."""

    loss_lm_sum: Tensor  # Supervised LM loss sum over the batch for the current step
    loss_q_value_sum: Tensor  # Auxiliary Q-value regression loss sum over the batch for the current step
    loss_actor_sum: Tensor  # Policy gradient (actor) loss sum over the batch for the current step
    loss_critic_sum: Tensor  # Value regression (critic) loss sum over the batch for the current step
    loss_entropy_sum: Tensor  # Entropy regularization loss sum over the batch for the current step

    @property
    def total(self) -> Tensor:
        """Total scalar loss for the step."""
        loss_rl = self.loss_actor_sum + self.loss_critic_sum + self.loss_entropy_sum
        loss_m = self.loss_lm_sum + self.loss_q_value_sum
        return loss_rl + loss_m


# =================================================================================================
@dataclass(frozen=True)
class HybridRLLossStep:
    """A single rollout/loss step produced by :class:`HybridRLLossHead`."""

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


# =================================================================================================
class HybridRLLossHead(nn.Module):
    """Hybrid RL batch-loss module: token-supervised LM loss plus actor-critic.

    The sole entry point is :meth:`compute_step`, which accepts a fully
    materialized :class:`HybridActorCriticBatch` assembled by the learner
    (e.g. :class:`~ehc_sn.training.actor_critic.TD0ActorCriticBatchBuilder`) and
    returns loss, metrics, and signals.  It is a pure function over its input.

    This class is **not** a rollout-scoring objective.  It does not implement
    ``evaluate_step`` or ``forward`` over rollout chunks.  Validation scoring
    is done by :class:`~ehc_sn.training.actor_critic.ZeroBootstrapActorCriticValidationScorer`.

    Used by the ``maze_hard`` HRM v2 training path.
    """

    def __init__(self, config: HybridRLLossConfig) -> None:
        """Create the batch-loss module from its configuration."""
        super().__init__()
        self._config = config

    @property
    def config(self) -> HybridRLLossConfig:
        """Return the batch-loss configuration."""
        return self._config

    # -- Loss function accessor -----------------------------------------------------------------

    @property
    def loss_fn(self) -> Any:
        """Return the configured token-level loss function."""
        return getattr(cross_entropy_module, self._config.function)

    def compute_lm_loss(self, logits_lm: Tensor, labels: Tensor, stats: AccuracyStats) -> Tensor:
        """Compute the summed supervised token loss for a step."""
        return compute_lm_loss_sum(self.loss_fn, logits_lm, labels, stats)

    # -- Primary training entry point -----------------------------------------------------------

    def compute_step(  # -----------------------------------------------------------------------
        self,
        batch: HybridActorCriticBatch,
        *,
        is_warmup: bool = False,
    ) -> HybridRLLossStep:
        """Score a pre-materialized actor-critic batch and return loss, metrics, and signals.

        This is the canonical entry point for the live HRM v2 training path.
        The caller (learner) owns TD(0) post-processing and provides a fully
        materialized batch; this method is a pure function over that input.
        """
        stats = compute_accuracy_stats(batch.task_logits, batch.labels)
        losses = self.compute_losses(batch, stats, is_warmup=is_warmup)
        extras = self._build_metric_ratios(losses, batch_size=int(batch.rewards.shape[0]))
        steps = (
            batch.steps if batch.steps is not None else torch.zeros(batch.rewards.shape[0], dtype=torch.long, device=batch.rewards.device)
        )
        metrics = build_token_step_metrics(steps, batch.halted, stats, extras)
        signals = self.compute_signals(batch, losses)
        return HybridRLLossStep(losses=losses, metrics=metrics, signals=signals)

    # -- Pure loss and signal computation -------------------------------------------------------

    def compute_losses(  # ---------------------------------------------------------------------
        self,
        batch: HybridActorCriticBatch,
        stats: AccuracyStats,
        *,
        is_warmup: bool = False,
        **_: Any,
    ) -> HybridRLLosses:
        """Compute supervised and hybrid RL loss terms from a pre-materialized batch.

        This method is pure: it does not instantiate distributions, compute
        log-probabilities, or compute advantages. All precomputed fields are
        consumed directly from ``batch``.
        """
        loss_lm_sum = self.compute_lm_loss(batch.task_logits, batch.labels, stats)
        if not is_warmup:
            loss_actor = -(batch.action_log_prob * batch.advantages).sum()
            loss_critic = F.mse_loss(batch.value_estimates, batch.returns, reduction="sum")
            loss_entropy = -batch.action_entropy.sum()
            q_a = batch.policy_logits.gather(1, batch.actions.unsqueeze(-1)).squeeze(-1)
            loss_q_value = F.mse_loss(q_a, batch.returns.detach(), reduction="sum")
        else:
            zero = batch.task_logits.new_zeros(())
            loss_actor = zero
            loss_critic = zero
            loss_entropy = zero
            loss_q_value = zero
        return HybridRLLosses(
            loss_lm_sum=loss_lm_sum,
            loss_actor_sum=self.config.c_actor * loss_actor,
            loss_critic_sum=self.config.c_critic * loss_critic,
            loss_entropy_sum=self.config.c_entropy * loss_entropy,
            loss_q_value_sum=self.config.c_q_value * loss_q_value,
        )

    def _build_metric_ratios(  # ---------------------------------------------------------------
        self, losses: HybridRLLosses, *, batch_size: int,
    ) -> dict[str, RatioStat]:  # fmt: skip
        """Pack hybrid RL loss terms into detached generic ratio metrics."""
        batch_count = losses.loss_lm_sum.new_tensor(batch_size, dtype=torch.float32)
        return {
            LOSS_LM: RatioStat(losses.loss_lm_sum.detach(), batch_count),
            RL_LOSS_ACTOR: RatioStat(losses.loss_actor_sum.detach(), batch_count),
            RL_LOSS_CRITIC: RatioStat(losses.loss_critic_sum.detach(), batch_count),
            RL_LOSS_ENTROPY: RatioStat(losses.loss_entropy_sum.detach(), batch_count),
            RL_LOSS_Q_VALUE: RatioStat(losses.loss_q_value_sum.detach(), batch_count),
        }

    def compute_signals(  # --------------------------------------------------------------------
        self,
        batch: HybridActorCriticBatch,
        losses: HybridRLLosses,
        **_: Any,
    ) -> dict[str, Tensor]:
        """Compute lightweight diagnostic signals for the hybrid RL objective."""
        return {
            S.REWARD_MEAN: batch.rewards.mean().detach(),
            S.REWARD_STD: batch.rewards.std().detach(),
            S.Q_MEAN: batch.policy_logits.detach().mean(),
            S.Q_STD: batch.policy_logits.detach().std(),
            S.RPE_MAGNITUDE: batch.advantages.abs().mean(),
            S.ACTION_ENTROPY: batch.action_entropy.mean().detach(),
            S.LOSS_ACTOR: losses.loss_actor_sum.detach(),
            S.LOSS_CRITIC: losses.loss_critic_sum.detach(),
            S.LOSS_ENTROPY: losses.loss_entropy_sum.detach(),
        }  # fmt: skip


# =================================================================================================
__all__ = [
    "HybridActorCriticBatch",
    "HybridRLLossConfig",
    "HybridRLLossHead",
    "HybridRLLosses",
    "HybridRLLossStep",
]
