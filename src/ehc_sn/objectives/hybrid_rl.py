"""Hybrid RL objective: token-supervised LM loss plus actor-critic RL.

This is the *hybrid* objective family for models that jointly train a
token-prediction head alongside an actor-critic RL head (e.g. the
``maze_hard`` HRM v2 training path).

For pure reward-first RL without token supervision, see
:mod:`ehc_sn.objectives.rl`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor
from torch.distributions import Categorical

from ehc_sn.controllers.rl import InteractionRecord, RLRolloutState
from ehc_sn.loss.cross_entropy import LossType
from ehc_sn.metrics import signals as S
from ehc_sn.metrics.keys import LOSS_LM, RL_LOSS_ACTOR, RL_LOSS_CRITIC, RL_LOSS_ENTROPY, RL_LOSS_Q_VALUE
from ehc_sn.objectives._token import AccuracyStats, TokenLossHeadBase, TokenSupervisionBinding
from ehc_sn.training.types import RatioStat, StepMetrics
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class HybridRLLossConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`HybridRLLossHead`.

    Attributes:
        function: Name of the token-level supervised loss function.
        c_actor: Coefficient for the policy gradient (actor) loss.
        c_critic: Coefficient for the value regression (critic) loss.
        c_entropy: Coefficient for entropy regularization.
        c_q_value: Coefficient for the auxiliary Q-value regression loss.
    """

    function: LossType = Field(
        default="stablemax_cross_entropy",
        description="The loss function to use for the modeling loss.",
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
class HybridRLObjectiveBinding[TargetsT](TokenSupervisionBinding[TargetsT], Protocol):
    """Hybrid RL extraction seam: token supervision plus actor and critic readouts."""

    def extract_policy_logits(self, step_output: Any) -> Tensor:
        """Return policy logits used for action selection and actor loss."""

    def extract_state_value(self, step_output: Any) -> Tensor:
        """Return critic state values used for the value loss."""


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
    outputs: Optional[InteractionRecord] = None  # Controller interaction record
    signals: dict[str, Tensor] = None  # Diagnostic signals (T2/T3); plain dict, no schema commitment

    def __post_init__(self) -> None:
        if self.signals is None:
            object.__setattr__(self, "signals", {})

    @property
    def loss(self) -> Tensor:
        """Scalar loss for this step, used for back-propagation."""
        return self.losses.total


# =================================================================================================
class HybridRLLossHead(TokenLossHeadBase[HybridRLLossConfig]):
    """Hybrid RL objective: token-supervised LM loss plus actor-critic scored over rollout steps.

    Used by the ``maze_hard`` HRM v2 training path. For pure reward-first RL
    without token supervision, use :class:`~ehc_sn.objectives.rl.RLLossHead`.
    """

    def __init__(
        self,
        config: HybridRLLossConfig,
        *,
        task_binding: HybridRLObjectiveBinding[Any],
    ) -> None:
        """Create a hybrid RL objective from its loss configuration and binding."""
        super().__init__(config=config, token_binding=task_binding)
        self._binding = task_binding

    def compute_losses(
        self,
        outputs: InteractionRecord,
        targets: Any,
        stats: AccuracyStats,
        *,
        logits: Tensor,
        is_warmup: bool = False,
        **_: Any,
    ) -> HybridRLLosses:
        """Compute supervised and hybrid RL loss terms for a step."""
        labels = getattr(targets, "labels", targets)
        if not isinstance(labels, Tensor):
            raise TypeError("HybridRLLossHead expects tensor labels from the bound hybrid RL targets.")

        lm_logits = logits
        policy_logits = outputs.policy_logits
        state_value = outputs.value_estimate.squeeze(-1)

        dist = Categorical(logits=policy_logits)
        logp = dist.log_prob(outputs.sampled_action)
        entropy = dist.entropy()

        reward = outputs.reward.squeeze(-1)
        advantage = (reward - state_value).detach()
        q_a = policy_logits.gather(1, outputs.sampled_action.unsqueeze(-1)).squeeze(-1)

        loss_lm_sum = self.compute_lm_loss(lm_logits, labels, stats)
        if not is_warmup:
            loss_actor = -(logp * advantage).sum()
            loss_critic = F.mse_loss(state_value, reward, reduction="sum")
            loss_entropy = -entropy.sum()
            loss_q_value = F.mse_loss(q_a, reward.detach(), reduction="sum")
        else:
            zero = torch.tensor(0.0, device=lm_logits.device)
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

    def _build_metric_ratios(  # -----------------------------------------------------------------
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

    def _build_step_output(  # -------------------------------------------------------------------
        self, losses: HybridRLLosses, metrics: Any, signals: dict[str, Tensor], outputs: InteractionRecord, **_: Any,
    ) -> HybridRLLossStep:  # fmt: skip
        """Wrap losses, metrics, and signals into a :class:`HybridRLLossStep`."""
        return HybridRLLossStep(losses=losses, metrics=metrics, outputs=outputs, signals=signals)

    def compute_signals(  # -----------------------------------------------------------------------
        self, batch: Batch, state: RLRolloutState, outputs: InteractionRecord, losses: HybridRLLosses, **_: Any,
    ) -> dict[str, Tensor]:  # fmt: skip
        """Compute lightweight diagnostic signals for the hybrid RL objective."""
        _ = batch
        policy_logits = outputs.policy_logits
        state_value = outputs.value_estimate.squeeze(-1)
        reward = outputs.reward.squeeze(-1)
        rpe = (reward - state_value).detach()
        dist = Categorical(logits=policy_logits.detach())
        return {
            S.REWARD_MEAN: reward.mean().detach(),
            S.REWARD_STD: reward.std().detach(),
            S.Q_MEAN: policy_logits.detach().mean(),
            S.Q_STD: policy_logits.detach().std(),
            S.RPE_MAGNITUDE: rpe.abs().mean(),
            S.ACTION_ENTROPY: dist.entropy().mean(),
            S.STEPS_MEAN: state.steps.float().mean().detach(),
            S.LOSS_ACTOR: losses.loss_actor_sum.detach(),
            S.LOSS_CRITIC: losses.loss_critic_sum.detach(),
            S.LOSS_ENTROPY: losses.loss_entropy_sum.detach(),
        }  # fmt: skip


# =================================================================================================
__all__ = [
    "HybridRLLossConfig",
    "HybridRLLossHead",
    "HybridRLLosses",
    "HybridRLLossStep",
    "HybridRLObjectiveBinding",
]
