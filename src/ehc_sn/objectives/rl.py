"""RL objective over executed rollout steps.

The RL objective keeps task-supervision extraction generic via the token
binding seam while keeping actor and critic extraction local to the RL path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor
from torch.distributions import Categorical

from ehc_sn.controllers.rl import RLRolloutState, RLStepOutput
from ehc_sn.loss.cross_entropy import LossType
from ehc_sn.metrics import signals as S
from ehc_sn.metrics.keys import LOSS_LM, RL_LOSS_ACTOR, RL_LOSS_CRITIC, RL_LOSS_ENTROPY, RL_LOSS_Q_VALUE
from ehc_sn.objectives._token import AccuracyStats, TokenLossHeadBase, TokenSupervisionBinding
from ehc_sn.training.types import RatioStat, StepMetrics
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class RLLossConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`RLLossHead`.

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
class RLObjectiveBinding[TargetsT](TokenSupervisionBinding[TargetsT], Protocol):
    """RL-local extraction seam for actor and critic readouts."""

    def extract_policy_logits(self, step_output: Any) -> Tensor:
        """Return policy logits used for action selection and actor loss."""

    def extract_state_value(self, step_output: Any) -> Tensor:
        """Return critic state values used for the value loss."""


# =================================================================================================
@dataclass(frozen=True)
class Losses(DetachMixin):
    """Bundle of per-step loss terms (summed over batch)."""

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
class RLLossStep:
    """A single rollout/loss step produced by :class:`RLLossHead`."""

    losses: Losses  # Combined losses for this step, kept live for backward()
    metrics: StepMetrics  # Aggregated metrics for this step, used for logging
    outputs: Optional[RLStepOutput] = None  # Raw controller outputs
    signals: dict[str, Tensor] = None  # Diagnostic signals (T2/T3); plain dict, no schema commitment

    def __post_init__(self) -> None:
        if self.signals is None:
            object.__setattr__(self, "signals", {})

    @property
    def loss(self) -> Tensor:
        """Scalar loss for this step, used for back-propagation."""
        return self.losses.total


# =================================================================================================
class RLLossHead(TokenLossHeadBase[RLLossConfig]):
    """Pure RL objective scored over executed rollout chunks."""

    def __init__(self, config: RLLossConfig, *, task_binding: RLObjectiveBinding[Any]) -> None:
        """Create an RL objective from its loss configuration and binding."""
        super().__init__(config=config, token_binding=task_binding)
        self._binding = task_binding

    def compute_losses(
        self,
        outputs: RLStepOutput,
        targets: Any,
        stats: AccuracyStats,
        *,
        logits: Tensor,
        is_warmup: bool = False,
        **_: Any,
    ) -> Losses:
        """Compute supervised and RL loss terms for a step."""
        labels = getattr(targets, "labels", targets)
        if not isinstance(labels, Tensor):
            raise TypeError("RLLossHead expects tensor labels from the bound RL targets.")

        lm_logits = logits
        policy_logits = self._binding.extract_policy_logits(outputs)
        state_value = self._binding.extract_state_value(outputs).squeeze(-1)

        dist = Categorical(logits=policy_logits)
        logp = dist.log_prob(outputs.action)
        entropy = dist.entropy()

        reward = outputs.reward.squeeze(-1)
        advantage = (reward - state_value).detach()
        q_a = policy_logits.gather(1, outputs.action.unsqueeze(-1)).squeeze(-1)

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

        return Losses(
            loss_lm_sum=loss_lm_sum,
            loss_actor_sum=self.config.c_actor * loss_actor,
            loss_critic_sum=self.config.c_critic * loss_critic,
            loss_entropy_sum=self.config.c_entropy * loss_entropy,
            loss_q_value_sum=self.config.c_q_value * loss_q_value,
        )

    def _build_metric_ratios(  # -----------------------------------------------------------------
        self, losses: Losses, *, batch_size: int,
    ) -> dict[str, RatioStat]:  # fmt: skip
        """Pack RL loss terms into detached generic ratio metrics."""
        batch_count = losses.loss_lm_sum.new_tensor(batch_size, dtype=torch.float32)
        return {
            LOSS_LM: RatioStat(losses.loss_lm_sum.detach(), batch_count),
            RL_LOSS_ACTOR: RatioStat(losses.loss_actor_sum.detach(), batch_count),
            RL_LOSS_CRITIC: RatioStat(losses.loss_critic_sum.detach(), batch_count),
            RL_LOSS_ENTROPY: RatioStat(losses.loss_entropy_sum.detach(), batch_count),
            RL_LOSS_Q_VALUE: RatioStat(losses.loss_q_value_sum.detach(), batch_count),
        }

    def _build_step_output(  # -------------------------------------------------------------------
        self, losses: Losses, metrics: Any, signals: dict[str, Tensor], outputs: RLStepOutput, **_: Any,
    ) -> RLLossStep:  # fmt: skip
        """Wrap losses, metrics, and signals into an :class:`RLLossStep`."""
        return RLLossStep(losses=losses, metrics=metrics, outputs=outputs, signals=signals)

    def compute_signals(  # -----------------------------------------------------------------------
        self, batch: Batch, state: RLRolloutState, outputs: RLStepOutput, losses: Losses, **_: Any,
    ) -> dict[str, Tensor]:  # fmt: skip
        """Compute lightweight diagnostic signals.

        Signals are intended for TensorBoard-style scalar logging.
        """
        _ = batch
        policy_logits = self._binding.extract_policy_logits(outputs)
        state_value = self._binding.extract_state_value(outputs).squeeze(-1)
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
__all__ = ["RLObjectiveBinding", "RLLossConfig", "RLLossHead", "RLLossStep"]
