"""Pure reward-first RL objective over executed rollout steps.

This module is the canonical pure RL objective family. It requires no token
labels, no token logits, and no task supervision targets.

For models that jointly train a token-prediction head alongside actor-critic
RL (e.g. the ``maze_hard`` HRM v2 path), see
:mod:`ehc_sn.objectives.hybrid_rl`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.metrics import signals as S
from ehc_sn.metrics.keys import RL_LOSS_ACTOR, RL_LOSS_CRITIC, RL_LOSS_ENTROPY
from ehc_sn.objectives._base import BaseObjective
from ehc_sn.policies._base import PolicyDecision
from ehc_sn.rollouts import EvaluatedChunk, ObservedStep, RolloutChunk, StepRecord
from ehc_sn.training.types import RatioStat, RolloutAgg, StepMetrics, TokenAgg, TransitionAgg
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
@dataclass(frozen=True)
class RewardBearingStep:
    """Generic reward-bearing transition bundle extracted at rollout time.

    All fields are batch-aligned with shape ``(B,)`` unless noted.

    Attributes:
        action: Selected action indices ``(B,)`` int64.
        reward: Environment reward ``(B,)`` float.
        terminated: Whether the episode terminated ``(B,)`` bool.
        truncated: Whether the episode was truncated ``(B,)`` bool.
        bootstrap_mask: True when value bootstrapping is appropriate ``(B,)``
            bool — typically ``~terminated``.
        value_estimate: Critic state-value at this step ``(B,)`` float.
        policy_decision: Rollout-time policy decision; carries ``log_prob``
            and ``entropy`` if the policy populated them.
    """

    action: Tensor
    reward: Tensor
    terminated: Tensor
    truncated: Tensor
    bootstrap_mask: Tensor
    value_estimate: Tensor
    policy_decision: PolicyDecision


# =================================================================================================
class RLObjectiveBinding[StepOutputT](Protocol):
    """Pure RL extraction seam.

    Does NOT inherit :class:`~ehc_sn.objectives._token.TokenSupervisionBinding`.
    No token labels, logits, or task supervision targets are required.
    """

    def extract_reward_step(
        self,
        batch: Batch,
        carry: Any,
        step_output: StepOutputT,
    ) -> RewardBearingStep:
        """Extract a :class:`RewardBearingStep` from one executed rollout step."""


# =================================================================================================
class RLLossConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`RLLossHead`.

    Attributes:
        c_actor: Coefficient for the policy gradient (actor) loss.
        c_critic: Coefficient for the value regression (critic) loss.
        c_entropy: Coefficient for entropy regularization.
    """

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


# =================================================================================================
@dataclass(frozen=True)
class RLLosses(DetachMixin):
    """Per-step RL loss terms (summed over batch)."""

    loss_actor_sum: Tensor  # Policy gradient loss sum over the batch
    loss_critic_sum: Tensor  # Value regression loss sum over the batch
    loss_entropy_sum: Tensor  # Entropy regularization loss sum over the batch

    @property
    def total(self) -> Tensor:
        """Total scalar loss for the step."""
        return self.loss_actor_sum + self.loss_critic_sum + self.loss_entropy_sum


# =================================================================================================
@dataclass(frozen=True)
class RLLossStep:
    """A single rollout/loss step produced by :class:`RLLossHead`."""

    losses: RLLosses
    metrics: StepMetrics
    signals: dict[str, Tensor] = field(default_factory=dict)
    outputs: Optional[Any] = None

    @property
    def loss(self) -> Tensor:
        """Scalar loss for this step, used for back-propagation."""
        return self.losses.total


# =================================================================================================
def _build_rl_step_metrics(carry: Any, losses: RLLosses) -> StepMetrics:
    """Build per-step metrics for the pure RL objective.

    Token accuracy fields are zeroed — pure RL produces no supervised token stats.
    """
    halted: Tensor = carry.halted
    steps: Tensor = carry.steps
    batch_size = int(halted.shape[0])
    device = halted.device
    dtype = torch.float32

    zero = torch.zeros(1, device=device, dtype=dtype)
    completed = halted
    completed_count = completed.to(dtype).sum()
    eligible_count = torch.tensor(batch_size, device=device, dtype=dtype)
    steps_sum = (steps.float() * completed.float()).sum()
    batch_count = torch.tensor(batch_size, device=device, dtype=dtype)

    return StepMetrics(
        episode=RolloutAgg(
            completed_count=completed_count,
            eligible_count=eligible_count,
            accuracy_sum=zero,
            exact_sum=zero,
            steps_sum=steps_sum,
        ),
        episode_tokens=TokenAgg(token_correct_sum=zero, token_count_sum=zero),
        step=TransitionAgg(
            evaluated_count=eligible_count,
            eligible_count=eligible_count,
            accuracy_sum=zero,
            exact_sum=zero,
            steps_sum=steps.float().sum(),
        ),
        step_tokens=TokenAgg(token_correct_sum=zero, token_count_sum=batch_count),
        extras={
            RL_LOSS_ACTOR: RatioStat(losses.loss_actor_sum.detach(), batch_count),
            RL_LOSS_CRITIC: RatioStat(losses.loss_critic_sum.detach(), batch_count),
            RL_LOSS_ENTROPY: RatioStat(losses.loss_entropy_sum.detach(), batch_count),
        },
    )


# =================================================================================================
class RLLossHead(BaseObjective[RLLossConfig]):
    """Pure reward-first RL objective scored over executed rollout steps.

    Does NOT inherit :class:`~ehc_sn.objectives._token.TokenLossHeadBase`.
    Does NOT require token labels, token logits, or task supervision targets.

    The main execution path is :meth:`forward` which processes a full rollout
    chunk. :meth:`evaluate_step` is a strict one-step TD(0) fallback for
    streaming / :class:`~ehc_sn.rollouts.SingleStepRunner` compatibility.
    """

    def __init__(
        self,
        config: RLLossConfig,
        *,
        binding: RLObjectiveBinding[Any],
    ) -> None:
        """Create a pure RL objective from its loss configuration and binding.

        Args:
            config: Pure RL loss configuration.
            binding: Binding that extracts :class:`RewardBearingStep` from controller outputs.
        """
        super().__init__(config=config)
        self._binding = binding

    @property
    def binding(self) -> RLObjectiveBinding[Any]:
        """Return the reward-bearing extraction binding."""
        return self._binding

    def forward(  # ------------------------------------------------------------------------------
        self, chunk: RolloutChunk, **options: Any,
    ) -> EvaluatedChunk:  # fmt: skip
        """Score an executed rollout chunk and return one step result per record."""
        observed_steps: list[ObservedStep] = []
        total_loss: Tensor | None = None

        for record in chunk.records:
            step_result = self.evaluate_step(record, **options)
            observed_steps.append(
                ObservedStep(
                    index=record.index,
                    batch=record.batch,
                    snapshot=record.snapshot,
                    outputs=step_result,
                )
            )
            total_loss = step_result.loss if total_loss is None else total_loss + step_result.loss

        if total_loss is None:
            raise ValueError("RLLossHead received an empty rollout chunk.")

        return EvaluatedChunk(
            steps=tuple(observed_steps),
            loss=total_loss,
            final_carry=chunk.final_carry,
            source_exhausted=chunk.source_exhausted,
        )

    def evaluate_step(  # ------------------------------------------------------------------------
        self, record: StepRecord, **options: Any,
    ) -> RLLossStep:  # fmt: skip
        """Score one executed rollout step (strict one-step TD(0)).

        This is the :class:`~ehc_sn.rollouts.SingleStepRunner` compatibility path.
        """
        is_warmup: bool = options.get("is_warmup", False)
        rb_step = self._binding.extract_reward_step(record.batch, record.carry, record.outputs)
        losses = self._compute_losses(rb_step, is_warmup=is_warmup)
        metrics = _build_rl_step_metrics(record.carry, losses)
        signals = self._compute_signals(record.carry, rb_step, losses)
        return RLLossStep(losses=losses, metrics=metrics, signals=signals, outputs=record.outputs)

    def _compute_losses(  # ----------------------------------------------------------------------
        self, rb: RewardBearingStep, *, is_warmup: bool = False,
    ) -> RLLosses:  # fmt: skip
        """Compute pure RL loss terms from a reward-bearing step."""
        value_estimate = rb.value_estimate
        reward = rb.reward
        advantage = (reward - value_estimate).detach()

        if not is_warmup:
            if rb.policy_decision.log_prob is None:
                raise ValueError("RLLossHead requires policy_decision.log_prob. " "Ensure the policy populates log_prob at rollout time.")
            logp = rb.policy_decision.log_prob
            entropy = rb.policy_decision.entropy if rb.policy_decision.entropy is not None else torch.zeros_like(logp)

            loss_actor = -(logp * advantage).sum()
            loss_critic = F.mse_loss(value_estimate, reward.detach(), reduction="sum")
            loss_entropy = -entropy.sum()
        else:
            zero = torch.tensor(0.0, device=reward.device)
            loss_actor = zero
            loss_critic = zero
            loss_entropy = zero

        return RLLosses(
            loss_actor_sum=self.config.c_actor * loss_actor,
            loss_critic_sum=self.config.c_critic * loss_critic,
            loss_entropy_sum=self.config.c_entropy * loss_entropy,
        )

    def _compute_signals(  # ---------------------------------------------------------------------
        self, carry: Any, rb: RewardBearingStep, losses: RLLosses,
    ) -> dict[str, Tensor]:  # fmt: skip
        """Compute lightweight diagnostic signals for the pure RL objective."""
        reward = rb.reward
        value_estimate = rb.value_estimate
        rpe = (reward - value_estimate).detach()
        return {
            S.REWARD_MEAN: reward.mean().detach(),
            S.REWARD_STD: reward.std().detach(),
            S.RPE_MAGNITUDE: rpe.abs().mean(),
            S.STEPS_MEAN: carry.steps.float().mean().detach(),
            S.LOSS_ACTOR: losses.loss_actor_sum.detach(),
            S.LOSS_CRITIC: losses.loss_critic_sum.detach(),
            S.LOSS_ENTROPY: losses.loss_entropy_sum.detach(),
        }


# =================================================================================================
__all__ = [
    "RewardBearingStep",
    "RLObjectiveBinding",
    "RLLossConfig",
    "RLLossHead",
    "RLLosses",
    "RLLossStep",
]
