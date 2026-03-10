"""RL loss head for HRM v2.

This module defines a loss head that wraps :class:`~ehc_sn.controllers.rl.RLController`
to produce a training/evaluation step with:
    - supervised token modeling loss (cross-entropy over maze tokens)
    - actor-critic losses computed from environment rewards
    - auxiliary vmPFC (Q-value) regression loss

The head reads controller outputs via *named properties* (``outputs.lm_logits``,
``outputs.q_logits``, ``outputs.value_logits``) — never via positional
``outputs.logits[N]`` index.

The head returns an :class:`RLLossStep` containing the live loss tensors (for
backprop), aggregated metrics, and diagnostic signals.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor
from torch.distributions import Categorical

import ehc_sn.loss.cross_entropy as cross_entropy_module
from ehc_sn.controllers.rl import RLController, RLOutput, RLState
from ehc_sn.heads._base import AccuracyStats, TokenLossHeadBase
from ehc_sn.loss.cross_entropy import LossType
from ehc_sn.metrics import signals as S
from ehc_sn.metrics.keys import LOSS_LM, RL_LOSS_ACTOR, RL_LOSS_CRITIC, RL_LOSS_ENTROPY, RL_LOSS_Q_VALUE
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
    c_actor: float = Field(default=1.0, ge=0.0, description="Actor loss coefficient.")
    c_critic: float = Field(default=0.5, ge=0.0, description="Critic loss coefficient.")
    c_entropy: float = Field(default=0.01, ge=0.0, description="Entropy regularization coefficient.")
    c_q_value: float = Field(default=0.5, ge=0.0, description="vmPFC auxiliary Q-predictor loss coefficient.")


# =================================================================================================
@dataclass(frozen=True)
class Losses(DetachMixin):
    """Bundle of per-step loss terms (summed over batch).

    All fields are *sums* (not means). Normalization (e.g. by local batch size)
    is performed by the Lightning module before backward.
    """

    loss_lm_sum: Tensor
    loss_q_value_sum: Tensor

    loss_actor_sum: Tensor
    loss_critic_sum: Tensor
    loss_entropy_sum: Tensor

    @property
    def total(self) -> Tensor:
        """Total scalar loss for the step."""
        loss_rl = self.loss_actor_sum + self.loss_critic_sum + self.loss_entropy_sum
        loss_m = self.loss_lm_sum + self.loss_q_value_sum
        return loss_rl + loss_m


# =================================================================================================
@dataclass(frozen=True)
class RLLossStep:
    """A single rollout/loss step produced by :class:`RLLossHead`.

    Attributes:
        losses: Live loss tensors (used for backward).
        metrics: Aggregated metrics detached for logging.
        outputs: Optional raw controller outputs for tracing.
        signals: Lightweight diagnostic signals.
    """

    losses: Losses  # Combined losses for this step, kept live for backward()
    metrics: StepMetrics  # Aggregated metrics for this step, used for logging
    outputs: Optional[RLOutput] = None  # Raw controller outputs
    signals: Dict[str, Tensor] = None  # Diagnostic signals (T2/T3); plain dict, no schema commitment

    def __post_init__(self) -> None:
        if self.signals is None:
            object.__setattr__(self, "signals", {})

    @property
    def loss(self) -> Tensor:
        """Scalar loss for this step, used for back-propagation."""
        return self.losses.total


# =================================================================================================
class RLLossHead(TokenLossHeadBase[RLController, RLLossConfig]):
    """Loss head wrapping :class:`~ehc_sn.controllers.rl.RLController`.

    The loss head is responsible for:
        - running one controller step
        - computing supervised + RL losses
        - producing step metrics and diagnostic signals
    """

    def __init__(  # ------------------------------------------------------------------------------
        self, controller: RLController, config: RLLossConfig,
    ) -> None:  # fmt: skip
        """Create a loss head.

        Args:
            controller: Controller responsible for forward pass + env stepping.
            config: Loss configuration.
        """
        super().__init__(controller=controller, config=config)

    @property
    def loss_fn(self) -> Any:
        """Return the configured token-level loss function."""
        return getattr(cross_entropy_module, self._config.function)

    def forward(  # -------------------------------------------------------------------------------
        self, batch: Batch, carry: RLState, *, is_warmup: bool = False, **options: Any,
    ) -> Tuple[RLLossStep, RLState, bool]:  # fmt: skip
        """Run one controller step and compute losses/metrics.

        Args:
            batch: Incoming batch used both for the step inputs and as a source
                of fresh samples for partial resets.
            carry: Current rollout state.
            is_warmup: If True, RL losses are suppressed (supervised loss only).
            **options: Forwarded to :meth:`RLController.step` (e.g. exploration).

        Returns:
            ``(step, new_carry, all_halted)`` where ``all_halted`` indicates that
            all slots are done for the current carry.
        """
        carry, outputs = self.controller.step(carry, batch, **options)
        step_output = self._run_token_step(batch, carry, outputs, is_warmup=is_warmup)
        return step_output, carry, bool(carry.halted.all())

    def compute_losses(  # ------------------------------------------------------------------------
        self, outputs: RLOutput, labels: Tensor, stats: AccuracyStats, *,
        is_warmup: bool = False, **_: Any,
    ) -> Losses:  # fmt: skip
        """Compute supervised and RL loss terms for a step.

        All returned losses are summed over the batch.
        """
        # --- Supervised LM loss (every step, as in ACT) ---
        loss_lm_sum = self.compute_lm_loss(outputs.lm_logits, labels, stats)

        # --- Reinforcement learning losses (vmPFC critic + actor) ---
        dist = Categorical(logits=outputs.q_logits)
        logp = dist.log_prob(outputs.action)  # (B,)
        entropy = dist.entropy()  # (B,)
        advantage = (outputs.reward.squeeze(-1) - outputs.value_logits.squeeze(-1)).detach()  # (B,)
        q_a = outputs.q_logits.gather(1, outputs.action.unsqueeze(-1)).squeeze(-1)  # (B,)

        if not is_warmup:  # Compute RL losses only after warmup phase
            loss_actor = -(logp * advantage).sum()
            loss_critic = F.mse_loss(outputs.value_logits, outputs.reward, reduction="sum")
            loss_entropy = -entropy.sum()  # negative so minimizing loss maximizes entropy
            loss_q_value = F.mse_loss(q_a, outputs.reward.squeeze(-1).detach(), reduction="sum")
        else:
            loss_actor = torch.tensor(0.0, device=outputs.lm_logits.device)
            loss_critic = torch.tensor(0.0, device=outputs.lm_logits.device)
            loss_entropy = torch.tensor(0.0, device=outputs.lm_logits.device)
            loss_q_value = torch.tensor(0.0, device=outputs.lm_logits.device)

        # --- Combine losses with coefficients from config ---
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
        self, losses: Losses, metrics: Any, signals: Dict[str, Tensor], outputs: RLOutput,
    ) -> RLLossStep:  # fmt: skip
        """Wrap losses, metrics, and signals into an :class:`RLLossStep`."""
        return RLLossStep(losses=losses, metrics=metrics, outputs=outputs, signals=signals)

    def compute_signals(  # -----------------------------------------------------------------------
        self, state: RLState, outputs: RLOutput, losses: Losses,
    ) -> Dict[str, Tensor]:  # fmt: skip
        """Compute lightweight diagnostic signals.

        Signals are intended for TensorBoard-style scalar logging.
        """
        reward = outputs.reward.squeeze(-1)  # (B,)
        rpe = (reward - outputs.value_logits.squeeze(-1)).detach()  # (B,) reward prediction error
        dist = Categorical(logits=outputs.q_logits.detach())
        return {
            S.REWARD_MEAN:    reward.mean().detach(),
            S.REWARD_STD:     reward.std().detach(),
            S.Q_MEAN:         outputs.q_logits.detach().mean(),
            S.Q_STD:          outputs.q_logits.detach().std(),
            S.RPE_MAGNITUDE:  rpe.abs().mean(),
            S.ACTION_ENTROPY: dist.entropy().mean(),
            S.STEPS_MEAN:     state.steps.float().mean().detach(),
            S.LOSS_ACTOR:     losses.loss_actor_sum.detach(),
            S.LOSS_CRITIC:    losses.loss_critic_sum.detach(),
            S.LOSS_ENTROPY:   losses.loss_entropy_sum.detach(),
        }  # fmt: skip


# =================================================================================================
__all__ = ["RLLossConfig", "RLLossHead", "RLLossStep"]
