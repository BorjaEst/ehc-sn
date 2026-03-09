"""RL loss head for HRM v2.

This module defines a loss head that wraps :class:`~ehc_sn.training.rl_controller.RLController`
to produce a training/evaluation step with:
    - supervised token modeling loss (cross-entropy over maze tokens)
    - actor-critic losses computed from environment rewards
    - auxiliary vmPFC (Q-value) regression loss

The head returns an :class:`RLLossStep` containing the live loss tensors (for
backprop), aggregated metrics, and diagnostic signals.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor, nn
from torch.distributions import Categorical

import ehc_sn.loss.cross_entropy as cross_entropy_module
from ehc_sn.loss.cross_entropy import LossType
from ehc_sn.metrics import signals as S
from ehc_sn.training.rl_controller import RLController, RLOutput, RLState
from ehc_sn.training.types import HaltedAgg, LossAgg, StepMetrics, TokenAgg
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin

# Token label ignored by supervised loss (padding / non-supervised positions).
IGNORE_LABEL_ID: int = -100


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
class AccuracyStats:
    """Token-level correctness statistics used for metrics.

    This is computed out-of-graph (no gradients through argmax) and used for
    logging and some halted-only aggregations.
    """

    mask: Tensor  # Boolean tensor indicating which tokens contribute to the loss (e.g., non-padding tokens).

    @property
    def loss_counts(self) -> Tensor:
        """Number of eligible tokens per sequence (shape ``(B,)``)."""
        return self.mask.sum(-1)

    @property
    def loss_divisor(self) -> Tensor:
        """Safe divisor for per-sequence averages (shape ``(B, 1)``)."""
        return self.loss_counts.clamp_min(1).unsqueeze(-1)

    is_correct: Tensor  # Indicates which tokens were predicted correctly (after masking).

    @property
    def seq_is_correct(self) -> Tensor:
        """Whether all eligible tokens were predicted correctly (shape ``(B,)``)."""
        return self.is_correct.sum(-1) == self.loss_counts


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
class RLLossHead(nn.Module):
    """Loss head wrapping :class:`~ehc_sn.training.rl_controller.RLController`.

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
        super().__init__()
        self._controller = controller
        self._config = config

    @property
    def controller(self) -> RLController:
        """Return the wrapped controller."""
        return self._controller

    @property
    def config(self) -> RLLossConfig:
        """Return the loss configuration."""
        return self._config

    @property
    def loss_fn(self) -> Any:
        """Return the configured token-level loss function implementation."""
        return getattr(cross_entropy_module, self._config.function)

    def initial_carry(  # -------------------------------------------------------------------------
        self, batch_sample: Batch,
    ) -> RLState:  # fmt: skip
        """Initialize the rollout carry/state from an example batch."""
        return self._controller.initial_state(batch_sample)

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
        carry, outputs = self._controller.step(carry, batch, **options)
        labels = carry.data["labels"]

        with torch.no_grad():
            # Correctness is used as a supervision signal for halting/continuation.
            # Keeping it out of the graph avoids gradients flowing through argmax.
            stats = self.compute_accuracy(outputs, labels)

        losses = self.compute_losses(outputs, labels, stats, is_warmup=is_warmup)
        metrics = self.compute_metrics(carry, outputs, stats, losses)
        signals = self.compute_signals(carry, outputs, losses)

        outputs = RLLossStep(losses=losses, metrics=metrics, outputs=outputs, signals=signals)
        return outputs, carry, bool(carry.halted.all())

    def compute_accuracy(  # ------------------------------------------------------------------
        self, outputs: RLOutput, labels: Tensor,
    ) -> AccuracyStats:  # fmt: skip
        """Compute masked token correctness statistics (out-of-graph)."""
        logits_lm, *_ = outputs.logits  # Unpack list of logits if backbone returns multiple heads
        mask = labels != IGNORE_LABEL_ID
        is_correct = mask & (torch.argmax(logits_lm, dim=-1) == labels)
        return AccuracyStats(mask=mask, is_correct=is_correct)

    def compute_losses(  # ------------------------------------------------------------------------
        self, outputs: RLOutput, labels: Tensor, stats: AccuracyStats, *,
        is_warmup: bool = False,
    ) -> Losses:  # fmt: skip
        """Compute supervised and RL loss terms for a step.

        All returned losses are summed over the batch.
        """
        logits_lm, logits_q, logits_r, *_ = outputs.logits  # Unpack list of logits multiple heads

        # --- Supervised LM loss (every step, as in ACT) ---
        loss_per_token = self.loss_fn(logits_lm, labels, ignore_index=IGNORE_LABEL_ID)  # (B,S)
        loss_per_seq = loss_per_token.sum(-1) / stats.loss_counts.clamp_min(1)  # (B,)
        loss_lm_sum = loss_per_seq.sum()

        # --- Reinforcement learning losses (vmPFC critic + actor) ---
        dist = Categorical(logits=logits_q)
        logp = dist.log_prob(outputs.action)  # (B,)
        entropy = dist.entropy()  # (B,)
        advantage = (outputs.reward.squeeze(-1) - logits_r.squeeze(-1)).detach()  # (B,)
        q_a = logits_q.gather(1, outputs.action.unsqueeze(-1)).squeeze(-1)  # (B,)

        if not is_warmup:  # Compute RL losses only after warmup phase
            loss_actor = -(logp * advantage).sum()
            loss_critic = F.mse_loss(logits_r, outputs.reward.squeeze(-1), reduction="sum")
            loss_entropy = -entropy.sum()  # negative so minimizing loss maximizes entropy
            loss_q_value = F.mse_loss(q_a, outputs.reward.squeeze(-1).detach(), reduction="sum")
        else:
            loss_actor = torch.tensor(0.0, device=logits_lm.device)
            loss_critic = torch.tensor(0.0, device=logits_lm.device)
            loss_entropy = torch.tensor(0.0, device=logits_lm.device)
            loss_q_value = torch.tensor(0.0, device=logits_lm.device)

        # --- Combine losses with coefficients from config ---
        return Losses(
            loss_lm_sum=loss_lm_sum,
            loss_actor_sum=self.config.c_actor * loss_actor,
            loss_critic_sum=self.config.c_critic * loss_critic,
            loss_entropy_sum=self.config.c_entropy * loss_entropy,
            loss_q_value_sum=self.config.c_q_value * loss_q_value,
        )

    def compute_metrics(  # -----------------------------------------------------------------------
        self, state: RLState, outputs: RLOutput, stats: AccuracyStats, losses: Losses,
    ) -> StepMetrics:  # fmt: skip
        """Aggregate per-step metrics for logging.

        Metrics are mostly reported over the subset of sequences that halted on
        this step (to match the "halted-only" reporting style used elsewhere).
        """
        eligible_mask = stats.loss_counts > 0
        halted_mask = state.halted & eligible_mask
        halted_weights = halted_mask.to(torch.float32)

        token_correct_per_seq = stats.is_correct.to(torch.float32).sum(-1)
        token_count_per_seq = stats.loss_counts.clamp_min(1).to(torch.float32)
        seq_accuracy = token_correct_per_seq / token_count_per_seq

        eligible_count = eligible_mask.to(torch.float32).sum()

        halted = HaltedAgg(
            halted_count=halted_weights.sum(),
            eligible_count=eligible_count,
            accuracy_sum=(seq_accuracy * halted_weights).sum(),
            exact_sum=(stats.seq_is_correct & halted_mask).to(torch.float32).sum(),
            steps_sum=(state.steps * halted_weights.to(state.steps.dtype)).sum(),
            q_halt_correct_sum=halted_weights.new_zeros(()),
            q_continue_correct_sum=halted_weights.new_zeros(()),
        )

        tokens = TokenAgg(
            token_correct_sum=(token_correct_per_seq * halted_weights).sum(),
            token_count_sum=(token_count_per_seq * halted_weights).sum(),
        )

        batch_size = int(outputs.logits[0].shape[0])  # logits[0] is features logits (B, S, V)
        q_halt_loss_sum=losses.loss_actor_sum + losses.loss_critic_sum + losses.loss_entropy_sum + losses.loss_q_value_sum  # fmt: skip
        loss = LossAgg(
            lm_loss_sum=losses.loss_lm_sum.detach(),
            q_halt_loss_sum=q_halt_loss_sum.detach(),
            q_continue_loss_sum=losses.loss_lm_sum.new_zeros(()),
            batch_count=losses.loss_lm_sum.new_tensor(batch_size, dtype=torch.float32),
            actor_loss_sum=losses.loss_actor_sum.detach(),
            critic_loss_sum=losses.loss_critic_sum.detach(),
            entropy_loss_sum=losses.loss_entropy_sum.detach(),
            q_value_loss_sum=losses.loss_q_value_sum.detach(),
        )

        return StepMetrics(halted=halted, tokens=tokens, loss=loss)

    def compute_signals(  # -----------------------------------------------------------------------
        self, state: RLState, outputs: RLOutput, losses: Losses,
    ) -> Dict[str, Tensor]:  # fmt: skip
        """Compute lightweight diagnostic signals.

        Signals are intended for TensorBoard-style scalar logging.
        """
        _feature, q_logits, logits_r = outputs.logits  # (B,S,V), (B,A), (B,1)
        reward = outputs.reward.squeeze(-1)  # (B,)
        rpe = (reward - logits_r.squeeze(-1)).detach()  # (B,) reward prediction error
        dist = Categorical(logits=q_logits.detach())
        return {
            S.REWARD_MEAN:    reward.mean().detach(),
            S.REWARD_STD:     reward.std().detach(),
            S.Q_MEAN:         q_logits.detach().mean(),
            S.Q_STD:          q_logits.detach().std(),
            S.RPE_MAGNITUDE:  rpe.abs().mean(),
            S.ACTION_ENTROPY: dist.entropy().mean(),
            S.STEPS_MEAN:     state.steps.float().mean().detach(),
            S.LOSS_ACTOR:     losses.loss_actor_sum.detach(),
            S.LOSS_CRITIC:    losses.loss_critic_sum.detach(),
            S.LOSS_ENTROPY:   losses.loss_entropy_sum.detach(),
        }  # fmt: skip


# =================================================================================================
__all__ = ["RLLossConfig", "RLLossHead", "RLLossStep"]
