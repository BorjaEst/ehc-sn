""" """

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
from ehc_sn.metrics import HaltedAgg, LossAgg, StepMetrics, TokenAgg
from ehc_sn.training.rl_controller import RLController, RLOutput, RLState
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin

# Token label ignored by supervised loss (padding / non-supervised positions).
IGNORE_LABEL_ID: int = -100


# =================================================================================================
class RLLossConfig(BaseModel, extra="forbid"):
    """ """

    function: LossType = Field(
        default="stablemax_cross_entropy",
        description="The loss function to use for the modeling loss.",
    )
    c_actor: float = Field(default=1.0, ge=0.0, description="Actor loss coefficient.")
    c_critic: float = Field(default=0.5, ge=0.0, description="Critic loss coefficient.")
    c_entropy: float = Field(default=0.01, ge=0.0, description="Entropy regularization coefficient.")
    c_vmPFC: float = Field(default=0.5, ge=0.0, description="vmPFC auxiliary Q-predictor loss coefficient.")


# =================================================================================================
@dataclass(frozen=True)
class AccuracyStats:
    """ """

    mask: Tensor  # Boolean tensor indicating which tokens contribute to the loss (e.g., non-padding tokens).

    @property
    def loss_counts(self) -> Tensor:
        """ """
        return self.mask.sum(-1)

    @property
    def loss_divisor(self) -> Tensor:
        """ """
        return self.loss_counts.clamp_min(1).unsqueeze(-1)

    is_correct: Tensor  # Indicates which tokens were predicted correctly (after masking).

    @property
    def seq_is_correct(self) -> Tensor:
        """ """
        return self.is_correct.sum(-1) == self.loss_counts


# =================================================================================================
@dataclass(frozen=True)
class Losses(DetachMixin):
    """ """

    loss_lm_sum: Tensor
    loss_vmPFC_sum: Tensor

    loss_actor_sum: Tensor
    loss_critic_sum: Tensor
    loss_entropy_sum: Tensor

    @property
    def total(self) -> Tensor:
        """ """
        loss_rl = self.loss_actor_sum + self.loss_critic_sum + self.loss_entropy_sum
        loss_m = self.loss_lm_sum + self.loss_vmPFC_sum
        return loss_rl + loss_m


# =================================================================================================
@dataclass(frozen=True)
class RLLossStep:
    """ """

    losses: Losses  # Combined losses for this step, kept live for backward()
    metrics: StepMetrics  # Aggregated metrics for this step, used for logging
    outputs: Optional[RLOutput] = None  # Raw controller outputs

    @property
    def loss(self) -> Tensor:
        """Scalar loss for this step, used for back-propagation."""
        return self.losses.total


# =================================================================================================
class RLLossHead(nn.Module):
    """ """

    def __init__(  # ------------------------------------------------------------------------------
        self, controller: RLController, config: RLLossConfig,
    ) -> None:  # fmt: skip
        """ """
        super().__init__()
        self._controller = controller
        self._config = config

    @property
    def controller(self) -> RLController:
        """ """
        return self._controller

    @property
    def config(self) -> RLLossConfig:
        """ """
        return self._config

    @property
    def loss_fn(self) -> Any:
        """ """
        return getattr(cross_entropy_module, self._config.function)

    def initial_carry(  # -------------------------------------------------------------------------
        self, batch_sample: Batch,
    ) -> RLState:  # fmt: skip
        """ """
        return self._controller.initial_state(batch_sample)

    def forward(  # -------------------------------------------------------------------------------
        self, batch: Batch, carry: RLState, *, is_warmup: bool = False, **options: Any,
    ) -> Tuple[RLLossStep, RLState, bool]:  # fmt: skip
        """ """
        carry, outputs = self._controller.step(carry, batch, **options)
        labels = carry.data["labels"]

        with torch.no_grad():
            # Correctness is used as a supervision signal for halting/continuation.
            # Keeping it out of the graph avoids gradients flowing through argmax.
            stats = self.compute_accuracy(outputs, labels)

        losses = self.compute_losses(outputs, labels, stats)
        metrics = self.compute_metrics(carry, outputs, stats, losses)

        outputs = RLLossStep(losses=losses, metrics=metrics, outputs=outputs)
        return outputs, carry, bool(carry.halted.all())

    def compute_accuracy(  # ------------------------------------------------------------------
        self, outputs: RLOutput, labels: Tensor,
    ) -> AccuracyStats:  # fmt: skip
        """ """
        logits_lm, *_ = outputs.logits  # Unpack list of logits if backbone returns multiple heads
        mask = labels != IGNORE_LABEL_ID
        is_correct = mask & (torch.argmax(logits_lm, dim=-1) == labels)
        return AccuracyStats(mask=mask, is_correct=is_correct)

    def compute_losses(  # ------------------------------------------------------------------------
        self, outputs: RLOutput, labels: Tensor, stats: AccuracyStats, *,
        is_warmup: bool = False,
    ) -> Losses:  # fmt: skip
        """ """
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
            loss_critic = F.mse_loss(logits_r, outputs.reward, reduction="sum")
            loss_entropy = -entropy.sum()  # negative so minimizing loss maximizes entropy
            loss_vmPFC = F.mse_loss(q_a, outputs.reward.squeeze(-1).detach(), reduction="sum")
        else:
            loss_actor = torch.tensor(0.0, device=logits_lm.device)
            loss_critic = torch.tensor(0.0, device=logits_lm.device)
            loss_entropy = torch.tensor(0.0, device=logits_lm.device)
            loss_vmPFC = torch.tensor(0.0, device=logits_lm.device)

        # --- Combine losses with coefficients from config ---
        return Losses(
            loss_lm_sum=loss_lm_sum,
            loss_actor_sum=self.config.c_actor * loss_actor,
            loss_critic_sum=self.config.c_critic * loss_critic,
            loss_entropy_sum=self.config.c_entropy * loss_entropy,
            loss_vmPFC_sum=self.config.c_vmPFC * loss_vmPFC,
        )

    def compute_metrics(  # -----------------------------------------------------------------------
        self, state: RLState, outputs: RLOutput, stats: AccuracyStats, losses: Losses,
    ) -> StepMetrics:  # fmt: skip
        """ """
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
        loss = LossAgg(
            lm_loss_sum=losses.loss_lm_sum.detach(),
            q_halt_loss_sum=(
                losses.loss_actor_sum
                + losses.loss_critic_sum
                + losses.loss_entropy_sum
                + losses.loss_vmPFC_sum
            ).detach(),
            q_continue_loss_sum=losses.loss_lm_sum.new_zeros(()),
            batch_count=losses.loss_lm_sum.new_tensor(batch_size, dtype=torch.float32),
        )

        return StepMetrics(halted=halted, tokens=tokens, loss=loss)


# =================================================================================================
__all__ = ["RLLossConfig", "RLLossHead", "RLLossStep"]
