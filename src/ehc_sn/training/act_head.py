""" """

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor, nn

import ehc_sn.loss.cross_entropy as cross_entropy_module
from ehc_sn.loss.cross_entropy import LossType
from ehc_sn.metrics import HaltedAgg, LossAgg, StepMetrics, TokenAgg
from ehc_sn.training.act_controller import ACTController, ACTOutput, ACTState
from ehc_sn.utils.detach import DetachMixin

Batch = Dict[str, Tensor]  # Generic batch type, can be specialized as needed
IGNORE_LABEL_ID = -100


# =================================================================================================
class ACTLossConfig(BaseModel, extra="forbid"):
    """ """

    function: LossType = Field(
        default="stablemax_cross_entropy",
        description="The loss function to use for the modeling loss.",
    )


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

    loss_sum: Tensor  # Per-step loss sum for the main task
    q_halt_loss_sum: Tensor  # Loss sum for the halting decision
    q_continue_loss_sum: Optional[Tensor]  # Loss sum for the continue decision (if applicable)

    @property
    def total(self) -> Tensor:
        """ """
        q_continue_loss_sum = self.q_continue_loss_sum
        if q_continue_loss_sum is None:
            q_continue_loss_sum = torch.tensor(0.0, device=self.loss_sum.device)
        return self.loss_sum + 0.5 * (self.q_halt_loss_sum + q_continue_loss_sum)


# =================================================================================================
@dataclass(frozen=True)
class ACTLossStep:
    """ """

    losses: Losses  # Combined losses for this step, kept live for backward()
    metrics: StepMetrics  # Aggregated metrics for this step, used for logging
    outputs: Optional[ACTOutput] = None  # Raw controller outputs
    signals: Dict[str, Any] = None  # Diagnostic signals (T2/T3); plain dict, no schema commitment

    def __post_init__(self) -> None:
        if self.signals is None:
            object.__setattr__(self, "signals", {})

    @property
    def loss(self) -> Tensor:
        """Scalar loss for this step, used for back-propagation."""
        return self.losses.total


# =================================================================================================
class ACTLossHead(nn.Module):
    """ """

    def __init__(  # ------------------------------------------------------------------------------
        self, controller: ACTController, config: ACTLossConfig,
    ) -> None:  # fmt: skip
        """ """
        super().__init__()
        self._controller = controller
        self._config = config

    @property
    def controller(self) -> ACTController:
        """ """
        return self._controller

    @property
    def config(self) -> ACTLossConfig:
        """ """
        return self._config

    @property
    def loss_fn(self) -> Any:
        """ """
        return getattr(cross_entropy_module, self._config.function)

    def initial_carry(  # -------------------------------------------------------------------------
        self, batch_sample: Batch, 
    ) -> ACTState:  # fmt: skip
        """ """
        return self.controller.initial_state(batch_sample)

    def forward(  # -------------------------------------------------------------------------------
        self, batch: Batch, carry: ACTState, **options: Any,
    ) -> Tuple[ACTLossStep, ACTState, bool]:  # fmt: skip
        """ """
        carry, outputs = self.controller.step(carry, batch, **options)
        labels = carry.data["labels"]

        with torch.no_grad():
            # Correctness is used as a supervision signal for halting/continuation.
            # Keeping it out of the graph avoids gradients flowing through argmax.
            stats = self.compute_accuracy(outputs, labels)

        losses = self.compute_losses(outputs, labels, stats)
        metrics = self.compute_metrics(carry, outputs, stats, losses)
        signals = self.compute_signals(carry, outputs, losses)

        outputs = ACTLossStep(losses=losses, metrics=metrics, outputs=outputs, signals=signals)
        return outputs, carry, bool(carry.halted.all())

    def compute_accuracy(  # ------------------------------------------------------------------
        self, outputs: ACTOutput, labels: Tensor
    ) -> AccuracyStats:  # fmt: skip
        """ """
        logits_lm, *_ = outputs.logits  # Unpack list of logits if backbone returns multiple heads
        mask = labels != IGNORE_LABEL_ID
        is_correct = mask & (torch.argmax(logits_lm, dim=-1) == labels)
        return AccuracyStats(mask=mask, is_correct=is_correct)

    def compute_losses(  # -----------------------------------------------------------------------
        self, outputs: ACTOutput, labels: Tensor, stats: AccuracyStats
    ) -> Losses:  # fmt: skip
        """ """
        loss_per_token = self.loss_fn(outputs.logits, labels, ignore_index=IGNORE_LABEL_ID)
        loss_per_seq = loss_per_token.sum(-1) / stats.loss_counts.clamp_min(1)
        loss_sum = loss_per_seq.sum()
        done_action = self.controller.config.done_action

        # Done-action loss: match Q(done) to sequence correctness.
        q_done_logits = outputs.q_values[..., done_action]  # (B,)
        q_done_loss = F.binary_cross_entropy_with_logits(
            input=q_done_logits,
            target=stats.seq_is_correct.to(q_done_logits.dtype),
            reduction="sum",
        )

        # Continue loss: optional auxiliary supervision from the controller's TD target.
        q_continue_loss: Tensor | None = None
        if outputs.target_q is not None:
            # Select the non-done logit(s). For 2-action, pick ~done_action.
            # Generalization: supervise all non-done actions toward TD target.
            n_actions = outputs.q_values.shape[-1]
            continue_actions = [a for a in range(n_actions) if a != done_action]
            if continue_actions:
                q_cont = outputs.q_values[..., continue_actions].mean(dim=-1)  # (B,)
                q_continue_loss = F.binary_cross_entropy_with_logits(
                    input=q_cont,
                    target=outputs.target_q,
                    reduction="sum",
                )

        return Losses(loss_sum, q_done_loss, q_continue_loss)

    def compute_metrics(  # -----------------------------------------------------------------------
        self, state: ACTState, outputs: ACTOutput, stats: AccuracyStats, losses: Losses,
    ) -> StepMetrics:  # fmt: skip
        """ """
        eligible_mask = stats.loss_counts > 0
        halted_mask = state.halted & eligible_mask  # (B,)
        halted_weights = halted_mask.to(torch.float32)

        token_correct_per_seq = stats.is_correct.to(torch.float32).sum(-1)  # (B,)
        token_count_per_seq = stats.loss_counts.clamp_min(1).to(torch.float32)  # (B,)
        seq_accuracy = token_correct_per_seq / token_count_per_seq  # (B,)

        done_action = self.controller.config.done_action
        pred_done = outputs.action == done_action  # (B,)
        q_done_correct = pred_done == stats.seq_is_correct  # (B,)

        q_continue_correct: Optional[Tensor] = None
        if outputs.target_q is not None:
            n_actions = outputs.q_values.shape[-1]
            continue_actions = [a for a in range(n_actions) if a != done_action]
            if continue_actions:
                q_cont = outputs.q_values[..., continue_actions].mean(dim=-1)
                pred_continue = q_cont >= 0
                q_continue_correct = pred_continue == stats.seq_is_correct

        eligible_count = eligible_mask.to(torch.float32).sum()
        halted = self._build_halted_agg(state, stats, halted_mask, halted_weights, eligible_count, seq_accuracy, q_done_correct, q_continue_correct)  # fmt: skip
        tokens = self._build_token_agg(token_correct_per_seq, token_count_per_seq, halted_weights)
        loss = self._build_loss_agg(losses, batch_size=outputs.logits.shape[0])

        return StepMetrics(halted=halted, tokens=tokens, loss=loss)

    def _build_halted_agg(  # --------------------------------------------------------------------
        self, state: ACTState, stats: AccuracyStats, halted_mask: Tensor, halted_weights: Tensor,
        eligible_count: Tensor, seq_accuracy: Tensor, q_halt_correct: Tensor,
        q_continue_correct: Optional[Tensor]=None,
    ) -> HaltedAgg:  # fmt: skip
        """ """
        if q_continue_correct is None:
            q_continue_correct = torch.zeros_like(halted_mask)  # (B,) bool

        return HaltedAgg(
            halted_count=halted_weights.sum(),
            eligible_count=eligible_count,
            accuracy_sum=(seq_accuracy * halted_weights).sum(),
            exact_sum=(stats.seq_is_correct & halted_mask).to(torch.float32).sum(),
            steps_sum=(state.steps * halted_weights.to(state.steps.dtype)).sum(),
            q_halt_correct_sum=(q_halt_correct & halted_mask).to(torch.float32).sum(),
            q_continue_correct_sum=(q_continue_correct & halted_mask).to(torch.float32).sum(),
        )

    def _build_token_agg(  # ---------------------------------------------------------------------
        self, token_correct_per_seq: Tensor, token_count_per_seq: Tensor, halted_weights: Tensor,
    ) -> TokenAgg:  # fmt: skip
        """ """
        return TokenAgg(
            token_correct_sum=(token_correct_per_seq * halted_weights).sum(),
            token_count_sum=(token_count_per_seq * halted_weights).sum(),
        )

    def _build_loss_agg(  # ----------------------------------------------------------------------
        self, losses: Losses, *, batch_size: int,
    ) -> LossAgg:  # fmt: skip
        """ """
        q_continue_loss_sum = losses.q_continue_loss_sum
        if q_continue_loss_sum is None:
            q_continue_loss_sum = losses.loss_sum.new_zeros(())

        return LossAgg(
            lm_loss_sum=losses.loss_sum.detach(),
            q_halt_loss_sum=losses.q_halt_loss_sum.detach(),
            q_continue_loss_sum=q_continue_loss_sum.detach(),
            batch_count=losses.loss_sum.new_tensor(batch_size, dtype=torch.float32),
            actor_loss_sum=torch.tensor(0.0, device=losses.loss_sum.device),
            critic_loss_sum=torch.tensor(0.0, device=losses.loss_sum.device),
            entropy_loss_sum=torch.tensor(0.0, device=losses.loss_sum.device),
            q_value_loss_sum=torch.tensor(0.0, device=losses.loss_sum.device),
        )

    def compute_signals(  # -----------------------------------------------------------------------
        self, state: ACTState, outputs: ACTOutput, losses: Losses,
    ) -> Dict[str, Tensor]:  # fmt: skip
        """ """
        signals: Dict[str, Tensor] = {
            "steps_mean": state.steps.float().mean().detach(),
            "theta_cls_norm": outputs.theta_cls.detach().norm(dim=-1).mean(),
            "loss_q_halt": losses.q_halt_loss_sum.detach(),
        }
        if outputs.target_q is not None:
            signals["target_q_mean"] = outputs.target_q.mean().detach()
            signals["target_q_std"] = outputs.target_q.std().detach()
        return signals
