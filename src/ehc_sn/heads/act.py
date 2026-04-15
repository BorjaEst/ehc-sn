"""ACT loss head for HRM v1.

This module defines :class:`ACTLossHead`, which wraps an
:class:`~ehc_sn.controllers.act.ACTController` and computes:
    - token-level supervised modeling loss (cross-entropy)
    - Q(done) binary supervision from sequence correctness
    - optional auxiliary Q(continue) supervision from TD bootstrap targets

The head reads controller outputs via *named properties* (``outputs.lm_logits``,
``outputs.q_logits``, etc.) — never via positional ``outputs.logits[N]`` index.

The head returns an :class:`ACTLossStep` with live loss tensors, aggregated
metrics, and diagnostic signals.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.controllers.act import ACTOutput, ACTRolloutState
from ehc_sn.heads._token import AccuracyStats, TokenLossHeadBase
from ehc_sn.loss.cross_entropy import LossType
from ehc_sn.metrics import signals as S
from ehc_sn.metrics.keys import ACT_LOSS_Q_CONTINUE, ACT_LOSS_Q_DONE, LOSS_LM
from ehc_sn.training.types import RatioStat, StepMetrics
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class ACTLossConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`ACTLossHead`.

    Attributes:
        function: Name of the token-level supervised loss function.
    """

    function: LossType = Field(
        default="stablemax_cross_entropy",
        description="The loss function to use for the modeling loss.",
    )


# =================================================================================================
@dataclass(frozen=True)
class Losses(DetachMixin):
    """Bundle of ACT loss terms (summed over batch)."""

    loss_sum: Tensor  # Per-step loss sum for the main task
    loss_q_done_sum: Tensor  # Loss sum for the done-action supervision
    loss_q_continue_sum: Optional[Tensor]  # Loss sum for the continue-action supervision (if applicable)

    @property
    def total(self) -> Tensor:
        """Total scalar loss for the step."""
        q_continue_loss_sum = self.loss_q_continue_sum
        if q_continue_loss_sum is None:
            q_continue_loss_sum = torch.tensor(0.0, device=self.loss_sum.device)
        return self.loss_sum + 0.5 * (self.loss_q_done_sum + q_continue_loss_sum)


# =================================================================================================
@dataclass(frozen=True)
class ACTLossStep:
    """A single rollout/loss step produced by :class:`ACTLossHead`."""

    losses: Losses  # Combined losses for this step, kept live for backward()
    metrics: StepMetrics  # Aggregated metrics for this step, used for logging
    outputs: Optional[ACTOutput] = None  # Raw controller outputs
    signals: dict[str, Any] | None = None  # Diagnostic signals (T2/T3); plain dict

    def __post_init__(self) -> None:
        if self.signals is None:
            object.__setattr__(self, "signals", {})

    @property
    def loss(self) -> Tensor:
        """Scalar loss for this step, used for back-propagation."""
        return self.losses.total


# =================================================================================================
class ACTLossHead(TokenLossHeadBase[ACTLossConfig]):
    """Pure ACT objective scored over executed rollout chunks."""

    def __init__(  # ------------------------------------------------------------------------------
        self, config: ACTLossConfig,
    ) -> None:  # fmt: skip
        """Create an ACT objective from its loss configuration."""
        super().__init__(config=config)

    def compute_losses(  # -----------------------------------------------------------------------
        self, outputs: ACTOutput, labels: Tensor, stats: AccuracyStats, **_: Any,
    ) -> Losses:  # fmt: skip
        """Compute supervised and halting-related losses for a step."""
        # Compute main modeling loss (e.g., cross-entropy) over all tokens, summed over batch.
        loss_sum = self.compute_lm_loss(outputs.lm_logits, labels, stats)

        # Done-action loss: match Q(done) to sequence correctness.
        done_action = outputs.done_action
        q_done_logits = outputs.q_logits[..., done_action]  # (B,)
        target = stats.seq_is_correct.to(q_done_logits.dtype)
        q_done_loss = F.binary_cross_entropy_with_logits(q_done_logits, target, reduction="sum")

        # Continue loss: optional auxiliary supervision from the controller's TD target.
        q_continue_loss: Tensor | None = None
        if outputs.target_q is not None:
            # Select the non-done logit(s). For 2-action, pick ~done_action.
            # Generalization: supervise all non-done actions toward TD target.
            n_actions = outputs.q_logits.shape[-1]
            continue_actions = [a for a in range(n_actions) if a != done_action]
            if continue_actions:
                q_cont = outputs.q_logits[..., continue_actions].mean(dim=-1)  # (B,)
                target = outputs.target_q
                q_continue_loss = F.binary_cross_entropy_with_logits(q_cont, target, reduction="sum")

        return Losses(loss_sum, q_done_loss, q_continue_loss)

    def _build_step_output(  # -------------------------------------------------------------------
        self, losses: Losses, metrics: Any, signals: dict[str, Any], outputs: ACTOutput,
    ) -> ACTLossStep:  # fmt: skip
        """Wrap losses, metrics, and signals into an :class:`ACTLossStep`."""
        return ACTLossStep(losses=losses, metrics=metrics, outputs=outputs, signals=signals)

    def _build_metric_ratios(  # -----------------------------------------------------------------
        self, losses: Losses, *, batch_size: int,
    ) -> dict[str, RatioStat]:  # fmt: skip
        """Build detached ratio metrics for logging."""
        batch_count = losses.loss_sum.new_tensor(batch_size, dtype=torch.float32)
        q_continue_loss_sum = losses.loss_q_continue_sum
        if q_continue_loss_sum is None:
            q_continue_loss_sum = losses.loss_sum.new_zeros(())

        return {
            LOSS_LM: RatioStat(losses.loss_sum.detach(), batch_count),
            ACT_LOSS_Q_DONE: RatioStat(losses.loss_q_done_sum.detach(), batch_count),
            ACT_LOSS_Q_CONTINUE: RatioStat(q_continue_loss_sum.detach(), batch_count),
        }

    def compute_signals(  # -----------------------------------------------------------------------
        self, batch: Batch, state: ACTRolloutState, outputs: ACTOutput, losses: Losses,
    ) -> dict[str, Tensor]:  # fmt: skip
        """Compute lightweight diagnostic signals for logging."""
        sig: dict[str, Tensor] = {
            S.STEPS_MEAN:     state.steps.float().mean().detach(),
            S.THETA_CLS_NORM: outputs.theta_cls.detach().norm(dim=-1).mean(),
            S.LOSS_Q_DONE:    losses.loss_q_done_sum.detach(),
        }  # fmt: skip
        if outputs.target_q is not None:
            sig[S.TARGET_Q_MEAN] = outputs.target_q.mean().detach()
            sig[S.TARGET_Q_STD] = outputs.target_q.std().detach()
        return sig
