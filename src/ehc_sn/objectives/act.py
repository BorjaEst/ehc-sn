"""ACT objective for HRM v1.

This module defines :class:`ACTLossHead`, which scores raw ACT execution steps
through an injected task binding. The objective computes token-level
supervision from the bound task payload, Q(done) supervision from exact
sequence correctness, and optional auxiliary Q(continue) supervision from an
objective-owned TD bootstrap target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.controllers.act import ACTController, ACTRolloutState, ACTStepOutput, collapse_act_halt_continue_logits
from ehc_sn.loss.cross_entropy import LossType
from ehc_sn.metrics import signals as S
from ehc_sn.metrics.keys import ACT_LOSS_Q_CONTINUE, ACT_LOSS_Q_DONE, LOSS_LM
from ehc_sn.objectives._token import AccuracyStats, TokenLossHeadBase, TokenSupervisionBinding
from ehc_sn.rollouts import StepRecord
from ehc_sn.training.types import RatioStat, StepMetrics
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin


# =================================================================================================
class ACTLossConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`ACTLossHead`."""

    function: LossType = Field(
        default="stablemax_cross_entropy",
        description="The loss function to use for the modeling loss.",
    )


# =================================================================================================
class ACTTaskBinding[TargetsT](TokenSupervisionBinding[TargetsT], Protocol):
    """Task-binding seam used by ACT objectives."""


# =================================================================================================
@dataclass(frozen=True)
class Losses(DetachMixin):
    """Bundle of ACT loss terms (summed over batch)."""

    loss_sum: Tensor
    loss_q_done_sum: Tensor
    loss_q_continue_sum: Optional[Tensor]

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

    losses: Losses
    metrics: StepMetrics
    outputs: Optional[ACTStepOutput] = None
    target_q: Tensor | None = None
    signals: dict[str, Any] | None = None

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

    def __init__(
        self,
        config: ACTLossConfig,
        task_binding: ACTTaskBinding[Any],
    ) -> None:
        """Create an ACT objective from its loss configuration and task binding."""
        super().__init__(config=config, token_binding=task_binding)

    def evaluate_step(
        self,
        record: StepRecord,
        **options: Any,
    ) -> ACTLossStep:
        """Score one ACT rollout step and attach any objective-owned TD target."""
        loss_options = dict(options)
        controller = loss_options.pop("controller", None)
        if not isinstance(controller, ACTController):
            raise TypeError("ACTLossHead requires controller=ACTController when scoring ACT rollout steps.")

        td_target = bool(loss_options.pop("td_target", True))
        target_q = self._compute_td_target(controller, record) if td_target and controller.config.max_steps > 1 else None
        return super().evaluate_step(record, controller=controller, target_q=target_q, **loss_options)

    def compute_losses(
        self,
        outputs: ACTStepOutput,
        targets: Any,
        stats: AccuracyStats,
        *,
        controller: ACTController,
        logits: Tensor,
        target_q: Tensor | None = None,
        **_: Any,
    ) -> Losses:
        """Compute supervised and halting-related losses for a step."""
        labels = getattr(targets, "labels", targets)
        if not isinstance(labels, Tensor):
            raise TypeError("ACTLossHead expects tensor labels from the bound ACT task targets.")

        loss_sum = self.compute_lm_loss(logits, labels, stats)

        done_action = controller.config.done_action
        q_logits = outputs.backbone_output.control.q_logits
        q_done_logits = q_logits[..., done_action]
        done_target = stats.seq_is_correct.to(q_done_logits.dtype)
        q_done_loss = F.binary_cross_entropy_with_logits(q_done_logits, done_target, reduction="sum")

        q_continue_loss: Tensor | None = None
        if target_q is not None:
            scores = collapse_act_halt_continue_logits(q_logits, done_action=done_action)
            q_continue_loss = F.binary_cross_entropy_with_logits(scores.continue_logit, target_q, reduction="sum")

        return Losses(loss_sum, q_done_loss, q_continue_loss)

    def _build_step_output(
        self,
        losses: Losses,
        metrics: Any,
        signals: dict[str, Any],
        outputs: ACTStepOutput,
        *,
        target_q: Tensor | None = None,
        **_: Any,
    ) -> ACTLossStep:
        """Wrap losses, metrics, and signals into an :class:`ACTLossStep`."""
        return ACTLossStep(losses=losses, metrics=metrics, outputs=outputs, target_q=target_q, signals=signals)

    def _build_metric_ratios(
        self,
        losses: Losses,
        *,
        batch_size: int,
    ) -> dict[str, RatioStat]:
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

    def compute_signals(
        self,
        batch: Batch,
        state: ACTRolloutState,
        outputs: ACTStepOutput,
        losses: Losses,
        *,
        target_q: Tensor | None = None,
        **_: Any,
    ) -> dict[str, Tensor]:
        """Compute lightweight diagnostic signals for logging."""
        _ = batch, outputs
        signals: dict[str, Tensor] = {
            S.STEPS_MEAN: state.steps.float().mean().detach(),
            S.LOSS_Q_DONE: losses.loss_q_done_sum.detach(),
        }
        if target_q is not None:
            signals[S.TARGET_Q_MEAN] = target_q.mean().detach()
            signals[S.TARGET_Q_STD] = target_q.std(unbiased=False).detach()
        return signals

    @staticmethod
    def _compute_td_target(controller: ACTController, record: StepRecord) -> Tensor:
        """Compute the TD bootstrap target from executed carry state."""
        data = record.carry.data
        model_state = record.carry.model_state
        steps = record.carry.steps
        if data is None or model_state is None or steps is None:
            raise ValueError("ACT TD target requires carry.data, carry.model_state, and carry.steps.")

        with torch.no_grad():
            _, backbone_output = controller.backbone(data, model_state)
            next_q = backbone_output.control.q_logits

        done_action = controller.config.done_action
        is_last_step = steps >= controller.config.max_steps
        scores = collapse_act_halt_continue_logits(next_q, done_action=done_action)
        target = torch.where(is_last_step, scores.halt_logit, scores.continue_logit)
        return torch.sigmoid(target)


# =================================================================================================
__all__ = ["ACTLossConfig", "ACTLossHead", "ACTLossStep", "ACTTaskBinding"]
