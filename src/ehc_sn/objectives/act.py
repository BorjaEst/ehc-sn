"""ACT objective for HRM v1.

This module defines :class:`ACTObjective`, which scores raw ACT execution steps
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

import ehc_sn.loss.cross_entropy as cross_entropy_module
from ehc_sn.controllers.deliberation.act import (
    ACTController,
    collapse_act_halt_continue_logits,
)
from ehc_sn.loss.cross_entropy import LossType
from ehc_sn.metrics import signals as S
from ehc_sn.metrics.keys import ACT_LOSS_Q_CONTINUE, ACT_LOSS_Q_DONE, LOSS_LM
from ehc_sn.objectives._base import BaseObjective
from ehc_sn.objectives._token import (
    IGNORE_LABEL_ID,
    AccuracyStats,
    build_token_step_metrics,
    compute_lm_loss_sum,
)
from ehc_sn.rollouts import CarrySnapshot, StepRecord
from ehc_sn.training.types import RatioStat, StepMetrics
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin


# =============================================================================
class ACTObjectiveConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`ACTObjective`."""

    function: LossType = Field(
        default="stablemax_cross_entropy",
        description="The loss function to use for the modeling loss.",
    )


# =============================================================================
class ACTStepOutput(Protocol):
    """Objective-facing output contract for ACT rollout steps."""

    task: object
    q_logits: Tensor


# =============================================================================
class ACTObjectiveBinding[TargetsT](Protocol):
    """Canonical task-binding protocol for the ACT objective.

    Implemented in the adapter layer so that :class:`ACTObjective` stays
    task-agnostic. The binding owns task-specific target extraction and
    sequence correctness evaluation; the objective owns loss math and metrics.

    The ``executed_batch`` input is the executed-step payload (``record.batch``),
    which is the authoritative source of current-step supervision. The
    ``snapshot`` input is a frozen post-step snapshot that should only supply
    continuity facts or lightweight post-step projections.
    """

    def extract_logits(  # ----------------------------------------------------
        self,
        executed_batch: Batch,
        snapshot: CarrySnapshot,
        step_output: ACTStepOutput,
    ) -> Tensor:
        """Return supervised logits for one executed step."""

    def extract_targets(  # ---------------------------------------------------
        self,
        executed_batch: Batch,
        snapshot: CarrySnapshot,
        step_output: ACTStepOutput,
    ) -> TargetsT:
        """Return task-owned supervision targets for one executed step."""

    def evaluate_sequences(  # ------------------------------------------------
        self,
        logits: Tensor,
        targets: TargetsT,
    ) -> AccuracyStats:
        """Return sequence-level accuracy statistics for one executed step."""


# =============================================================================
@dataclass(frozen=True)
class ACTLosses(DetachMixin):
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
        return self.loss_sum + 0.5 * (
            self.loss_q_done_sum + q_continue_loss_sum
        )


# =============================================================================
@dataclass(frozen=True)
class ACTTerms:
    """Per-example ACT loss terms scored for one executed step."""

    loss_lm: Tensor
    loss_q_done: Tensor
    loss_q_continue: Tensor | None

    @property
    def loss_sum(self) -> Tensor:
        """Return the aggregate summed token loss."""
        return self.loss_lm.sum()

    @property
    def loss_q_done_sum(self) -> Tensor:
        """Return the aggregate summed q(done) loss."""
        return self.loss_q_done.sum()

    @property
    def loss_q_continue_sum(self) -> Tensor | None:
        """Return the aggregate summed q(continue) loss if present."""
        if self.loss_q_continue is None:
            return None
        return self.loss_q_continue.sum()


# =============================================================================
@dataclass(frozen=True)
class ACTContext:
    """Shared objective-scoring context resolved once per ACT step."""

    executed_batch: Batch
    snapshot: CarrySnapshot
    outputs: ACTStepOutput
    targets: Any
    logits: Tensor
    stats: AccuracyStats
    done_action: int
    target_q: Tensor | None


# =============================================================================
@dataclass(frozen=True)
class ACTObjectiveStep:
    """A single rollout/loss step produced by :class:`ACTObjective`."""

    losses: ACTLosses
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


# =============================================================================
class ACTObjective(BaseObjective[ACTObjectiveConfig]):
    """Pure ACT objective scored over executed rollout chunks."""

    def __init__(  # ----------------------------------------------------------
        self,
        config: ACTObjectiveConfig,
        task_binding: ACTObjectiveBinding[Any],
    ) -> None:
        """Create an ACT objective from its loss configuration and task binding."""
        super().__init__(config=config)
        self._task_binding = task_binding

    @property
    def loss_fn(self) -> Any:
        """Return the configured token-level loss function."""
        return getattr(cross_entropy_module, self._config.function)

    def evaluate_step(  # -----------------------------------------------------
        self,
        record: StepRecord,
        **options: Any,
    ) -> ACTObjectiveStep:
        """Score one ACT rollout step and attach any objective-owned TD target."""
        loss_options = dict(options)
        controller = loss_options.pop("controller", None)
        if not isinstance(controller, ACTController):
            raise TypeError(
                "ACTObjective requires controller=ACTController when scoring ACT rollout steps."
            )

        td_target = bool(loss_options.pop("td_target", True))
        target_q = (
            self._compute_td_target(controller, record) if td_target else None
        )

        step_output = record.outputs
        context = self.build_context(
            record,
            step_output,
            controller=controller,
            target_q=target_q,
            **loss_options,
        )
        terms = self.compute_terms(context, **loss_options)
        losses = self.compute_losses(terms)
        metrics = self._build_step_metrics(
            context.stats,
            losses,
            batch_size=int(context.logits.shape[0]),
            steps=context.snapshot.steps,
            completed=context.snapshot.halted,
        )
        signals = self.compute_signals(
            context.executed_batch,
            context.snapshot,
            step_output,
            losses,
            target_q=target_q,
            **loss_options,
        )
        return self._build_step_output(
            losses,
            metrics,
            signals,
            step_output,
            target_q=target_q,
        )

    def compute_lm_loss(  # ---------------------------------------------------
        self,
        logits_lm: Tensor,
        labels: Tensor,
        stats: AccuracyStats,
    ) -> Tensor:
        """Compute the summed supervised token loss for a step."""
        return compute_lm_loss_sum(self.loss_fn, logits_lm, labels, stats)

    def build_context(  # -----------------------------------------------------
        self,
        record: StepRecord,
        outputs: ACTStepOutput,
        *,
        controller: ACTController,
        target_q: Tensor | None = None,
        **_: Any,
    ) -> ACTContext:
        """Resolve task targets, logits, and sequence stats for one ACT step."""
        executed_batch = record.batch
        snapshot = record.carry
        logits = self._task_binding.extract_logits(
            executed_batch=executed_batch,
            snapshot=snapshot,
            step_output=outputs,
        )
        targets = self._task_binding.extract_targets(
            executed_batch=executed_batch,
            snapshot=snapshot,
            step_output=outputs,
        )
        stats = self._task_binding.evaluate_sequences(logits, targets)
        return ACTContext(
            executed_batch=executed_batch,
            snapshot=snapshot,
            outputs=outputs,
            targets=targets,
            logits=logits,
            stats=stats,
            done_action=controller.config.done_action,
            target_q=target_q,
        )

    def compute_terms(  # -----------------------------------------------------
        self,
        context: ACTContext,
        **_: Any,
    ) -> ACTTerms:
        """Score per-example ACT loss terms for one executed step."""
        labels = self._require_labels(context.targets)
        loss_lm = self._compute_lm_loss_per_seq(
            context.logits, labels, context.stats
        )

        q_logits = context.outputs.q_logits
        q_done_logits = q_logits[..., context.done_action]
        done_target = context.stats.seq_is_correct.to(q_done_logits.dtype)
        loss_q_done = F.binary_cross_entropy_with_logits(
            q_done_logits, done_target, reduction="none"
        )

        loss_q_continue: Tensor | None = None
        if context.target_q is not None:
            scores = collapse_act_halt_continue_logits(
                q_logits, done_action=context.done_action
            )
            loss_q_continue = F.binary_cross_entropy_with_logits(
                scores.continue_logit, context.target_q, reduction="none"
            )

        return ACTTerms(
            loss_lm=loss_lm,
            loss_q_done=loss_q_done,
            loss_q_continue=loss_q_continue,
        )

    def compute_losses(  # ----------------------------------------------------
        self,
        terms: ACTTerms,
    ) -> ACTLosses:
        """Aggregate per-example terms into summed ACT losses."""
        return ACTLosses(
            terms.loss_sum,
            terms.loss_q_done_sum,
            terms.loss_q_continue_sum,
        )

    def _build_step_output(  # ------------------------------------------------
        self,
        losses: ACTLosses,
        metrics: Any,
        signals: dict[str, Any],
        outputs: ACTStepOutput,
        *,
        target_q: Tensor | None = None,
        **_: Any,
    ) -> ACTObjectiveStep:
        """Wrap losses, metrics, and signals into an :class:`ACTObjectiveStep`."""
        return ACTObjectiveStep(
            losses=losses,
            metrics=metrics,
            outputs=outputs,
            target_q=target_q,
            signals=signals,
        )

    def _build_metric_ratios(  # ----------------------------------------------
        self,
        losses: ACTLosses,
        *,
        batch_size: int,
    ) -> dict[str, RatioStat]:
        """Build detached ratio metrics for logging."""
        batch_count = losses.loss_sum.new_tensor(
            batch_size, dtype=torch.float32
        )
        q_continue_loss_sum = losses.loss_q_continue_sum
        if q_continue_loss_sum is None:
            q_continue_loss_sum = losses.loss_sum.new_zeros(())

        return {
            LOSS_LM: RatioStat(losses.loss_sum.detach(), batch_count),
            ACT_LOSS_Q_DONE: RatioStat(
                losses.loss_q_done_sum.detach(), batch_count
            ),
            ACT_LOSS_Q_CONTINUE: RatioStat(
                q_continue_loss_sum.detach(), batch_count
            ),
        }

    def compute_signals(  # ---------------------------------------------------
        self,
        batch: Batch,
        state: CarrySnapshot,
        outputs: ACTStepOutput,
        losses: ACTLosses,
        *,
        target_q: Tensor | None = None,
        **_: Any,
    ) -> dict[str, Tensor]:
        """Compute lightweight diagnostic signals for logging."""
        _ = batch, outputs
        steps = state.steps
        if steps is None:
            steps = losses.loss_sum.new_zeros((1,))

        signals: dict[str, Tensor] = {
            S.STEPS_MEAN: steps.float().mean().detach(),
            S.LOSS_Q_DONE: losses.loss_q_done_sum.detach(),
        }
        if target_q is not None:
            signals[S.TARGET_Q_MEAN] = target_q.mean().detach()
            signals[S.TARGET_Q_STD] = target_q.std(unbiased=False).detach()
        return signals

    def _build_step_metrics(  # ----------------------------------------------
        self,
        stats: AccuracyStats,
        losses: ACTLosses,
        *,
        batch_size: int,
        steps: Tensor | None,
        completed: Tensor,
    ) -> StepMetrics:
        """Assemble per-step token metrics for the current ACT step."""
        extras = self._build_metric_ratios(losses, batch_size=batch_size)
        if steps is None:
            steps = losses.loss_sum.new_zeros((batch_size,), dtype=torch.long)
        return build_token_step_metrics(steps, completed, stats, extras)

    @staticmethod
    def _require_labels(targets: Any) -> Tensor:
        labels = getattr(targets, "labels", targets)
        if not isinstance(labels, Tensor):
            raise TypeError(
                "ACTObjective expects tensor labels from the bound ACT task targets."
            )
        return labels

    def _compute_lm_loss_per_seq(  # -----------------------------------------
        self,
        logits_lm: Tensor,
        labels: Tensor,
        stats: AccuracyStats,
    ) -> Tensor:
        """Compute per-sequence supervised token loss for a step."""
        loss_per_token = self.loss_fn(
            logits_lm, labels, ignore_index=IGNORE_LABEL_ID
        )
        return loss_per_token.sum(-1) / stats.loss_counts.clamp_min(1)

    @staticmethod
    def _compute_td_target(  # ------------------------------------------------
        controller: ACTController, record: StepRecord
    ) -> Tensor:
        """Compute the TD bootstrap target from executed carry state."""
        data = record.carry.data
        model_state = record.carry.model_state
        steps = record.carry.steps
        if data is None or model_state is None or steps is None:
            raise ValueError(
                "ACT TD target requires carry.data, carry.model_state, and carry.steps."
            )
        del steps  # no longer used for forced-halt boundary; kept for carry validation only

        with torch.no_grad():
            backbone_output, _ = controller.backbone(data, model_state)
            next_q = backbone_output.control.q_logits

        done_action = controller.config.done_action
        scores = collapse_act_halt_continue_logits(
            next_q, done_action=done_action
        )
        return torch.sigmoid(scores.continue_logit)


# =============================================================================
__all__ = [
    "ACTObjectiveConfig",
    "ACTObjective",
    "ACTObjectiveStep",
    "ACTObjectiveBinding",
    "ACTObjectiveBinding",
    "ACTStepOutput",
    "ACTLosses",
]
