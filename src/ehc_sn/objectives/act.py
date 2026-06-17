"""ACT objective for HRM v1.

This module defines :class:`ACTObjective`, which scores raw ACT execution steps
through an injected task binding. The objective computes token-level
supervision from the bound task payload, Q(done) supervision from exact
sequence correctness, and optional auxiliary Q(continue) supervision from an
objective-owned TD bootstrap target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, cast

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor

import ehc_sn.loss.cross_entropy as cross_entropy_module
import ehc_sn.metrics.signals as S
from ehc_sn.controllers.deliberation.act import (
    ACTController,
    collapse_act_halt_continue_logits,
)
from ehc_sn.loss.cross_entropy import LossType
from ehc_sn.metrics.keys import (
    ACT_LOSS_Q_CONTINUE,
    ACT_LOSS_Q_DONE,
    LOSS_TOKEN,
)
from ehc_sn.metrics.step_metrics import RatioStat, StepMetrics
from ehc_sn.objectives._base import BaseObjective
from ehc_sn.objectives._token import (
    IGNORE_LABEL_ID,
    AccuracyStats,
    build_token_step_metrics,
)
from ehc_sn.rollouts.runtime import CarrySnapshot, StepRecord
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin


# =============================================================================
class ACTObjectiveConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`ACTObjective`."""

    token_loss: LossType = Field(
        default="softmax_cross_entropy",
        description="The loss function to use for the modeling loss.",
    )
    use_token_weights: bool = Field(
        default=False,
        description=(
            "Whether to apply task-provided per-token weights when computing "
            "token supervision loss."
        ),
    )


# =============================================================================
class ACTStepOutput(Protocol):
    """Objective-facing output contract for ACT rollout steps."""

    task: object
    q_logits: Tensor
    done_action: int


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

    def extract_loss_labels(  # -----------------------------------------------
        self,
        targets: TargetsT,
    ) -> Tensor:
        """Return a label tensor suitable for the token-level loss function.

        Non-supervised positions must be replaced with an ignore index
        (typically ``IGNORE_LABEL_ID = -100``) recognized by the loss function's
        ``ignore_index`` parameter.
        """


# =============================================================================
class _TokenWeightBinding(Protocol):
    """Optional task binding surface for per-token Token weights."""

    def build_token_weights(  # -----------------------------------------------
        self,
        labels: Tensor,
    ) -> Tensor:
        """Return per-token loss weights aligned with Token labels."""


# =============================================================================
@dataclass(frozen=True)
class ACTLosses(DetachMixin):
    """Bundle of ACT loss terms (summed over batch)."""

    loss_token_sum: Tensor
    loss_q_done_sum: Tensor
    loss_q_continue_sum: Optional[Tensor]

    @property
    def loss_sum(self) -> Tensor:
        """Return the summed token-supervision loss for the step."""
        return self.loss_token_sum

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

    loss_token: Tensor
    loss_q_done: Tensor
    loss_q_continue: Tensor | None


# =============================================================================
@dataclass
class ACTContext:
    """Shared objective-scoring context resolved once per ACT step."""

    executed_batch: Batch
    snapshot: CarrySnapshot
    outputs: ACTStepOutput
    targets: Any
    logits: Tensor
    stats: AccuracyStats
    token_weights: Tensor | None
    target_q: Tensor | None

    terms: ACTTerms | None = field(default=None, init=False)
    losses: ACTLosses | None = field(default=None, init=False)
    metrics: StepMetrics | None = field(default=None, init=False)
    signals: dict[str, Tensor] = field(default_factory=dict, init=False)


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

    @property
    def step_output(self) -> Optional[ACTStepOutput]:
        """Canonical name for the underlying controller/backbone step output.

        Preferred over the bare ``outputs`` field, which collides with
        :attr:`ObservedStep.outputs` and forces consumers to write
        ``outputs.outputs`` when traversing from an ``EvaluatedChunk``.
        """
        return self.outputs


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
    def _token_loss_fn(self) -> Any:
        """Return the configured token-level loss function."""
        return getattr(cross_entropy_module, self.config.token_loss)

    def evaluate_step(  # -----------------------------------------------------
        self,
        record: StepRecord,
        controller: ACTController | None = None,
        td_target: bool = False,
        target_q: Tensor | None = None,
        target_backbone: Any | None = None,
        **options: Any,
    ) -> ACTObjectiveStep:
        """Score one ACT rollout step and attach any objective-owned TD target."""
        if td_target:
            if controller is None:
                raise ValueError(
                    "ACTObjective: td_target=True requires a controller."
                )
            target_q = self._compute_td_target(
                controller=controller,
                record=record,
                target_backbone=target_backbone,
            )

        # --- single scored-step computation ---
        context = self.build_context(record, target_q=target_q, **options)
        context.terms = self.compute_terms(context)
        context.losses = self.compute_losses(context)

        # --- metrics from precomputed losses/terms ---
        context.metrics = self.evaluate_metrics(context)

        # --- signals from precomputed losses/context ---
        context.signals = self.compute_signals(context)

        return self.build_output(context)

    def build_context(  # -----------------------------------------------------
        self,
        record: StepRecord,
        *,
        target_q: Tensor | None = None,
        use_token_weights: bool = False,
        **_: Any,
    ) -> ACTContext:
        """Resolve task targets, logits, and sequence stats for one ACT step."""

        outputs = cast(ACTStepOutput, record.outputs)

        # Extract supervised logits and targets from the task binding.
        logits = self._task_binding.extract_logits(
            executed_batch=record.batch,
            snapshot=record.carry,
            step_output=outputs,
        )
        targets = self._task_binding.extract_targets(
            executed_batch=record.batch,
            snapshot=record.carry,
            step_output=outputs,
        )

        # Compute sequence-level correctness stats for the step from the task
        # binding.
        stats = self._task_binding.evaluate_sequences(logits, targets)

        # Optional token weights for token loss scaling (e.g., for mazehard).
        token_weights = None
        if use_token_weights:
            if not hasattr(self._task_binding, "build_token_weights"):
                raise RuntimeError(
                    "ACTObjective: use_token_weights=True but the task binding "
                    "does not implement build_token_weights."
                )
            labels = self._task_binding.extract_loss_labels(targets)
            token_weight_binding = cast(_TokenWeightBinding, self._task_binding)
            token_weights = token_weight_binding.build_token_weights(labels)

        # Return a bundled context for the step.
        return ACTContext(
            executed_batch=record.batch,
            snapshot=record.carry,
            outputs=outputs,
            targets=targets,
            logits=logits,
            stats=stats,
            token_weights=token_weights,
            target_q=target_q,
        )

    def compute_terms(  # -----------------------------------------------------
        self,
        context: ACTContext,
        **_: Any,
    ) -> ACTTerms:
        """Score per-example ACT loss terms for one executed step."""
        outputs = context.outputs
        logits_q_done = outputs.q_logits[..., outputs.done_action]
        labels = self._task_binding.extract_loss_labels(context.targets)

        return ACTTerms(
            # Compute the token loss per sequence
            loss_token=self.token_loss_fn(
                logits_token=context.logits,
                labels=labels,
                stats=context.stats,
                token_weights=context.token_weights,
            ),
            # Compute the Q(done) loss per sequence
            loss_q_done=self.q_done_loss_fn(
                logits_q_done=logits_q_done,
                done_target=context.stats.seq_is_correct,
            ),
            # Compute the Q(continue) loss per sequence
            loss_q_continue=(
                self.q_continue_loss_fn(
                    logits_q_done=outputs.q_logits,
                    done_action=outputs.done_action,
                    continue_target=context.target_q,
                )
                if context.target_q is not None
                else None
            ),
        )

    def token_loss_fn(  # -----------------------------------------------------
        self,
        logits_token: Tensor,
        labels: Tensor,
        stats: AccuracyStats,
        token_weights: Tensor | None = None,
    ) -> Tensor:
        """Compute per-sequence supervised token loss for a step."""
        loss_per_token = self._token_loss_fn(
            logits_token, labels, ignore_index=IGNORE_LABEL_ID
        )
        if token_weights is not None:
            if token_weights.shape != labels.shape:
                raise ValueError(
                    "token_weights must match labels shape for Token loss "
                    "weighting."
                )
            loss_per_token = loss_per_token * token_weights.to(
                loss_per_token.dtype
            )
        return loss_per_token.sum(-1) / stats.loss_counts.clamp_min(1)

    def q_done_loss_fn(  # ----------------------------------------------------
        self,
        logits_q_done: Tensor,
        done_target: Tensor,
    ) -> Tensor:
        """Compute per-sequence Q(done) loss for a step."""
        return F.binary_cross_entropy_with_logits(
            input=logits_q_done,
            target=done_target.to(logits_q_done.dtype),
            reduction="none",
        )

    def q_continue_loss_fn(  # ------------------------------------------------
        self,
        logits_q_done: Tensor,
        done_action: int,
        continue_target: Tensor,
    ) -> Tensor:
        """Compute per-sequence Q(continue) loss for a step."""
        scores = collapse_act_halt_continue_logits(
            logits_q_done, done_action=done_action
        )
        return F.binary_cross_entropy_with_logits(
            input=scores.continue_logit,
            target=continue_target,
            reduction="none",
        )

    def compute_losses(  # ----------------------------------------------------
        self,
        context: ACTContext,
        **_: Any,
    ) -> ACTLosses:
        """Aggregate per-example terms into summed ACT losses."""
        terms = context.terms
        if terms is None:
            raise ValueError(
                "ACTContext.terms must be set before computing losses."
            )
        return ACTLosses(
            loss_token_sum=terms.loss_token.sum(),
            loss_q_done_sum=terms.loss_q_done.sum(),
            loss_q_continue_sum=_maybe_sum(terms.loss_q_continue),
        )

    def evaluate_metrics(  # --------------------------------------------------
        self,
        context: ACTContext,
        **_: Any,
    ) -> StepMetrics:
        """Assemble per-step token metrics for the current ACT step."""
        losses = context.losses
        if losses is None:
            raise ValueError(
                "ACTContext.losses must be set before evaluating metrics."
            )
        stats = context.stats
        batch_size = int(context.logits.shape[0])
        extras = self._build_metric_ratios(losses, batch_size=batch_size)
        steps = context.snapshot.steps
        if steps is None:
            steps = losses.loss_sum.new_zeros((batch_size,), dtype=torch.long)
        completed = context.snapshot.halted
        return build_token_step_metrics(steps, completed, stats, extras)

    def compute_signals(  # ---------------------------------------------------
        self,
        context: ACTContext,
        **_: Any,
    ) -> dict[str, Tensor]:
        """Compute lightweight diagnostic signals for logging."""
        losses = context.losses
        if losses is None:
            raise ValueError(
                "ACTContext.losses must be set before computing signals."
            )
        steps = context.snapshot.steps
        if steps is None:
            steps = losses.loss_sum.new_zeros((1,))
        scores = collapse_act_halt_continue_logits(
            context.outputs.q_logits,
            done_action=context.outputs.done_action,
        )

        signals: dict[str, Tensor] = {
            S.STEPS_MEAN: steps.float().mean().detach(),
            S.LOSS_Q_DONE: losses.loss_q_done_sum.detach(),
            S.HALT_LOGIT_MEAN: scores.halt_logit.mean().detach(),
            S.CONTINUE_LOGIT_MEAN: scores.continue_logit.mean().detach(),
            S.GREEDY_HALT_RATE: (
                scores.greedy_halt.to(dtype=torch.float32).mean().detach()
            ),
        }
        target_q = context.target_q
        if target_q is not None:
            signals[S.TARGET_Q_MEAN] = target_q.mean().detach()
            signals[S.TARGET_Q_STD] = target_q.std(unbiased=False).detach()
        return signals

    def build_output(  # ------------------------------------------------------
        self,
        context: ACTContext,
        **_: Any,
    ) -> ACTObjectiveStep:
        """Wrap losses, metrics, and signals into an :class:`ACTObjectiveStep`."""
        if context.losses is None:
            raise ValueError(
                "ACTContext.losses must be set before building output."
            )
        if context.metrics is None:
            raise ValueError(
                "ACTContext.metrics must be set before building output."
            )
        if context.signals is None:
            raise ValueError(
                "ACTContext.signals must be set before building output."
            )
        return ACTObjectiveStep(
            losses=context.losses,
            metrics=context.metrics,
            outputs=context.outputs,
            target_q=context.target_q,
            signals=context.signals,
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
            LOSS_TOKEN: RatioStat(losses.loss_sum.detach(), batch_count),
            ACT_LOSS_Q_DONE: RatioStat(
                losses.loss_q_done_sum.detach(), batch_count
            ),
            ACT_LOSS_Q_CONTINUE: RatioStat(
                q_continue_loss_sum.detach(), batch_count
            ),
        }

    @staticmethod
    def _compute_td_target(  # ------------------------------------------------
        controller: ACTController,
        record: StepRecord,
        *,
        target_backbone: Any | None = None,
    ) -> Tensor:
        """Compute the TD bootstrap target from executed carry state.

        When a *target_backbone* is provided, a fresh target model state is
        derived by cloning the online model state from the record, resetting it
        with the same post-rollout halted mask, and stepping the target backbone
        on the *same executed inputs* (``record.carry.data``) that the online
        backbone consumed.  This guarantees input alignment — the key invariant
        that the previous implementation violated by stepping the target on the
        raw incoming batch.

        When *target_backbone* is *None*, falls back to the online controller's
        backbone (current self-bootstrap behaviour).

        Args:
            controller: The online ACT controller (used as fallback when
                ``target_backbone`` is *None*).
            record: The executed step record containing carry/model_state.
            target_backbone: Optional frozen lagged backbone.  Must expose
                ``init_state(batch_size)``, ``reset_state(flag, state)``, and a
                forward call ``(data, state) -> (output, next_state)``.  If
                *None*, uses ``controller.backbone``.

        Returns:
            Sigmoid-squashed TD bootstrap target tensor of shape ``(B,)``.
        """
        data = record.carry.data
        model_state = record.carry.model_state
        steps = record.carry.steps
        if data is None or model_state is None or steps is None:
            raise ValueError(
                "ACT TD target requires carry.data, carry.model_state, and "
                "carry.steps."
            )

        if target_backbone is not None:
            # Build a fresh target state from the online state: clone, reset
            # with the same halted mask, then step on the same executed data.
            with torch.no_grad():
                batch_size = int(
                    next(iter(data.values())).shape[0]
                    if isinstance(data, dict)
                    else model_state.steps.shape[0]
                )
                online_halted = record.carry.halted
                target_state = model_state.detach()
                target_state = target_backbone.target.reset_state(
                    online_halted, target_state
                )
                backbone_output, _ = target_backbone(data, target_state)
                next_q = backbone_output.control.q_logits
        else:
            with torch.no_grad():
                backbone_output, _ = controller.backbone(data, model_state)
                next_q = backbone_output.control.q_logits

        done_action = controller.config.done_action
        max_halt_steps = controller.config.max_halt_steps
        scores = collapse_act_halt_continue_logits(
            next_q, done_action=done_action
        )
        is_last_step = steps >= max_halt_steps
        target_q = torch.where(
            is_last_step,
            scores.halt_logit,
            torch.maximum(scores.halt_logit, scores.continue_logit),
        )
        return torch.sigmoid(target_q)


# =============================================================================
def _maybe_sum(  # ----------------------------------------------------------
    tensor: Tensor | None,
) -> Tensor | None:
    """Sum a tensor if it's not None, otherwise return None."""
    if tensor is not None:
        return tensor.sum()
    return None


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
