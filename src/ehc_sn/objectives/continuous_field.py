"""Continuous field objective — MSE supervision with Q(done) from field quality.

This module defines :class:`ContinuousFieldObjective`, which scores ACT
execution steps for any task producing continuous-valued predictions in
``[0, 1]`` (e.g. firing fields, probability maps, occupancy grids).
Unlike :class:`~ehc_sn.objectives.act.ACTObjective` (token-level
cross-entropy), this objective computes:

1. **Field loss**: masked MSE between predicted and target continuous field.
2. **Q(done) loss**: BCE on halting quality (field MSE below a threshold).
3. **Q(continue)**: deferred — no TD bootstrap in v1.

The goal is to keep the same interface shape as ``ACTObjective`` so
``ACTSupervisedModule`` (and its evaluation infrastructure) can consume
it without modification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, cast, runtime_checkable

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor

from ehc_sn.metrics.signals import (
    CONTINUE_LOGIT_MEAN,
    HALT_LOGIT_MEAN,
    LOSS_Q_DONE,
)
from ehc_sn.metrics.step_metrics import (
    RatioStat,
    RolloutAgg,
    StepMetrics,
    TokenAgg,
    TransitionAgg,
)
from ehc_sn.objectives._base import BaseObjective
from ehc_sn.rollouts.runtime import CarrySnapshot, StepRecord
from ehc_sn.types import Batch
from ehc_sn.utils.detach import DetachMixin


# =============================================================================
class ContinuousFieldObjectiveConfig(BaseModel, extra="forbid"):
    """Configuration for :class:`ContinuousFieldObjective`.

    Attributes:
        field_loss: Loss function for field supervision.  Only ``"mse"``
            is supported in v1.
        field_done_threshold: Field MSE below which the Q(done) target
            is 1.0.  Controls how strict halting quality is.
    """

    field_loss: str = Field(
        default="mse",
        description="Loss function for field supervision.  Only 'mse' supported.",
    )
    field_done_threshold: float = Field(
        default=0.01,
        ge=0.0,
        description="Field MSE below which the Q(done) target is 1.0.",
    )


# =============================================================================
class ContinuousFieldStepOutput(Protocol):
    """Objective-facing output contract for continuous-field ACT rollout steps.

    The bridge adapter duck-types this via
    ``GoaltraceHRMV1BridgeOutput`` (``.task.firing_field``, ``.task``,
    ``.control.q_logits``).
    """

    task: object
    q_logits: Tensor
    done_action: int


# =============================================================================
@runtime_checkable
class ContinuousFieldObjectiveBinding(Protocol):
    """Task-binding protocol for the continuous field objective.

    Implemented by the adapter layer so the objective stays task-agnostic.
    The binding owns field extraction from steps and field-quality computation;
    the objective owns loss math and metrics.
    """

    def extract_field(
        self,
        executed_batch: Batch,
        snapshot: CarrySnapshot,
        step_output: ContinuousFieldStepOutput,
    ) -> Tensor:
        """Return the predicted field for one executed step.

        Expected shape ``(B, N)``, values in ``[0, 1]``.
        """

    def extract_target_field(
        self,
        executed_batch: Batch,
        snapshot: CarrySnapshot,
        step_output: ContinuousFieldStepOutput,
    ) -> Tensor:
        """Return the target field for one executed step.

        Expected shape ``(B, N)``, values in ``[0, 1]``.
        """

    def extract_node_mask(
        self,
        executed_batch: Batch,
    ) -> Tensor:
        """Return the valid-node mask for one executed step.

        Expected shape ``(B, N)`` bool.
        """

    def compute_field_quality(
        self,
        pred_field: Tensor,
        target_field: Tensor,
        node_mask: Tensor,
        threshold: float,
    ) -> Tensor:
        """Return per-sample field quality flag for Q(done) supervision.

        Returns a bool tensor of shape ``(B,)`` where ``True`` means
        ``field_mse < threshold``.
        """


# =============================================================================
@dataclass(frozen=True)
class ContinuousFieldLosses(DetachMixin):
    """Bundle of continuous field loss terms (summed over batch)."""

    loss_field_sum: Tensor
    loss_q_done_sum: Tensor
    loss_q_continue_sum: Optional[Tensor]

    @property
    def loss_sum(self) -> Tensor:
        """Return the summed field loss for the step."""
        return self.loss_field_sum

    @property
    def total(self) -> Tensor:
        """Total scalar loss for the step."""
        q_continue_loss_sum = self.loss_q_continue_sum
        if q_continue_loss_sum is None:
            q_continue_loss_sum = torch.tensor(0.0, device=self.loss_sum.device)
        return self.loss_sum + 0.5 * self.loss_q_done_sum


# =============================================================================
@dataclass(frozen=True)
class ContinuousFieldTerms:
    """Per-example field loss terms scored for one executed step."""

    loss_field: Tensor
    loss_q_done: Tensor


# =============================================================================
@dataclass
class ContinuousFieldContext:
    """Shared objective-scoring context resolved once per ACT step."""

    executed_batch: Batch
    snapshot: CarrySnapshot
    outputs: ContinuousFieldStepOutput
    pred_field: Tensor
    target_field: Tensor
    node_mask: Tensor
    field_quality: Tensor
    target_q: Tensor | None = None

    terms: ContinuousFieldTerms | None = field(default=None, init=False)
    losses: ContinuousFieldLosses | None = field(default=None, init=False)
    metrics: StepMetrics | None = field(default=None, init=False)
    signals: dict[str, Tensor] = field(default_factory=dict, init=False)


# =============================================================================
@dataclass(frozen=True)
class ContinuousFieldObjectiveStep:
    """A single rollout/loss step produced by :class:`ContinuousFieldObjective`."""

    losses: ContinuousFieldLosses
    metrics: StepMetrics
    outputs: Optional[ContinuousFieldStepOutput] = None
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
    def step_output(self) -> Optional[ContinuousFieldStepOutput]:
        """Canonical name for the underlying controller/backbone step output."""
        return self.outputs


# =============================================================================
class ContinuousFieldObjective(BaseObjective[ContinuousFieldObjectiveConfig]):
    """Continuous field objective scored over executed rollout chunks.

    Computes masked MSE on the firing field and BCE(Q(done)) based on
    field quality.  Designed to slot into ``ACTSupervisedModule`` via
    ``ACTSupervisedBindings.objective_cls``.
    """

    def __init__(
        self,
        config: ContinuousFieldObjectiveConfig,
        task_binding: ContinuousFieldObjectiveBinding,
    ) -> None:
        """Create a continuous field objective from its config and task binding.

        Args:
            config: Field objective configuration.
            task_binding: Task-specific binding implementing
                :class:`ContinuousFieldObjectiveBinding`.

        Raises:
            TypeError: If *task_binding* does not implement
                :class:`ContinuousFieldObjectiveBinding`.
        """
        super().__init__(config=config)
        if not isinstance(task_binding, ContinuousFieldObjectiveBinding):
            raise TypeError(
                f"ContinuousFieldObjective requires a task_binding that "
                f"implements ContinuousFieldObjectiveBinding, "
                f"got {type(task_binding).__name__}."
            )
        self._task_binding = task_binding

    @property
    def _field_loss_fn(self) -> str:
        """Return the configured field loss function name."""
        loss = self.config.field_loss
        if loss != "mse":
            raise ValueError(f"Unsupported field_loss: {loss!r}.")
        return loss

    def evaluate_step(
        self,
        record: StepRecord,
        controller: Any | None = None,
        td_target: bool = False,
        target_q: Tensor | None = None,
        target_backbone: Any | None = None,
        **options: Any,
    ) -> ContinuousFieldObjectiveStep:
        """Score one ACT rollout step for continuous field prediction."""
        _ = controller, td_target, target_backbone, options, target_q

        context = self.build_context(record)
        context.terms = self.compute_terms(context)
        context.losses = self.compute_losses(context)
        context.metrics = self.evaluate_metrics(context)
        context.signals = self.compute_signals(context)

        return self.build_output(context)

    def build_context(
        self,
        record: StepRecord,
        *,
        target_q: Tensor | None = None,
        **_: Any,
    ) -> ContinuousFieldContext:
        """Resolve field predictions, targets, and quality for one ACT step."""
        outputs = cast(ContinuousFieldStepOutput, record.outputs)

        pred_field = self._task_binding.extract_field(
            executed_batch=record.batch,
            snapshot=record.carry,
            step_output=outputs,
        )
        target_field = self._task_binding.extract_target_field(
            executed_batch=record.batch,
            snapshot=record.carry,
            step_output=outputs,
        )
        node_mask = self._task_binding.extract_node_mask(record.batch)
        field_quality = self._task_binding.compute_field_quality(
            pred_field=pred_field,
            target_field=target_field,
            node_mask=node_mask,
            threshold=self.config.field_done_threshold,
        )

        return ContinuousFieldContext(
            executed_batch=record.batch,
            snapshot=record.carry,
            outputs=outputs,
            pred_field=pred_field,
            target_field=target_field,
            node_mask=node_mask,
            field_quality=field_quality,
            target_q=target_q,
        )

    def compute_terms(
        self,
        context: ContinuousFieldContext,
        **_: Any,
    ) -> ContinuousFieldTerms:
        """Score per-example field loss terms for one executed step."""
        diff = context.pred_field - context.target_field
        squared = diff**2
        valid_count = context.node_mask.sum(dim=1).clamp(min=1)
        loss_field = (squared * context.node_mask).sum(dim=1) / valid_count

        outputs = context.outputs
        done_logit = outputs.q_logits[..., outputs.done_action]
        done_target = context.field_quality.to(done_logit.dtype)
        loss_q_done = F.binary_cross_entropy_with_logits(
            input=done_logit,
            target=done_target,
            reduction="none",
        )

        return ContinuousFieldTerms(
            loss_field=loss_field,
            loss_q_done=loss_q_done,
        )

    def compute_losses(
        self,
        context: ContinuousFieldContext,
    ) -> ContinuousFieldLosses:
        """Aggregate per-example terms into summed losses."""
        terms = context.terms
        if terms is None:
            raise RuntimeError("compute_losses requires compute_terms first.")

        return ContinuousFieldLosses(
            loss_field_sum=terms.loss_field.sum(),
            loss_q_done_sum=terms.loss_q_done.sum(),
            loss_q_continue_sum=None,
        )

    def evaluate_metrics(
        self,
        context: ContinuousFieldContext,
    ) -> StepMetrics:
        """Build step metrics from precomputed losses and context."""
        losses = context.losses
        terms = context.terms

        if losses is None or terms is None:
            B = 1
            extras = {}
        else:
            B = max(terms.loss_field.shape[0], 1)
            extras = {
                "field_mse": RatioStat(
                    numerator_sum=losses.loss_field_sum.detach(),
                    denominator_sum=torch.tensor(float(B)),
                ),
                "q_done_accuracy": RatioStat(
                    numerator_sum=(
                        (
                            context.outputs.q_logits[
                                ..., context.outputs.done_action
                            ]
                            > 0.0
                        )
                        == context.field_quality
                    )
                    .sum()
                    .float()
                    .detach(),
                    denominator_sum=torch.tensor(float(B)),
                ),
            }

        return StepMetrics(
            episode=RolloutAgg(
                completed_count=torch.tensor(0.0),
                eligible_count=torch.tensor(0.0),
                accuracy_sum=torch.tensor(0.0),
                exact_sum=torch.tensor(0.0),
                steps_sum=torch.tensor(0.0),
            ),
            episode_tokens=TokenAgg(
                token_correct_sum=torch.tensor(0.0),
                token_count_sum=torch.tensor(0.0),
            ),
            step=TransitionAgg(
                evaluated_count=torch.tensor(float(B)),
                eligible_count=torch.tensor(float(B)),
                accuracy_sum=torch.tensor(0.0),
                exact_sum=torch.tensor(0.0),
                steps_sum=torch.tensor(0.0),
            ),
            step_tokens=TokenAgg(
                token_correct_sum=torch.tensor(0.0),
                token_count_sum=torch.tensor(0.0),
            ),
            extras=extras,
        )

    def compute_signals(
        self,
        context: ContinuousFieldContext,
    ) -> dict[str, Tensor]:
        """Compute scalar telemetry signals for the step."""
        losses = context.losses
        if losses is None:
            return {}

        outputs = context.outputs
        done_logit = outputs.q_logits[..., outputs.done_action]
        non_done = [
            i
            for i in range(outputs.q_logits.shape[-1])
            if i != outputs.done_action
        ]
        continue_logit = outputs.q_logits[..., non_done].max(dim=-1).values

        return {
            LOSS_Q_DONE: losses.loss_q_done_sum.detach(),
            HALT_LOGIT_MEAN: done_logit.detach().mean(),
            CONTINUE_LOGIT_MEAN: continue_logit.detach().mean(),
        }

    def build_output(
        self,
        context: ContinuousFieldContext,
    ) -> ContinuousFieldObjectiveStep:
        """Package the scored step into a :class:`ContinuousFieldObjectiveStep`."""
        losses = context.losses
        metrics = context.metrics

        if losses is None:
            raise RuntimeError("build_output requires compute_losses first.")
        if metrics is None:
            metrics = StepMetrics()

        return ContinuousFieldObjectiveStep(
            losses=losses,
            metrics=metrics,
            outputs=context.outputs,
            target_q=context.target_q,
            signals=context.signals,
        )


# =============================================================================
__all__ = [
    "ContinuousFieldLosses",
    "ContinuousFieldObjective",
    "ContinuousFieldObjectiveBinding",
    "ContinuousFieldObjectiveConfig",
    "ContinuousFieldObjectiveStep",
    "ContinuousFieldTerms",
]
