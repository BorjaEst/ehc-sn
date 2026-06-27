"""Task-agnostic contracts for task-step evaluation and ACT control.

The types in this module are the unified boundary between:

- task-owned evaluators (which produce ``TaskStepEvaluation``),
- controller-produced control predictions (``ACTControlPrediction``),
- the generic ACT scorer (which consumes both).

No task-specific attribute names, tensor shapes beyond ``(B,)``, or
modality-specific fields appear here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Protocol, TypeVar

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ehc_sn.metrics.token import AccuracyStats

# TODO: Relocate RatioStat to a neutral contracts module
# (e.g. src/ehc_sn/contracts/metrics.py) to avoid coupling
# objective contracts to the full step-metrics transport layer.
from ehc_sn.metrics.step_metrics import RatioStat  # noqa: TCH

TaskOutputT = TypeVar("TaskOutputT", contravariant=True)
SupervisionT = TypeVar("SupervisionT", contravariant=True)


def _empty_metrics() -> dict[str, Tensor]:
    """Return an empty metrics dict — default factory for frozen dataclass."""
    return {}


def _empty_task_extras() -> dict[str, RatioStat]:
    """Return an empty task-extras dict — default factory for frozen dataclass."""
    return {}


# =============================================================================
@dataclass(frozen=True)
class TaskStepEvaluation:
    """Task-agnostic evaluation result for one ACT step.

    Attributes:
        task_loss_sum: Scalar sum of task loss over the batch — used for
            back-propagation.
        task_loss_count: Scalar number of elements that contributed to
            *task_loss_sum* — used for normalization.
        completion_target: Per-sample readiness to halt, shape ``(B,)``
            float tensor with values in ``[0, 1]``.  Higher = more ready.
        continuation_target: Optional per-sample continuation signal,
            shape ``(B,)`` or ``None``.
        accuracy_stats: Optional token-level correctness statistics
            produced by ``compute_accuracy_stats``.  When provided, the
            generic ACT scorer uses them to populate per-step token and
            sequence accuracy in ``StepMetrics``.  ``None`` for non-token
            tasks (Goaltrace, Routebind, SeqMaze).
        metrics: Named detached tensors for logging — no task-specific
            dict keys prohibited, but the ACT scorer treats them opaquely.
    """

    task_loss_sum: Tensor
    task_loss_count: Tensor
    completion_target: Tensor
    continuation_target: Tensor | None = None
    accuracy_stats: AccuracyStats | None = None
    metrics: Mapping[str, Tensor] = field(default_factory=_empty_metrics)
    """Raw evaluator diagnostics. Not automatically routed or reduced."""

    task_extras: Mapping[str, RatioStat] = field(
        default_factory=_empty_task_extras
    )
    """Canonical aggregation-ready ratio statistics for inclusion in
    ``StepMetrics.extras``.  Keys must match route-table entries for the
    active metric profile (e.g. ``"field_mse"``, ``"loss_token"``).
    The ACT scorer forwards these structurally — it does not interpret
    individual key names."""

    def __post_init__(self) -> None:
        _validate_completion_target(self.completion_target)


def _validate_completion_target(t: Tensor) -> None:
    """Validate completion_target shape and value range."""
    if t.ndim != 1:
        raise ValueError(
            f"completion_target must be 1-D, got {t.ndim}-D with shape "
            f"{tuple(t.shape)}."
        )
    if t.is_floating_point():
        if (t < 0.0).any() or (t > 1.0).any():
            raise ValueError(
                "completion_target values must be in [0, 1], got "
                f"[{t.min().item():.4f}, {t.max().item():.4f}]."
            )


# =============================================================================
@dataclass(frozen=True)
class ACTControlPrediction:
    """Generic control prediction for one ACT step.

    Extracted from any model-family-specific control output by the
    Lightning orchestration layer, not by the ACT scorer.

    Attributes:
        halt_logit: Raw halt logit per sample, shape ``(B,)``.
        continue_logit: Optional raw continue logit per sample,
            shape ``(B,)`` or ``None``.
    """

    halt_logit: Tensor
    continue_logit: Tensor | None = None

    def __post_init__(self) -> None:
        if self.halt_logit.ndim != 1:
            raise ValueError(
                f"halt_logit must be 1-D, got {self.halt_logit.ndim}-D "
                f"with shape {tuple(self.halt_logit.shape)}."
            )
        if self.continue_logit is not None and self.continue_logit.ndim != 1:
            raise ValueError(
                f"continue_logit must be 1-D or None, got "
                f"{self.continue_logit.ndim}-D with shape "
                f"{tuple(self.continue_logit.shape)}."
            )


# =============================================================================
@dataclass(frozen=True)
class ACTSupervisedScoringInput:
    """Named scoring input for the ACT-supervised scorer.

    Carries a pre-computed task evaluation and a control prediction.
    This is the typed input expected by
    ``ACTSupervisedScorer.evaluate_step(record, *, inputs=...)``.

    Attributes:
        task: Task-agnostic evaluation result for this step.
        control: Generic control prediction for this step.
    """

    task: TaskStepEvaluation
    control: ACTControlPrediction


# =============================================================================
class TaskStepEvaluator(Protocol[TaskOutputT, SupervisionT]):
    """Protocol for task-owned step evaluators.

    A conforming type:

    - Receives one predicted task output and one supervision struct,
    - Returns a task-agnostic evaluation containing only loss sums,
      counts, a completion target, optional continuation target, and
      opaque metrics.

    The ACT scorer does not inspect *task_output* or *supervision* —
    only the returned ``TaskStepEvaluation``.
    """

    def evaluate(
        self,
        *,
        task_output: TaskOutputT,
        supervision: SupervisionT,
    ) -> TaskStepEvaluation:
        """Evaluate one step of task predictions against supervision.

        Args:
            task_output: Task-owned prediction output from the model.
            supervision: Task-owned supervision targets.

        Returns:
            A task-agnostic evaluation result.
        """
        ...


# =============================================================================
__all__ = [
    "ACTControlPrediction",
    "ACTSupervisedScoringInput",
    "TaskStepEvaluation",
    "TaskStepEvaluator",
]
