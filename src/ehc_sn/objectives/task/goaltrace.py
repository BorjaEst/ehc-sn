"""Goaltrace task-step evaluator for ACT training regimes.

Wraps ``FieldRegressionObjective`` and ``FieldQualityHaltTarget`` to
produce a ``TaskStepEvaluation`` suitable for the generic ACT scorer.

Ownership boundary:
- This evaluator knows about ``GoaltraceTaskOutput`` and
  ``GoaltraceFieldSupervision``.
- It does not know about HRM, ACT controllers, rollouts, or Lightning.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn

from ehc_sn.objectives.contracts import (
    RatioStat,
    TaskStepEvaluation,
    TaskStepEvaluator,
)
from ehc_sn.objectives.supervised.field import (
    FieldObjectiveInput,
    FieldRegressionObjective,
)
from ehc_sn.targets.halt import (
    FieldQualityHaltTarget,
    FieldQualityHaltTargetConfig,
    HaltTargetBuilder,
)
from ehc_sn.tasks.goaltrace.contracts import GoaltraceTaskOutput
from ehc_sn.tasks.goaltrace.supervision import GoaltraceFieldSupervision


class GoaltraceTaskEvaluatorConfig(BaseModel, extra="forbid"):
    """Configuration for Goaltrace field task evaluation.

    Attributes:
        field_loss: Field regression loss variant.
        acceptable_error: MSE at which the halt sigmoid midpoint is centered.
        halt_temperature: Sigmoid temperature for the completion target.
    """

    field_loss: Literal["mse"] = "mse"
    acceptable_error: float = 0.1
    halt_temperature: float = 0.1


class GoaltraceFieldEvaluator:
    """Evaluates Goaltrace field predictions for ACT training.

    Reproduces the exact loss and halt-target semantics that the old
    ``ACTSupervisedScorer`` produced for field-modality tasks.
    """

    def __init__(
        self,
        config: GoaltraceTaskEvaluatorConfig | None = None,
        field_objective: FieldRegressionObjective | None = None,
        completion_policy: HaltTargetBuilder | None = None,
    ) -> None:
        self._field = field_objective or FieldRegressionObjective()
        c = config or GoaltraceTaskEvaluatorConfig()
        self._completion = completion_policy or FieldQualityHaltTarget(
            FieldQualityHaltTargetConfig(
                acceptable_error=c.acceptable_error,
                temperature=c.halt_temperature,
            )
        )

    # ── TaskStepEvaluator protocol ──────────────────────────────────────────

    def evaluate(
        self,
        *,
        task_output: GoaltraceTaskOutput,
        supervision: GoaltraceFieldSupervision,
    ) -> TaskStepEvaluation:
        """Evaluate one step of goaltrace field predictions.

        Args:
            task_output: Must have ``firing_field`` shape ``(B, N)``.
            supervision: Must have ``target`` and ``mask``.

        Returns:
            TaskStepEvaluation with single-field MSE loss and field-quality
            completion target.
        """
        result = self._field(
            FieldObjectiveInput(
                prediction=task_output.firing_field,
                target=supervision.target,
                mask=supervision.mask,
            )
        )
        per_sample = result.terms["field_per_sample"]
        per_element_sum = result.terms["field_per_element"].sum()
        loss_sum = per_sample.sum()
        valid_count = supervision.mask.sum()

        if valid_count.item() == 0:
            raise ValueError(
                "Field evaluation received no valid spatial elements"
            )

        completion_target = self._completion.build(
            pred_field=task_output.firing_field,
            target_field=supervision.target,
            node_mask=supervision.mask,
        )

        return TaskStepEvaluation(
            task_loss_sum=loss_sum,
            task_loss_count=valid_count,
            completion_target=completion_target,
            continuation_target=1.0 - completion_target,
            metrics={
                "field_mse_sum": loss_sum.detach(),
                "field_mse_count": valid_count.detach(),
            },
            task_extras={
                "field_mse": RatioStat(
                    numerator_sum=per_element_sum.detach(),
                    denominator_sum=valid_count.detach(),
                ),
            },
        )


__all__ = ["GoaltraceFieldEvaluator"]
