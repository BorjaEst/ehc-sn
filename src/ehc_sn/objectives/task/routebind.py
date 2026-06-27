"""Routebind trajectory-bootstrap evaluator for ACT training.

Trains only the trajectory field component of Routebind.  Waypoint,
direction, and observation outputs are decoded but receive no loss
gradient in the bootstrap evaluator.

The evaluator operates on the complete ``RoutebindSupervision`` (with
canonical corpus key names) but consumes only ``target_trajectory``
and ``spatial_mask`` for loss and completion.

Ownership boundary:
- This evaluator knows about ``RoutebindTaskOutput`` and
  ``RoutebindSupervision``.
- It does not know about HRM, ACT controllers, rollouts, or Lightning.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from pydantic import BaseModel, Field
from torch import Tensor

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
from ehc_sn.tasks.routebind.contracts import RoutebindTaskOutput
from ehc_sn.tasks.routebind.supervision import RoutebindSupervision


class RoutebindTaskEvaluatorConfig(BaseModel, extra="forbid"):
    """Configuration for Routebind trajectory-bootstrap evaluation.

    Attributes:
        trajectory_loss: Trajectory field regression loss variant.
        acceptable_error: MSE at which the halt sigmoid midpoint is centered.
        halt_temperature: Sigmoid temperature for the completion target.
    """

    trajectory_loss: Literal["mse"] = "mse"
    acceptable_error: float = 0.1
    halt_temperature: float = 0.1


class RoutebindTrajectoryBootstrapEvaluator:
    """Bootstrap evaluator training only the routebind trajectory field.

    Completion target is derived from trajectory field quality via the
    same ``FieldQualityHaltTarget`` logic used by Goaltrace.
    """

    def __init__(
        self,
        config: RoutebindTaskEvaluatorConfig | None = None,
        field_objective: FieldRegressionObjective | None = None,
        completion_policy: HaltTargetBuilder | None = None,
    ) -> None:
        self._field = field_objective or FieldRegressionObjective()
        c = config or RoutebindTaskEvaluatorConfig()
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
        task_output: RoutebindTaskOutput,
        supervision: RoutebindSupervision,
    ) -> TaskStepEvaluation:
        """Evaluate one step of routebind trajectory field predictions.

        Args:
            task_output: Must have ``trajectory_field`` shape ``(B, S)``.
            supervision: Must have ``target_trajectory`` and
                ``spatial_mask``.

        Returns:
            TaskStepEvaluation with masked MSE on the trajectory field
            and field-quality completion target.
        """
        result = self._field(
            FieldObjectiveInput(
                prediction=task_output.trajectory_field,
                target=supervision.target_trajectory,
                mask=supervision.spatial_mask,
            )
        )
        per_sample = result.terms["field_per_sample"]
        per_element_sum = result.terms["field_per_element"].sum()
        loss_sum = per_sample.sum()
        valid_count = supervision.spatial_mask.sum()

        if valid_count.item() == 0:
            raise ValueError(
                "Field evaluation received no valid spatial elements"
            )

        completion_target = self._completion.build(
            pred_field=task_output.trajectory_field,
            target_field=supervision.target_trajectory,
            node_mask=supervision.spatial_mask,
        )

        return TaskStepEvaluation(
            task_loss_sum=loss_sum,
            task_loss_count=valid_count,
            completion_target=completion_target,
            continuation_target=1.0 - completion_target,
            metrics={
                "trajectory_mse_sum": loss_sum.detach(),
                "trajectory_mse_count": valid_count.detach(),
            },
            task_extras={
                "field_mse": RatioStat(
                    numerator_sum=per_element_sum.detach(),
                    denominator_sum=valid_count.detach(),
                ),
            },
        )


__all__ = ["RoutebindTrajectoryBootstrapEvaluator"]
