# SPDX-License-Identifier: MIT
"""Task-owned step-evaluation contract for fixed-instance RL feedback.

Implement :class:`TaskStepEvaluator` in ``tasks/<task>/evaluators/step.py``
when a reasoning task should receive RL feedback: the controller calls
:meth:`TaskStepEvaluator.evaluate_step` after each model step to obtain
reward, termination, truncation, optional metrics, and optional next runtime
state.

Use :class:`~ehc_sn.contracts.task_environment.TaskEnvironmentAdapter` instead
when the task exposes an interactive environment whose observations, rewards,
and termination arise from stepping external state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

from torch import Tensor

from ehc_sn.types import Batch


# =============================================================================
@dataclass(frozen=True)
class StepEvaluation:
    """Task-side evaluation of one controller step.

    Attributes:
        reward: Task-finalized reward tensor of shape ``(B, 1)``.
        terminated: Per-slot episode termination flag of shape ``(B,)``.
        truncated: Per-slot episode truncation flag of shape ``(B,)``.
        metrics: Optional per-step audit tensors (e.g. step scores,
            accuracy components).  Mutable by convention; frozen dataclass
            only prevents field reassignment.
        next_runtime_state: Updated lightweight runtime carry, or ``None`` if
            the evaluator is stateless.
    """

    reward: Tensor  # (B, 1)
    terminated: Tensor  # (B,)
    truncated: Tensor  # (B,)
    metrics: Mapping[str, Tensor] = field(default_factory=dict)
    next_runtime_state: object | None = None


# =============================================================================
class TaskStepEvaluator(Protocol):
    """Task-owned seam for evaluating one deliberation controller step.

    Implement this for fixed-instance reasoning tasks where the controller
    performs internal recurrent steps over a batch and the task can score
    each proposed action or output against known structure, targets, or
    validators.

    This seam owns reward, termination, truncation, optional metrics, and
    optional task runtime-state updates.

    Use :class:`~ehc_sn.contracts.task_environment.TaskEnvironmentAdapter`
    together with an ``EnvBase`` implementation for closed-loop online
    environment tasks.
    """

    def evaluate_step(
        self,
        data: Batch,
        task_output: object,
        action: Tensor,
        steps: Tensor,
        runtime_state: object | None,
    ) -> StepEvaluation:
        """Produce reward and termination flags for the current step.

        Args:
            data: Current per-slot batch.
            task_output: Task-side model output from ``backbone_output.task``.
            action: Sampled action tensor of shape ``(B,)``.
            steps: Per-slot step counters of shape ``(B,)`` after advancing.
            runtime_state: Lightweight runtime carry from the previous step.

        Returns:
            :class:`StepEvaluation` with reward, termination flags,
            optional metrics, and the updated runtime state.
        """
        ...


# =============================================================================
__all__ = [
    "StepEvaluation",
    "TaskStepEvaluator",
]
