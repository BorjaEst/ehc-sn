"""Deliberation actor-critic controller contracts.

Defines the shared result and finalizer protocol used by deliberation
actor-critic controllers and task-capability bindings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from torch import Tensor

from ehc_sn.types import Batch


# =============================================================================
@dataclass(frozen=True)
class DeliberationStepResult:
    """Per-slot deliberation step result carrying reward and termination signals.

    Attributes:
        reward: Per-slot reward tensor of shape ``(B, 1)`` or ``(B,)``.
        terminated: Per-slot termination flag of shape ``(B,)``. ``True`` when
            the model emits the task-defined halt action.
        truncated: Per-slot truncation flag of shape ``(B,)``. ``True`` when
            ``steps >= task_config.episode_horizon``.
        next_runtime_state: Optional task-owned carry threaded across steps.
    """

    reward: Tensor
    terminated: Tensor
    truncated: Tensor
    next_runtime_state: object | None = None


# =============================================================================
class DeliberationStepFinalizer(Protocol):
    """Protocol implemented by task capabilities that finalize deliberation steps."""

    def finalize_step(
        self,
        data: Batch,
        task_output: object,
        action: Tensor,
        steps: Tensor,
        runtime_state: object | None,
    ) -> DeliberationStepResult: ...


# =============================================================================
__all__ = [
    "DeliberationStepFinalizer",
    "DeliberationStepResult",
]
