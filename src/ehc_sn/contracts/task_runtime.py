"""Runtime-backed value-control controller-to-task boundary.

This module defines the canonical runtime protocol that decouples
value-control controllers from task-specific reward, termination, and
observation dynamics.

TaskRuntime is the architectural successor to :class:`TaskStepEvaluator`:
it owns reset, partial-reset, and step semantics with full runtime state
and observation construction.  TaskStepEvaluator remains for legacy
static-evaluation consumers; new code should implement TaskRuntime.

Canonical import path::

    from ehc_sn.contracts.task_runtime import (
        TaskRuntime, RuntimeReset, StepFeedback,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Mapping, Protocol, TypeVar

from torch import Tensor

from ehc_sn.types import Batch

# =============================================================================
RuntimeState = TypeVar("RuntimeState")


# =============================================================================
@dataclass(frozen=True)
class RuntimeReset(Generic[RuntimeState]):
    """Result of initialising or resetting runtime slots.

    Attributes:
        observation: Model input for the current step, shape ``(B, ...)``.
            Covers ALL batch slots — reset slots get fresh observations from
            the incoming batch, continuing slots derive observations from the
            preserved runtime state.
        state: Updated runtime state object.
        metrics: Optional per-slot audit tensors.
    """

    observation: Batch
    state: RuntimeState
    metrics: Mapping[str, Tensor] = field(default_factory=dict)


# =============================================================================
@dataclass(frozen=True)
class StepFeedback(Generic[RuntimeState]):
    """Result of one runtime step.

    Attributes:
        reward: Scalar reward per slot of shape ``(B, 1)``, ``float32``.
        terminated: Per-slot termination flag of shape ``(B,)``, ``bool``.
        truncated: Per-slot truncation flag of shape ``(B,)``, ``bool``.
        next_observation: Model input for the next step, shape ``(B, ...)``.
            For static-instance tasks this is the same observation each step.
            For dynamic tasks it reflects the updated task state.
        next_state: Updated runtime state after the step.
        metrics: Optional per-step audit tensors.
    """

    reward: Tensor  # (B, 1) float32
    terminated: Tensor  # (B,) bool
    truncated: Tensor  # (B,) bool
    next_observation: Batch
    next_state: RuntimeState
    metrics: Mapping[str, Tensor] = field(default_factory=dict)


# =============================================================================
class TaskRuntime(Protocol[RuntimeState]):
    """Task-owned runtime boundary for value-control controllers.

    Owns:
        - Observation construction from task data.
        - Task-state transitions (may be degenerate for static tasks).
        - Reward semantics (correctness, progress, cost).
        - Termination and truncation rules.
        - Partial reset (halted slots get new episodes; continuing slots
          persist).

    Does **not** own:
        - Model forward pass.
        - Action selection.
        - Loss computation.
        - Optimizer stepping.

    The controller interacts with the runtime only through ``reset``,
    ``reset_slots``, and ``step``.  No task-specific types cross this
    boundary.
    """

    def reset(self, batch: Batch) -> RuntimeReset[RuntimeState]:
        """Initialise runtime for a full batch of new episodes.

        Args:
            batch: Model-input mapping from the dataloader.

        Returns:
            :class:`RuntimeReset` with the initial observation and runtime state.
        """
        ...

    def reset_slots(
        self,
        reset_mask: Tensor,  # (B,) bool — True = reset this slot
        batch: Batch,  # fresh dataloader batch (used for reset slots only)
        state: RuntimeState,  # current runtime state (continuing slots preserved)
    ) -> RuntimeReset[RuntimeState]:
        """Reset only halted slots; continuing slots keep their runtime state.

        The returned ``observation`` covers ALL batch slots — reset slots get
        fresh observations from *batch*, continuing slots derive observations
        from the preserved portion of *state*.

        Args:
            reset_mask: Boolean mask of shape ``(B,)``; ``True`` means reset the slot.
            batch: Incoming batch used to initialise reset slots.
            state: Current runtime state.  Continuing slots must be preserved.

        Returns:
            :class:`RuntimeReset` with observations across all B slots and the
            updated runtime state.
        """
        ...

    def step(
        self,
        state: RuntimeState,
        task_output: object,
        action: Tensor,  # (B,) int64
        steps: Tensor,  # (B,) int32 — step counter AFTER advancing
    ) -> StepFeedback[RuntimeState]:
        """Advance the runtime by one deliberation step.

        The runtime owns interpreting what ``action`` means (halt, continue,
        other), computing reward from ``task_output`` correctness, determining
        termination/truncation, and producing the next observation.

        Args:
            state: Current runtime state from the previous step or reset.
            task_output: Opaque backbone output; the runtime casts it internally.
            action: Sampled action indices of shape ``(B,)``.
            steps: Per-slot step counters *after* advancing of shape ``(B,)``.

        Returns:
            :class:`StepFeedback` with reward, termination flags, next observation,
            and updated runtime state.
        """
        ...


# =============================================================================
__all__ = [
    "RuntimeReset",
    "RuntimeState",
    "StepFeedback",
    "TaskRuntime",
]
