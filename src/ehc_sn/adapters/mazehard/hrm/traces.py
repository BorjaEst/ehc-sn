"""MazeHard HRM-family trace extras.

These fields bind MazeHard task semantics to HRM-family-specific output
surfaces. They live here because they read task-facing logits from the adapter
boundary while the generic trace infrastructure stays model-agnostic.

Usage
-----
    from ehc_sn.adapters.mazehard.hrm import (
        MAZE_HARD_HRM_ACTOR_CRITIC_TRACE_FIELDS,
        MAZE_HARD_HRM_ACT_TRACE_FIELDS,
    )
    self.trace_specs = build_trace_spec("act", extra_fields=MAZE_HARD_HRM_ACT_TRACE_FIELDS)
    self.trace_specs = build_trace_spec(
        "actor_critic",
        extra_fields=MAZE_HARD_HRM_ACTOR_CRITIC_TRACE_FIELDS,
    )
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from ehc_sn.data.schema import O_ID
from ehc_sn.traces import TraceField, TraceValue

# =================================================================================================
# Minimal typed context for MazeHard+HRM ACT trace getters
# =================================================================================================


class _MazeHardTaskLogits(Protocol):
    """Minimal task payload exposing MazeHard supervised logits."""

    task_logits: Tensor


class _MazeHardHRMACTTraceOutputs(Protocol):
    """Minimal raw ACTStepOutput surface consumed by MazeHard+HRM trace getters."""

    class _Backbone(Protocol):
        task: _MazeHardTaskLogits

    backbone_output: _Backbone


class _MazeHardHRMACTTraceContext(Protocol):
    """Trace context expected by MazeHard+HRM ACT trace fields."""

    outputs: _MazeHardHRMACTTraceOutputs


class _MazeHardHRMActorCriticTraceOutputs(Protocol):
    """Minimal actor-critic record surface consumed by MazeHard+HRM trace getters."""

    task_output: _MazeHardTaskLogits


class _MazeHardHRMActorCriticTraceContext(Protocol):
    """Trace context expected by MazeHard+HRM actor-critic trace fields."""

    outputs: _MazeHardHRMActorCriticTraceOutputs


# =================================================================================================
# Getter function
# =================================================================================================


def _solution_overlay_from_task_logits(task_logits: Tensor) -> TraceValue:
    """Return a binary overlay for the MazeHard solution-path token."""
    pred = torch.argmax(task_logits.detach(), dim=-1)  # (B, S)
    return (pred == O_ID).to(torch.uint8)


def _get_maze_hard_solution_overlay_act(ctx: _MazeHardHRMACTTraceContext) -> TraceValue:
    """Read MazeHard solution-overlay traces from ACT backbone task logits."""
    return _solution_overlay_from_task_logits(ctx.outputs.backbone_output.task.task_logits)


def _get_maze_hard_solution_overlay_actor_critic(
    ctx: _MazeHardHRMActorCriticTraceContext,
) -> TraceValue:
    """Read MazeHard solution-overlay traces from actor-critic task output logits."""
    return _solution_overlay_from_task_logits(ctx.outputs.task_output.task_logits)


# =================================================================================================
# Named field objects
# =================================================================================================

MAZE_HARD_HRM_TRACE_SOLUTION_OVERLAY = TraceField(
    name="pred/solution_overlay",
    get=_get_maze_hard_solution_overlay_act,
)

MAZE_HARD_HRM_ACT_TRACE_FIELDS: tuple[TraceField, ...] = (MAZE_HARD_HRM_TRACE_SOLUTION_OVERLAY,)

_MAZE_HARD_HRM_TRACE_SOLUTION_OVERLAY_ACTOR_CRITIC = TraceField(
    name="pred/solution_overlay",
    get=_get_maze_hard_solution_overlay_actor_critic,
)

MAZE_HARD_HRM_ACTOR_CRITIC_TRACE_FIELDS: tuple[TraceField, ...] = (_MAZE_HARD_HRM_TRACE_SOLUTION_OVERLAY_ACTOR_CRITIC,)


# =================================================================================================
__all__ = [
    "MAZE_HARD_HRM_ACTOR_CRITIC_TRACE_FIELDS",
    "MAZE_HARD_HRM_ACT_TRACE_FIELDS",
    "MAZE_HARD_HRM_TRACE_SOLUTION_OVERLAY",
]
