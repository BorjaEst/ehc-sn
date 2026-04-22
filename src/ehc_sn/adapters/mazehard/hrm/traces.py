"""MazeHard HRM-family ACT trace fields.

These fields bind MazeHard task semantics to HRM-family-specific ACT output
surfaces.  They live here — in the shared MazeHard+HRM bridge namespace —
because they read from ``backbone_output.task.task_logits``, which is a
MazeHard+HRM-specific surface.  Generic trace infrastructure must stay
model-agnostic.

Usage
-----
    from ehc_sn.adapters.maze_hard.bridges.hrm.traces import MAZE_HARD_HRM_ACT_TRACE_FIELDS
    self.trace_specs = build_trace_spec("act", extra_fields=MAZE_HARD_HRM_ACT_TRACE_FIELDS)
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


class _MazeHardHRMACTTraceOutputs(Protocol):
    """Minimal raw ACTStepOutput surface consumed by MazeHard+HRM trace getters."""

    class _Backbone(Protocol):
        class _Task(Protocol):
            task_logits: Tensor

        task: _Task

    backbone_output: _Backbone


class _MazeHardHRMACTTraceContext(Protocol):
    """Trace context expected by MazeHard+HRM ACT trace fields."""

    outputs: _MazeHardHRMACTTraceOutputs


# =================================================================================================
# Getter function
# =================================================================================================


def _get_maze_hard_solution_overlay(ctx: _MazeHardHRMACTTraceContext) -> TraceValue:
    """Binary mask: 1 where the model predicts the MazeHard solution-path token.

    Reads ``ctx.outputs.backbone_output.task.task_logits`` so the
    generic trace infrastructure is not coupled to MazeHard task shapes.
    """
    task_logits: Tensor = ctx.outputs.backbone_output.task.task_logits  # (B, S, vocab)
    pred = torch.argmax(task_logits.detach(), dim=-1)  # (B, S)
    return (pred == O_ID).to(torch.uint8)


# =================================================================================================
# Named field objects
# =================================================================================================

MAZE_HARD_HRM_TRACE_SOLUTION_OVERLAY = TraceField(
    name="pred/solution_overlay",
    get=_get_maze_hard_solution_overlay,
)

MAZE_HARD_HRM_ACT_TRACE_FIELDS: tuple[TraceField, ...] = (MAZE_HARD_HRM_TRACE_SOLUTION_OVERLAY,)


# =================================================================================================
__all__ = [
    "MAZE_HARD_HRM_ACT_TRACE_FIELDS",
    "MAZE_HARD_HRM_TRACE_SOLUTION_OVERLAY",
]
