"""MazeHard ACT adapter-owned trace fields.

These fields are intentionally separate from the generic trace vocabulary in
:mod:`ehc_sn.metrics.traces` because they read from ``backbone_output.task``,
which is a MazeHard-specific surface.  Generic trace infrastructure must stay
model-agnostic.

Usage
-----
    from ehc_sn.adapters.maze_hard.traces import MAZE_HARD_ACT_TRACE_FIELDS
    self.trace_specs = build_trace_spec("act", extra_fields=MAZE_HARD_ACT_TRACE_FIELDS)
"""

from __future__ import annotations

from typing import Any, Protocol

import torch
from torch import Tensor

from ehc_sn.data.schema import O_ID
from ehc_sn.traces import TraceField, TraceValue


# =================================================================================================
class _MazeHardACTTraceOutputs(Protocol):
    """Minimal raw ACTStepOutput surface consumed by MazeHard trace getters."""

    class _Backbone(Protocol):
        class _Task(Protocol):
            task_logits: Tensor

        task: _Task

    backbone_output: _Backbone


class _MazeHardACTTraceContext(Protocol):
    """Trace context expected by MazeHard ACT trace fields."""

    outputs: _MazeHardACTTraceOutputs


# =================================================================================================
def _get_maze_hard_solution_overlay(ctx: Any) -> TraceValue:
    """Binary mask: 1 where the model predicts the MazeHard solution-path token.

    Reads ``ctx.outputs.backbone_output.task.task_logits`` so the
    generic trace infrastructure is not coupled to MazeHard task shapes.
    """
    task_logits: Tensor = ctx.outputs.backbone_output.task.task_logits  # (B, S, vocab)
    pred = torch.argmax(task_logits.detach(), dim=-1)  # (B, S)
    return (pred == O_ID).to(torch.uint8)


MAZE_HARD_TRACE_SOLUTION_OVERLAY = TraceField(
    name="pred/solution_overlay",
    get=_get_maze_hard_solution_overlay,
)

MAZE_HARD_ACT_TRACE_FIELDS: tuple[TraceField, ...] = (MAZE_HARD_TRACE_SOLUTION_OVERLAY,)


# =================================================================================================
__all__ = [
    "MAZE_HARD_ACT_TRACE_FIELDS",
    "MAZE_HARD_TRACE_SOLUTION_OVERLAY",
]
