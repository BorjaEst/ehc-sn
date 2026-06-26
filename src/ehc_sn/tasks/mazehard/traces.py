"""MazeHard producer-side trace supplement API.

This module owns:

- :class:`MazeHardEvaluationSourceContext` — typed, frozen provider context for MazeHard cases.
- :class:`MazeHardTraceSupplements` — canonical supplement content ready to attach.
- :func:`build_mazehard_trace_supplements` — constructs supplements from source context.
- :func:`apply_mazehard_trace_supplements` — attaches supplement data to a :class:`TraceTree`.

MazeHard does not currently require spatial geometry supplements (no rate-map or
world-geometry enrichment is needed for existing figure consumers).  The build/apply
functions are explicit no-ops today, documented as the canonical task-local supplement
seam for future work.

The surface mirrors the Arena supplement pattern exactly so that Lightning families
can wire MazeHard through the same :func:`_maybe_apply_X_supplements` pattern if
needed in the future without changing the evaluation contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from ehc_sn.tasks.mazehard.runtime import PATH_ID
from ehc_sn.traces.observer import TraceField, TraceValue
from ehc_sn.traces.trace_tree import TraceTree


# =============================================================================
@dataclass(frozen=True)
class MazeHardEvaluationSourceContext:
    """Typed, frozen provider-side context for a MazeHard evaluation case batch.

    Attributes:
        task_family: Always ``"mazehard"``. Used as a discriminant for isinstance checks.
        dataset_path: Absolute path to the processed MazeHard task corpus root.
        split: Dataset split the samples belong to (e.g. ``"val"``).
        sample_ids: Ordered sample ids in this batch.
    """

    task_family: str
    dataset_path: Path
    split: str
    sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.task_family != "mazehard":
            raise ValueError(
                "MazeHardEvaluationSourceContext.task_family must be 'mazehard', "
                f"got {self.task_family!r}"
            )
        if not self.sample_ids:
            raise ValueError(
                "MazeHardEvaluationSourceContext.sample_ids must not be empty",
            )


# =============================================================================
@dataclass(frozen=True)
class MazeHardTraceSupplements:
    """Canonical supplement content for MazeHard traces.

    MazeHard does not currently produce spatial geometry supplements.
    This dataclass is intentionally empty and serves as the typed seam for
    future work.
    """


# =============================================================================
def build_mazehard_trace_supplements(  # --------------------------------------
    source_context: MazeHardEvaluationSourceContext,
    trace_length: int,
) -> MazeHardTraceSupplements:
    """Build MazeHard trace supplements from a typed source context.

    Currently a no-op: MazeHard does not require spatial geometry supplements for
    existing figure consumers.  This function is the canonical task-local supplement
    seam for future enrichment.

    Args:
        source_context: Typed MazeHard evaluation source context.
        trace_length: Number of time steps to cover.

    Returns:
        Empty :class:`MazeHardTraceSupplements`.
    """
    return MazeHardTraceSupplements()


# =============================================================================
def apply_mazehard_trace_supplements(  # --------------------------------------
    trace: TraceTree,
    supplements: MazeHardTraceSupplements,
) -> None:
    """Attach MazeHard supplement content to *trace* in-place.

    Currently a no-op: no supplement keys are attached because MazeHard does not
    produce spatial geometry supplements for existing figure consumers.

    Args:
        trace: The :class:`~ehc_sn.traces.trace_tree.TraceTree` to modify in-place.
        supplements: The supplement content to attach.
    """


# =============================================================================
# Task-level trace fields — shared vocabulary names with MazeHard-specific getters
# =============================================================================


def _get_pred_solution_overlay(ctx) -> TraceValue:
    """Decoded solution overlay from model task logits.

    Reads ``task_logits`` from the controller output surface.
    Supports ACT-style (``ctx.outputs.task.task_logits``) and
    actor-critic-style (``ctx.outputs.task_output.task_logits``).
    """
    outputs = getattr(ctx, "outputs", None)
    if outputs is None:
        return None

    # ACT path: outputs.task.task_logits
    task = getattr(outputs, "task", None)
    if task is not None:
        task_logits: Tensor | None = getattr(task, "task_logits", None)
        if task_logits is not None:
            pred = torch.argmax(task_logits.detach(), dim=-1)
            return (pred == PATH_ID).to(torch.uint8).cpu()

    # Actor-critic fallback: outputs.task_output.task_logits
    task_output = getattr(outputs, "task_output", None)
    if task_output is not None:
        task_logits: Tensor | None = getattr(task_output, "task_logits", None)
        if task_logits is not None:
            pred = torch.argmax(task_logits.detach(), dim=-1)
            return (pred == PATH_ID).to(torch.uint8).cpu()

    return None


def _get_target_solution_overlay(ctx) -> TraceValue:
    """Oracle solution overlay from the batch labels.

    Labels are sample-constant and held in ``ctx.carry.data["labels"]``
    (the carry's slot-data mirror of the batch).  Extracted once as
    metadata.
    """
    carry = getattr(ctx, "carry", None)
    if carry is None:
        return None
    data = getattr(carry, "data", None)
    if data is None:
        return None
    labels = data.get("labels")
    if labels is None:
        return None
    return (labels.detach() == PATH_ID).to(torch.uint8).cpu()


def trace_task_fields() -> tuple[TraceField, ...]:
    """Return TraceField objects for shared vocabulary names with MazeHard getters."""
    return (
        TraceField(
            name="pred/solution_overlay",
            get=_get_pred_solution_overlay,
            storage="dense",
        ),
        TraceField(
            name="target/solution_overlay",
            get=_get_target_solution_overlay,
            storage="meta",
        ),
    )


# =============================================================================
__all__ = [
    "MazeHardEvaluationSourceContext",
    "MazeHardTraceSupplements",
    "build_mazehard_trace_supplements",
    "apply_mazehard_trace_supplements",
    "trace_task_fields",
]
