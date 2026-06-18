"""Goaltrace producer-side trace supplement API.

This module owns:

- :class:`GoaltraceEvaluationSourceContext` — typed, frozen provider context for
  Goaltrace evaluation cases.
- :class:`GoaltraceTraceSupplements` — canonical supplement content ready to attach.
- :func:`build_goaltrace_trace_supplements` — constructs supplements from source context.
- :func:`apply_goaltrace_trace_supplements` — attaches supplement data to a
  :class:`~ehc_sn.traces.trace_tree.TraceTree`.

Goaltrace does not currently require spatial geometry supplements (no rate-map
or world-geometry enrichment is needed for existing figure consumers).  The
build/apply functions are explicit no-ops today, documented as the canonical
task-local supplement seam for future work.

The surface mirrors the MazeHard supplement pattern exactly so that Lightning
families can wire Goaltrace through the same ``_maybe_apply_X_supplements``
pattern if needed in the future without changing the evaluation contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ehc_sn.traces.trace_tree import TraceTree


# =============================================================================
@dataclass(frozen=True)
class GoaltraceEvaluationSourceContext:
    """Typed, frozen provider-side context for a Goaltrace evaluation case batch.

    Attributes:
        task_family: Always ``"goaltrace"``. Used as a discriminant for isinstance checks.
        dataset_path: Absolute path to the processed Goaltrace task corpus root.
        split: Dataset split the samples belong to (e.g. ``"val"``).
        sample_ids: Ordered sample ids in this batch.
    """

    task_family: str
    dataset_path: Path
    split: str
    sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.task_family != "goaltrace":
            raise ValueError(
                "GoaltraceEvaluationSourceContext.task_family must be 'goaltrace', "
                f"got {self.task_family!r}"
            )
        if not self.sample_ids:
            raise ValueError(
                "GoaltraceEvaluationSourceContext.sample_ids must not be empty"
            )


# =============================================================================
@dataclass(frozen=True)
class GoaltraceTraceSupplements:
    """Canonical supplement content for Goaltrace traces.

    Goaltrace does not currently produce spatial geometry supplements.
    This dataclass is intentionally empty and serves as the typed seam for
    future work.
    """


# =============================================================================
def build_goaltrace_trace_supplements(
    source_context: GoaltraceEvaluationSourceContext,
    trace_length: int,
) -> GoaltraceTraceSupplements:
    """Build Goaltrace trace supplements from a typed source context.

    Currently a no-op: Goaltrace does not require spatial geometry supplements
    for existing figure consumers.  This function is the canonical task-local
    supplement seam for future enrichment.

    Args:
        source_context: Typed Goaltrace evaluation source context.
        trace_length: Number of time steps to cover.

    Returns:
        Empty :class:`GoaltraceTraceSupplements`.
    """
    return GoaltraceTraceSupplements()


# =============================================================================
def apply_goaltrace_trace_supplements(
    trace: TraceTree,
    supplements: GoaltraceTraceSupplements,
) -> None:
    """Attach Goaltrace supplement content to *trace* in-place.

    Currently a no-op: no supplement keys are attached because Goaltrace does
    not produce spatial geometry supplements for existing figure consumers.

    Args:
        trace: The :class:`~ehc_sn.traces.trace_tree.TraceTree` to modify in-place.
        supplements: The supplement content to attach.
    """


# =============================================================================
__all__ = [
    "GoaltraceEvaluationSourceContext",
    "GoaltraceTraceSupplements",
    "build_goaltrace_trace_supplements",
    "apply_goaltrace_trace_supplements",
]
