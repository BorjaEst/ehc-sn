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

from ehc_sn.traces.trace_tree import TraceTree


# =================================================================================================
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
                f"MazeHardEvaluationSourceContext.task_family must be 'mazehard', got {self.task_family!r}"
            )
        if not self.sample_ids:
            raise ValueError("MazeHardEvaluationSourceContext.sample_ids must not be empty")


# =================================================================================================
@dataclass(frozen=True)
class MazeHardTraceSupplements:
    """Canonical supplement content for MazeHard traces.

    MazeHard does not currently produce spatial geometry supplements.
    This dataclass is intentionally empty and serves as the typed seam for future work.
    """


# =================================================================================================
def build_mazehard_trace_supplements(
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


# =================================================================================================
def apply_mazehard_trace_supplements(
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


# =================================================================================================
__all__ = [
    "MazeHardEvaluationSourceContext",
    "MazeHardTraceSupplements",
    "build_mazehard_trace_supplements",
    "apply_mazehard_trace_supplements",
]
