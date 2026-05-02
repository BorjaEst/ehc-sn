"""Countwalk producer-side trace supplement API.

This module owns:

- :class:`CountwalkEvaluationSourceContext` — typed, frozen provider context for Countwalk cases.
- :class:`CountwalkTraceSupplements` — canonical supplement content ready to attach.
- :func:`build_countwalk_trace_supplements` — constructs supplements from source context.
- :func:`apply_countwalk_trace_supplements` — attaches supplement data to a :class:`TraceTree`.

Countwalk does not currently require spatial geometry supplements (no rate-map or
world-geometry enrichment is needed for existing figure consumers).  The build/apply
functions are explicit no-ops today, documented as the canonical task-local supplement
seam for future work.

The surface mirrors the Arena supplement pattern exactly so that Lightning families
can wire Countwalk through the same supplement pattern if needed in the future without
changing the evaluation contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ehc_sn.traces.trace_tree import TraceTree


# =================================================================================================
@dataclass(frozen=True)
class CountwalkEvaluationSourceContext:
    """Typed, frozen provider-side context for a Countwalk evaluation case batch.

    Attributes:
        task_family: Always ``"countwalk"``. Used as a discriminant for isinstance checks.
        dataset_path: Absolute path to the processed Countwalk dataset root.
        split: Dataset split the samples belong to (e.g. ``"val"``).
        sample_ids: Ordered sample ids in this batch.
    """

    task_family: str
    dataset_path: Path
    split: str
    sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.task_family != "countwalk":
            raise ValueError(
                f"CountwalkEvaluationSourceContext.task_family must be 'countwalk', got {self.task_family!r}"
            )
        if not self.sample_ids:
            raise ValueError("CountwalkEvaluationSourceContext.sample_ids must not be empty")


# =================================================================================================
@dataclass(frozen=True)
class CountwalkTraceSupplements:
    """Canonical supplement content for Countwalk traces.

    Countwalk does not currently produce spatial geometry supplements.
    This dataclass is intentionally empty and serves as the typed seam for future work.
    """


# =================================================================================================
def build_countwalk_trace_supplements(
    source_context: CountwalkEvaluationSourceContext,
    trace_length: int,
) -> CountwalkTraceSupplements:
    """Build Countwalk trace supplements from a typed source context.

    Currently a no-op: Countwalk does not require spatial geometry supplements for
    existing figure consumers.  This function is the canonical task-local supplement
    seam for future enrichment.

    Args:
        source_context: Typed Countwalk evaluation source context.
        trace_length: Number of time steps to cover.

    Returns:
        Empty :class:`CountwalkTraceSupplements`.
    """
    return CountwalkTraceSupplements()


# =================================================================================================
def apply_countwalk_trace_supplements(
    trace: TraceTree,
    supplements: CountwalkTraceSupplements,
) -> None:
    """Attach Countwalk supplement content to *trace* in-place.

    Currently a no-op: no supplement keys are attached because Countwalk does not
    produce spatial geometry supplements for existing figure consumers.

    Args:
        trace: The :class:`~ehc_sn.traces.trace_tree.TraceTree` to modify in-place.
        supplements: The supplement content to attach.
    """


# =================================================================================================
__all__ = [
    "CountwalkEvaluationSourceContext",
    "CountwalkTraceSupplements",
    "build_countwalk_trace_supplements",
    "apply_countwalk_trace_supplements",
]
