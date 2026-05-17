"""Dungeon producer-side trace supplement API.

This module owns:

- :class:`DungeonEvaluationSourceContext` — typed, frozen provider context for Dungeon cases.
- :class:`DungeonTraceSupplements` — canonical supplement content ready to attach.
- :func:`build_dungeon_trace_supplements` — constructs supplements from source context.
- :func:`apply_dungeon_trace_supplements` — attaches supplement data to a :class:`TraceTree`.

Dungeon does not currently require spatial geometry supplements (no rate-map or
world-geometry enrichment is needed for existing figure consumers).  The build/apply
functions are explicit no-ops today, documented as the canonical task-local supplement
seam for future work.

The surface mirrors the Arena supplement pattern exactly so that Lightning families
can wire Dungeon through the same supplement pattern if needed in the future without
changing the evaluation contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ehc_sn.traces.trace_tree import TraceTree


# =============================================================================
@dataclass(frozen=True)
class DungeonEvaluationSourceContext:
    """Typed, frozen provider-side context for a Dungeon evaluation case batch.

    Attributes:
        task_family: Always ``"dungeon"``. Used as a discriminant for isinstance checks.
        dataset_path: Absolute path to the processed Dungeon dataset root.
        split: Dataset split the samples belong to (e.g. ``"val"``).
        sample_ids: Ordered sample ids in this batch.
    """

    task_family: str
    dataset_path: Path
    split: str
    sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.task_family != "dungeon":
            raise ValueError(
                "DungeonEvaluationSourceContext.task_family must be 'dungeon', "
                f"got {self.task_family!r}"
            )
        if not self.sample_ids:
            raise ValueError(
                "DungeonEvaluationSourceContext.sample_ids must not be empty",
            )


# =============================================================================
@dataclass(frozen=True)
class DungeonTraceSupplements:
    """Canonical supplement content for Dungeon traces.

    Dungeon does not currently produce spatial geometry supplements.
    This dataclass is intentionally empty and serves as the typed seam for future work.
    """


# =============================================================================
def build_dungeon_trace_supplements(  # ---------------------------------------
    source_context: DungeonEvaluationSourceContext,
    trace_length: int,
) -> DungeonTraceSupplements:
    """Build Dungeon trace supplements from a typed source context.

    Currently a no-op: Dungeon does not require spatial geometry supplements for
    existing figure consumers.  This function is the canonical task-local supplement
    seam for future enrichment.

    Args:
        source_context: Typed Dungeon evaluation source context.
        trace_length: Number of time steps to cover.

    Returns:
        Empty :class:`DungeonTraceSupplements`.
    """
    return DungeonTraceSupplements()


# =============================================================================
def apply_dungeon_trace_supplements(  # ---------------------------------------
    trace: TraceTree,
    supplements: DungeonTraceSupplements,
) -> None:
    """Attach Dungeon supplement content to *trace* in-place.

    Currently a no-op: no supplement keys are attached because Dungeon does not
    produce spatial geometry supplements for existing figure consumers.

    Args:
        trace: The :class:`~ehc_sn.traces.trace_tree.TraceTree` to modify in-place.
        supplements: The supplement content to attach.
    """


# =============================================================================
__all__ = [
    "DungeonEvaluationSourceContext",
    "DungeonTraceSupplements",
    "build_dungeon_trace_supplements",
    "apply_dungeon_trace_supplements",
]
