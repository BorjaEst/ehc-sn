"""Rollout trace observation helpers."""

from __future__ import annotations

from collections.abc import Mapping

from ehc_sn.rollouts.runtime import RolloutChunk
from ehc_sn.traces.observer import TraceObserver, TraceSpec
from ehc_sn.traces.trace_tree import TraceTree


# =============================================================================
def observe_rollout_chunk(
    chunk: RolloutChunk,
    spec: TraceSpec,
    *,
    trace_meta: Mapping[str, object] | None = None,
) -> TraceTree:
    """Materialize a TraceTree from a rollout chunk and trace spec."""
    trace = TraceTree()
    observer = TraceObserver(trace, spec)
    observer.observe_records(chunk.records)
    if trace_meta is not None:
        trace.attach_meta(trace_meta, overwrite=True)
    trace.finalize()
    return trace


# =============================================================================
__all__ = ["observe_rollout_chunk"]
