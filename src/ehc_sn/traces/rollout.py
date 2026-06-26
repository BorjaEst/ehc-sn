"""Rollout trace observation helpers."""

from __future__ import annotations

from collections.abc import Mapping

from ehc_sn.rollouts.runtime import RolloutChunk
from ehc_sn.traces.observer import TraceSpec
from ehc_sn.traces.trace_tree import TraceTree


# =============================================================================
def observe_rollout_chunk(
    chunk: RolloutChunk,
    spec: TraceSpec,
    *,
    trace_meta: Mapping[str, object] | None = None,
) -> TraceTree:
    """Materialize a TraceTree from a rollout chunk and trace spec.

    .. note::
        This function is preserved for backward compatibility.
        New code should prefer the streaming evaluation path via
        ``score_rollout_streaming_with_trace``.
    """
    tree = TraceTree()
    tree.config.metadata_paths.update(
        (field.name,) for field in spec.fields if field.storage == "meta"
    )
    for record in chunk.records:
        payload: dict = {"t": record.index}
        for field in spec.fields:
            payload[field.name] = field.get(record)
        tree.append(payload)
    if trace_meta is not None:
        tree.attach_meta(trace_meta, overwrite=True)
    tree.finalize()
    return tree


# =============================================================================
__all__ = [
    "observe_rollout_chunk",
]
