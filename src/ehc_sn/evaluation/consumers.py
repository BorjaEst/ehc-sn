"""Trace consumer wrapping ``TraceObserver + TraceSink`` as an ``EvaluationConsumer``."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from ehc_sn.contracts.dependencies import model_view
from ehc_sn.evaluation.contracts import (
    ArtifactKey,
    ArtifactKind,
    EvaluationConsumer,
    ProducedArtifact,
)
from ehc_sn.traces.observer import StepContext, TraceObserver
from ehc_sn.traces.sink import TraceSink

# =============================================================================
_TRACE_INDEX_SCHEMA = "ehp_sn.evaluation.trace_index.v1"


def _to_json_serializable(value: object) -> object:
    """Convert a value to JSON-serializable form, handling tensors and arrays."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, torch.Tensor):
        return _to_json_serializable(value.detach().cpu().numpy().tolist())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        return _to_json_serializable(value.item())
    if isinstance(value, (list, tuple)):
        return [_to_json_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_json_serializable(v) for k, v in value.items()}
    return str(value)


def _build_trace_index(
    *,
    archive_path: Path,
    field_names: list[str],
    case_boundaries: dict[str, tuple[int, int]],
    case_meta: dict[str, dict[str, object]] | None = None,
) -> dict[str, Any]:
    """Build the trace index dictionary from case boundary information."""
    cases = [
        {"case_id": cid, "start": start, "length": length}
        for cid, (start, length) in case_boundaries.items()
    ]
    # Attach per-case metadata if available.
    if case_meta:
        for entry in cases:
            cid = entry["case_id"]
            if cid in case_meta:
                entry["meta"] = case_meta[cid]
    return {
        "schema": _TRACE_INDEX_SCHEMA,
        "archive": archive_path.name,
        "case_count": len(cases),
        "fields": sorted(field_names),
        "cases": cases,
    }


# =============================================================================
class TraceConsumer(EvaluationConsumer):
    """Adapter that wraps ``TraceObserver + TraceSink`` as an ``EvaluationConsumer``.

    Tracks per-case step boundaries and writes a ``trace_index.json`` sidecar
    alongside the Zarr archive during ``finalize()``.  The sidecar maps case
    IDs to start/length offsets within the concatenated Zarr arrays, and
    carries per-case metadata (task data like cell_type, start_flag, etc.)
    that figures require.

    An optional ``meta_fn`` may be provided to extract per-case metadata from
    a case batch during ``begin_case()``.  This function receives the
    ``source_context`` of the case and must return a JSON-serializable dict.
    """

    def __init__(
        self,
        observer: TraceObserver,
        sink: TraceSink,
        *,
        name: str = "trace",
        meta_fn: Callable[[object], dict[str, object]] | None = None,
    ) -> None:
        """Configure a trace consumer.

        Args:
            observer: Trace observer that extracts values from each step.
            sink: Trace sink that accumulates and finalizes trace data.
            name: Consumer identifier.
            meta_fn: Optional callable that returns per-case metadata
                from ``source_context``.  Called during ``begin_case()``.
        """
        self._observer = observer
        self._sink = sink
        self._meta_fn = meta_fn
        self._total_steps: int = 0
        self._case_boundaries: dict[str, tuple[int, int]] = {}
        self._current_case_id: str | None = None
        self._case_start_step: int = 0
        self._case_meta: dict[str, dict[str, object]] = {}

        super().__init__(name=name)

    @property
    def dependencies(self) -> frozenset:
        """Return typed dependencies from the trace spec."""
        raw = self._observer.spec.resolved_dependencies()
        return frozenset(model_view(dep.name) for dep in raw)

    # -- Lifecycle -----------------------------------------------------------

    def begin_case(self, context: Any) -> None:
        """Record the start of a new case for boundary tracking."""
        cid = getattr(context, "case_id", str(context))
        self._current_case_id = cid
        self._case_start_step = self._total_steps

        # Extract per-case metadata if meta_fn is provided.
        # The context may carry a source_context with batch info.
        if self._meta_fn is not None:
            try:
                # Try to get source_context from the context (it may be
                # on EvaluationCaseContext or on the case/batch).
                src = getattr(context, "source_context", None)
                if src is not None:
                    meta = self._meta_fn(src)
                    if isinstance(meta, dict):
                        self._case_meta[cid] = meta
            except Exception:
                pass

    def update(self, ctx: StepContext) -> None:
        """Extract and store one step's trace values."""
        self._observer.observe(ctx, step_index=ctx.index, sink=self._sink)
        self._total_steps += 1

    def end_case(self, context: Any) -> None:
        """Record the end of a case and store boundary info."""
        cid = self._current_case_id
        if cid is not None:
            length = self._total_steps - self._case_start_step
            self._case_boundaries[cid] = (self._case_start_step, length)
        self._current_case_id = None

    def attach_case_meta(
        self,
        case_id: str,
        meta: dict[str, object],
    ) -> None:
        """Attach per-case metadata for the trace index.

        Called by the orchestrator after case execution completes, when
        the case result's source_context is available for meta extraction.
        Values are converted to JSON-serializable form.
        """
        self._case_meta[case_id] = {
            k: _to_json_serializable(v) for k, v in meta.items()
        }

    def finalize(self) -> tuple[ProducedArtifact, ...]:
        """Finalize the trace sink and write the trace index sidecar."""
        result = self._sink.finalize()
        # ZarrTraceSink returns a Path; InMemoryTraceSink returns a TraceTree.
        source_path: Path | None = None
        if isinstance(result, Path):
            source_path = result

        # Write trace_index.json sidecar inside the traces directory.
        if source_path is not None and self._case_boundaries:
            index_path = source_path / "trace_index.json"
            index = _build_trace_index(
                archive_path=source_path,
                field_names=list(self._observer.spec.keys()),
                case_boundaries=self._case_boundaries,
                case_meta=self._case_meta or None,
            )
            # Include per-case meta data if collected.
            if self._case_meta:
                for entry in index["cases"]:
                    cid = entry["case_id"]
                    if cid in self._case_meta:
                        entry["meta"] = _to_json_serializable(
                            self._case_meta[cid]
                        )
            index_path.write_text(
                json.dumps(index, indent=2, sort_keys=True),
                encoding="utf-8",
            )

        return (
            ProducedArtifact(
                key=ArtifactKey(ArtifactKind.TRACE, self._name),
                schema_version=1,
                path=Path("traces"),
                media_type="application/vnd+zarr",
                producer_digest="",
                content_digest="",
                source_path=source_path,
            ),
        )

    def close(self) -> None:
        """Release the sink (currently a no-op for InMemory sinks)."""
        if hasattr(self._sink, "close"):
            self._sink.close()  # type: ignore[union-attr]


# =============================================================================
__all__ = [
    "TraceConsumer",
    "_build_trace_index",
    "_to_json_serializable",
]
