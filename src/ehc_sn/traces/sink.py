"""Trace sink protocol and concrete implementations.

A trace sink decouples trace retention policy from trace field extraction.
The observer emits extracted payloads one step at a time;
the sink decides whether to retain them in memory, write to disk, sample,
or window.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import numpy as np

from ehc_sn.traces.trace_tree import TraceTree
from ehc_sn.types import MultiScaleView

if TYPE_CHECKING:
    from ehc_sn.traces.observer import TraceSpec, TraceValue

# =============================================================================


@runtime_checkable
class TraceSink(Protocol):
    """Protocol for trace-value retention policies.

    Implementations decide how to store extracted trace payloads:
    in-memory, disk-backed, windowed, or sampled.
    """

    def append(self, payload: dict[str, TraceValue]) -> None:
        """Store one extracted step payload.

        The payload dict contains at least ``"t"`` (the step index) plus
        one entry per trace field resolved in the current step.
        """

    def finalize(self) -> Any:
        """Finalize collection and return the finalized trace product.

        The return type is implementation-specific.  ``InMemoryTraceSink``
        returns a ``TraceTree`` suitable for export, slicing, and
        meta attachment.  Other implementations may return file paths,
        chunk references, or opaque handles.
        """


# =============================================================================
class InMemoryTraceSink:
    """Bounded in-memory ``TraceSink`` that wraps a ``TraceTree``.

    Retains at most ``max_steps`` extracted step payloads.  Exceeding the
    bound raises ``RuntimeError`` (or drops oldest when ``overflow="stop"``).
    Returns the finalized tree from ``finalize()``.

    For diagnostic-only unbounded use, see ``UnboundedInMemoryTraceSink``.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        spec: TraceSpec,
        *,
        max_steps: int,
        overflow: Literal["error", "stop"] = "error",
    ) -> None:
        """Initialize a bounded in-memory sink from a trace specification.

        Args:
            spec: Trace specification defining the expected fields.
            max_steps: Maximum number of step payloads to retain.  Must be
                strictly positive.
            overflow: ``"error"`` raises ``RuntimeError`` on append beyond
                the bound; ``"stop"`` silently ignores further payloads and
                records truncation metadata.

        Raises:
            ValueError: If ``max_steps`` is not strictly positive.
        """
        if max_steps <= 0:
            raise ValueError(
                f"InMemoryTraceSink max_steps must be positive, got {max_steps}."
            )
        self._max_steps = max_steps
        self._overflow = overflow
        self._truncated = False
        self._tree = TraceTree()
        self._tree.config.metadata_paths.update(
            (field.name,) for field in spec.fields if field.storage == "meta"
        )
        self._finalized = False

    def append(self, payload: dict[str, TraceValue]) -> None:
        """Append one step payload to the internal trace tree.

        Raises ``RuntimeError`` when the bound is exceeded and
        ``overflow="error"``.
        """
        if self._finalized:
            raise RuntimeError("InMemoryTraceSink is already finalized.")
        if self._tree.length >= self._max_steps:
            if self._overflow == "error":
                raise RuntimeError(
                    f"InMemoryTraceSink: max_steps={self._max_steps} exceeded."
                )
            # overflow == "stop": silently drop and record truncation.
            self._truncated = True
            return
        self._tree.append(payload)

    def finalize(self) -> TraceTree:
        """Finalize the internal trace tree and return it directly."""
        if self._finalized:
            raise RuntimeError("InMemoryTraceSink is already finalized.")
        self._finalized = True
        self._tree.finalize()
        if self._truncated:
            self._tree.attach_meta(
                {"trace_truncated": True, "trace_max_steps": self._max_steps}
            )
        return self._tree


# =============================================================================
class UnboundedInMemoryTraceSink(InMemoryTraceSink):
    """Diagnostic-only unbounded in-memory trace sink.

    Prefer ``InMemoryTraceSink`` with an explicit ``max_steps`` bound for
    normal evaluation use.  This variant exists for diagnostic introspection
    where every step must be retained.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        spec: TraceSpec,
    ) -> None:
        """Initialize an unbounded in-memory sink.

        Args:
            spec: Trace specification defining the expected fields.
        """
        super().__init__(spec, max_steps=2**63, overflow="stop")


# =============================================================================
class ZarrTraceSink:
    """Disk-backed ``TraceSink`` that writes chunked Zarr arrays.

    Buffers per-step payloads in memory until a configurable chunk size is
    reached, then writes the chunk to a Zarr group on disk with compression.
    ``finalize()`` flushes any partial chunk and returns the path.

    Requires ``zarr>=2.17``.  Raises ``ImportError`` at construction time if
    ``zarr`` is not installed.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        *,
        path: str | Path,
        spec: TraceSpec,
        chunk_size: int = 256,
        compressor: str = "zstd",
        clevel: int = 3,
    ) -> None:
        """Initialize a Zarr-backed trace sink.

        Args:
            path: Directory path for the Zarr group.  Will be created as a
                directory (not a single file).
            spec: Trace specification defining the expected field names and
                storage types.
            chunk_size: Number of steps to buffer before flushing to disk.
                Default 256.
            compressor: Blosc compressor name (``"zstd"``, ``"lz4"``, etc.).
            clevel: Blosc compression level (1–9).

        Raises:
            ImportError: If ``zarr`` is not installed.
            ValueError: If ``chunk_size <= 0``.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
        try:
            import zarr
        except ImportError as exc:
            raise ImportError(
                "zarr is required for ZarrTraceSink. "
                "Install with: pip install 'ehp-sn[evaluation]'"
            ) from exc

        self._path = Path(path)
        self._spec = spec
        self._chunk_size = chunk_size
        self._zarr = zarr
        self._compressor = dict(compressor=compressor, clevel=clevel)
        self._buffer: list[dict[str, Any]] = []
        self._finalized = False
        self._field_names: list[str] = []
        self._meta_values: dict[str, Any] = {}

    # ── public protocol ─────────────────────────────────────────────────────

    def append(self, payload: dict[str, TraceValue]) -> None:
        """Buffer one step payload; flush to Zarr when chunk is full."""
        if self._finalized:
            raise RuntimeError("ZarrTraceSink is already finalized.")

        # Capture field names from the first payload.
        # MultiScaleView fields are expanded into per-band names so each
        # band occupies its own Zarr array with uniform shape.
        if not self._field_names:
            field_names: list[str] = []
            for k, v in payload.items():
                if isinstance(v, (str, bytes, bool)):
                    continue
                if isinstance(v, np.ndarray) and v.dtype.kind == "U":
                    continue
                if isinstance(v, MultiScaleView):
                    # Expand each band as a separate field.
                    for band_key in v.values:
                        field_names.append(f"{k}._band_{band_key}")
                else:
                    field_names.append(k)
            self._field_names = sorted(field_names)

        self._buffer.append(payload)

        if len(self._buffer) >= self._chunk_size:
            self._flush_buffer()

    def finalize(self) -> Path:
        """Flush remaining buffer, finalize Zarr arrays, return group path."""
        if self._finalized:
            raise RuntimeError("ZarrTraceSink is already finalized.")
        self._finalized = True

        if self._buffer:
            self._flush_buffer()

        return self._path.resolve()

    # ── internal ────────────────────────────────────────────────────────────

    def _flush_buffer(self) -> None:
        """Write buffered payloads as compressed Zarr arrays."""
        import numcodecs

        if not self._buffer:
            return

        n = len(self._buffer)
        compressor = numcodecs.Blosc(
            cname=self._compressor["compressor"],
            clevel=self._compressor["clevel"],
        )

        for name in self._field_names:
            # Collect all values for this field across the buffer.
            arrays: list[np.ndarray] = []

            # Check if this is a band-expanded name (multi-scale field).
            band_tag: str | None = None
            parent_name: str = name
            if "._band_" in name:
                parent_name, sep, band_tag = name.rpartition("._band_")

            for payload in self._buffer:
                if band_tag is not None:
                    # Extract the specific band from a MultiScaleView parent.
                    v_parent = payload.get(parent_name)
                    if isinstance(v_parent, MultiScaleView):
                        v = v_parent.values.get(band_tag)
                        if v is not None:
                            arrays.append(v.detach().cpu().numpy())
                    continue

                v = payload.get(name)
                if v is None:
                    continue
                if isinstance(v, np.ndarray):
                    arrays.append(v)
                elif isinstance(v, (int, float)):
                    arrays.append(np.array([v], dtype=np.float32))
                elif isinstance(v, list):
                    # List of tensors (multi-scale code) — flatten into a
                    # single array.  Each element is a per-frequency tensor.
                    stacked = [
                        (
                            x.detach().cpu().numpy()
                            if hasattr(x, "detach")
                            else np.asarray(x)
                        )
                        for x in v
                    ]
                    arrays.append(np.stack(stacked, axis=0))
                elif isinstance(v, MultiScaleView):
                    # MultiScaleView fields were expanded into band keys
                    # during ``_field_names`` capture; each band is stored
                    # under its own ``{name}._band_{key}`` name and will be
                    # processed as a separate field in the outer loop.
                    # Skip the parent name here.
                    continue
                elif hasattr(v, "detach"):
                    arrays.append(v.detach().cpu().numpy())
                else:
                    arrays.append(np.asarray(v, dtype=np.float32))

            if not arrays:
                continue

            # Stack along a new leading axis (time).
            try:
                chunk = np.stack(arrays, axis=0)
            except ValueError:
                # Variable shapes — skip this field for Zarr storage.
                continue

            key = name.replace("/", ".")
            zarr_path = self._path / key
            if zarr_path.exists():
                # Append to existing array.
                z = self._zarr.open_array(
                    str(zarr_path),
                    mode="a",
                )
                z.append(chunk, axis=0)
            else:
                # Create new array (zarr_format=2 for Blosc compressor compat).
                zarr_path.parent.mkdir(parents=True, exist_ok=True)
                chunks = (min(self._chunk_size, n),) + tuple(
                    s if s > 0 else 1 for s in chunk.shape[1:]
                )
                self._zarr.open_array(
                    str(zarr_path),
                    mode="w",
                    shape=chunk.shape,
                    chunks=chunks,
                    dtype=chunk.dtype,
                    compressor=compressor,
                    zarr_format=2,
                    fill_value=None,
                )[:] = chunk

        self._buffer.clear()


# =============================================================================
__all__ = [
    "EvaluationEvent",
    "EventSink",
    "InMemoryTraceSink",
    "ParquetEventSink",
    "TraceSink",
    "UnboundedInMemoryTraceSink",
    "ZarrTraceSink",
]


# =============================================================================
# Phase 5 — Sparse event capture
# =============================================================================


import dataclasses
import json
import time


@dataclasses.dataclass(frozen=True)
class EvaluationEvent:
    """One sparse evaluation event emitted during a rollout step.

    ``payload`` values must be JSON-serializable (int, float, bool, str,
    None, or simple lists/dicts of those).  Non-serializable values are
    stringified via ``repr()`` before JSON encoding.

    ``event_type`` follows a dotted namespace convention:
    ``"controller.halt"``, ``"memory.write"``, ``"memory.retrieval_fail"``,
    ``"model.nan_detect"``, ``"prediction.incorrect"``.
    """

    step: int
    case_id: str
    event_type: str
    payload: dict[str, object]
    timestamp_ns: int | None = None

    def __post_init__(self) -> None:
        if self.timestamp_ns is None:
            object.__setattr__(self, "timestamp_ns", time.monotonic_ns())


# =============================================================================
@runtime_checkable
class EventSink(Protocol):
    """Protocol for sparse event retention during evaluation.

    Complements dense per-step ``TraceSink`` by capturing targeted events
    (halt, memory write, retrieval failure, NaN detection, incorrect
    prediction) without requiring per-step trace storage.
    """

    def emit(self, event: EvaluationEvent) -> None:
        """Record one evaluation event."""

    def close(self) -> None:
        """Finalize and release resources."""


# =============================================================================
class ParquetEventSink:
    """Concrete ``EventSink`` that persists events as a Parquet table.

    Buffers events in memory and flushes to ``<root>/events.parquet`` in
    chunks.  ``close()`` writes any remaining buffered events.

    Schema:
        case_id (string), step (int64), event_type (string),
        payload_json (string), timestamp_ns (int64, nullable)

    Requires ``pyarrow``.  Raises ``ImportError`` at construction time if
    not installed.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        root: Path,
        *,
        case_id: str = "",
        chunk_size: int = 1024,
    ) -> None:
        """Initialize a Parquet-backed event sink.

        Args:
            root: Artifact root directory.  ``events.parquet`` is written
                directly under this path.
            case_id: Default case_id prepended to events that do not carry
                their own.
            chunk_size: Number of events to buffer before flushing to disk.

        Raises:
            ImportError: If ``pyarrow`` is not installed.
            ValueError: If ``chunk_size <= 0``.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
        try:
            import pyarrow as pa  # noqa: F401
            import pyarrow.parquet as pq  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "pyarrow is required for ParquetEventSink. "
                "Install with: pip install 'ehp-sn[evaluation]'"
            ) from exc

        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._case_id = case_id
        self._chunk_size = chunk_size
        self._buffer: list[EvaluationEvent] = []
        self._closed = False

        # Schema: case_id (string), step (int64), event_type (string),
        #         payload_json (string), timestamp_ns (int64, nullable)
        self._schema = pa.schema(
            [
                pa.field("case_id", pa.string()),
                pa.field("step", pa.int64()),
                pa.field("event_type", pa.string()),
                pa.field("payload_json", pa.string()),
                pa.field("timestamp_ns", pa.int64(), nullable=True),
            ]
        )

    # ── public protocol ─────────────────────────────────────────────────────

    def emit(self, event: EvaluationEvent) -> None:
        """Buffer one event; flush to Parquet when chunk_size is reached."""
        if self._closed:
            raise RuntimeError("ParquetEventSink is already closed.")
        self._buffer.append(event)
        if len(self._buffer) >= self._chunk_size:
            self._flush()

    def close(self) -> None:
        """Flush buffered events and release resources."""
        if self._closed:
            raise RuntimeError("ParquetEventSink is already closed.")
        self._closed = True
        if self._buffer:
            self._flush()
        self._buffer = []

    # ── internal ────────────────────────────────────────────────────────────

    def _flush(self) -> None:
        """Write buffered events to ``events.parquet`` (append or create)."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows: list[dict[str, object]] = []
        for ev in self._buffer:
            payload_str = _json_serializable_payload(ev.payload)
            rows.append(
                {
                    "case_id": ev.case_id or self._case_id,
                    "step": ev.step,
                    "event_type": ev.event_type,
                    "payload_json": payload_str,
                    "timestamp_ns": ev.timestamp_ns,
                }
            )
        self._buffer.clear()

        table = pa.Table.from_pylist(rows, schema=self._schema)
        file_path = self._root / "events.parquet"

        if file_path.exists():
            existing = pq.read_table(str(file_path), schema=self._schema)
            combined = pa.concat_tables([existing, table])
            pq.write_table(combined, str(file_path))
        else:
            pq.write_table(table, str(file_path))


# =============================================================================
def _json_serializable_payload(payload: dict[str, object]) -> str:
    """Serialize *payload* to JSON, stringifying non-serializable values."""
    clean: dict[str, object] = {}
    for key, value in payload.items():
        try:
            json.dumps(value)
            clean[key] = value
        except (TypeError, ValueError):
            clean[key] = repr(value)
    return json.dumps(clean, sort_keys=True)
