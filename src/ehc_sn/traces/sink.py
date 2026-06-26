"""Trace sink protocol and concrete implementations.

A trace sink decouples trace retention policy from trace field extraction.
The observer emits extracted payloads one step at a time;
the sink decides whether to retain them in memory, write to disk, sample,
or window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ehc_sn.traces.trace_tree import TraceTree

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
    """In-memory ``TraceSink`` that wraps a ``TraceTree``.

    This is the default sink for backward-compatible .npz export.
    It retains all extracted fields in RAM via the existing ``TraceTree``
    machinery and returns the finalized tree directly from ``finalize()``.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        spec: TraceSpec,
    ) -> None:
        """Initialize an in-memory sink from a trace specification.

        Args:
            spec: Trace specification defining the expected fields.
        """
        self._tree = TraceTree()
        self._tree.config.metadata_paths.update(
            (field.name,) for field in spec.fields if field.storage == "meta"
        )
        self._finalized = False

    def append(self, payload: dict[str, TraceValue]) -> None:
        """Append one step payload to the internal trace tree.

        The payload dict should contain ``"t"`` (step index) plus one
        entry per resolved trace field.
        """
        if self._finalized:
            raise RuntimeError("InMemoryTraceSink is already finalized.")
        self._tree.append(payload)

    def finalize(self) -> TraceTree:
        """Finalize the internal trace tree and return it directly."""
        if self._finalized:
            raise RuntimeError("InMemoryTraceSink is already finalized.")
        self._finalized = True
        self._tree.finalize()
        return self._tree


# =============================================================================
__all__ = [
    "InMemoryTraceSink",
    "TraceSink",
]
