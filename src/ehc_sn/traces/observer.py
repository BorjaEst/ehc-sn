"""Trace observers over executed rollout data."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import (
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
)

import numpy as np
import torch

from ehc_sn.contracts.dependencies import Dependency
from ehc_sn.rollouts.runtime import StepRecord
from ehc_sn.traces.sink import TraceSink
from ehc_sn.types import MultiScaleView


# =============================================================================
class IndexedTraceContext(Protocol):
    """Minimal traced context carrying a stable step index."""

    index: int


Context = TypeVar("Context", bound=IndexedTraceContext)


TraceLeaf: TypeAlias = int | float | np.ndarray | torch.Tensor
TraceValue: TypeAlias = (
    TraceLeaf | None | Mapping[str, "TraceValue"] | Sequence["TraceValue"]
)
TraceGetter: TypeAlias = Callable[[Context], TraceValue]
TraceStorage: TypeAlias = Literal["dense", "meta"]


# =============================================================================
def _detach_to_cpu(value: TraceValue) -> TraceValue:
    """Detach and transfer a trace value to CPU for sink consumption.

    Getters return torch-native values on the model device.  The observer
    calls this function once per field to produce CPU-ready values for sinks
    (Zarr, Parquet, TraceTree).  This is the single device-transfer boundary.

    Supports:
        ``torch.Tensor`` → ``.detach().cpu()``
        ``list[torch.Tensor]`` → ``[t.detach().cpu() for t in lst]``
        ``dict[str, torch.Tensor]`` → ``{k: v.detach().cpu() for k, v in d.items()}``
        ``MultiScaleView`` → detached + CPU per band, metadata preserved
        ``None``, scalars, ``np.ndarray`` → unchanged
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [_detach_to_cpu(v) for v in value]
    if isinstance(value, Mapping):
        return {k: _detach_to_cpu(v) for k, v in value.items()}
    if isinstance(value, MultiScaleView):
        return MultiScaleView(
            values={k: v.detach().cpu() for k, v in value.values.items()},
            metadata=dict(value.metadata),
        )
    # None, int, float, bool, np.ndarray — pass through unchanged.
    return value


# =============================================================================
@dataclass(frozen=True)
class StepContext:
    """Frozen per-step context carrying the record and extracted trace views.

    Replaces the previous pattern of cloning the full recurrent carry into every
    ``CarrySnapshot``.  The executor extracts only the requested semantic views
    from the model via ``trace_views()`` and places them here for trace field
    getters to consume.

    ``views`` is a dict of semantic view name → detached tensor on the model
    device.  Trace field getters read from ``ctx.views[key]`` (for model
    observations), ``ctx.record.snapshot.*`` (for carry-side data), and
    ``ctx.record.outputs.*`` (for controller outputs).
    """

    index: int
    record: StepRecord
    views: dict[str, torch.Tensor | MultiScaleView] = field(
        default_factory=dict
    )


# =============================================================================
@dataclass(frozen=True)
class TraceField(Generic[Context]):
    """One named field in a trace specification.

    Fields:
        name: Unique trace-field key.
        get: Callable that extracts the field from an execution context.
        storage: ``"dense"`` = per-step tensor appended to trace; ``"meta"`` =
            static value captured once and reused.
        dependencies: Semantic view names required by this field's getter.
            The executor uses ``resolved_dependencies()`` on ``TraceSpec`` to
            determine which views to request from ``model.trace_views()``.
    """

    name: str
    get: TraceGetter[Context]
    storage: TraceStorage = "dense"
    dependencies: frozenset[Dependency] = field(default_factory=frozenset)


# =============================================================================
@dataclass(frozen=True)
class TraceSpec(Generic[Context]):
    """Specification describing which values to collect into a trace."""

    fields: Sequence[TraceField[Context]]

    def keys(  # --------------------------------------------------------------
        self,
    ) -> set[str]:
        """Return the set of field names in this spec."""
        return {field.name for field in self.fields}

    def resolved_dependencies(  # --------------------------------------------
        self,
    ) -> frozenset[Dependency]:
        """Return the union of all non-empty ``dependencies`` across fields."""
        union: set[Dependency] = set()
        for field in self.fields:
            if field.dependencies:
                union.update(field.dependencies)
        return frozenset(union)


# =============================================================================
class TraceObserver(Generic[Context]):
    """Extract trace field values from indexed execution contexts.

    Unlike the original implementation, the observer no longer owns a
    ``TraceTree``.  Instead it writes extracted step payloads into a
    caller-supplied :class:`TraceSink`, decoupling extraction from
    retention policy.
    """

    def __init__(  # ----------------------------------------------------------
        self,
        spec: TraceSpec[Context],
    ) -> None:
        """Initialize a trace observer from a trace specification.

        Args:
            spec: Trace specification defining the expected fields.
        """
        self.spec = spec

    def observe(  # -----------------------------------------------------------
        self,
        ctx: Context,
        *,
        step_index: int,
        sink: TraceSink,
    ) -> None:
        """Extract one timestep payload and write it to *sink*.

        Args:
            ctx: Execution context (e.g. a ``StepRecord``).
            step_index: Rollout step index for the ``"t"`` field.
            sink: Target sink that receives the extracted payload.
        """
        payload: dict[str, TraceValue] = {"t": step_index}
        payload.update(
            {
                field.name: _detach_to_cpu(field.get(ctx))
                for field in self.spec.fields
            }
        )
        sink.append(payload)

    def observe_records(  # ---------------------------------------------------
        self,
        records: Sequence[Context],
        sink: TraceSink,
    ) -> None:
        """Extract an ordered sequence of execution records into *sink*."""
        for record in records:
            self.observe(record, step_index=record.index, sink=sink)


# =============================================================================
__all__ = [
    "IndexedTraceContext",
    "TraceField",
    "TraceObserver",
    "TraceSpec",
    "TraceValue",
]
