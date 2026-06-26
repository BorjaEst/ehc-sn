"""Trace observers over executed rollout data."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import (
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
)

import numpy as np
import torch

from ehc_sn.traces.sink import TraceSink


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
@dataclass(frozen=True)
class TraceField(Generic[Context]):
    """One named field in a trace specification."""

    name: str
    get: TraceGetter[Context]
    storage: TraceStorage = "dense"
    requires_model_state: bool = False


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

    def requires_model_state(  # ---------------------------------------------
        self,
    ) -> bool:
        """Return whether any selected field needs carry.model_state."""
        return any(field.requires_model_state for field in self.fields)


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
            {field.name: field.get(ctx) for field in self.spec.fields}
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
