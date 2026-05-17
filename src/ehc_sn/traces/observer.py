"""Trace observers over executed rollout data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    Generic,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TypeAlias,
    TypeVar,
)

import numpy as np
import torch

from ehc_sn.traces.trace_tree import TraceTree


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


# =============================================================================
@dataclass(frozen=True)
class TraceSpec(Generic[Context]):
    """Specification describing which values to collect into a :class:`TraceTree`."""

    fields: Sequence[TraceField[Context]]

    def keys(  # --------------------------------------------------------------
        self,
    ) -> set[str]:
        """Return the set of field names in this spec."""
        return {field.name for field in self.fields}


# =============================================================================
class TraceObserver(Generic[Context]):
    """Observe indexed execution contexts into a :class:`TraceTree`."""

    def __init__(  # ----------------------------------------------------------
        self,
        tree: TraceTree,
        spec: TraceSpec[Context],
    ) -> None:
        """Initialize a trace observer with a target tree and trace specification."""
        self.tree = tree
        self.spec = spec
        self.tree.config.metadata_paths.update(
            (field.name,) for field in spec.fields if field.storage == "meta"
        )

    def observe(  # -----------------------------------------------------------
        self,
        ctx: Context,
        *,
        step_index: int,
    ) -> None:
        """Append one timestep payload extracted from the given context."""
        payload: dict[str, TraceValue] = {"t": step_index}
        payload.update(
            {field.name: field.get(ctx) for field in self.spec.fields}
        )
        self.tree.append(payload)

    def observe_records(  # ---------------------------------------------------
        self,
        records: Sequence[Context],
    ) -> None:
        """Append an ordered sequence of execution records."""
        for record in records:
            self.observe(record, step_index=record.index)


# =============================================================================
__all__ = [
    "IndexedTraceContext",
    "TraceField",
    "TraceObserver",
    "TraceSpec",
    "TraceValue",
]
