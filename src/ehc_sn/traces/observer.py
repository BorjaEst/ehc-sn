"""Trace observers over executed or evaluated rollout data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Literal, Mapping, Sequence, TypeAlias, TypeVar

import numpy as np
import torch

from ehc_sn.rollouts import EvaluatedChunk
from ehc_sn.traces.trace_tree import TraceTree

# =================================================================================================
Context = TypeVar("Context")

TraceLeaf: TypeAlias = int | float | np.ndarray | torch.Tensor
TraceValue: TypeAlias = TraceLeaf | None | Mapping[str, "TraceValue"] | Sequence["TraceValue"]
TraceGetter: TypeAlias = Callable[[Context], TraceValue]
TraceStorage: TypeAlias = Literal["dense", "meta"]


# =================================================================================================
@dataclass(frozen=True)
class TraceField(Generic[Context]):
    """One named field in a trace specification."""

    name: str
    get: TraceGetter[Context]
    storage: TraceStorage = "dense"


# =================================================================================================
@dataclass(frozen=True)
class TraceSpec(Generic[Context]):
    """Specification describing which values to collect into a :class:`TraceTree`."""

    fields: Sequence[TraceField[Context]]

    def keys(self) -> set[str]:
        """Return the set of field names in this spec."""
        return {field.name for field in self.fields}


# =================================================================================================
class TraceObserver(Generic[Context]):
    """Observe rollout or objective contexts into a :class:`TraceTree`."""

    def __init__(self, tree: TraceTree, spec: TraceSpec[Context]):
        self.tree = tree
        self.spec = spec
        self.tree.config.metadata_paths.update((field.name,) for field in spec.fields if field.storage == "meta")

    def observe(self, ctx: Context, *, step_index: int) -> None:
        """Append one timestep payload extracted from the given context."""
        payload: dict[str, TraceValue] = {"t": step_index}
        payload.update({field.name: field.get(ctx) for field in self.spec.fields})
        self.tree.append(payload)

    def observe_chunk(self, chunk: EvaluatedChunk) -> None:
        """Append all scored steps from a chunk in order."""
        for step in chunk.steps:
            self.observe(step, step_index=step.index)


# =================================================================================================
__all__ = ["TraceField", "TraceObserver", "TraceSpec", "TraceValue"]
