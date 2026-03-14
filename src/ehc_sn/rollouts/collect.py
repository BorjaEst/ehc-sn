from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Literal, Mapping, Sequence, TypeAlias, TypeVar

import numpy as np
import torch

from ehc_sn.rollouts.trace_tree import TraceTree

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
        return {f.name for f in self.fields}


# =================================================================================================
class TraceCollector(Generic[Context]):
    """Collect per-step values into a :class:`~ehc_sn.rollouts.trace_tree.TraceTree`."""

    def __init__(self, tree: TraceTree, spec: TraceSpec[Context]):
        """Create a collector bound to a tree and a spec."""
        self.tree = tree
        self.spec = spec
        self.tree.config.metadata_paths.update(tuple(field.name.split("/")) for field in spec.fields if field.storage == "meta")  # fmt: skip

    def append(self, t: int, ctx: Context) -> None:
        """Append one timestep payload extracted from the given context."""
        payload: dict[str, TraceValue] = {"t": t}
        payload.update({f.name: f.get(ctx) for f in self.spec.fields})
        self.tree.append(payload)


# =================================================================================================
__all__ = ["TraceField", "TraceSpec", "TraceCollector"]
