from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Mapping, Sequence, TypeAlias, TypeVar

import numpy as np
import torch

from ehc_sn.rollouts.trace_tree import TraceTree

Context = TypeVar("Context")

TraceLeaf: TypeAlias = int | float | np.ndarray | torch.Tensor
TraceValue: TypeAlias = TraceLeaf | None | Mapping[str, "TraceValue"] | Sequence["TraceValue"]
TraceGetter: TypeAlias = Callable[[Context], TraceValue]


@dataclass(frozen=True)
class TraceField(Generic[Context]):
    """ """

    name: str
    get: TraceGetter[Context]


@dataclass(frozen=True)
class TraceSpec(Generic[Context]):
    """ """

    fields: Sequence[TraceField[Context]]

    def keys(self) -> set[str]:
        """ """
        return {f.name for f in self.fields}


class TraceCollector(Generic[Context]):
    """ """

    def __init__(self, tree: TraceTree, spec: TraceSpec[Context]):
        """ """
        self.tree = tree
        self.spec = spec

    def append(self, t: int, ctx: Context) -> None:
        """ """
        payload: dict[str, TraceValue] = {"t": t}
        payload.update({f.name: f.get(ctx) for f in self.spec.fields})
        self.tree.append(payload)
