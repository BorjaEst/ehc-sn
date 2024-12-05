"""Sequential Navigation (SN) module for the Entorhinal–Hippocampal circuit (EHC)"""

from dataclasses import dataclass
from typing import List, Callable

import torch
from torch import nn
import numpy as np
import numpy.typing as npt

# Type alias for integer space arrays
Array1DInt = npt.NDArray[np.int_]
Array2DInt = npt.NDArray[np.int_]

Item = Callable[[int], Array1DInt]
Sequence = Callable[[int], Array2DInt]
Map = Callable[[], Array1DInt]


@dataclass
class HGModelParams:
    """The param set class example for model configuration."""

    items_size: int = 32
    sequences_size: int = 32
    maps_size: int = 32


class HierarchicalGenerativeModel(nn.Module):
    """Model for representing hierarchical spatiotemporal data."""

    def __init__(
        self,
        observation_size: int,
        n_maps: int = 1,
        config: HGModelParams | None = None,
    ):
        """Construct for model."""
        config = config or HGModelParams()  # default params
        super().__init__()
        self.items_layer = nn.Linear(
            in_features=observation_size,
            out_features=config.items_size,
            bias=True,
        )
        self.sequences_layer = nn.Linear(
            in_features=config.items_size,
            out_features=config.sequences_size,
            bias=True,
        )
        self.maps_layer = nn.Linear(
            in_features=config.sequences_size,
            out_features=config.maps_size,
            bias=True,
        )
        self.cluster_layer = nn.Linear(
            in_features=config.maps_size,
            out_features=n_maps,
            bias=True,
        )

    def forward(self, observation: torch.Tensor):
        """Inference function."""
        raise NotImplementedError

    def item_gen(self, ξ: List[Array1DInt]) -> Item:
        """Return the hidden code for item."""
        if self.items_layer.out_features != len(ξ[0]):
            raise ValueError("Len ξ should match model items.")
        def item(t: int) -> Array1DInt:  # fmt: skip
            return ξ[t - 1]
        return item

    def sequence_gen(self, x: Item, δ: float = 0.7) -> Sequence:
        """Return the hidden code for sequence."""
        if δ < 0 or δ > 1:
            raise ValueError("The δ value should be in [0, 1].")
        def sequence(T: int) -> Array2DInt:  # fmt: skip
            return np.array([x(t) * δ ** (T - t) for t in range(1, T + 1)])
        return sequence

    def map_gen(self) -> Map:
        """Return the hidden code for map."""
        raise NotImplementedError

    def p_sequence(
        self,
        y: Sequence,
        pθ: Callable[[int], Callable[[Sequence], int]],
        p: Callable[[int], float],
    ) -> Array2DInt:
        """Return the hidden code for sequence probabilities."""
        K = self.cluster_layer.out_features  # number of clusters
        py_k = np.array([pθ(k)(y) * p(k) for k in range(1, K + 1)])
        return py_k.sum(axis=0)
