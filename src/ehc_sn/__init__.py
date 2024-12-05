"""Sequential Navigation (SN) module for the Entorhinal–Hippocampal circuit (EHC)"""

from dataclasses import dataclass
from typing import List, Callable

import torch
from torch import nn
import numpy as np
import numpy.typing as npt

# Type alias for integer space arrays
Item = npt.NDArray[np.float32]  # Navigation Item
Sequence = npt.NDArray[np.float32]  # Navigation Sequence
Map = npt.NDArray[np.float32]  # Cognitive Map


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


def get_sequence(items: List[Item], δ: float = 0.7) -> Sequence:
    """Return the hidden code for sequence."""
    if δ < 0 or δ > 1:
        raise ValueError("The δ value should be in [0, 1].")
    T = len(items)
    discounted = [x * δ ** (T - t) for t, x in enumerate(items, 1)]
    return np.array(discounted).sum(axis=0)  # Eq. (2)


def p_sequence(y: Sequence, Θ: dict[Map, float]) -> float:
    """Return the probability of a sequence."""
    a = [_p_sequence(y, θ) * p for θ, p in Θ.items()]
    return np.array(a).sum(axis=0)  # Eq. (3)


def _p_sequence(y: Sequence, θ: Map) -> float:
    """Calculate the probability of sequence given a map."""
    lnpΘ = y @ np.log(θ)  # Eq. (5)
    # Note lnp(y|Θ_k) actually proportional to y·ln(θ_k)
    return np.exp(lnpΘ)
