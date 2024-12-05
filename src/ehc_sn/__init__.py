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

    def get_sequence(
        self, items: List[Item], δ: float = 0.7  # fmt: skip
    ) -> Sequence:
        """Return the hidden code for sequence."""
        if δ < 0 or δ > 1:
            raise ValueError("The δ value should be in [0, 1].")
        T = len(items)
        discounted = [x * δ ** (T - t) for t, x in enumerate(items, 1)]
        return np.array(discounted).sum(axis=0)

    def p_sequence(
        self, y: List[Sequence], pΘ: List[Callable], p: List[float]
    ) -> float:
        """Return the probability of sequence."""
        probability_array = [fΘ(y) * p[k] for k, fΘ in enumerate(pΘ)]
        return np.array(probability_array).sum(axis=0)
