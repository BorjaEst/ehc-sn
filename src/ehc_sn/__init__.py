"""Sequential Navigation (SN) module for the Entorhinal–Hippocampal circuit (EHC)"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class HGModelParams:
    """The param set class example for model configuration."""

    items_size: int = 32
    sequences_size: int = 32
    maps_size: int = 32


class HierarchicalGenerativeModel(nn.Module):
    """Model for representing hierarchical spatiotemporal data."""

    def __init__(self, observation_size, n_maps=1, config=None):
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
