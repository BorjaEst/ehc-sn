"""Module for the model class."""

from abc import ABC

import torch
from ehc_sn import layers, parameters
from torch import nn

# pylint: disable=too-few-public-methods
# pylint: disable=non-ascii-name
# pylint: disable=arguments-differ


class Network(nn.Module, ABC):
    """The connectivity settings of the EI model."""

    def __init__(self, p: parameters.Network):
        super().__init__()
        self.excitatory = layers.EILayer(p.layers["excitatory"])
        self.inhibitory = layers.EILayer(p.layers["inhibitory"])
        self.eval()

    def reset(self) -> None:
        """Reset the state of the network."""
        self.excitatory.neurons.reset()  # Reset excitatory layer
        self.inhibitory.neurons.reset()  # Reset inhibitory layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network for a given input current."""
        with torch.no_grad():
            xe = self.excitatory.neurons.spikes
            xi = self.inhibitory.neurons.spikes
            xe = self.excitatory(x, xe, xi)
            xi = self.inhibitory(0, xe, xi)
        return xe
