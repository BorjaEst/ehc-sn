"""Module for the model class."""

from abc import ABC
from typing import Any

import norse.torch as snn
import torch
from ehc_sn import config, parameters
from ehc_sn import layers
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

    def reset(self) -> None:
        """Reset the state of the network."""
        self.excitatory.neurons.reset()  # Reset excitatory layer
        self.inhibitory.neurons.reset()  # Reset inhibitory layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network for a given input current."""
        xe = torch.concatenate([x, self.excitatory.neurons.spikes])
        xi = self.inhibitory.neurons.spikes
        xe = self.excitatory(xe, xi)
        xi = self.inhibitory(xe, xi)
        return xe
