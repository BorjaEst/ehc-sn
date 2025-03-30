"""Module for the model class."""

import torch
from ehc_sn import layers, parameters
from torch import nn
from torch import Tensor

# pylint: disable=too-few-public-methods
# pylint: disable=non-ascii-name
# pylint: disable=arguments-differ


class EINetwork(nn.Module):
    """The connectivity settings of the EI model."""

    def __init__(self, p: parameters.Network):
        super().__init__()
        self.excitatory = layers.EILayer(p.layers["excitatory"])
        self.inhibitory = layers.EILayer(p.layers["inhibitory"])
        self.eval()

    def reset(self) -> None:
        """Reset the state of the network."""
        self.excitatory.neurons.reset()
        self.inhibitory.neurons.reset()

    def forward(self, x: Tensor) -> Tensor:
        """Run the network for a given input current."""
        with torch.no_grad():
            xe = self.excitatory.neurons.spikes
            xi = self.inhibitory.neurons.spikes
            xe = self.excitatory(x, xe, xi)
            xi = self.inhibitory(0, xe, xi)
        return xe


class EHCNetwork(nn.Module):
    """The main class for the EHC model."""

    def __init__(self, p: parameters.Network):
        super().__init__()
        self.mapping = layers.EILayer(p.layers["mapping"])
        self.inhibitory = layers.EILayer(p.layers["inhibitory"])
        self.embedding = layers.EILayer(p.layers["embedding"])
        self.eval()

    def reset(self) -> None:
        """Reset the state of the network."""
        self.mapping.reset()
        self.inhibitory.reset()
        self.embedding.reset()

    @property
    def xe(self) -> Tensor:
        """Return the cognitive map and embeddings of the network."""
        signals = (self.mapping.spikes, self.embedding.spikes)
        return torch.concatenate(signals)

    @property
    def xi(self) -> Tensor:
        """Return the inhibitory layer of the network."""
        return self.inhibitory.spikes

    def forward(self, sens: Tensor, mem: Tensor, stdp=True) -> tuple[Tensor, Tensor]:
        """Run the network for a given input current."""
        with torch.no_grad():
            self.embedding(mem, self.xe, self.xi, stdp=stdp)
            self.mapping(sens, self.xe, self.xi, stdp=stdp)
            self.inhibitory(0, self.xe, self.xi, stdp=stdp)
        return self.mapping.spikes, self.embedding.spikes
