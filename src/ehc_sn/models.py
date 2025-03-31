"""Module for the model class."""

import torch
from ehc_sn import config, layers, parameters
from torch import Tensor, jit, nn

# pylint: disable=too-few-public-methods
# pylint: disable=non-ascii-name
# pylint: disable=arguments-differ


class EINetwork(nn.Module):
    """The connectivity settings of the EI model."""

    def __init__(self, p: parameters.Network, **kwds):
        super().__init__()
        self.excitatory = layers.EILayer(p.layers["excitatory"], **kwds)
        self.inhibitory = layers.EILayer(p.layers["inhibitory"], **kwds)
        self.eval()

    def reset(self) -> None:
        """Reset the state of the network."""
        self.excitatory.neurons.reset()
        self.inhibitory.neurons.reset()

    @property
    def xe(self) -> Tensor:
        """Return the excitatory layer of the network."""
        return self.excitatory.neurons.spikes

    @property
    def xi(self) -> Tensor:
        """Return the inhibitory layer of the network."""
        return self.inhibitory.neurons.spikes

    def forward(self, x: Tensor, stdp: bool = True) -> Tensor:
        """Run the network for a given input current."""
        with torch.no_grad():
            for task in [
                jit.fork(self.excitatory, x, self.xe, self.xi, stdp=stdp),
                jit.fork(self.inhibitory, 0, self.xe, self.xi, stdp=stdp),
            ]:
                jit.wait(task)
        return self.excitatory.neurons.spikes


class EHCNetwork(nn.Module):
    """The main class for the EHC model."""

    def __init__(self, p: parameters.Network, **kwds):
        super().__init__()
        self.mapping = layers.EILayer(p.layers["mapping"], **kwds)
        self.inhibitory = layers.EILayer(p.layers["inhibitory"], **kwds)
        self.embedding = layers.EILayer(p.layers["embedding"], **kwds)
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
            for task in [
                jit.fork(self.embedding, mem, self.xe, self.xi, stdp=stdp),
                jit.fork(self.mapping, sens, self.xe, self.xi, stdp=stdp),
                jit.fork(self.inhibitory, 0, self.xe, self.xi, stdp=stdp),
            ]:
                jit.wait(task)
        return self.mapping.spikes, self.embedding.spikes
