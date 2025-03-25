"""Module for the model class."""

from abc import ABC

import torch
from ehc_sn import config, parameters
from norse.torch.functional import stdp
from torch import nn
import norse.torch as snn

# pylint: disable=too-few-public-methods
# pylint: disable=non-ascii-name


class Layer:
    """The layer settings of the EI model."""

    def __init__(self, p: parameters.Layer):
        self.population = p.population
        self.cell = p.cell.cell()
        zeros = torch.zeros(self.population).to(config.device)
        self.nodes = self.cell(zeros, None)

    def reset(self):
        """Reset the state of the layer."""
        zeros = torch.zeros(self.population).to(config.device)
        self.nodes = self.cell(zeros, None)


class Network(nn.Module, ABC):
    """The connectivity settings of the EI model."""

    def __init__(self, p: parameters.Network):
        super().__init__()
        self.layers = {l1: Layer(p) for l1, p in p.layers.items()}
        self.w = {l1: {l2: p.mask(l1, l2) * p.weights(l1, l2)
                       for l2 in p.synapses[l1].w}
                  for l1 in p.synapses} # fmt: skip
        self.p_stdp = p.plasticity.parameters()
        self.stdp_state = {l1: {l2: p.stdp_state(l1, l2)
                                for l2 in p.synapses[l1].w}
                           for l1 in p.synapses} # fmt: skip

    def reset(self) -> None:
        """Reset the state of the network."""
        for layer in self.layers.values():
            layer.reset()

    def plasticity(self, l1: str, l2: str) -> None:
        """Update the weights of the network."""
        self.w[l1][l2], self.stdp_state[l1][l2] = stdp.stdp_step_linear(
            z_pre=self.layers[l1].nodes[0].unsqueeze(0),
            z_post=self.layers[l2].nodes[0].unsqueeze(0),
            w=self.w[l1][l2],
            state_stdp=self.stdp_state[l1][l2],
            p_stdp=self.p_stdp,
        )

    def run(self, exc_current):
        """Run the model for a given input current."""
        return torch.stack([self(x) for x in exc_current])


class SumDecoder(nn.Module):
    """The decoder settings of the EI model."""

    def forward(self, x):
        """Decode the input current."""
        return x.sum(dim=0)


class EHCModel(nn.Module):
    """The encoder extension of the EI model."""

    def __init__(self, network: Network, seq_length: int = 32):
        super().__init__()
        self.encoder = snn.ConstantCurrentLIFEncoder(seq_length)
        self.network = network
        self.decoder = SumDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model for a given input current."""
        x = self.encoder(x)
        x = self.network.run(x)
        return self.decoder(x)
