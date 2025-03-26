"""Module for the model class."""

from abc import ABC
from collections import deque
from typing import Any

import norse.torch as snn
import torch
from ehc_sn import config, parameters
from norse.torch.functional import stdp
from norse.torch.module.encode import PoissonEncoderStep
from torch import nn

# pylint: disable=too-few-public-methods
# pylint: disable=non-ascii-name
# pylint: disable=arguments-differ


class Layer(snn.LIFRefracCell):
    """The layer settings of the model."""

    def __init__(self, p: parameters.Layer):
        super().__init__(p=p.cell_parameters())
        self.w = nn.Parameter(p.spawn_weights())
        self.population = p.population
        self.input_size = p.input_size
        zeros = torch.zeros(self.population).to(config.device)
        self.nodes = super().forward(zeros, None)

    @property
    def states(self) -> torch.Tensor:
        """Return the state of the layer."""
        return self.nodes[1]

    @property
    def spikes(self) -> torch.Tensor:
        """Return the spikes of the layer."""
        return self.nodes[0]

    def reset(self) -> None:
        """Reset the state of the layer."""
        zeros = torch.zeros(self.population).to(config.device)
        self.nodes = super().forward(zeros, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the layer for a given input current."""
        self.nodes = super().forward(x @ self.w.T, self.states)
        return self.spikes


class Network(nn.Module, ABC):
    """The connectivity settings of the EI model."""

    def __init__(self, p: parameters.Network):
        super().__init__()
        self.layers = nn.ModuleDict({l1: Layer(v) for l1, v in p.layers.items()})
        # self.p_stdp = p.plasticity.parameters()
        # self.stdp_state = {l1: {l2: p.stdp_state(l1, l2)
        #                         for l2 in p.synapses[l1].w}
        #                    for l1 in p.synapses} # fmt: skip

    def reset(self) -> None:
        """Reset the state of the network."""
        for layer in self.layers.values():
            layer.reset()

    @property
    def spikes(self) -> list[torch.Tensor]:
        """Return the spikes of the network."""
        return [layer.spikes for layer in self.layers.values()]

    # def plasticity(self, l1: str, l2: str) -> None:
    #     """Update the weights of the network."""
    #     self.w[l1][l2], self.stdp_state[l1][l2] = stdp.stdp_step_linear(
    #         z_pre=self.layers[l1].nodes[0].unsqueeze(0),
    #         z_post=self.layers[l2].nodes[0].unsqueeze(0),
    #         w=self.w[l1][l2],
    #         state_stdp=self.stdp_state[l1][l2],
    #         p_stdp=self.p_stdp,
    #     )

    def run(self, exc_currents: torch.Tensor) -> torch.Tensor:
        """Do something."""
        return torch.stack([self(x) for x in exc_currents])


class SumDecoder(nn.Module):
    """The decoder settings of the EI model."""

    def __init__(self, window: int):
        super().__init__()
        self._acc: deque = deque(maxlen=window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode the input current."""
        self._acc.append(x)
        return sum(self._acc) / len(self._acc)


class EHCModel(nn.Module):
    """The encoder extension of the EI model."""

    def __init__(self, network: Network, decode_win: int = 32):
        super().__init__()
        self.encoder = PoissonEncoderStep()
        self.network = network
        self.decoder = SumDecoder(window=decode_win)

    def forward(self, x: torch.Tensor) -> tuple[Any, torch.Tensor]:
        """Run the model for a given input current."""
        x = self.encoder(x)
        x = self.network(x)
        return self.decoder(x), x
