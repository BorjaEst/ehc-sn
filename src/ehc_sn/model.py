"""Module for the model class."""

from abc import ABC
from typing import Any

import norse.torch as snn
import torch
from ehc_sn import config, parameters
from ehc_sn.decoders import HannDecoder
from norse.torch.functional import stdp
from norse.torch.module.encode import PoissonEncoderStep
from torch import nn

# pylint: disable=too-few-public-methods
# pylint: disable=non-ascii-name
# pylint: disable=arguments-differ


class Layer(snn.LIFRefracCell):
    """The layer main class for the the model."""

    def __init__(self, p: parameters.Layer):
        super().__init__(p=p.cell_parameters())
        self.mask = p.spawn_connections()
        self.w = self.mask * p.init_weight
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
        spikes_in = x @ (self.mask * self.w).T
        self.nodes = super().forward(spikes_in, self.states)
        return self.spikes


class STDPLayer(Layer):
    """The STDP class attributes of the model."""

    def __init__(self, p: parameters.STDPLayer):
        super().__init__(p)
        self.stdp_state = p.plasticity_state()
        self.plasticity = p.plasticity_parameters()

    def stdp(self, x: torch.Tensor) -> None:
        """Update the weights of the network."""
        self.w[:], self.stdp_state = stdp.stdp_step_linear(
            z_pre=x.unsqueeze(0),
            z_post=self.spikes.unsqueeze(0),
            w=self.w,
            state_stdp=self.stdp_state,
            p_stdp=self.plasticity,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the layer for a given input current."""
        spikes = super().forward(x)  # Run the layer
        self.stdp(x)  # Calculate weight update
        return spikes


class Network(nn.Module, ABC):
    """The connectivity settings of the EI model."""

    def __init__(self, p: parameters.Network):
        super().__init__()
        self.excitatory = STDPLayer(p.layers["excitatory"])
        self.inhibitory = STDPLayer(p.layers["inhibitory"])

    def reset(self) -> None:
        """Reset the state of the network."""
        self.excitatory.reset()
        self.inhibitory.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network for a given input current."""
        xe, xi = self.excitatory.spikes, self.inhibitory.spikes
        xe = self.excitatory(torch.concatenate([x, xe, -xi]))
        xi = self.inhibitory(torch.concatenate([xe, -xi]))
        return xe


class EHCModel(nn.Module):
    """The encoder extension of the EI model."""

    def __init__(self, network: Network, decode_win: int = 100):
        super().__init__()
        self.encoder = PoissonEncoderStep()
        self.network = network
        self.decoder = HannDecoder(decode_win)
        self.decoder_win = decode_win

    def reset(self) -> None:
        """Reset the state of the model."""
        self.network.reset()
        self.decoder = HannDecoder(self.decoder_win)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model for a given input current."""
        x = self.encoder(x)
        x = self.network(x)
        return self.decoder(x)
