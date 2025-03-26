"""Module for the model class."""

from abc import ABC
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
    """The layer main class for the the model."""

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
        self.layers = nn.ModuleDict({
            "excitatory": STDPLayer(p.layers["excitatory"]),
            "inhibitory": STDPLayer(p.layers["inhibitory"]),
        }) # fmt: skip

    def reset(self) -> None:
        """Reset the state of the network."""
        for layer in self.layers.values():
            layer.reset()

    @property
    def spikes(self) -> list[torch.Tensor]:
        """Return the spikes of the network."""
        return [layer.spikes for layer in self.layers.values()]

    def run(self, exc_currents: torch.Tensor) -> torch.Tensor:
        """Do something."""
        return torch.stack([self(x) for x in exc_currents])


class SumDecoder(nn.Module):
    """The decoder settings of the EI model."""

    def __init__(self, window: int):
        super().__init__()
        self.window = window
        self._index = 0
        self._kernel = torch.ones(window, 1).to(config.device) / window
        self._buffer: torch.Tensor | None = None

    def _initialize_buffer(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize the rolling buffer based on the input tensor shape."""
        buffer_shape = (self.window, *x.shape)
        return torch.zeros(buffer_shape, device=x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode the input current."""
        if self._buffer is None:  # Initialize the buffer
            self._buffer = self._initialize_buffer(x)

        self._buffer[self._index] = x  # Store the input in buffer
        self._index = (self._index + 1) % self.window
        return (self._buffer * self._kernel).sum(dim=0)


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
