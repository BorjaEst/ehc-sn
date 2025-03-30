"""Module for the model class."""

import norse.torch as snn
import torch
from ehc_sn import config, parameters
from norse.torch.functional import stdp
from torch import Tensor, nn

# pylint: disable=too-few-public-methods
# pylint: disable=arguments-differ
# pylint: disable=redefined-outer-name


class BaseLayer(snn.LIFRefracCell):
    """The layer main class for the the model."""

    def __init__(self, n: int, p: parameters.CellParameters):
        super().__init__(p=p.parameters())
        self.size = n
        zeros = torch.zeros(self.size).to(config.device)
        self.nodes = super().forward(zeros, None)

    @property
    def states(self) -> Tensor:
        """Return the state of the layer."""
        return self.nodes[1]

    @property
    def spikes(self) -> Tensor:
        """Return the spikes of the layer."""
        return self.nodes[0]

    def reset(self) -> None:
        """Reset the state of the layer."""
        zeros = torch.zeros(self.size).to(config.device)
        self.nodes = super().forward(zeros, None)

    def forward(self, x: Tensor) -> Tensor:
        """Run the layer for a given input current."""
        self.nodes = super().forward(x, self.states)
        return self.spikes


class Inputs(nn.Module):
    """Excitatory layer class."""

    def __init__(self, neurons: BaseLayer, p: parameters.Synapses):
        super().__init__()
        self.mask = p.make_mask(neurons.size)
        self.init_value = w0 = p.init_value
        self.w = nn.Parameter(w0 * self.mask, requires_grad=False)
        self.plasticity = p.stdp.parameters()
        self.state = p.state(neurons.size)

    def stdp(self, z_pre: Tensor, z_post: Tensor) -> None:
        """Update the weights of the network."""
        w, self.state = stdp.stdp_step_linear(
            z_pre=z_pre.unsqueeze(0),
            z_post=z_post.unsqueeze(0),
            w=self.w,
            state_stdp=self.state,
            p_stdp=self.plasticity,
        )
        self.w[:] = w * self.mask

    def reset(self) -> None:
        """Reset the state of the layer."""
        self.state.t_post = torch.zeros(1, self.w.shape[0]).to(config.device)
        self.state.t_pre = torch.zeros(1, self.w.shape[1]).to(config.device)
        self.w[:] = self.init_value * self.mask

    def forward(self, x: Tensor) -> Tensor:
        """Run the layer for a given input current."""
        return x @ (self.mask * self.w).T


class EILayer(nn.Module):
    """Excitatory layer class."""

    def __init__(self, p: parameters.Layer):
        super().__init__()
        self.neurons = BaseLayer(n=p.population, p=p.cells)
        self.ampa = Inputs(self.neurons, p=p.synapses["ampa"])
        self.gaba = Inputs(self.neurons, p=p.synapses["gaba"])

    def reset(self) -> None:
        """Reset the state of the layer."""
        self.neurons.reset()
        self.ampa.reset()
        self.gaba.reset()

    @property
    def spikes(self) -> Tensor:
        """Return the spikes of the layer."""
        return self.neurons.spikes

    def forward(self, *x: Tensor, stdp: bool = True) -> Tensor:
        """Run the layer for excitatory and inhibitory synapses."""
        y = self.neurons(x[0] + self.ampa(x[1]) - self.gaba(x[2]))
        if stdp:  # Update the weights of the network.
            self.ampa.stdp(z_pre=x[1], z_post=y)
            self.gaba.stdp(z_pre=x[2], z_post=y)
        return y
