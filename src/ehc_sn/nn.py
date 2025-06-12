from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn
from torch.nn import Module

from ehc_sn import parameters
from ehc_sn.settings import config


class Neurons(Module):
    def __init__(self, p: parameters.Neurons, **kwds):
        super().__init__(**kwds)
        self.activation = torch.zeros(p.size, device=config.device)

    @property
    def size(self) -> int:
        return self.activation.shape[0]

    def forward(self, x: Tensor) -> Tensor:
        self.activation = torch.relu(x)
        return self.activation


class Synapse(Module):
    def __init__(self, neurons: Neurons, p: parameters.Synapses, **kwds):
        super().__init__(**kwds)
        # Initialize weight matrix with ones scaled by provided initialization
        self.w = torch.ones((neurons.size, p.size), device=config.device)
        self.w *= p.w_init  # Initialize weights

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.w.T


class Synapses(Module):
    def __init__(self, neurons: Neurons, p: parameters.Synapses, **kwds):
        super().__init__(**kwds)
        # Register synapse types as submodules; here we register "ampa"
        self.add_module("ampa", Synapse(neurons, p["ampa"], **kwds))

    def forward(self, x: Tensor) -> Tensor:
        # This fallback forward applies the "ampa" synapse by default.
        # In practice, the Layer class is expected to index specific synapses.
        return self.ampa(x)


class Layer(Module):
    def __init__(self, p: parameters.Layer, **kwds):
        super().__init__(**kwds)
        self.neurons = Neurons(p=p.neurons, **kwds)
        self.synapses = Synapses(self.neurons, p=p.synapses, **kwds)

    def forward(self, x: Tensor) -> Tensor:
        return self.synapses(x)


class Network(Module, ABC):
    def __init__(self, p: parameters.Network, **kwds):
        super().__init__(**kwds)
        self.layers = nn.ModuleDict({n: Layer(p.layers[n], **kwds) for n in p.layers})
        self.eval()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
