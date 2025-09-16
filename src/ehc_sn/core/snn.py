from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import init

from ehc_sn import parameters, utils

nn.RNN


class Layer(nn.Module, ABC):
    """Base class for neuron in the EHC spatial navigation model.
    This class is not meant to be instantiated directly.
    """

    def __init__(self, p: parameters.Neuron, population_size: int):
        super().__init__()

        # Collect synapse parameters from the provided Neuron object
        self.description = p.description
        self.activation_lim = p.activation_lim
        self.activation_function = p.activation_function

        # Initialize the neuron with a fixed population size.
        self.register_buffer("activations", torch.zeros(1, population_size))

    def clamp(self, tensor: Tensor) -> Tensor:
        """Clamp the activations to the specified min and max values."""
        if self.activation_lim:
            tensor = tensor.clamp(-self.activation_lim, self.activation_lim)
        return tensor

    @property
    def activation_function(self) -> Tensor:
        """Get the current activations of the neuron."""
        return self.__activation_name

    @activation_function.setter
    def activation_function(self, value: str) -> None:
        """Set the activation function for the neuron."""
        self.__activation_name = value
        self.__activation_fn = utils.torch_function(value)

    def forward(self, current: Tensor) -> Tensor:
        """Forward pass through the neuron."""
        self.activations = self.__activation_fn(current)
        self.activations = self.clamp(self.activations)
        return self.activations

    def extra_repr(self) -> str:
        return f"activation_function={self.activation_function}"


class Sigmoid(Layer):
    """Sigmoid neuron activation function."""

    def __init__(self, population_size: int):
        super().__init__(parameters.neurons["sigmoid"], population_size)


class ReLU(Layer):
    """ReLU neuron activation function."""

    def __init__(self, population_size: int):
        super().__init__(parameters.neurons["relu"], population_size)


class Linear(Layer):
    """Linear neuron activation function."""

    def __init__(self, population_size: int):
        super().__init__(parameters.neurons["linear"], population_size)


class Synapse(nn.Module, ABC):
    """Base class for synapses in the EHC spatial navigation model.
    This class is not meant to be instantiated directly.
    """

    def __init__(self, p: parameters.Synapse, weight: Tensor, bias: Optional[Tensor] = None):
        super().__init__()

        # Collect synapse parameters from the provided Synapse object
        self.description = p.description
        self.weight_max = p.weight_max
        self.weight_min = p.weight_min
        self.normalize = p.normalize
        self.learning_rate = p.learning_rate

        # Initialize the synapse with fixed weights and optional bias.
        self.weight = nn.Parameter(weight.clone())
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.register_parameter("bias", None)

        # Apply clamping to the weights and bias
        self.weight.data = self.clamp(self.weight.data)
        if bias is not None:
            self.bias.data = self.clamp(self.bias.data)

        # Freeze the parameters to prevent backpropagation training
        self.weight.requires_grad = False
        if bias is not None:
            self.bias.requires_grad = False

    def clamp(self, tensor: Tensor) -> Tensor:
        """Clamp the weights to the specified min and max values."""
        if self.weight_min or self.weight_max:
            tensor = tensor.clamp(self.weight_min, self.weight_max)
        if self.normalize:
            tensor = F.normalize(tensor, p=2, dim=tensor.ndim - 1)
        return tensor

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass through the synapse."""
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        out_feat, in_feat = self.weight.shape
        return f"in_features={in_feat}, out_features={out_feat}, bias={self.bias is not None}"

    @classmethod
    def normal(cls, in_features: int, out_features: int, bias: bool = False) -> "Synapse":
        """Create a synapse with normally distributed weights."""
        weight = torch.randn(out_features, in_features)
        bias_tensor = torch.randn(out_features) if bias else None
        return cls(weight, bias_tensor)

    @classmethod
    def xavier(cls, in_features: int, out_features: int, bias: bool = False) -> "Synapse":
        """Create a synapse with Xavier initialized weights."""
        weight = torch.empty(out_features, in_features)
        init.xavier_normal_(weight)
        bias_tensor = torch.zeros(out_features) if bias else None
        return cls(weight, bias_tensor)


class Hybrid(Synapse):
    """Hybrid synapse that combines features of different synapse types."""

    def __init__(self, weight: Tensor, bias: Optional[Tensor] = None):
        super().__init__(parameters.synapses["hybrid"], weight, bias)


class Silent(Synapse):
    """Silent synapse that does not transmit signals but can be used for weight storage."""

    def __init__(self, weight: Tensor, bias: Optional[Tensor] = None):
        super().__init__(parameters.synapses["silent"], weight, bias)


class AMPA(Synapse):
    """AMPA synapse that transmits excitatory signals."""

    def __init__(self, weight: Tensor, bias: Optional[Tensor] = None):
        super().__init__(parameters.synapses["ampa"], weight, bias)


class GABA(Synapse):
    """GABA synapse that transmits inhibitory signals."""

    def __init__(self, weight: Tensor, bias: Optional[Tensor] = None):
        super().__init__(parameters.synapses["gaba"], weight, bias)
