from abc import ABC

import torch
from torch import Tensor, nn

from ehc_sn import parameters, utils


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
