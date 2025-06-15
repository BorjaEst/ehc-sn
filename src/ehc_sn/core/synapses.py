from typing import Optional

import torch
from torch import nn


class FixedRandom(nn.Linear):
    """Fixed random linear layer with fixed synaptic weights.
    Cannot be trained, weights are fixed at initialization.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias)
        # Initialize with random weights
        nn.init.xavier_normal_(self.weight)  # Paper mentions N(0, 1) initialization
        # Freeze the parameters
        self.weight.requires_grad = False
        if bias:
            self.bias.requires_grad = False


class FixedNonRandom(nn.Linear):
    """Fixed non random linear layer with fixed synaptic weights.
    Cannot be trained, weights are fixed at initialization.
    """

    def __init__(self, weights: torch.Tensor, bias: Optional[torch.Tensor] = None):
        out_features, in_features = weights.shape
        has_bias = bias is not None
        super().__init__(in_features, out_features, has_bias)

        # Set the provided weights
        with torch.no_grad():
            self.weight.copy_(weights)
            if has_bias:
                self.bias.copy_(bias)

        # Freeze the parameters
        self.weight.requires_grad = False
        if has_bias:
            self.bias.requires_grad = False


class Plastic(nn.Linear):
    """Plastic linear layer with learnable synaptic weights.
    Trainable only through Hebbian learning.
    """

    def __init__(self, in_features, out_features, learning_rate=0.01, decay=0.0, bias=False):
        super().__init__(in_features, out_features, bias)
        # Initialize weights
        nn.init.xavier_normal_(self.weight)
        # Freeze parameters for backprop (we'll update them manually)
        self.weight.requires_grad = False
        if bias:
            self.bias.requires_grad = False

        self.learning_rate = learning_rate
        self.decay = decay

    def forward(self, x):
        # Store input for Hebbian learning
        self.last_input = x
        # Use nn.Linear's forward pass
        output = super().forward(x)
        self.last_output = output
        return output

    def update_weights(self):
        """Apply Hebbian learning rule"""
        if hasattr(self, "last_input") and hasattr(self, "last_output"):
            batch_size = self.last_input.size(0)

            # Compute weight updates using Hebbian learning
            for i in range(batch_size):
                x = self.last_input[i].unsqueeze(0)
                y = self.last_output[i].unsqueeze(0)

                # Hebbian update: presynaptic * postsynaptic activity
                dw = torch.mm(y.t(), x)

                # Apply update with learning rate and decay
                with torch.no_grad():
                    self.weight.data = (1 - self.decay) * self.weight.data + self.learning_rate * dw

            # Clear stored values after update
            del self.last_input
            del self.last_output
