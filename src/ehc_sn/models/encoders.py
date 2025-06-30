from math import prod
from typing import Tuple

import torch
from pydantic import BaseModel, Field, model_validator
from torch import Tensor, nn

from ehc_sn.hooks.drtp import DRTPLayer


class EncoderParams(BaseModel):
    """Configuration parameters for the neural network encoder."""

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    input_shape: Tuple[int, int, int] = Field(..., description="Shape input (chn,h,w)")
    latent_dim: int = Field(..., description="Dimensionality of latent representation z")
    activation_fn: object = Field(..., description="Activation function")


class BaseEncoder(nn.Module):
    """Base class for neural network encoders.

    This class defines the basic structure and properties of an encoder, which
    transforms input data into latent representations.
    """

    def __init__(self, params: EncoderParams):
        super().__init__()
        self.params = params

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the encoder."""
        raise NotImplementedError("Subclasses must implement forward method")

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        """Returns the shape of the input feature map."""
        return self.params.input_shape

    @property
    def input_channels(self) -> int:
        """Returns the number of input channels."""
        return self.input_shape[0]

    @property
    def spatial_dimensions(self) -> Tuple[int, int]:
        """Returns the output shape as (height, width)."""
        return self.input_shape[1], self.input_shape[2]

    @property
    def latent_dim(self) -> int:
        """Returns the dimensionality of the latent representation."""
        return self.params.latent_dim


class Linear(BaseEncoder):
    """Linear neural network encoder that transforms flattened inputs into fixed-size embeddings.

    This encoder flattens multi-dimensional inputs and passes them through a sequence of
    linear layers with non-linear activations to produce a sparse embedding vector.
    It's suitable for processing structured obstacle maps where sparse representations
    are biologically plausible.
    """

    def __init__(self, params: EncoderParams):
        super().__init__(params)
        in_features = prod(params.input_shape)

        # First layer: 1000 units
        self.layer1 = nn.Linear(in_features, 1000, bias=True)
        self.activation1 = params.activation_fn()  # Usually GELU

        # Second layer: 1000 units
        self.layer2 = nn.Linear(1000, 1000, bias=True)
        self.activation2 = params.activation_fn()  # Usually GELU

        # Output layer: latent_dim units
        self.layer3 = nn.Linear(1000, params.latent_dim, bias=True)
        self.output_activation = nn.ReLU()

    def forward(self, x: Tensor, target: Tensor = None) -> Tensor:
        # Flatten the input tensor to a single feature vector
        x = x.reshape(x.shape[0], -1)  # Reshape to (batch_size, num_features)

        # First layer
        h1 = self.layer1(x)
        h1 = self.activation1(h1)

        # Second layer
        h2 = self.layer2(h1)
        h2 = self.activation2(h2)

        # Output layer
        output = self.layer3(h2)
        output = self.output_activation(output)

        return output


class DRTPLinear(BaseEncoder):

    def __init__(self, params: EncoderParams):
        super().__init__(params)

        # Define layers separately since we need to apply DRTP manually
        self.layer1 = nn.Linear(params.latent_dim, out_features=1000)
        self.activation1 = params.activation_fn()  # Usually Tanh
        self.drtp1 = DRTPLayer(target_dim=self.input_shape, hidden_dim=1000)

        # Second layer: 1000 units
        self.layer2 = nn.Linear(params.latent_dim * 2, params.latent_dim * 4)
        self.activation2 = params.activation_fn()  # Usually Tanh
        self.drtp2 = DRTPLayer(target_dim=self.input_shape, hidden_dim=1000)

        # Output layer (no DRTP - uses standard gradients)
        self.layer3 = nn.Linear(1000, out_features=prod(params.input_shape))
        self.output_activation = nn.Sigmoid()

    def forward(self, x: Tensor, target: Tensor = None) -> Tensor:
        # First layer
        h1 = self.layer1(x)
        h1 = self.activation1(h1)
        h1 = self.drtp1(h1, target)

        # Second layer
        h2 = self.layer2(h1)
        h2 = self.activation2(h2)
        h2 = self.drtp2(h2, target)

        # Output layer (no DRTP - uses standard gradients)
        output = self.layer3(h2)
        output = self.output_activation(output)

        # Reshape to original input shape
        return output


class Conv2D(BaseEncoder):
    def __init__(self, params: EncoderParams):
        super().__init__(params)

        # Conv layer 1: 32 channels, 5x5 kernel, stride=1, padding=2
        self.conv1 = nn.Conv2d(self.input_channels, out_channels=32, kernel_size=5, stride=1, padding=2, bias=True)
        self.activation1 = params.activation_fn()  # Usually GELU
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Linear layer 2: 1000 units
        self.layer2 = nn.Linear(32000, 1000, bias=True)
        self.activation2 = params.activation_fn()  # Usually GELU

        # Linear layer 3 (output): 10 units
        self.layer3 = nn.Linear(1000, self.latent_dim, bias=True)
        self.output_activation = nn.Sigmoid()

    def forward(self, x: Tensor, target: Tensor = None) -> Tensor:
        # First layer
        h1 = self.conv1(x)
        h1 = self.activation1(h1)
        h1 = self.pool1(h1)

        # Flatten for FC layers
        h1 = h1.reshape(h1.size(0), -1)

        # Second layer
        h2 = self.layer2(h1)
        h2 = self.activation2(h2)

        # Output layer
        output = self.layer3(h2)
        output = self.output_activation(output)

        return output


class DRTPConv2D(BaseEncoder):
    def __init__(self, params: EncoderParams):
        super().__init__(params)

        # Conv layer 1: 32 channels, 5x5 kernel, stride=1, padding=2
        self.conv1 = nn.Conv2d(self.input_channels, out_channels=32, kernel_size=5, stride=1, padding=2, bias=True)
        self.activation1 = params.activation_fn()  # Usually Tanh
        self.drtp1 = DRTPLayer(target_dim=self.input_shape, hidden_dim=[32, 32, 16])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Linear layer 2: 1000 units
        self.layer2 = nn.Linear(32000, 1000, bias=True)
        self.activation2 = params.activation_fn()  # Usually Tanh
        self.drtp2 = DRTPLayer(target_dim=self.input_shape, hidden_dim=[1000])

        # Linear layer 3 (output): 10 units
        self.layer3 = nn.Linear(1000, self.latent_dim, bias=True)
        self.output_activation = nn.Sigmoid()
        # No DRTP on output layer (uses standard gradients)

    def forward(self, x: Tensor, target: Tensor = None) -> Tensor:
        # First layer
        h1 = self.conv1(x)
        h1 = self.activation1(h1)
        h1 = self.drtp1(h1, target)  # Apply DRTP to conv layer
        h1 = self.pool1(h1)

        # Flatten for FC layers
        h1 = h1.reshape(h1.size(0), -1)

        # Second layer
        h2 = self.layer2(h1)
        h2 = self.activation2(h2)
        h2 = self.drtp2(h2, target)  # Apply DRTP to linear layer

        # Output layer
        output = self.layer3(h2)
        output = self.output_activation(output)

        return output


# Example usage of the Encoder class
if __name__ == "__main__":
    # Create encoder parameters for a simple case:
    params = EncoderParams(
        input_shape=(1, 32, 16),  # 1 channel, 32x16 grid
        latent_dim=128,
    )
