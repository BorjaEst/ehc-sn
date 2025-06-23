from math import prod
from typing import Tuple

import torch
from pydantic import BaseModel, Field, model_validator
from torch import Tensor, nn


class EncoderParams(BaseModel):
    """Configuration parameters for the neural network encoder."""

    model_config = {"extra": "forbid"}  # Forbid extra fields not defined in the model

    input_shape: Tuple[int, int, int] = Field(
        ...,
        description="Shape of the input feature map (channels, height, width). ",
    )
    base_channel_size: int = Field(
        default=16,
        description="Base number of channels used in the first convolutional layers",
    )
    latent_dim: int = Field(
        ...,
        description="Dimensionality of latent representation z",
    )
    activation_fn: object = Field(
        default=nn.GELU,
        description="Activation function used throughout the encoder network",
    )

    @model_validator(mode="after")
    def validate_convolution_after(self) -> "EncoderParams":
        expected_features = 2 * 16 * self.base_channel_size
        if expected_features < self.latent_dim:
            raise ValueError(
                f"Base channel size {self.base_channel_size} "
                f"does not scale properly to the latent dimension {self.latent_dim}. "
                f"Expected at least {self.latent_dim / 32} base channels."
            )
        return self


class Encoder(nn.Module):
    """Neural network encoder that transforms multi-dimensional inputs into fixed-size embeddings.

    This encoder flattens multi-dimensional inputs and passes them through a sequence of
    linear layers with non-linear activations to produce an embedding vector.
    """

    def __init__(self, params: EncoderParams):
        super().__init__()
        self._input_shape = params.input_shape  # Store input shape for later reference
        c_hid = params.base_channel_size  # Base number of channels used in the first convolutional layers
        self.net = nn.Sequential(
            nn.Conv2d(self.input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            params.activation_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),  # 16x16 => 16x16
            params.activation_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            params.activation_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),  # 8x8 => 8x8
            params.activation_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            params.activation_fn(),
        )
        self.flatten = nn.Flatten()  # Flatten the output of the convolutional layers
        self.linear = nn.Sequential(
            nn.Linear(2 * c_hid * prod(self.spatial_dimensions) // 64, params.latent_dim),
            nn.ReLU(),  # Activation after linear layer
        )

    def forward(self, x):
        x = self.net(x)  # Pass through convolutional layers
        x = self.flatten(x)  # Flatten the output to a single feature vector
        x = self.linear(x)  # Pass through linear layer to get embedding
        return x

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        """Returns the shape of the input feature map."""
        return self._input_shape

    @property
    def input_channels(self) -> int:
        """Returns the number of input channels."""
        return self.input_shape[0]

    @property
    def spatial_dimensions(self) -> Tuple[int, int]:
        """Returns the output shape as (height, width)."""
        return self.input_shape[1], self.input_shape[2]

    @property
    def base_channel_size(self) -> int:
        """Returns the base channel size used in the encoder."""
        return self.net[0].out_channels

    @property
    def latent_dim(self) -> int:
        """Returns the dimensionality of the latent representation."""
        return self.linear[-2].out_features


# Example usage of the Encoder class
if __name__ == "__main__":
    # Create encoder parameters for a simple case:
    params = EncoderParams(
        input_shape=(1, 32, 16),  # 1 channel, 32x16 grid
        base_channel_size=16,
        latent_dim=128,
    )

    # Create the encoder
    encoder = Encoder(params)

    # Create a sample input (batch size 4, 1 channel, 16x32 grid)
    sample_input = torch.randn(4, *params.input_shape)

    # Forward pass
    embedding = encoder(sample_input)

    # Print encoder details
    print(f"Encoder architecture: {encoder}")
    print(f"Input shape: {sample_input.shape}")
    print(f"Output embedding shape: {embedding.shape}")

    # Verify embedding dimension matches expected
    assert embedding.shape == (4, 128)
    print("Encoder works as expected!")
