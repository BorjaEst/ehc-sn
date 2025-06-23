from math import prod
from typing import Tuple

import torch
from pydantic import BaseModel, Field, model_validator
from torch import nn


class DecoderParams(BaseModel):
    """Configuration parameters for the neural network decoder."""

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
    def validate_convolution_after(self) -> "DecoderParams":
        expected_features = 2 * 16 * self.base_channel_size
        if expected_features < self.latent_dim:
            raise ValueError(
                f"Base channel size {self.base_channel_size} "
                f"does not scale properly to the latent dimension {self.latent_dim}. "
                f"Expected at least {self.latent_dim / 32} base channels."
            )
        return self


class Decoder(nn.Module):
    """Neural network decoder that transforms embeddings into reconstructed features.

    This decoder takes embedding vectors and passes them through a sequence of
    linear layers with non-linear activations to produce a reconstructed feature output
    which is then reshaped to the original feature dimensions.
    """

    def __init__(self, params: DecoderParams):
        super().__init__()
        self._input_shape = params.input_shape  # Shape of the input feature map (channels, height, width)
        c_hid = params.base_channel_size  # Base number of channels used in the first convolutional layers
        first_dim = 2 * c_hid * prod(self.spatial_dimensions) // 64  # First dimension based on input shape
        self.linear = nn.Sequential(nn.Linear(params.latent_dim, first_dim), params.activation_fn())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            params.activation_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            params.activation_fn(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            params.activation_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            params.activation_fn(),
            nn.ConvTranspose2d(c_hid, self.input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.Sigmoid(),  # Changed from Softmax to Sigmoid for continuous value reconstruction
        )

    def forward(self, x):
        x = self.linear(x)  # Needs reshape
        x = x.reshape(x.shape[0], -1, *[s // 8 for s in self.spatial_dimensions])
        x = self.net(x)
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
        """Returns the base channel size used in the decoder."""
        return self.net[-1].in_channels

    @property
    def latent_dim(self) -> int:
        """Returns the dimensionality of the latent representation."""
        return self.linear[0].in_features


# Example usage of the Decoder class
if __name__ == "__main__":
    # Create decoder parameters for a simple case:
    params = DecoderParams(
        input_shape=(1, 32, 16),  # 1 channel, 32x16 grid
        base_channel_size=16,
        latent_dim=128,
    )

    # Create the decoder
    decoder = Decoder(params)

    # Create a sample input (batch size 4, latent dimension)
    sample_input = torch.randn(4, params.latent_dim)

    # Forward pass
    reconstruction = decoder(sample_input)

    # Print decoder details
    print(f"Decoder architecture: {decoder}")
    print(f"Input shape: {sample_input.shape}")
    print(f"Output reconstruction shape: {reconstruction.shape}")

    # Verify reconstruction dimension matches expected
    assert reconstruction.shape == (4, *params.input_shape)
    print("Decoder works as expected!")
