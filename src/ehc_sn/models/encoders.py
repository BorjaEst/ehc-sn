from math import prod
from typing import Tuple

import torch
from pydantic import BaseModel, Field, model_validator
from torch import Tensor, nn


class EncoderParams(BaseModel):
    """Configuration parameters for the neural network encoder."""

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    input_shape: Tuple[int, int, int] = Field(
        ...,
        description="Shape of the input feature map (channels, height, width). ",
    )
    scale_factor: int = Field(
        default=16,
        description="Scaling factor to determine the number of nurons in hidden layers",
    )
    latent_dim: int = Field(
        ...,
        description="Dimensionality of latent representation z",
    )
    activation_fn: object = Field(
        default=nn.GELU,
        description="Activation function used throughout the encoder network",
    )


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

    def reinit_weights(self, init_fn=None):
        """Reinitialize all weights in the encoder.

        Args:
            init_fn: Optional initialization function. If None, uses Xavier uniform.
        """
        if init_fn is None:
            init_fn = nn.init.xavier_uniform_

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                init_fn(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(init_weights)

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
    def scale_factor(self) -> int:
        """Returns the scaling factor used in the encoder."""
        return self.params.scale_factor

    @property
    def latent_dim(self) -> int:
        """Returns the dimensionality of the latent representation."""
        return self.params.latent_dim


class LinearEncoder(BaseEncoder):
    """Linear neural network encoder that transforms flattened inputs into fixed-size embeddings.

    This encoder flattens multi-dimensional inputs and passes them through a sequence of
    linear layers with non-linear activations to produce a sparse embedding vector.
    It's suitable for processing structured obstacle maps where sparse representations
    are biologically plausible.
    """

    def __init__(self, params: EncoderParams):
        super().__init__(params)
        in_features = prod(params.input_shape)
        self.linear = nn.Sequential(
            nn.Linear(in_features, params.latent_dim * 4),
            self.params.activation_fn(),
            nn.Linear(params.latent_dim * 4, params.latent_dim * 2),
            self.params.activation_fn(),
            nn.Linear(params.latent_dim * 2, params.latent_dim),
            self.params.activation_fn(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape(x.size(0), -1)  # Flatten
        return self.linear(x)


class ConvEncoder(BaseEncoder):
    """Convolutional neural network encoder that transforms multi-dimensional inputs into fixed-size embeddings.

    This encoder uses convolutional layers to process spatial data and transform it into a latent representation.
    """

    def __init__(self, params: EncoderParams):
        super().__init__(params)
        c_hid = params.scale_factor  # Base number of channels used in the first convolutional layers
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

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)  # Pass through convolutional layers
        x = self.flatten(x)  # Flatten the output to a single feature vector
        x = self.linear(x)  # Pass through linear layer to get embedding
        return x


# Example usage of the Encoder class
if __name__ == "__main__":
    # Create encoder parameters for a simple case:
    params = EncoderParams(
        input_shape=(1, 32, 16),  # 1 channel, 32x16 grid
        scale_factor=16,
        latent_dim=128,
    )

    # Create the encoder
    encoder = LinearEncoder(params)

    # Test both encoders
    sample_input = torch.randn(4, *params.input_shape)

    # Forward passes
    embeddings = encoder(sample_input)

    # Print encoder details
    print(f"Encoder architecture: {encoder}")
    print(f"Input shape: {sample_input.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")

    # Verify embeddings dimension matches expected
    assert embeddings.shape == (4, *params.input_shape)
    print("Encoder works as expected!")
