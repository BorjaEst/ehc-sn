from math import prod
from typing import List, Tuple

import torch
from pydantic import BaseModel, Field, model_validator
from torch import Tensor, nn

from ehc_sn.core.drtp import DRTPLayer


class DecoderParams(BaseModel):
    """Configuration parameters for the neural network decoder."""

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
        description="Activation function used throughout the decoder network",
    )


class BaseDecoder(nn.Module):
    """Base class for neural network decoders.

    This class defines the basic structure and properties of a decoder, which
    transforms latent representations into reconstructed features.
    """

    def __init__(self, params: DecoderParams):
        super().__init__()
        self.params = params

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the decoder."""
        raise NotImplementedError("Subclasses must implement forward method")

    def reinit_weights(self, init_fn=None):
        """Reinitialize all weights in the decoder.

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
        """Returns the scaling factor used in the decoder."""
        return self.params.scale_factor

    @property
    def latent_dim(self) -> int:
        """Returns the dimensionality of the latent representation."""
        return self.params.latent_dim


class LinearDecoder(BaseDecoder):
    """Neural network decoder that transforms embeddings into reconstructed features.

    This decoder takes embedding vectors and passes them through a sequence of
    linear layers with non-linear activations to produce a reconstructed feature output.
    """

    def __init__(self, params: DecoderParams):
        super().__init__(params)
        in_features = prod(params.input_shape)
        self.linear = nn.Sequential(
            nn.Linear(params.latent_dim, params.latent_dim * 2),
            self.params.activation_fn(),
            nn.Linear(params.latent_dim * 2, params.latent_dim * 4),
            self.params.activation_fn(),
            nn.Linear(params.latent_dim * 4, in_features),
            nn.Sigmoid(),  # Keep sigmoid for final layer to ensure output is in [0,1]
        )

    def forward(self, x):
        x = self.linear(x)  # Pass through linear layers to get reconstructed features
        return x.reshape(x.shape[0], *self.input_shape)  # Reshape to original input shape


class DRTPDecoder(BaseDecoder):
    """Neural network decoder that transforms embeddings into reconstructed features using DRTP.

    This decoder takes embedding vectors and passes them through a sequence of
    linear layers with DRTP modulation to produce a reconstructed feature output.

    Unlike the LinearDecoder which uses standard backpropagation, this decoder
    applies DRTP (Direct Random Target Projection) to hidden layers, providing
    biologically plausible learning signals.

    The target for DRTP should be provided during the forward pass.
    """

    def __init__(self, params: DecoderParams):
        super().__init__(params)
        in_features = prod(params.input_shape)

        # Define layers separately since we need to apply DRTP manually
        self.layer1 = nn.Linear(params.latent_dim, params.latent_dim * 2)
        self.activation1 = params.activation_fn()
        self.drtp1 = DRTPLayer(target_dim=in_features, hidden_dim=params.latent_dim * 2)

        self.layer2 = nn.Linear(params.latent_dim * 2, params.latent_dim * 4)
        self.activation2 = params.activation_fn()
        self.drtp2 = DRTPLayer(target_dim=in_features, hidden_dim=params.latent_dim * 4)

        self.layer3 = nn.Linear(params.latent_dim * 4, in_features)
        self.output_activation = nn.Sigmoid()

    # -----------------------------------------------------------------------------------
    def forward(self, x: Tensor, target: Tensor = None) -> Tensor:
        """
        Forward pass through the DRTP decoder.

        Args:
            x: Input embeddings with shape (batch_size, latent_dim)
            target: Target for DRTP training with shape (batch_size, in_features).
                   If None, DRTP layers are bypassed (inference mode).

        Returns:
            Reconstructed features with shape (batch_size, *input_shape)
        """
        # First layer
        h1 = self.layer1(x)
        h1 = self.activation1(h1)

        # Apply DRTP to first hidden layer if target is provided
        if target is not None:
            # Flatten target to match in_features dimension
            target_flat = target.view(target.size(0), -1)
            h1 = self.drtp1(h1, target_flat)

        # Second layer
        h2 = self.layer2(h1)
        h2 = self.activation2(h2)

        # Apply DRTP to second hidden layer if target is provided
        if target is not None:
            h2 = self.drtp2(h2, target_flat)

        # Output layer (no DRTP - uses standard gradients)
        output = self.layer3(h2)
        output = self.output_activation(output)

        # Reshape to original input shape
        return output.reshape(output.shape[0], *self.input_shape)


class ConvDecoder(BaseDecoder):
    """Neural network decoder that transforms embeddings into reconstructed features.

    This decoder takes embedding vectors and passes them through a sequence of
    linear layers with non-linear activations to produce a reconstructed feature output
    which is then reshaped to the original feature dimensions.
    """

    def __init__(self, params: DecoderParams):
        super().__init__(params)
        c_hid = self.scale_factor  # Base number of channels used in the first convolutional layers
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


# Example usage of the Decoder classes
if __name__ == "__main__":
    # Create decoder parameters for a simple case:
    params = DecoderParams(
        input_shape=(1, 32, 16),  # 1 channel, 32x16 grid
        scale_factor=16,
        latent_dim=128,
    )

    print("=== Testing LinearDecoder ===")
    # Create the linear decoder
    linear_decoder = LinearDecoder(params)

    # Create a sample input (batch size 4, latent dimension)
    sample_input = torch.randn(4, params.latent_dim)

    # Forward pass
    reconstruction = linear_decoder(sample_input)

    print(f"Linear Decoder:")
    print(f"  Input shape: {sample_input.shape}")
    print(f"  Output reconstruction shape: {reconstruction.shape}")
    assert reconstruction.shape == (4, *params.input_shape)
    print("  ✓ Linear decoder works as expected!")

    print("\n=== Testing DRTPDecoder ===")
    # Create the DRTP decoder
    drtp_decoder = DRTPDecoder(params)

    # Create target for DRTP training (same shape as expected output)
    target = torch.randn(4, *params.input_shape)

    # Forward pass with target (training mode)
    reconstruction_drtp = drtp_decoder(sample_input, target)

    print(f"DRTP Decoder (training mode):")
    print(f"  Input shape: {sample_input.shape}")
    print(f"  Target shape: {target.shape}")
    print(f"  Output reconstruction shape: {reconstruction_drtp.shape}")
    assert reconstruction_drtp.shape == (4, *params.input_shape)
    print("  ✓ DRTP decoder training mode works!")

    # Forward pass without target (inference mode)
    reconstruction_inference = drtp_decoder(sample_input, target=None)

    print(f"DRTP Decoder (inference mode):")
    print(f"  Input shape: {sample_input.shape}")
    print(f"  Output reconstruction shape: {reconstruction_inference.shape}")
    assert reconstruction_inference.shape == (4, *params.input_shape)
    print("  ✓ DRTP decoder inference mode works!")

    print("\n=== Testing ConvDecoder ===")
    # Create the convolutional decoder
    conv_decoder = ConvDecoder(params)

    # Forward pass
    reconstruction_conv = conv_decoder(sample_input)

    print(f"Conv Decoder:")
    print(f"  Input shape: {sample_input.shape}")
    print(f"  Output reconstruction shape: {reconstruction_conv.shape}")
    assert reconstruction_conv.shape == (4, *params.input_shape)
    print("  ✓ Conv decoder works as expected!")

    print("\nAll decoders work correctly!")
