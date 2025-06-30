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
        in_features = prod(params.input_shape)

        # Define layers separately since we need to apply DRTP manually
        self.layer1 = nn.Linear(in_features, out_features=1000)
        self.activation1 = params.activation_fn()  # Usually Tanh
        self.drtp1 = DRTPLayer(target_dim=self.input_shape, hidden_dim=1000)

        # Second layer: 1000 units
        self.layer2 = nn.Linear(1000, 1000)
        self.activation2 = params.activation_fn()  # Usually Tanh
        self.drtp2 = DRTPLayer(target_dim=self.input_shape, hidden_dim=1000)

        # Output layer (no DRTP - uses standard gradients)
        self.layer3 = nn.Linear(1000, out_features=self.latent_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, x: Tensor, target: Tensor = None) -> Tensor:
        # Flatten the input tensor to a single feature vector
        x = x.reshape(x.shape[0], -1)  # Reshape to (batch_size, num_features)

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

        return output


class Conv2D(BaseEncoder):
    def __init__(self, params: EncoderParams):
        super().__init__(params)

        # Conv layer 1: 32 channels, 5x5 kernel, stride=1, padding=2
        self.conv1 = nn.Conv2d(self.input_channels, out_channels=32, kernel_size=5, stride=1, padding=2, bias=True)
        self.activation1 = params.activation_fn()  # Usually GELU
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the output size after conv and pooling
        h, w = self.spatial_dimensions
        # After conv (same size due to padding=2, kernel=5): h x w
        # After maxpool (stride=2): h//2 x w//2
        conv_output_size = 32 * (h // 2) * (w // 2)

        # Linear layer 2: 1000 units
        self.layer2 = nn.Linear(conv_output_size, 1000, bias=True)
        self.activation2 = params.activation_fn()  # Usually GELU

        # Linear layer 3 (output): latent_dim units
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

        # Calculate the output size after conv and pooling
        h, w = self.spatial_dimensions
        # After conv (same size due to padding=2, kernel=5): h x w
        # After maxpool (stride=2): h//2 x w//2
        conv_output_size = 32 * (h // 2) * (w // 2)

        # Linear layer 2: 1000 units
        self.layer2 = nn.Linear(conv_output_size, 1000, bias=True)
        self.activation2 = params.activation_fn()  # Usually Tanh
        self.drtp2 = DRTPLayer(target_dim=self.input_shape, hidden_dim=[1000])

        # Linear layer 3 (output): latent_dim units
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


# -------------------------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Setup and parameters
    # -----------------------------------------------------------------------------------

    print("=== Encoder Examples ===\n")

    # Example parameters for cognitive maps (32x16 obstacle grid, single channel)
    params = EncoderParams(
        input_shape=(1, 32, 16),  # 1 channel, 32x16 grid (512 total features)
        latent_dim=128,  # Compress to 128-dimensional representation
        activation_fn=nn.GELU,  # Using GELU activation for non-DRTP encoders
    )

    # Parameters for DRTP encoders (typically use Tanh)
    drtp_params = EncoderParams(input_shape=(1, 32, 16), latent_dim=128, activation_fn=nn.Tanh)

    # Create sample input batch (4 samples)
    batch_size = 4
    sample_input = torch.randn(batch_size, *params.input_shape)

    print(f"Input shape: {sample_input.shape}")
    print(f"Target latent dimension: {params.latent_dim}\n")

    # -----------------------------------------------------------------------------------
    # Linear Encoder Example
    # -----------------------------------------------------------------------------------

    print("1. Linear Encoder:")
    linear_encoder = Linear(params)

    # Forward pass
    with torch.no_grad():
        linear_output = linear_encoder(sample_input)

    print(f"   Input features: {prod(params.input_shape)}")
    print(f"   Output shape: {linear_output.shape}")
    print(f"   Output range: [{linear_output.min():.3f}, {linear_output.max():.3f}]")
    print(f"   Sparsity (zeros): {(linear_output == 0).float().mean():.3f}\n")

    # -----------------------------------------------------------------------------------
    # DRTP Linear Encoder Example
    # -----------------------------------------------------------------------------------

    print("2. DRTP Linear Encoder:")
    drtp_linear_encoder = DRTPLinear(drtp_params)

    # Forward pass (requires target for DRTP)
    target = torch.randn(batch_size, *drtp_params.input_shape)
    with torch.no_grad():
        drtp_linear_output = drtp_linear_encoder(sample_input, target)

    print(f"   Input features: {prod(drtp_params.input_shape)}")
    print(f"   Output shape: {drtp_linear_output.shape}")
    print(f"   Output range: [{drtp_linear_output.min():.3f}, {drtp_linear_output.max():.3f}]")
    print(f"   Sparsity (zeros): {(drtp_linear_output == 0).float().mean():.3f}\n")

    # -----------------------------------------------------------------------------------
    # Conv2D Encoder Example
    # -----------------------------------------------------------------------------------

    print("3. Conv2D Encoder:")
    conv_encoder = Conv2D(params)

    # Calculate expected conv output size
    h, w = params.input_shape[1], params.input_shape[2]
    expected_conv_features = 32 * (h // 2) * (w // 2)

    with torch.no_grad():
        conv_output = conv_encoder(sample_input)

    print(f"   Spatial dimensions: {h}x{w}")
    print(f"   After conv+pool: 32x{h//2}x{w//2} = {expected_conv_features} features")
    print(f"   Output shape: {conv_output.shape}")
    print(f"   Output range: [{conv_output.min():.3f}, {conv_output.max():.3f}]")
    print(f"   Sparsity (zeros): {(conv_output == 0).float().mean():.3f}\n")

    # -----------------------------------------------------------------------------------
    # DRTP Conv2D Encoder Example
    # -----------------------------------------------------------------------------------

    print("4. DRTP Conv2D Encoder:")
    drtp_conv_encoder = DRTPConv2D(drtp_params)

    with torch.no_grad():
        drtp_conv_output = drtp_conv_encoder(sample_input, target)

    print(f"   Spatial dimensions: {h}x{w}")
    print(f"   After conv+pool: 32x{h//2}x{w//2} = {expected_conv_features} features")
    print(f"   Output shape: {drtp_conv_output.shape}")
    print(f"   Output range: [{drtp_conv_output.min():.3f}, {drtp_conv_output.max():.3f}]")
    print(f"   Sparsity (zeros): {(drtp_conv_output == 0).float().mean():.3f}\n")

    # -----------------------------------------------------------------------------------
    # Model comparison
    # -----------------------------------------------------------------------------------

    print("5. Model Size Comparison:")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    models = {
        "Linear": linear_encoder,
        "DRTP Linear": drtp_linear_encoder,
        "Conv2D": conv_encoder,
        "DRTP Conv2D": drtp_conv_encoder,
    }

    for name, model in models.items():
        param_count = count_parameters(model)
        print(f"   {name}: {param_count:,} parameters")

    print("\n=== Examples completed ===")
