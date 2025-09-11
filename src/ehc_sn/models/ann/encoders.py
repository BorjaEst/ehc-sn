"""Neural network encoders for entorhinal-hippocampal circuit spatial navigation modeling.

This module implements various encoder architectures that transform sensory inputs
(e.g., obstacle maps, cognitive maps) into latent representations suitable for spatial
navigation and memory tasks. The encoders support multiple biologically-inspired
training methods including standard backpropagation (BP), Direct Random Target
Projection (DRTP), and Direct Feedback Alignment (DFA).

The encoder architectures are designed to mimic the encoding properties of the
medial entorhinal cortex (MEC), which processes spatial information and projects
to hippocampal regions for memory formation and spatial navigation.

Key Features:
    - Pydantic-based parameter validation and configuration management
    - Biologically plausible sparse representations mimicking place cell activity
    - Support for standard BP, DRTP, and DFA training algorithms
    - Consistent interface across all encoder implementations
    - Configurable activation functions optimized for each training method

Classes:
    EncoderParams: Configuration parameters for encoder initialization
    BaseEncoder: Abstract base class defining the encoder interface
    Linear: Fully connected encoder for flattened spatial inputs
    DRTPLinear: DRTP-enabled encoder with target projection learning
    DFALinear: DFA-enabled encoder with random feedback alignment

Architecture Pattern:
    All encoders follow a consistent three-layer architecture:
    - Input processing layer (flattening of spatial inputs)
    - Two hidden layers (1024 and 512 units) for feature extraction
    - Output projection layer to specified latent dimension
    - Appropriate activation functions for biological plausibility

Examples:
    >>> # Standard encoder for spatial navigation tasks
    >>> params = EncoderParams(
    ...     input_shape=(1, 32, 16),  # Single-channel obstacle map
    ...     latent_dim=128,           # Compressed representation
    ...     activation_fn=nn.GELU     # Smooth activation for BP training
    ... )
    >>> encoder = Linear(params)
    >>> latent = encoder(spatial_input)
    >>>
    >>> # DRTP encoder with biologically motivated activation
    >>> drtp_params = EncoderParams(
    ...     input_shape=(1, 32, 16),
    ...     latent_dim=128,
    ...     activation_fn=nn.Tanh  # Bounded activation for DRTP stability
    ... )
    >>> drtp_encoder = DRTPLinear(drtp_params)

References:
    - Lillicrap, T. P., et al. (2016). Random synaptic feedback weights support
      error backpropagation for deep learning. Nature Communications, 7, 13276.
    - O'Keefe, J., & Nadel, L. (1978). The hippocampus as a cognitive map.
      Oxford University Press.
"""

from math import prod
from typing import Optional, Tuple

import torch
from pydantic import BaseModel, Field, model_validator
from torch import Tensor, nn

from ehc_sn.hooks.registry import registry
from ehc_sn.modules import dfa
from ehc_sn.modules.drtp import DRTPLayer


class EncoderParams(BaseModel):
    """Configuration parameters for neural network encoders.

    This class defines the essential parameters needed to configure any encoder
    in the EHC library. It uses Pydantic for automatic validation and type checking.

    Attributes:
        input_shape: 3D tensor shape as (channels, height, width). For cognitive
            maps, typically (1, 32, 16) representing a single-channel obstacle grid.
        latent_dim: Size of the compressed latent representation. Should be smaller
            than the input dimension to achieve meaningful compression. Typical
            values range from 64 to 512 depending on the complexity of the task.
        activation_fn: PyTorch activation function class (not instance). Use nn.GELU
            for standard encoders and nn.Tanh for DRTP encoders, as these have been
            shown to work well in practice.

    Examples:
        >>> # For a 32x16 obstacle map compressed to 128 dimensions
        >>> params = EncoderParams(
        ...     input_shape=(1, 32, 16),
        ...     latent_dim=128,
        ...     activation_fn=nn.GELU
        ... )

        >>> # For DRTP training (use Tanh activation)
        >>> drtp_params = EncoderParams(
        ...     input_shape=(1, 32, 16),
        ...     latent_dim=128,
        ...     activation_fn=nn.Tanh
        ... )

    Note:
        The activation_fn should be a class reference (e.g., nn.GELU), not an
        instance (e.g., nn.GELU()). The encoder will instantiate it as needed.
    """

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    input_shape: Tuple[int, int, int] = Field(..., description="3D input shape as (channels, height, width)")
    latent_dim: int = Field(..., description="Dimensionality of the latent representation", gt=0)
    activation_fn: object = Field(..., description="PyTorch activation function class (not instance)")


class BaseEncoder(nn.Module):
    """Abstract base class for neural network encoders in the EHC library.

    This class defines the common interface and utility properties for all encoder
    implementations. It provides a consistent API for working with different encoder
    architectures while maintaining biological plausibility principles.

    All encoders in this library follow the pattern of transforming high-dimensional
    sensory inputs (such as obstacle maps or visual scenes) into lower-dimensional
    latent representations that can be processed by the hippocampal circuit.

    Args:
        params: EncoderParams instance containing configuration parameters.

    Attributes:
        params: Stored configuration parameters for the encoder.

    Properties:
        input_shape: 3D shape of input tensors (channels, height, width).
        input_channels: Number of input channels.
        spatial_dimensions: Spatial dimensions as (height, width) tuple.
        latent_dim: Dimensionality of the output latent representation.

    Abstract Methods:
        forward: Must be implemented by subclasses to define the encoding process.

    Example:
        This is an abstract class and cannot be instantiated directly. Use one of
        the concrete implementations:

        >>> params = EncoderParams(...)
        >>> encoder = Linear(params)  # or DRTPLinear, DFALinear
        >>> latent = encoder(input_tensor)
    """

    def __init__(self, params: EncoderParams):
        super().__init__()
        self.params = params

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the encoder.

        Args:
            x: Input tensor of shape (batch_size, *input_shape).

        Returns:
            Latent representation tensor of shape (batch_size, latent_dim).

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement forward method")

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        """Returns the expected input tensor shape (channels, height, width)."""
        return self.params.input_shape

    @property
    def input_channels(self) -> int:
        """Returns the number of input channels."""
        return self.input_shape[0]

    @property
    def spatial_dimensions(self) -> Tuple[int, int]:
        """Returns the spatial dimensions as (height, width)."""
        return self.input_shape[1], self.input_shape[2]

    @property
    def latent_dim(self) -> int:
        """Returns the dimensionality of the latent representation."""
        return self.params.latent_dim


class Linear(BaseEncoder):
    """Linear neural network encoder using standard backpropagation training.

    This encoder transforms flattened multi-dimensional inputs into fixed-size
    embeddings through a sequence of fully connected layers. It's designed for
    processing structured obstacle maps where spatial relationships can be captured
    through dense connectivity patterns.

    Architecture:
        Input (flattened) → Linear(1024) → GELU → Linear(512) → GELU → Linear(latent_dim) → ReLU

    The architecture uses:
    - Two hidden layers: first with 1024 units, second with 512 units for progressive dimensionality reduction
    - GELU activation functions in hidden layers for smooth, differentiable non-linearity
    - ReLU activation in the output layer to ensure non-negative sparse representations
    - Bias terms in all layers for improved expressiveness

    This encoder is suitable for:
    - Cognitive map encoding where spatial structure is less critical
    - Baseline comparisons with more complex architectures
    - Cases where computational efficiency is important
    - Initial prototyping and development

    Args:
        params: EncoderParams with activation_fn typically set to nn.GELU.

    Example:
        >>> params = EncoderParams(
        ...     input_shape=(1, 32, 16),  # 512 input features when flattened
        ...     latent_dim=128,
        ...     activation_fn=nn.GELU
        ... )
        >>> encoder = Linear(params)
        >>> input_batch = torch.randn(4, 1, 32, 16)  # batch of 4 samples
        >>> latent = encoder(input_batch)  # shape: (4, 128)

    Note:
        The target parameter in forward() is ignored but kept for API consistency
        with DRTP encoders.
    """

    def __init__(self, params: EncoderParams):
        super().__init__(params)
        in_features = prod(params.input_shape)

        # First layer: 1024 units
        self.layer1 = nn.Linear(in_features, 1024, bias=True)
        self.activation1 = params.activation_fn()  # Usually GELU

        # Second layer: 512 units
        self.layer2 = nn.Linear(1024, 512, bias=True)
        self.activation2 = params.activation_fn()  # Usually GELU

        # Output layer: latent_dim units
        self.layer3 = nn.Linear(512, params.latent_dim, bias=True)
        self.output_activation = nn.GELU()

    def forward(self, x: Tensor, target: Optional[Tensor] = None) -> Tensor:
        """Forward pass through the linear encoder.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            target: Ignored for standard encoder, kept for API consistency.

        Returns:
            Latent representation of shape (batch_size, latent_dim) with
            non-negative values due to ReLU output activation.
        """
        # Flatten the input tensor to a single feature vector
        x = x.reshape(x.shape[0], -1)  # Reshape to (batch_size, num_features)

        # First layer
        h1 = self.layer1(x)
        h1 = self.activation1(h1)

        # Store in registry for SRTP decoders
        registry.set_activation("encoder.h1", h1, detach=True)

        # Second layer
        h2 = self.layer2(h1)
        h2 = self.activation2(h2)

        # Store in registry for SRTP decoders
        registry.set_activation("encoder.h2", h2, detach=True)

        # Output layer
        output = self.layer3(h2)
        output = self.output_activation(output)

        return output


class DRTPLinear(BaseEncoder):
    """Linear encoder using Direct Random Target Projection (DRTP) training.

    This encoder implements the DRTP algorithm, which uses random feedback weights
    instead of symmetric weight updates for biologically plausible learning. DRTP
    enables learning without requiring precise error backpropagation through the
    network, making it more similar to biological neural networks.

    Architecture:
        Input (flattened) → Linear(1024) → Tanh → DRTP → Linear(512) → Tanh → DRTP → Linear(latent_dim) → Sigmoid

    Key differences from standard Linear encoder:
    - Uses Tanh activation functions (work better with DRTP)
    - Two hidden layers: first with 1024 units, second with 512 units for progressive dimensionality reduction
    - Applies DRTP layers after hidden layers but not output layer
    - Uses Sigmoid output activation for bounded outputs
    - Target tensor is only required during backward pass for DRTP gradient computation
    - Forward pass accepts optional target parameter for consistent API

    The DRTP mechanism:
    - Generates random feedback weights instead of using transpose of forward weights
    - Computes gradients using these random weights for biologically plausible learning
    - Maintains learning performance comparable to standard backpropagation
    - Provides insights into how biological neural networks might learn

    Args:
        params: EncoderParams with activation_fn typically set to nn.Tanh.

    Example:
        >>> params = EncoderParams(
        ...     input_shape=(1, 32, 16),
        ...     latent_dim=128,
        ...     activation_fn=nn.Tanh
        ... )
        >>> encoder = DRTPLinear(params)
        >>> input_batch = torch.randn(4, 1, 32, 16)
        >>> latent = encoder(input_batch)  # shape: (4, 128), target=None is default
        >>> # During training: latent = encoder(input_batch, target_batch)

    References:
        Lillicrap, T. P., et al. (2016). Random synaptic feedback weights support
        error backpropagation for deep learning. Nature Communications, 7, 13276.
    """

    def __init__(self, params: EncoderParams):
        super().__init__(params)
        in_features = prod(params.input_shape)

        # First layer: 1024 units
        self.layer1 = nn.Linear(in_features, out_features=1024)
        self.activation1 = params.activation_fn()  # Usually Tanh
        self.drtp1 = DRTPLayer(target_dim=self.latent_dim, hidden_dim=1024)

        # Second layer: 512 units
        self.layer2 = nn.Linear(in_features=1024, out_features=512)
        self.activation2 = params.activation_fn()  # Usually Tanh
        self.drtp2 = DRTPLayer(target_dim=self.latent_dim, hidden_dim=512)

        # Output layer (no DRTP - uses standard gradients)
        self.layer3 = nn.Linear(512, out_features=self.latent_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, x: Tensor, target: Optional[Tensor] = None) -> Tensor:
        """Forward pass through the DRTP linear encoder.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            target: Target tensor for DRTP gradient computation during backward pass.
                Can be None during forward-only inference.

        Returns:
            Latent representation of shape (batch_size, latent_dim) with values
            in range [0, 1] due to Sigmoid output activation.
        """
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


class DFALinear(BaseEncoder):
    """Linear encoder using Direct Feedback Alignment (DFA) training.

    This encoder implements the DFA algorithm, which uses random feedback weights
    to propagate error signals directly from the output layer to hidden layers,
    bypassing the need for symmetric weight transport. DFA is biologically plausible
    and provides an alternative to both standard backpropagation and DRTP.

    Architecture:
        Input (flattened) → Linear(1024) → Tanh → DFA → Linear(512) → Tanh → DFA → Linear(latent_dim) → Sigmoid

    Key differences from standard Linear encoder:
    - Uses Tanh activation functions (work well with DFA)
    - Two hidden layers: first with 1024 units, second with 512 units for progressive dimensionality reduction
    - Applies DFA layers after hidden layers but not output layer
    - Uses Sigmoid output activation for bounded outputs
    - Error signal from output layer is propagated directly to hidden layers via random weights
    - Automatically registers DFA hooks during forward pass

    The DFA mechanism:
    - Uses random feedback weights to project output errors to hidden layers
    - Error signals are computed as (output - target) and propagated via random matrices
    - Maintains learning performance while being more biologically plausible than backprop
    - Does not require symmetric weight transport like standard backpropagation

    Args:
        params: EncoderParams with activation_fn typically set to nn.Tanh.

    Example:
        >>> params = EncoderParams(
        ...     input_shape=(1, 32, 16),
        ...     latent_dim=128,
        ...     activation_fn=nn.Tanh
        ... )
        >>> encoder = DFALinear(params)
        >>> input_batch = torch.randn(4, 1, 32, 16)
        >>> latent = encoder(input_batch)  # shape: (4, 128)

    References:
        Lillicrap, T. P., et al. (2016). Random synaptic feedback weights support
        error backpropagation for deep learning. Nature Communications, 7, 13276.
    """

    def __init__(self, params: EncoderParams):
        super().__init__(params)
        in_features = prod(params.input_shape)

        # First layer: 1024 units
        self.layer1 = nn.Linear(in_features, out_features=1024)
        self.activation1 = params.activation_fn()  # Usually Tanh
        self.dfa1 = dfa.DFALayer(output_dim=self.latent_dim, hidden_dim=1024)

        # Second layer: 512 units
        self.layer2 = nn.Linear(in_features=1024, out_features=512)
        self.activation2 = params.activation_fn()  # Usually Tanh
        self.dfa2 = dfa.DFALayer(output_dim=self.latent_dim, hidden_dim=512)

        # Output layer (no DFA - uses standard gradients)
        self.layer3 = nn.Linear(512, out_features=self.latent_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, x: Tensor, target: Optional[Tensor] = None) -> Tensor:
        """Forward pass through the DFA linear encoder.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            target: Ignored for standard encoder, kept for API consistency.

        Returns:
            Latent representation of shape (batch_size, latent_dim) with values
            in range [0, 1] due to Sigmoid output activation.

        Note:
            Hook management for DFA error signal capture is now handled by
            the DFATrainer, not in this forward method. This provides better
            separation of concerns between model computation and training logic.
        """
        # Flatten the input tensor to a single feature vector
        x = x.reshape(x.shape[0], -1)  # Reshape to (batch_size, num_features)

        # First layer
        h1 = self.layer1(x)
        h1 = self.activation1(h1)
        h1 = self.dfa1(h1)

        # Second layer
        h2 = self.layer2(h1)
        h2 = self.activation2(h2)
        h2 = self.dfa2(h2)

        # Output layer (no DFA - uses standard gradients)
        output = self.layer3(h2)
        output = self.output_activation(output)

        return output


# -------------------------------------------------------------------------------------------
# Examples and demonstrations
# -------------------------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Demonstration script showing how to use all encoder types.

    This script provides complete examples of:
    1. Parameter configuration for different encoder types
    2. Model instantiation and forward passes
    3. Performance comparison between architectures
    4. Expected input/output shapes and characteristics

    Run this script directly to see the encoders in action:
        python -m ehc_sn.models.encoders
    """
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

    # Parameters for DFA encoders (typically use Tanh)
    dfa_params = EncoderParams(input_shape=(1, 32, 16), latent_dim=128, activation_fn=nn.Tanh)

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

    # Forward pass (no target needed for forward pass)
    with torch.no_grad():
        drtp_linear_output = drtp_linear_encoder(sample_input)

    print(f"   Input features: {prod(drtp_params.input_shape)}")
    print(f"   Output shape: {drtp_linear_output.shape}")
    print(f"   Output range: [{drtp_linear_output.min():.3f}, {drtp_linear_output.max():.3f}]")
    print(f"   Sparsity (zeros): {(drtp_linear_output == 0).float().mean():.3f}\n")

    # -----------------------------------------------------------------------------------
    # DFA Linear Encoder Example
    # -----------------------------------------------------------------------------------

    print("3. DFA Linear Encoder:")
    dfa_linear_encoder = DFALinear(dfa_params)

    # Forward pass (no error signal needed for forward pass)
    with torch.no_grad():
        dfa_linear_output = dfa_linear_encoder(sample_input)

    print(f"   Input features: {prod(dfa_params.input_shape)}")
    print(f"   Output shape: {dfa_linear_output.shape}")
    print(f"   Output range: [{dfa_linear_output.min():.3f}, {dfa_linear_output.max():.3f}]")
    print(f"   Sparsity (zeros): {(dfa_linear_output == 0).float().mean():.3f}\n")

    # -----------------------------------------------------------------------------------
    # Model comparison
    # -----------------------------------------------------------------------------------

    print("4. Model Size Comparison:")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    models = {
        "Linear": linear_encoder,
        "DRTP Linear": drtp_linear_encoder,
        "DFA Linear": dfa_linear_encoder,
    }

    for name, model in models.items():
        param_count = count_parameters(model)
        print(f"   {name}: {param_count:,} parameters")

    print("\n=== Examples completed ===")
