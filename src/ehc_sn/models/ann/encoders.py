"""
Neural network encoders for the entorhinal-hippocampal circuit (EHC) spatial navigation library.

This module implements various encoder architectures that transform sensory inputs
(e.g., obstacle maps, cognitive maps) into latent representations suitable for spatial
navigation and memory tasks. The encoders support both standard backpropagation (BP),
Direct Random Target Projection (DRTP), and Direct Feedback Alignment (DFA) training methods.

The module provides six main encoder types:
1. Linear: Fully connected layers for flattened inputs
2. DRTPLinear: DRTP-enabled fully connected encoder
3. DFALinear: DFA-enabled fully connected encoder
4. Conv2D: Convolutional encoder for spatial data
5. DRTPConv2D: DRTP-enabled convolutional encoder
6. DFAConv2D: DFA-enabled convolutional encoder

All encoders follow a consistent architecture pattern:
- Input processing (flattening or convolution)
- Two hidden layers with 1024 and 512 units respectively
- Output layer projecting to latent dimension
- Appropriate activation functions for each training method

Key Features:
- Pydantic-based parameter validation and configuration
- Dynamic dimension calculation for convolutional layers
- Biologically plausible sparse representations
- Support for standard, DRTP, and DFA training algorithms
- Consistent interface across all encoder types

Example:
    >>> params = EncoderParams(
    ...     input_shape=(1, 32, 16),
    ...     latent_dim=128,
    ...     activation_fn=nn.GELU
    ... )
    >>> encoder = Linear(params)
    >>> latent = encoder(input_tensor)

References:
    - Lillicrap, T. P., et al. (2016). Random synaptic feedback weights support
      error backpropagation for deep learning. Nature Communications, 7, 13276.
    - O'Keefe, J., & Nadel, L. (1978). The hippocampus as a cognitive map.
"""

from math import prod
from typing import Optional, Tuple

import torch
from pydantic import BaseModel, Field, model_validator
from torch import Tensor, nn

from ehc_sn.hooks.registry import registry
from ehc_sn.modules.dfa import DFALayer, clear_dfa_error, register_dfa_hook
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
        >>> encoder = Linear(params)  # or Conv2D, DRTPLinear, etc.
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
        self.dfa1 = DFALayer(output_dim=self.latent_dim, hidden_dim=1024)

        # Second layer: 512 units
        self.layer2 = nn.Linear(in_features=1024, out_features=512)
        self.activation2 = params.activation_fn()  # Usually Tanh
        self.dfa2 = DFALayer(output_dim=self.latent_dim, hidden_dim=512)

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

        # Register DFA hook on the current output every forward (per batch)
        if output.requires_grad:
            clear_dfa_error()
            register_dfa_hook(output)

        return output


class Conv2D(BaseEncoder):
    """Convolutional neural network encoder using standard backpropagation training.

    This encoder uses convolutional layers to process spatial data while preserving
    spatial relationships in the input. It's particularly effective for obstacle maps,
    cognitive maps, and other spatially structured data where local patterns and
    spatial features are important.

    Architecture:
        Input → Conv2d(5x5, 32) → GELU → MaxPool(2x2) → Flatten → Linear(512) → GELU → Linear(latent_dim) → Sigmoid

    Design choices:
    - 5x5 convolution kernel captures local spatial patterns effectively
    - 32 output channels provide sufficient feature diversity
    - Padding=2 with kernel=5 preserves spatial dimensions before pooling
    - MaxPool with 2x2 stride reduces spatial dimensions by half
    - Single fully connected layer (512 units) for final feature processing
    - GELU activations for smooth gradients and improved training

    The encoder automatically calculates the correct linear layer input size based
    on the spatial dimensions and convolution/pooling operations, making it adaptable
    to different input sizes.

    Args:
        params: EncoderParams with activation_fn typically set to nn.GELU.

    Example:
        >>> params = EncoderParams(
        ...     input_shape=(1, 32, 16),  # After conv+pool: 32 channels × 16×8 = 4096 features
        ...     latent_dim=128,
        ...     activation_fn=nn.GELU
        ... )
        >>> encoder = Conv2D(params)
        >>> input_batch = torch.randn(4, 1, 32, 16)
        >>> latent = encoder(input_batch)  # shape: (4, 128)

    Note:
        Spatial dimensions must be even numbers to work correctly with MaxPool(2x2).
        The target parameter in forward() is ignored but kept for API consistency.
    """

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

        # Linear layer 2: 512 units
        self.layer2 = nn.Linear(conv_output_size, 512, bias=True)
        self.activation2 = params.activation_fn()  # Usually GELU

        # Linear layer 3 (output): latent_dim units
        self.layer3 = nn.Linear(in_features=512, out_features=self.latent_dim, bias=True)
        self.output_activation = nn.Sigmoid()

    def forward(self, x: Tensor, target: Optional[Tensor] = None) -> Tensor:
        """Forward pass through the convolutional encoder.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            target: Ignored for standard encoder, kept for API consistency.

        Returns:
            Latent representation of shape (batch_size, latent_dim) with values
            in range [0, 1] due to Sigmoid output activation.
        """
        # First layer: convolution → activation → pooling
        h1 = self.conv1(x)
        h1 = self.activation1(h1)
        h1 = self.pool1(h1)

        # Flatten for FC layers
        h1 = h1.reshape(h1.size(0), -1)

        # Second layer: fully connected
        h2 = self.layer2(h1)
        h2 = self.activation2(h2)

        # Output layer
        output = self.layer3(h2)
        output = self.output_activation(output)

        return output


class DRTPConv2D(BaseEncoder):
    """Convolutional encoder using Direct Random Target Projection (DRTP) training.

    This encoder combines the spatial processing capabilities of convolutional layers
    with the biologically plausible learning mechanism of DRTP. It's designed for
    scenarios where both spatial feature extraction and biological learning
    constraints are important.

    Architecture:
        Input → Conv2d(5x5, 32) → Tanh → DRTP → MaxPool(2x2) → Flatten → Linear(512) → Tanh → DRTP → Linear(latent_dim) → Sigmoid

    Key features:
    - Convolutional layer processes spatial patterns while preserving local structure
    - DRTP applied to both convolutional and fully connected hidden layers
    - Uses Tanh activation functions for compatibility with DRTP
    - Random feedback weights enable biologically plausible learning
    - Output layer uses standard backpropagation for final projection

    The combination of convolution and DRTP makes this encoder suitable for:
    - Spatial navigation tasks requiring biological plausibility
    - Investigating how spatial processing might work in biological neural networks
    - Research comparing biological vs. standard learning mechanisms
    - Applications where spatial features and learning constraints both matter

    Args:
        params: EncoderParams with activation_fn typically set to nn.Tanh.

    Example:
        >>> params = EncoderParams(
        ...     input_shape=(1, 32, 16),
        ...     latent_dim=128,
        ...     activation_fn=nn.Tanh
        ... )
        >>> encoder = DRTPConv2D(params)
        >>> input_batch = torch.randn(4, 1, 32, 16)
        >>> latent = encoder(input_batch)  # shape: (4, 128), target=None is default
        >>> # During training: latent = encoder(input_batch, target_batch)

    Note:
        Target tensor is only required during backward pass for DRTP computation.
        Forward pass accepts optional target parameter for consistent API.
        Spatial dimensions should be even numbers for MaxPool compatibility.

    References:
        Lillicrap, T. P., et al. (2016). Random synaptic feedback weights support
        error backpropagation for deep learning. Nature Communications, 7, 13276.
    """

    def __init__(self, params: EncoderParams):
        super().__init__(params)

        # Get spatial dimensions
        h, w = self.spatial_dimensions

        # Conv layer 1: 32 channels, 5x5 kernel, stride=1, padding=2
        self.conv1 = nn.Conv2d(self.input_channels, out_channels=32, kernel_size=5, stride=1, padding=2, bias=True)
        self.activation1 = params.activation_fn()  # Usually Tanh
        self.drtp1 = DRTPLayer(target_dim=self.latent_dim, hidden_dim=[32, h, w])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the output size after conv and pooling
        # After conv (same size due to padding=2, kernel=5): h x w
        # After maxpool (stride=2): h//2 x w//2
        conv_output_size = 32 * (h // 2) * (w // 2)

        # Linear layer 2: 512 units
        self.layer2 = nn.Linear(conv_output_size, out_features=512, bias=True)
        self.activation2 = params.activation_fn()  # Usually Tanh
        self.drtp2 = DRTPLayer(target_dim=self.latent_dim, hidden_dim=[512])

        # Linear layer 3 (output): latent_dim units
        self.layer3 = nn.Linear(in_features=512, out_features=self.latent_dim, bias=True)
        self.output_activation = nn.Sigmoid()
        # No DRTP on output layer (uses standard gradients)

    def forward(self, x: Tensor, target: Optional[Tensor] = None) -> Tensor:
        """Forward pass through the DRTP convolutional encoder.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            target: Target tensor for DRTP gradient computation during backward pass.
                Can be None during forward-only inference.

        Returns:
            Latent representation of shape (batch_size, latent_dim) with values
            in range [0, 1] due to Sigmoid output activation.
        """
        # First layer: convolution → activation → DRTP → pooling
        h1 = self.conv1(x)
        h1 = self.activation1(h1)
        h1 = self.drtp1(h1, target)  # Apply DRTP to conv layer
        h1 = self.pool1(h1)

        # Flatten for FC layers
        h1 = h1.reshape(h1.size(0), -1)

        # Second layer: fully connected with DRTP
        h2 = self.layer2(h1)
        h2 = self.activation2(h2)
        h2 = self.drtp2(h2, target)  # Apply DRTP to linear layer

        # Output layer (standard backpropagation)
        output = self.layer3(h2)
        output = self.output_activation(output)

        return output


class DFAConv2D(BaseEncoder):
    """Convolutional encoder using Direct Feedback Alignment (DFA) training.

    This encoder combines the spatial processing capabilities of convolutional layers
    with the biologically plausible learning mechanism of DFA. It's designed for
    scenarios where both spatial feature extraction and biological learning
    constraints are important, providing an alternative to both standard backprop and DRTP.

    Architecture:
        Input → Conv2d(5x5, 32) → Tanh → DFA → MaxPool(2x2) → Flatten → Linear(512) → Tanh → DFA → Linear(latent_dim) → Sigmoid

    Key features:
    - Convolutional layer processes spatial patterns while preserving local structure
    - DFA applied to both convolutional and fully connected hidden layers
    - Uses Tanh activation functions for compatibility with DFA
    - Random feedback weights enable biologically plausible learning
    - Output layer uses standard backpropagation for final projection
    - Error signals are propagated directly from output to hidden layers

    The combination of convolution and DFA makes this encoder suitable for:
    - Spatial navigation tasks requiring biological plausibility
    - Investigating how spatial processing might work in biological neural networks
    - Research comparing biological vs. standard learning mechanisms
    - Applications where spatial features and learning constraints both matter

    Args:
        params: EncoderParams with activation_fn typically set to nn.Tanh.

    Example:
        >>> params = EncoderParams(
        ...     input_shape=(1, 32, 16),
        ...     latent_dim=128,
        ...     activation_fn=nn.Tanh
        ... )
        >>> encoder = DFAConv2D(params)
        >>> input_batch = torch.randn(4, 1, 32, 16)
        >>> latent = encoder(input_batch)  # shape: (4, 128), grad_output=None is default
        >>> # During training: latent = encoder(input_batch, grad_output_batch)

    Note:
        Error signal is only required during backward pass for DFA computation.
        Forward pass accepts optional grad_output parameter for consistent API.
        Spatial dimensions should be even numbers for MaxPool compatibility.

    References:
        Lillicrap, T. P., et al. (2016). Random synaptic feedback weights support
        error backpropagation for deep learning. Nature Communications, 7, 13276.
    """

    def __init__(self, params: EncoderParams):
        super().__init__(params)

        # Get spatial dimensions
        h, w = self.spatial_dimensions

        # Conv layer 1: 32 channels, 5x5 kernel, stride=1, padding=2
        self.conv1 = nn.Conv2d(self.input_channels, out_channels=32, kernel_size=5, stride=1, padding=2, bias=True)
        self.activation1 = params.activation_fn()  # Usually Tanh
        self.dfa1 = DFALayer(output_dim=self.latent_dim, hidden_dim=[32, h, w])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the output size after conv and pooling
        # After conv (same size due to padding=2, kernel=5): h x w
        # After maxpool (stride=2): h//2 x w//2
        conv_output_size = 32 * (h // 2) * (w // 2)

        # Linear layer 2: 512 units
        self.layer2 = nn.Linear(conv_output_size, out_features=512, bias=True)
        self.activation2 = params.activation_fn()  # Usually Tanh
        self.dfa2 = DFALayer(output_dim=self.latent_dim, hidden_dim=[512])

        # Linear layer 3 (output): latent_dim units
        self.layer3 = nn.Linear(in_features=512, out_features=self.latent_dim, bias=True)
        self.output_activation = nn.Sigmoid()
        # No DFA on output layer (uses standard gradients)

    def forward(self, x: Tensor, target: Optional[Tensor] = None) -> Tensor:
        """Forward pass through the DFA convolutional encoder.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            target: Ignored for standard encoder, kept for API consistency.

        Returns:
            Latent representation of shape (batch_size, latent_dim) with values
            in range [0, 1] due to Sigmoid output activation.
        """
        # First layer: convolution → activation → DFA → pooling
        h1 = self.conv1(x)
        h1 = self.activation1(h1)
        h1 = self.dfa1(h1)  # Apply DFA to conv layer
        h1 = self.pool1(h1)

        # Flatten for FC layers
        h1 = h1.reshape(h1.size(0), -1)

        # Second layer: fully connected with DFA
        h2 = self.layer2(h1)
        h2 = self.activation2(h2)
        h2 = self.dfa2(h2)  # Apply DFA to linear layer

        # Output layer (standard backpropagation)
        output = self.layer3(h2)
        output = self.output_activation(output)

        # Register DFA hook on the current output every forward (per batch)
        if output.requires_grad:
            clear_dfa_error()
            register_dfa_hook(output)

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
    # Conv2D Encoder Example
    # -----------------------------------------------------------------------------------

    print("4. Conv2D Encoder:")
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

    print("5. DRTP Conv2D Encoder:")
    drtp_conv_encoder = DRTPConv2D(drtp_params)

    with torch.no_grad():
        drtp_conv_output = drtp_conv_encoder(sample_input)

    print(f"   Spatial dimensions: {h}x{w}")
    print(f"   After conv+pool: 32x{h//2}x{w//2} = {expected_conv_features} features")
    print(f"   Output shape: {drtp_conv_output.shape}")
    print(f"   Output range: [{drtp_conv_output.min():.3f}, {drtp_conv_output.max():.3f}]")
    print(f"   Sparsity (zeros): {(drtp_conv_output == 0).float().mean():.3f}\n")

    # -----------------------------------------------------------------------------------
    # DFA Conv2D Encoder Example
    # -----------------------------------------------------------------------------------

    print("6. DFA Conv2D Encoder:")
    dfa_conv_encoder = DFAConv2D(dfa_params)

    with torch.no_grad():
        dfa_conv_output = dfa_conv_encoder(sample_input)

    print(f"   Spatial dimensions: {h}x{w}")
    print(f"   After conv+pool: 32x{h//2}x{w//2} = {expected_conv_features} features")
    print(f"   Output shape: {dfa_conv_output.shape}")
    print(f"   Output range: [{dfa_conv_output.min():.3f}, {dfa_conv_output.max():.3f}]")
    print(f"   Sparsity (zeros): {(dfa_conv_output == 0).float().mean():.3f}\n")

    # -----------------------------------------------------------------------------------
    # Model comparison
    # -----------------------------------------------------------------------------------

    print("7. Model Size Comparison:")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    models = {
        "Linear": linear_encoder,
        "DRTP Linear": drtp_linear_encoder,
        "DFA Linear": dfa_linear_encoder,
        "Conv2D": conv_encoder,
        "DRTP Conv2D": drtp_conv_encoder,
        "DFA Conv2D": dfa_conv_encoder,
    }

    for name, model in models.items():
        param_count = count_parameters(model)
        print(f"   {name}: {param_count:,} parameters")

    print("\n=== Examples completed ===")
