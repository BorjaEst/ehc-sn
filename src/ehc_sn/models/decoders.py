"""
Neural network decoders for the entorhinal-hippocampal circuit (EHC) spatial navigation library.

This module implements various decoder architectures that transform latent representations
back into reconstructed outputs (e.g., cognitive maps, obstacle maps). The decoders complement
the encoders and support both standard backpropagation (BP), Direct Random Target
Projection (DRTP), and Direct Feedback Alignment (DFA) training methods.

The module provides six main decoder types:
1. Linear: Fully connected layers for reconstructing flattened outputs
2. DRTPLinear: DRTP-enabled fully connected decoder
3. DFALinear: DFA-enabled fully connected decoder
4. Conv2D: Transpose convolutional decoder for spatial data reconstruction
5. DRTPConv2D: DRTP-enabled transpose convolutional decoder
6. DFAConv2D: DFA-enabled transpose convolutional decoder

All decoders follow a consistent architecture pattern:
- Input from latent representation
- Progressive expansion through hidden layers
- Output reconstruction matching original input dimensions
- Appropriate activation functions for each training method

Key Features:
- Pydantic-based parameter validation and configuration
- Dynamic dimension calculation for transpose convolutions
- Symmetric architectures complementing encoder designs
- Support for standard, DRTP, and DFA training algorithms
- Consistent interface across all decoder types

Example:
    >>> params = DecoderParams(
    ...     output_shape=(1, 32, 16),
    ...     latent_dim=128,
    ...     activation_fn=nn.GELU
    ... )
    >>> decoder = Linear(params)
    >>> reconstruction = decoder(latent_tensor)

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

from ehc_sn.hooks.dfa import DFALayer, clear_dfa_error, register_dfa_hook
from ehc_sn.hooks.drtp import DRTPLayer


class DecoderParams(BaseModel):
    """Configuration parameters for neural network decoders.

    This class defines the essential parameters needed to configure any decoder
    in the EHC library. It uses Pydantic for automatic validation and type checking.

    Attributes:
        output_shape: 3D tensor shape as (channels, height, width) that the decoder
            should reconstruct. For cognitive maps, typically (1, 32, 16) representing
            a single-channel obstacle grid.
        latent_dim: Size of the input latent representation. Should match the latent
            dimension of the corresponding encoder.
        activation_fn: PyTorch activation function class (not instance). Use nn.GELU
            for standard decoders and nn.Tanh for DRTP decoders, as these have been
            shown to work well in practice.

    Examples:
        >>> # For reconstructing a 32x16 obstacle map from 128-dim latent
        >>> params = DecoderParams(
        ...     output_shape=(1, 32, 16),
        ...     latent_dim=128,
        ...     activation_fn=nn.GELU
        ... )

        >>> # For DRTP training (use Tanh activation)
        >>> drtp_params = DecoderParams(
        ...     output_shape=(1, 32, 16),
        ...     latent_dim=128,
        ...     activation_fn=nn.Tanh
        ... )

    Note:
        The activation_fn should be a class reference (e.g., nn.GELU), not an
        instance (e.g., nn.GELU()). The decoder will instantiate it as needed.
    """

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    output_shape: Tuple[int, int, int] = Field(..., description="3D output shape as (channels, height, width)")
    latent_dim: int = Field(..., description="Dimensionality of the input latent representation", gt=0)
    activation_fn: object = Field(..., description="PyTorch activation function class (not instance)")


class BaseDecoder(nn.Module):
    """Abstract base class for neural network decoders in the EHC library.

    This class defines the common interface and utility properties for all decoder
    implementations. It provides a consistent API for working with different decoder
    architectures while maintaining biological plausibility principles.

    All decoders in this library follow the pattern of transforming lower-dimensional
    latent representations back into higher-dimensional reconstructed outputs that
    match the original input dimensions.

    Args:
        params: DecoderParams instance containing configuration parameters.

    Attributes:
        params: Stored configuration parameters for the decoder.

    Properties:
        output_shape: 3D shape of output tensors (channels, height, width).
        output_channels: Number of output channels.
        spatial_dimensions: Spatial dimensions as (height, width) tuple.
        latent_dim: Dimensionality of the input latent representation.

    Abstract Methods:
        forward: Must be implemented by subclasses to define the decoding process.

    Example:
        This is an abstract class and cannot be instantiated directly. Use one of
        the concrete implementations:

        >>> params = DecoderParams(...)
        >>> decoder = Linear(params)  # or Conv2D, DRTPLinear, etc.
        >>> reconstruction = decoder(latent_tensor)
    """

    def __init__(self, params: DecoderParams):
        super().__init__()
        self.params = params

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the decoder.

        Args:
            x: Input latent tensor of shape (batch_size, latent_dim).

        Returns:
            Reconstructed output tensor of shape (batch_size, *output_shape).

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement forward method")

    @property
    def output_shape(self) -> Tuple[int, int, int]:
        """Returns the expected output tensor shape (channels, height, width)."""
        return self.params.output_shape

    @property
    def output_channels(self) -> int:
        """Returns the number of output channels."""
        return self.output_shape[0]

    @property
    def spatial_dimensions(self) -> Tuple[int, int]:
        """Returns the spatial dimensions as (height, width)."""
        return self.output_shape[1], self.output_shape[2]

    @property
    def latent_dim(self) -> int:
        """Returns the dimensionality of the input latent representation."""
        return self.params.latent_dim


class Linear(BaseDecoder):
    """Linear neural network decoder using standard backpropagation training.

    This decoder transforms latent representations into reconstructed outputs through
    a sequence of fully connected layers. It mirrors the Linear encoder architecture
    in reverse, expanding from the latent dimension back to the full output size.

    Architecture:
        Latent input → Linear(512) → GELU → Linear(1024) → GELU → Linear(output_size) → Sigmoid

    The architecture uses:
    - Two hidden layers: first with 512 units, second with 1024 units for progressive expansion
    - GELU activation functions in hidden layers for smooth, differentiable non-linearity
    - Sigmoid activation in the output layer to ensure bounded outputs in [0,1]
    - Bias terms in all layers for improved expressiveness
    - Final reshape to match the original spatial dimensions

    This decoder is suitable for:
    - Reconstructing cognitive maps where spatial structure is captured implicitly
    - Baseline comparisons with more complex architectures
    - Cases where computational efficiency is important
    - Autoencoder training with Linear encoders

    Args:
        params: DecoderParams with activation_fn typically set to nn.GELU.

    Example:
        >>> params = DecoderParams(
        ...     output_shape=(1, 32, 16),  # 512 output features when flattened
        ...     latent_dim=128,
        ...     activation_fn=nn.GELU
        ... )
        >>> decoder = Linear(params)
        >>> latent_batch = torch.randn(4, 128)  # batch of 4 latent vectors
        >>> reconstruction = decoder(latent_batch)  # shape: (4, 1, 32, 16)
    """

    def __init__(self, params: DecoderParams):
        super().__init__(params)
        output_features = prod(params.output_shape)

        # First layer: expand from latent to first hidden layer (512 units)
        self.layer1 = nn.Linear(params.latent_dim, out_features=512, bias=True)
        self.activation1 = params.activation_fn()  # Usually GELU

        # Second layer: 1024 units (progressive expansion)
        self.layer2 = nn.Linear(in_features=512, out_features=1024, bias=True)
        self.activation2 = params.activation_fn()  # Usually GELU

        # Output layer: expand to full output size
        self.layer3 = nn.Linear(in_features=1024, out_features=output_features, bias=True)
        self.output_activation = nn.Sigmoid()

    def forward(self, x: Tensor, target: Optional[Tensor] = None) -> Tensor:
        """Forward pass through the linear decoder.

        Args:
            x: Input latent tensor of shape (batch_size, latent_dim).
            target: Ignored for standard decoder, kept for API consistency.

        Returns:
            Reconstructed output of shape (batch_size, *output_shape) with
            values in range [0, 1] due to Sigmoid output activation.
        """
        # First layer
        h1 = self.layer1(x)
        h1 = self.activation1(h1)  # Usually GELU

        # Second layer
        h2 = self.layer2(h1)
        h2 = self.activation2(h2)  # Usually GELU

        # Output layer
        output = self.layer3(h2)
        output = self.output_activation(output)

        # Reshape to original spatial dimensions
        return output.reshape(output.shape[0], *self.output_shape)


class DRTPLinear(BaseDecoder):
    """Linear decoder using Direct Random Target Projection (DRTP) training.

    This decoder implements the DRTP algorithm, which uses random feedback weights
    instead of symmetric weight updates for biologically plausible learning. It
    mirrors the DRTPLinear encoder architecture in reverse.

    Architecture:
        Latent input → Linear(512) → Tanh → DRTP → Linear(1024) → Tanh → DRTP → Linear(output_size) → Sigmoid

    Key differences from standard Linear decoder:
    - Uses Tanh activation functions (work better with DRTP)
    - Two hidden layers: first with 512 units, second with 1024 units for progressive expansion
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
        params: DecoderParams with activation_fn typically set to nn.Tanh.

    Example:
        >>> params = DecoderParams(
        ...     output_shape=(1, 32, 16),
        ...     latent_dim=128,
        ...     activation_fn=nn.Tanh
        ... )
        >>> decoder = DRTPLinear(params)
        >>> latent_batch = torch.randn(4, 128)
        >>> reconstruction = decoder(latent_batch)  # shape: (4, 1, 32, 16), target=None is default
        >>> # During training: reconstruction = decoder(latent_batch, target_batch)

    References:
        Lillicrap, T. P., et al. (2016). Random synaptic feedback weights support
        error backpropagation for deep learning. Nature Communications, 7, 13276.
    """

    def __init__(self, params: DecoderParams):
        super().__init__(params)
        output_features = prod(params.output_shape)

        # First layer: expand from latent to first hidden layer (512 units)
        self.layer1 = nn.Linear(params.latent_dim, out_features=512, bias=True)
        self.activation1 = params.activation_fn()  # Usually Tanh
        self.drtp1 = DRTPLayer(target_dim=params.output_shape, hidden_dim=512)

        # Second layer: 1024 units
        self.layer2 = nn.Linear(in_features=512, out_features=1024, bias=True)
        self.activation2 = params.activation_fn()  # Usually Tanh
        self.drtp2 = DRTPLayer(target_dim=params.output_shape, hidden_dim=1024)

        # Output layer (no DRTP - uses standard gradients)
        self.layer3 = nn.Linear(in_features=1024, out_features=output_features)
        self.output_activation = nn.Sigmoid()

    def forward(self, x: Tensor, target: Optional[Tensor] = None) -> Tensor:
        """Forward pass through the DRTP linear decoder.

        Args:
            x: Input latent tensor of shape (batch_size, latent_dim).
            target: Target tensor for DRTP gradient computation during backward pass.
                Can be None during forward-only inference.

        Returns:
            Reconstructed output of shape (batch_size, *output_shape) with values
            in range [0, 1] due to Sigmoid output activation.
        """
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

        # Reshape to original spatial dimensions
        return output.reshape(output.shape[0], *self.output_shape)


class DFALinear(BaseDecoder):
    """Linear decoder using Direct Feedback Alignment (DFA) training.

    This decoder implements the DFA algorithm, which uses random feedback weights
    to propagate error signals directly from the output layer to hidden layers,
    bypassing the need for symmetric weight transport. DFA is biologically plausible
    and provides an alternative to both standard backpropagation and DRTP.

    Architecture:
        Latent input → Linear(512) → Tanh → DFA → Linear(1024) → Tanh → DFA → Linear(output_size) → Sigmoid

    Key differences from standard Linear decoder:
    - Uses Tanh activation functions (work well with DFA)
    - Two hidden layers: first with 512 units, second with 1024 units for progressive expansion
    - Applies DFA layers after hidden layers but not output layer
    - Uses Sigmoid output activation for bounded outputs
    - Error signal from output layer is propagated directly to hidden layers via random weights
    - Forward pass accepts optional error_signal parameter for DFA computation

    The DFA mechanism:
    - Uses random feedback weights to project output errors to hidden layers
    - Error signals are computed as (output - target) and propagated via random matrices
    - Maintains learning performance while being more biologically plausible than backprop
    - Does not require symmetric weight transport like standard backpropagation

    Args:
        params: DecoderParams with activation_fn typically set to nn.Tanh.

    Example:
        >>> params = DecoderParams(
        ...     output_shape=(1, 32, 16),
        ...     latent_dim=128,
        ...     activation_fn=nn.Tanh
        ... )
        >>> decoder = DFALinear(params)
        >>> latent_batch = torch.randn(4, 128)
        >>> reconstruction = decoder(latent_batch)  # shape: (4, 1, 32, 16), error_signal=None is default
        >>> # During training: reconstruction = decoder(latent_batch, grad_output_batch)

    References:
        Lillicrap, T. P., et al. (2016). Random synaptic feedback weights support
        error backpropagation for deep learning. Nature Communications, 7, 13276.
    """

    def __init__(self, params: DecoderParams):
        super().__init__(params)
        output_features = prod(params.output_shape)

        # First layer: expand from latent to first hidden layer (512 units)
        self.layer1 = nn.Linear(params.latent_dim, out_features=512, bias=True)
        self.activation1 = params.activation_fn()  # Usually Tanh
        self.dfa1 = DFALayer(output_dim=params.output_shape, hidden_dim=512)

        # Second layer: 1024 units
        self.layer2 = nn.Linear(in_features=512, out_features=1024, bias=True)
        self.activation2 = params.activation_fn()  # Usually Tanh
        self.dfa2 = DFALayer(output_dim=params.output_shape, hidden_dim=1024)

        # Output layer (no DFA - uses standard gradients)
        self.layer3 = nn.Linear(in_features=1024, out_features=output_features)
        self.output_activation = nn.Sigmoid()

    def forward(self, x: Tensor, target: Optional[Tensor] = None) -> Tensor:
        """Forward pass through the DFA linear decoder.

        Args:
            x: Input latent tensor of shape (batch_size, latent_dim).
            target: Ignored for standard decoder, kept for API consistency.

        Returns:
            Reconstructed output of shape (batch_size, *output_shape) with values
            in range [0, 1] due to Sigmoid output activation.
        """
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

        # Reshape to original spatial dimensions
        return output.reshape(output.shape[0], *self.output_shape)


class Conv2D(BaseDecoder):
    """Transpose convolutional neural network decoder using standard backpropagation training.

    This decoder uses transpose convolutional layers to reconstruct spatial data while
    preserving and enhancing spatial relationships. It mirrors the Conv2D encoder
    architecture in reverse, expanding from latent space back to full spatial dimensions.

    Architecture:
        Latent input → Linear(512) → GELU → Linear(intermediate_features) → GELU → Reshape → TransposeConv2d(5x5, 1) → Sigmoid

    Design choices:
    - First linear layer expands latent to 512 units
    - Second linear layer expands to match flattened conv feature size (calculated dynamically)
    - Reshape to spatial dimensions that transpose conv can process
    - 5x5 transpose convolution kernel reconstructs local spatial patterns
    - Stride=2 and appropriate padding to double spatial dimensions
    - GELU activations for smooth gradients and improved training
    - Sigmoid output activation for bounded reconstruction values

    The decoder automatically calculates the correct intermediate dimensions based
    on the output spatial dimensions and transpose convolution operations.

    Args:
        params: DecoderParams with activation_fn typically set to nn.GELU.

    Example:
        >>> params = DecoderParams(
        ...     output_shape=(1, 32, 16),  # Final output shape
        ...     latent_dim=128,
        ...     activation_fn=nn.GELU
        ... )
        >>> decoder = Conv2D(params)
        >>> latent_batch = torch.randn(4, 128)
        >>> reconstruction = decoder(latent_batch)  # shape: (4, 1, 32, 16)

    Note:
        Output spatial dimensions must be even numbers to work correctly with the
        transpose convolution stride settings.
    """

    def __init__(self, params: DecoderParams):
        super().__init__(params)

        # Calculate intermediate dimensions (mirror of encoder conv calculations)
        h, w = self.spatial_dimensions
        # Before transpose conv: h//2 x w//2 (what encoder conv+pool produces)
        intermediate_h, intermediate_w = h // 2, w // 2
        intermediate_features = 32 * intermediate_h * intermediate_w

        # Linear layer: expand from latent to 512 units
        self.layer1 = nn.Linear(params.latent_dim, out_features=512, bias=True)
        self.activation1 = params.activation_fn()  # Usually GELU

        # Second linear layer to match conv input size
        self.layer2 = nn.Linear(in_features=512, out_features=intermediate_features, bias=True)
        self.activation2 = params.activation_fn()  # Usually GELU

        # Store intermediate dimensions for reshape
        self.intermediate_shape = (32, intermediate_h, intermediate_w)

        # Transpose convolution: 32 channels -> 1 channel, 5x5 kernel, stride=2
        kwds = {"stride": 2, "padding": 2, "output_padding": 1, "bias": True}
        self.conv3 = nn.ConvTranspose2d(32, self.output_channels, kernel_size=5, **kwds)
        self.output_activation = nn.Sigmoid()

    def forward(self, x: Tensor, target: Optional[Tensor] = None) -> Tensor:
        """Forward pass through the transpose convolutional decoder.

        Args:
            x: Input latent tensor of shape (batch_size, latent_dim).
            target: Ignored for standard decoder, kept for API consistency.

        Returns:
            Reconstructed output of shape (batch_size, *output_shape) with values
            in range [0, 1] due to Sigmoid output activation.
        """
        # First linear layer
        h1 = self.layer1(x)
        h1 = self.activation1(h1)

        # Second linear layer
        h2 = self.layer2(h1)
        h2 = self.activation2(h2)

        # Reshape to spatial format for transpose convolution
        h2 = h2.reshape(h2.size(0), *self.intermediate_shape)

        # Transpose convolution to final output size
        output = self.conv3(h2)
        output = self.output_activation(output)

        return output


class DRTPConv2D(BaseDecoder):
    """Transpose convolutional decoder using Direct Random Target Projection (DRTP) training.

    This decoder combines the spatial reconstruction capabilities of transpose
    convolutional layers with the biologically plausible learning mechanism of DRTP.
    It mirrors the DRTPConv2D encoder architecture in reverse.

    Architecture:
        Latent input → Linear(512) → Tanh → DRTP → Linear(intermediate_features) → Tanh → DRTP → Reshape → TransposeConv2d(5x5, 1) → Sigmoid

    Key features:
    - Transpose convolutional layer reconstructs spatial patterns from feature maps
    - DRTP applied to both fully connected hidden layers
    - Uses Tanh activation functions for compatibility with DRTP
    - Random feedback weights enable biologically plausible learning
    - Output layer uses standard backpropagation for final reconstruction
    - Second linear layer size is calculated dynamically based on output dimensions

    The combination of transpose convolution and DRTP makes this decoder suitable for:
    - Spatial navigation tasks requiring biological plausibility
    - Investigating how spatial reconstruction might work in biological neural networks
    - Research comparing biological vs. standard learning mechanisms
    - Applications where spatial features and learning constraints both matter

    Args:
        params: DecoderParams with activation_fn typically set to nn.Tanh.

    Example:
        >>> params = DecoderParams(
        ...     output_shape=(1, 32, 16),
        ...     latent_dim=128,
        ...     activation_fn=nn.Tanh
        ... )
        >>> decoder = DRTPConv2D(params)
        >>> latent_batch = torch.randn(4, 128)
        >>> reconstruction = decoder(latent_batch)  # shape: (4, 1, 32, 16), target=None is default
        >>> # During training: reconstruction = decoder(latent_batch, target_batch)

    Note:
        Target tensor is only required during backward pass for DRTP computation.
        Forward pass accepts optional target parameter for consistent API.
        Output spatial dimensions should be even numbers for transpose conv compatibility.

    References:
        Lillicrap, T. P., et al. (2016). Random synaptic feedback weights support
        error backpropagation for deep learning. Nature Communications, 7, 13276.
    """

    def __init__(self, params: DecoderParams):
        super().__init__(params)

        # Calculate intermediate dimensions (mirror of encoder conv calculations)
        h, w = self.spatial_dimensions
        intermediate_h, intermediate_w = h // 2, w // 2
        intermediate_features = 32 * intermediate_h * intermediate_w

        # Define layers separately since we need to apply DRTP manually
        self.layer1 = nn.Linear(in_features=params.latent_dim, out_features=512, bias=True)
        self.activation1 = params.activation_fn()  # Usually Tanh
        self.drtp1 = DRTPLayer(target_dim=params.output_shape, hidden_dim=512)

        # Second linear layer to match conv input size
        self.layer2 = nn.Linear(in_features=512, out_features=intermediate_features, bias=True)
        self.activation2 = params.activation_fn()  # Usually Tanh
        self.drtp2 = DRTPLayer(target_dim=params.output_shape, hidden_dim=intermediate_features)

        # Store intermediate dimensions for reshape
        self.intermediate_shape = (32, intermediate_h, intermediate_w)

        # Transpose convolution: 32 channels -> 1 channel, 5x5 kernel, stride=2
        # No DRTP on transpose conv layer (uses standard gradients)
        kwds = {"stride": 2, "padding": 2, "output_padding": 1, "bias": True}
        self.conv3 = nn.ConvTranspose2d(32, self.output_channels, kernel_size=5, **kwds)
        self.output_activation = nn.Sigmoid()

    def forward(self, x: Tensor, target: Optional[Tensor] = None) -> Tensor:
        """Forward pass through the DRTP transpose convolutional decoder.

        Args:
            x: Input latent tensor of shape (batch_size, latent_dim).
            target: Target tensor for DRTP gradient computation during backward pass.
                Can be None during forward-only inference.

        Returns:
            Reconstructed output of shape (batch_size, *output_shape) with values
            in range [0, 1] due to Sigmoid output activation.
        """
        # First layer with DRTP
        h1 = self.layer1(x)
        h1 = self.activation1(h1)
        h1 = self.drtp1(h1, target)

        # Second layer with DRTP
        h2 = self.layer2(h1)
        h2 = self.activation2(h2)
        h2 = self.drtp2(h2, target)

        # Reshape to spatial format for transpose convolution
        h2 = h2.reshape(h2.size(0), *self.intermediate_shape)

        # Transpose convolution to final output size (no DRTP)
        output = self.conv3(h2)
        output = self.output_activation(output)

        return output


class DFAConv2D(BaseDecoder):
    """Transpose convolutional decoder using Direct Feedback Alignment (DFA) training.

    This decoder combines the spatial reconstruction capabilities of transpose
    convolutional layers with the biologically plausible learning mechanism of DFA.
    It provides an alternative to both standard backpropagation and DRTP while
    maintaining spatial processing capabilities.

    Architecture:
        Latent input → Linear(512) → Tanh → DFA → Linear(intermediate_features) → Tanh → DFA → Reshape → TransposeConv2d(5x5, 1) → Sigmoid

    Key features:
    - Transpose convolutional layer reconstructs spatial patterns from feature maps
    - DFA applied to both fully connected hidden layers
    - Uses Tanh activation functions for compatibility with DFA
    - Random feedback weights enable biologically plausible learning
    - Output layer uses standard backpropagation for final reconstruction
    - Second linear layer size is calculated dynamically based on output dimensions
    - Error signals are propagated directly from output to hidden layers

    The combination of transpose convolution and DFA makes this decoder suitable for:
    - Spatial navigation tasks requiring biological plausibility
    - Investigating how spatial reconstruction might work in biological neural networks
    - Research comparing biological vs. standard learning mechanisms
    - Applications where spatial features and learning constraints both matter

    Args:
        params: DecoderParams with activation_fn typically set to nn.Tanh.

    Example:
        >>> params = DecoderParams(
        ...     output_shape=(1, 32, 16),
        ...     latent_dim=128,
        ...     activation_fn=nn.Tanh
        ... )
        >>> decoder = DFAConv2D(params)
        >>> latent_batch = torch.randn(4, 128)
        >>> reconstruction = decoder(latent_batch)  # shape: (4, 1, 32, 16), error_signal=None is default
        >>> # During training: reconstruction = decoder(latent_batch, grad_output_batch)

    Note:
        Error signal is only required during backward pass for DFA computation.
        Forward pass accepts optional error_signal parameter for consistent API.
        Output spatial dimensions should be even numbers for transpose conv compatibility.

    References:
        Lillicrap, T. P., et al. (2016). Random synaptic feedback weights support
        error backpropagation for deep learning. Nature Communications, 7, 13276.
    """

    def __init__(self, params: DecoderParams):
        super().__init__(params)

        # Calculate intermediate dimensions (mirror of encoder conv calculations)
        h, w = self.spatial_dimensions
        intermediate_h, intermediate_w = h // 2, w // 2
        intermediate_features = 32 * intermediate_h * intermediate_w

        # Define layers separately since we need to apply DFA manually
        self.layer1 = nn.Linear(in_features=params.latent_dim, out_features=512, bias=True)
        self.activation1 = params.activation_fn()  # Usually Tanh
        self.dfa1 = DFALayer(output_dim=params.output_shape, hidden_dim=512)

        # Second linear layer to match conv input size
        self.layer2 = nn.Linear(in_features=512, out_features=intermediate_features, bias=True)
        self.activation2 = params.activation_fn()  # Usually Tanh
        self.dfa2 = DFALayer(output_dim=params.output_shape, hidden_dim=intermediate_features)

        # Store intermediate dimensions for reshape
        self.intermediate_shape = (32, intermediate_h, intermediate_w)

        # Transpose convolution: 32 channels -> 1 channel, 5x5 kernel, stride=2
        # No DFA on transpose conv layer (uses standard gradients)
        kwds = {"stride": 2, "padding": 2, "output_padding": 1, "bias": True}
        self.conv3 = nn.ConvTranspose2d(32, self.output_channels, kernel_size=5, **kwds)
        self.output_activation = nn.Sigmoid()

    def forward(self, x: Tensor, target: Optional[Tensor] = None) -> Tensor:
        """Forward pass through the DFA transpose convolutional decoder.

        Args:
            x: Input latent tensor of shape (batch_size, latent_dim).
            target: Ignored for standard decoder, kept for API consistency.

        Returns:
            Reconstructed output of shape (batch_size, *output_shape) with values
            in range [0, 1] due to Sigmoid output activation.
        """
        # First layer with DFA
        h1 = self.layer1(x)
        h1 = self.activation1(h1)
        h1 = self.dfa1(h1)

        # Second layer with DFA
        h2 = self.layer2(h1)
        h2 = self.activation2(h2)
        h2 = self.dfa2(h2)

        # Reshape to spatial format for transpose convolution
        h2 = h2.reshape(h2.size(0), *self.intermediate_shape)

        # Transpose convolution to final output size (no DFA)
        output = self.conv3(h2)
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
    Demonstration script showing how to use all decoder types.

    This script provides complete examples of:
    1. Parameter configuration for different decoder types
    2. Model instantiation and forward passes
    3. Performance comparison between architectures
    4. Expected input/output shapes and characteristics
    5. Encoder-decoder compatibility testing

    Run this script directly to see the decoders in action:
        python -m ehc_sn.models.decoders
    """
    # -----------------------------------------------------------------------------------
    # Setup and parameters
    # -----------------------------------------------------------------------------------

    print("=== Decoder Examples ===\n")

    # Example parameters for cognitive maps (32x16 obstacle grid, single channel)
    params = DecoderParams(
        output_shape=(1, 32, 16),  # 1 channel, 32x16 grid (512 total features)
        latent_dim=128,  # Decompress from 128-dimensional representation
        activation_fn=nn.GELU,  # Using GELU activation for non-DRTP decoders
    )

    # Parameters for DRTP decoders (typically use Tanh)
    drtp_params = DecoderParams(output_shape=(1, 32, 16), latent_dim=128, activation_fn=nn.Tanh)

    # Parameters for DFA decoders (typically use Tanh)
    dfa_params = DecoderParams(output_shape=(1, 32, 16), latent_dim=128, activation_fn=nn.Tanh)

    # Create sample latent input batch (4 samples)
    batch_size = 4
    sample_latent = torch.randn(batch_size, params.latent_dim)

    print(f"Latent input shape: {sample_latent.shape}")
    print(f"Target output shape: (batch_size, *{params.output_shape})\n")

    # -----------------------------------------------------------------------------------
    # Linear Decoder Example
    # -----------------------------------------------------------------------------------

    print("1. Linear Decoder:")
    linear_decoder = Linear(params)

    # Forward pass
    with torch.no_grad():
        linear_output = linear_decoder(sample_latent)

    print(f"   Output features: {prod(params.output_shape)}")
    print(f"   Output shape: {linear_output.shape}")
    print(f"   Output range: [{linear_output.min():.3f}, {linear_output.max():.3f}]")
    print(f"   Mean activation: {linear_output.mean():.3f}\n")

    # -----------------------------------------------------------------------------------
    # DRTP Linear Decoder Example
    # -----------------------------------------------------------------------------------

    print("2. DRTP Linear Decoder:")
    drtp_linear_decoder = DRTPLinear(drtp_params)

    # Forward pass (no target needed for forward pass)
    with torch.no_grad():
        drtp_linear_output = drtp_linear_decoder(sample_latent)

    print(f"   Output features: {prod(drtp_params.output_shape)}")
    print(f"   Output shape: {drtp_linear_output.shape}")
    print(f"   Output range: [{drtp_linear_output.min():.3f}, {drtp_linear_output.max():.3f}]")
    print(f"   Mean activation: {drtp_linear_output.mean():.3f}\n")

    # -----------------------------------------------------------------------------------
    # DFA Linear Decoder Example
    # -----------------------------------------------------------------------------------

    print("3. DFA Linear Decoder:")
    dfa_linear_decoder = DFALinear(dfa_params)

    # Forward pass (no error signal needed for forward pass)
    with torch.no_grad():
        dfa_linear_output = dfa_linear_decoder(sample_latent)

    print(f"   Output features: {prod(dfa_params.output_shape)}")
    print(f"   Output shape: {dfa_linear_output.shape}")
    print(f"   Output range: [{dfa_linear_output.min():.3f}, {dfa_linear_output.max():.3f}]")
    print(f"   Mean activation: {dfa_linear_output.mean():.3f}\n")

    # -----------------------------------------------------------------------------------
    # Conv2D Decoder Example
    # -----------------------------------------------------------------------------------

    print("4. Conv2D Decoder:")
    conv_decoder = Conv2D(params)

    # Calculate expected intermediate dimensions
    h, w = params.output_shape[1], params.output_shape[2]
    intermediate_features = 32 * (h // 2) * (w // 2)

    with torch.no_grad():
        conv_output = conv_decoder(sample_latent)

    print(f"   Spatial dimensions: {h}x{w}")
    print(f"   Intermediate features: 32x{h//2}x{w//2} = {intermediate_features}")
    print(f"   Output shape: {conv_output.shape}")
    print(f"   Output range: [{conv_output.min():.3f}, {conv_output.max():.3f}]")
    print(f"   Mean activation: {conv_output.mean():.3f}\n")

    # -----------------------------------------------------------------------------------
    # DRTP Conv2D Decoder Example
    # -----------------------------------------------------------------------------------

    print("5. DRTP Conv2D Decoder:")
    drtp_conv_decoder = DRTPConv2D(drtp_params)

    with torch.no_grad():
        drtp_conv_output = drtp_conv_decoder(sample_latent)

    print(f"   Spatial dimensions: {h}x{w}")
    print(f"   Intermediate features: 32x{h//2}x{w//2} = {intermediate_features}")
    print(f"   Output shape: {drtp_conv_output.shape}")
    print(f"   Output range: [{drtp_conv_output.min():.3f}, {drtp_conv_output.max():.3f}]")
    print(f"   Mean activation: {drtp_conv_output.mean():.3f}\n")

    # -----------------------------------------------------------------------------------
    # DFA Conv2D Decoder Example
    # -----------------------------------------------------------------------------------

    print("6. DFA Conv2D Decoder:")
    dfa_conv_decoder = DFAConv2D(dfa_params)

    with torch.no_grad():
        dfa_conv_output = dfa_conv_decoder(sample_latent)

    print(f"   Spatial dimensions: {h}x{w}")
    print(f"   Intermediate features: 32x{h//2}x{w//2} = {intermediate_features}")
    print(f"   Output shape: {dfa_conv_output.shape}")
    print(f"   Output range: [{dfa_conv_output.min():.3f}, {dfa_conv_output.max():.3f}]")
    print(f"   Mean activation: {dfa_conv_output.mean():.3f}\n")

    # -----------------------------------------------------------------------------------
    # Model comparison
    # -----------------------------------------------------------------------------------

    print("7. Model Size Comparison:")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    models = {
        "Linear": linear_decoder,
        "DRTP Linear": drtp_linear_decoder,
        "DFA Linear": dfa_linear_decoder,
        "Conv2D": conv_decoder,
        "DRTP Conv2D": drtp_conv_decoder,
        "DFA Conv2D": dfa_conv_decoder,
    }

    for name, model in models.items():
        param_count = count_parameters(model)
        print(f"   {name}: {param_count:,} parameters")

    print("\n=== Examples completed ===")
