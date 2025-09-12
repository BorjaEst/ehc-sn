"""Neural network decoders for entorhinal-hippocampal circuit spatial navigation modeling.

This module implements various decoder architectures that transform latent representations
back into reconstructed spatial outputs (e.g., cognitive maps, obstacle maps). The decoders
complement the encoders in autoencoder architectures and support multiple biologically-inspired
training methods including standard backpropagation (BP), Direct Random Target Projection
(DRTP), Direct Feedback Alignment (DFA), and Symmetric Random Target Projection (SRTP).

The decoder architectures are designed to mimic the decoding properties of hippocampal
regions (DG, CA3, CA2, CA1) that reconstruct spatial representations from compressed
neural codes for spatial navigation and memory retrieval.

Key Features:
    - Pydantic-based parameter validation and configuration management
    - Support for multiple biologically-motivated training algorithms
    - Consistent interface across all decoder implementations
    - Progressive expansion from latent to full spatial dimensions
    - Appropriate activation functions optimized for each training method

Classes:
    DecoderParams: Configuration parameters for decoder initialization
    BaseDecoder: Abstract base class defining the decoder interface
    Linear: Standard fully connected decoder for spatial reconstruction
    DRTPLinear: DRTP-enabled decoder with target projection learning
    DFALinear: DFA-enabled decoder with random feedback alignment
    SRTPLinear: SRTP-enabled decoder with symmetric random projections
    ZOLinear: Zero-Order decoder with gradient-free learning via perturbations

Architecture Pattern:
    All decoders follow a consistent expansion architecture:
    - Input from latent representation (compressed spatial features)
    - Progressive expansion through hidden layers
    - Output reconstruction matching original spatial input dimensions
    - Sigmoid activation for spatial probability maps

Examples:
    >>> # Standard decoder for spatial map reconstruction
    >>> params = DecoderParams(
    ...     output_shape=(1, 32, 16),  # Target spatial map dimensions
    ...     latent_dim=128,            # Input latent dimension
    ...     activation_fn=nn.GELU      # Smooth activation for BP training
    ... )
    >>> decoder = Linear(params)
    >>> reconstruction = decoder(latent_tensor)
    >>>
    >>> # DRTP decoder with bounded activation
    >>> drtp_params = DecoderParams(
    ...     output_shape=(1, 32, 16),
    ...     latent_dim=128,
    ...     activation_fn=nn.Tanh  # Bounded activation for DRTP stability
    ... )
    >>> drtp_decoder = DRTPLinear(drtp_params)

References:
    - Lillicrap, T. P., et al. (2016). Random synaptic feedback weights support
      error backpropagation for deep learning. Nature Communications, 7, 13276.
    - O'Keefe, J., & Nadel, L. (1978). The hippocampus as a cognitive map.
      Oxford University Press.
"""

from math import prod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from pydantic import BaseModel, Field, model_validator
from torch import Tensor, nn

from ehc_sn.hooks.registry import registry
from ehc_sn.modules import dfa, zo
from ehc_sn.modules.drtp import DRTPLayer
from ehc_sn.modules.srtp import SRTPLayer


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
        >>> decoder = Linear(params)  # or DRTPLinear, DFALinear, SRTPLinear
        >>> reconstruction = decoder(latent_tensor)
    """

    def __init__(self, params: DecoderParams):
        super().__init__()
        self.params = params

    def forward(self, x: Tensor, *args: Any, **kwds: Any) -> Tensor:
        """Forward pass through the decoder.

        Args:
            x: Input latent tensor of shape (batch_size, latent_dim).
            *args: Additional positional arguments for interface compatibility
                with different decoder architectures and training strategies.
            **kwds: Additional keyword arguments including:
                - target: Target tensor for some training methods (e.g., DFA, DRTP)
                - seed: Random seed for stochastic methods (e.g., Zero-Order)
                - Other method-specific parameters

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

    def forward(self, x: Tensor, *args: Any, **kwds: Any) -> Tensor:
        """Forward pass through the linear decoder.

        Args:
            x: Input latent tensor of shape (batch_size, latent_dim).
            *args: Additional positional arguments for interface compatibility.
            **kwds: Additional keyword arguments including:
                - target: Ignored for standard decoder, kept for API consistency
                - seed: Ignored for standard decoder
                - Other method-specific parameters

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

    def forward(self, x: Tensor, *args, target: Optional[Tensor] = None, **kwds) -> Tensor:
        """Forward pass through the DRTP linear decoder.

        Args:
            x: Input latent tensor of shape (batch_size, latent_dim).
            *args: Additional positional arguments for interface compatibility.
            target: Target tensor for DRTP gradient computation during backward pass.
            **kwds: Additional keyword arguments including:
                - seed: Ignored for DRTP decoder
                - Other method-specific parameters

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
        self.dfa1 = dfa.DFALayer(output_dim=params.output_shape, hidden_dim=512)

        # Second layer: 1024 units
        self.layer2 = nn.Linear(in_features=512, out_features=1024, bias=True)
        self.activation2 = params.activation_fn()  # Usually Tanh
        self.dfa2 = dfa.DFALayer(output_dim=params.output_shape, hidden_dim=1024)

        # Output layer (no DFA - uses standard gradients)
        self.layer3 = nn.Linear(in_features=1024, out_features=output_features)
        self.output_activation = nn.Sigmoid()

    def forward(self, x: Tensor, *args: Any, **kwds: Any) -> Tensor:
        """Forward pass through the DFA linear decoder.

        Args:
            x: Input latent tensor of shape (batch_size, latent_dim).
            *args: Additional positional arguments for interface compatibility.
            **kwds: Additional keyword arguments including:
                - target: Ignored for DFA decoder, kept for API consistency
                - seed: Ignored for DFA decoder
                - Other method-specific parameters

        Returns:
            Reconstructed output of shape (batch_size, *output_shape) with values
            in range [0, 1] due to Sigmoid output activation.

        Note:
            Hook management for DFA error signal capture is now handled by
            the DFATrainer, not in this forward method. This provides better
            separation of concerns between model computation and training logic.
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

        # Reshape to original spatial dimensions
        return output.reshape(output.shape[0], *self.output_shape)


class SRTPLinear(BaseDecoder):
    """Linear decoder using Selective Random Target Projection (SRTP) training.

    This decoder implements the SRTP algorithm which uses fixed random feedback weights
    with encoder activation targets for biologically plausible learning. The decoder
    mirrors the Linear architecture but applies SRTP modulatory signals at hidden layers.

    Architecture:
        Latent input → Linear(512) → GELU → SRTP → Linear(1024) → GELU → SRTP → Linear(output_size) → Sigmoid

    Key differences from standard Linear decoder:
    - Uses SRTP layers that accept encoder activations as targets
    - SRTP layers use fixed random projection matrices (non-trainable)
    - Backward pass projects error through random weights instead of transpose weights
    - Supports lazy initialization of SRTP layers based on encoder activation shapes
    - Maintains compatibility with standard training when encoder activations unavailable

    The SRTP mechanism:
    - Maps encoder activations to decoder hidden layer dimensions via fixed random weights
    - Computes gradients as: hidden_activation - projected_encoder_target
    - Provides modulatory signals that guide learning without symmetric weight transport
    - Preserves computational graph for standard gradient flow

    Args:
        params: DecoderParams with activation_fn typically set to nn.GELU.

    Example:
        >>> params = DecoderParams(
        ...     output_shape=(1, 32, 16),
        ...     latent_dim=128,
        ...     activation_fn=nn.GELU
        ... )
        >>> decoder = SRTPLinear(params)
        >>> # With encoder activations
        >>> reconstruction = decoder(latent_batch, encoder_acts={'h1': enc_h1, 'h2': enc_h2})
        >>> # Without encoder activations (fallback to standard forward)
        >>> reconstruction = decoder(latent_batch)

    References:
        Based on the SRTP mechanism which extends DFA principles with target-specific
        random projections for enhanced biological plausibility.
    """

    def __init__(self, params: DecoderParams):
        super().__init__(params)
        output_features = prod(params.output_shape)

        # Linear layers matching the standard Linear decoder architecture
        self.fc1 = nn.Linear(params.latent_dim, 512, bias=True)
        self.act1 = params.activation_fn()

        self.fc2 = nn.Linear(512, 1024, bias=True)
        self.act2 = params.activation_fn()

        self.out = nn.Linear(1024, output_features, bias=True)
        self.output_activation = nn.Sigmoid()

        # SRTP layers - initialized lazily when encoder activations are first provided
        self.srtp1: Optional[SRTPLayer] = None
        self.srtp2: Optional[SRTPLayer] = None
        self._srtp_initialized = False

    def _initialize_srtp_layers(self) -> None:
        """Initialize SRTP layers based on encoder activations from registry.

        This method tries to get encoder activations from the registry to
        determine the dimensions for SRTP layers. If activations are not available,
        it uses default dimensions.
        """
        # Try to get encoder activations from registry
        h1_activation = registry.get_activation("encoder.h1")
        h2_activation = registry.get_activation("encoder.h2")

        enc_h1_flat = h1_activation.flatten(1).shape[1]
        enc_h2_flat = h2_activation.flatten(1).shape[1]

        # Create SRTP layers
        self.srtp1 = SRTPLayer(target_dim=enc_h2_flat, hidden_dim=512)
        self.srtp2 = SRTPLayer(target_dim=enc_h1_flat, hidden_dim=1024)

        # Ensure layers are on the same device as the model
        self.srtp1 = self.srtp1.to(device=self.fc1.weight.device)
        self.srtp2 = self.srtp2.to(device=self.fc2.weight.device)

        # Register as proper submodules
        self.add_module("srtp1", self.srtp1)
        self.add_module("srtp2", self.srtp2)

        # Set initialization flat to true
        self._srtp_initialized = True

    def forward(self, x: Tensor, *args: Any, **kwds: Any) -> Tensor:
        """Forward pass through SRTP linear decoder.

        Args:
            x: Latent tensor of shape (batch_size, latent_dim)
            *args: Additional positional arguments for interface compatibility.
            **kwds: Additional keyword arguments including:
                - target: Ignored for SRTP decoder, kept for API consistency
                - seed: Ignored for SRTP decoder
                - Other method-specific parameters

        Returns:
            Reconstructed output tensor of shape (batch_size, *output_shape)
        """
        # Initialize SRTP layers on first forward pass
        if not self._srtp_initialized:
            self._initialize_srtp_layers()

        # First hidden layer
        h1 = self.fc1(x)
        h1 = self.act1(h1)

        # Apply SRTP1 with encoder h2 as target
        target1 = registry.get_activation("encoder.h2").flatten(1).detach()
        target1 = target1.to(device=h1.device, dtype=h1.dtype)
        h1 = self.srtp1(h1, target=target1)

        # Second hidden layer
        h2 = self.fc2(h1)
        h2 = self.act2(h2)

        # Apply SRTP2 with encoder h1 as target
        target2 = registry.get_activation("encoder.h1").flatten(1).detach()
        target2 = target2.to(device=h2.device, dtype=h2.dtype)
        h2 = self.srtp2(h2, target=target2)

        # Output layer (no SRTP)
        output = self.out(h2)
        output = self.output_activation(output)

        # Reshape to match expected output shape
        return output.reshape(output.shape[0], *self.output_shape)


class ZOLinear(BaseDecoder):
    """Linear decoder using Zero-Order (ZO) optimization for gradient-free learning.

    This decoder implements the Memory-efficient Zero-Order (MeZO) algorithm, which
    estimates gradients using finite differences with random perturbations instead
    of traditional backpropagation. It provides a biologically plausible learning
    mechanism that doesn't require gradient computation or storage.

    Architecture:
        Latent input → ZOLinear(512) → GELU → ZOLinear(1024) → GELU → ZOLinear(output_size) → Sigmoid

    Key features of ZO optimization:
    - Gradient-free learning using finite difference approximation
    - Memory-efficient implementation (no backward pass storage required)
    - Biologically plausible learning mechanism without weight transport
    - Compatible with standard PyTorch optimizers after gradient estimation
    - Two-phase forward pass: +ε and -ε perturbations for gradient estimation

    The ZO mechanism:
    - Applies random perturbations to layer weights during forward passes
    - Estimates gradients as: ∇θ ≈ (L(θ + εz) - L(θ - εz)) / (2ε) * z
    - Uses deterministic perturbation generation with seed control
    - Provides gradients via feedback mechanism for optimizer compatibility

    Training process:
    1. Forward pass with +ε perturbation (seed provided)
    2. Forward pass with -ε perturbation (same seed)
    3. Compute loss difference and apply feedback to all layers
    4. Standard optimizer step using estimated gradients

    Args:
        params: DecoderParams with activation_fn typically set to nn.GELU.

    Example:
        >>> params = DecoderParams(
        ...     output_shape=(1, 32, 16),
        ...     latent_dim=128,
        ...     activation_fn=nn.GELU
        ... )
        >>> decoder = ZOLinear(params)
        >>>
        >>> # Standard inference
        >>> reconstruction = decoder(latent_batch)
        >>>
        >>> # Training with ZO optimization
        >>> output_1 = decoder(latent_batch, seed=42)  # +ε perturbation
        >>> output_2 = decoder(latent_batch, seed=42)  # -ε perturbation
        >>>
        >>> # Apply feedback with loss difference
        >>> loss_diff = loss_1 - loss_2
        >>> decoder.feedback(loss_diff.item())

    References:
        - MeZO: Fine-Tuning Language Models with Just Forward Passes (Malladi et al., 2023)
        - Zero-Order Optimization in Machine Learning (Chen et al., 2020)
    """

    def __init__(self, params: DecoderParams, epsilon: float = 1e-3):
        """Initialize Zero-Order linear decoder.

        Args:
            params: DecoderParams containing configuration parameters.
            epsilon: Perturbation magnitude for finite difference gradient estimation.
                Smaller values provide more accurate gradients but may suffer from
                numerical precision issues. Typical range: 1e-4 to 1e-2. Default: 1e-3.
        """
        super().__init__(params)
        output_features = prod(params.output_shape)

        # Create ZO Linear layers with specified epsilon
        self.layer1 = zo.Linear(params.latent_dim, 512, bias=True, epsilon=epsilon)
        self.activation1 = params.activation_fn()  # Usually GELU

        self.layer2 = zo.Linear(512, 1024, bias=True, epsilon=epsilon)
        self.activation2 = params.activation_fn()  # Usually GELU

        self.layer3 = zo.Linear(1024, output_features, bias=True, epsilon=epsilon)
        self.output_activation = nn.Sigmoid()

        # Store epsilon for reference
        self.epsilon = epsilon

    def forward(self, x: Tensor, *args: Any, seed: Optional[int] = None, **kwds) -> Tensor:
        """Forward pass through the ZO linear decoder.

        This method supports both inference mode (seed=None) and training mode
        (seed provided). During training, the method expects to be called twice
        with the same seed for proper gradient estimation via finite differences.

        Args:
            *args: Additional positional arguments for interface compatibility.
            seed: Random seed for perturbation generation during training.
            **kwds: Additional keyword arguments including:
                - target: Ignored for SRTP decoder, kept for API consistency
                - Other method-specific parameters

        Returns:
            Reconstructed output of shape (batch_size, *output_shape) with
            values in range [0, 1] due to Sigmoid output activation.

        Note:
            During training, this method must be called twice with the same seed:
            1. First call: applies +ε perturbation to all layers
            2. Second call: applies -ε perturbation to all layers
            The gradient estimation is completed by calling feedback() on all layers.
        """

        # Generate independent seeds for each layer if not provided
        if seed is None:
            seeds = [None for _ in range(3)]
        else:
            gen = torch.Generator(device=x.device)
            gen.manual_seed(self._seed if seed is None else seed)
            seeds = [gen.randint(0, 2**32 - 1).item() for _ in range(3)]

        # First layer
        h1 = self.layer1(x, seed=seeds[0])
        h1 = self.activation1(h1)

        # Second layer
        h2 = self.layer2(h1, seed=seeds[1])
        h2 = self.activation2(h2)

        # Output layer
        output = self.layer3(h2, seed=seeds[2])
        output = self.output_activation(output)

        # Reshape to original spatial dimensions
        return output.reshape(output.shape[0], *self.output_shape)

    def feedback(self, projected_grad: float) -> None:
        """Apply gradient feedback to all ZO layers using finite difference estimate.

        This method propagates the projected gradient (loss difference) to all
        ZO layers in the decoder, allowing them to compute parameter gradients
        for the subsequent optimizer step.

        Args:
            projected_grad: The finite difference value (loss_1 - loss_2) from
                the two perturbed forward passes. This represents the directional
                derivative along the random perturbation directions.

        Note:
            This method must be called after completing the two-phase forward pass
            (with +ε and -ε perturbations) and before the optimizer step.
        """
        self.layer1.feedback(projected_grad)
        self.layer2.feedback(projected_grad)
        self.layer3.feedback(projected_grad)


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

    # Parameters for ZO decoders (typically use GELU)
    zo_params = DecoderParams(output_shape=(1, 32, 16), latent_dim=128, activation_fn=nn.GELU)

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
    # ZO Linear Decoder Example
    # -----------------------------------------------------------------------------------

    print("4. ZO Linear Decoder:")
    zo_linear_decoder = ZOLinear(zo_params, epsilon=1e-3)

    # Forward pass (inference mode)
    with torch.no_grad():
        zo_linear_output = zo_linear_decoder(sample_latent)

    print(f"   Output features: {prod(zo_params.output_shape)}")
    print(f"   Output shape: {zo_linear_output.shape}")
    print(f"   Output range: [{zo_linear_output.min():.3f}, {zo_linear_output.max():.3f}]")
    print(f"   Mean activation: {zo_linear_output.mean():.3f}")
    print(f"   Epsilon (perturbation): {zo_linear_decoder.epsilon}\n")

    # -----------------------------------------------------------------------------------
    # Model comparison
    # -----------------------------------------------------------------------------------

    print("5. Model Size Comparison:")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    models = {
        "Linear": linear_decoder,
        "DRTP Linear": drtp_linear_decoder,
        "DFA Linear": dfa_linear_decoder,
        "ZO Linear": zo_linear_decoder,
    }

    for name, model in models.items():
        param_count = count_parameters(model)
        print(f"   {name}: {param_count:,} parameters")

    # -----------------------------------------------------------------------------------
    # ZO Training Example
    # -----------------------------------------------------------------------------------

    print("\n6. ZO Training Example:")

    # Create sample data for ZO training demonstration
    sample_target = torch.randn(batch_size, *params.output_shape)

    # Simulate ZO training steps
    print("   Demonstrating ZO gradient estimation...")

    # Two forward passes with same seed for gradient estimation
    output_plus = zo_linear_decoder(sample_latent, seed=42)  # +ε perturbation
    output_minus = zo_linear_decoder(sample_latent, seed=42)  # -ε perturbation

    # Compute losses
    criterion = nn.MSELoss()
    loss_plus = criterion(output_plus, sample_target)
    loss_minus = criterion(output_minus, sample_target)
    loss_diff = loss_plus - loss_minus

    print(f"   Loss (+ε): {loss_plus.item():.6f}")
    print(f"   Loss (-ε): {loss_minus.item():.6f}")
    print(f"   Loss difference: {loss_diff.item():.6f}")

    # Apply feedback for gradient estimation
    zo_linear_decoder.feedback(loss_diff.item())
    print("   ✓ Gradient feedback applied to all ZO layers")

    # Check that gradients were computed
    grad_norms = []
    for name, param in zo_linear_decoder.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())

    if grad_norms:
        avg_grad_norm = sum(grad_norms) / len(grad_norms)
        print(f"   Average gradient norm: {avg_grad_norm:.6f}")
    else:
        print("   No gradients computed (unexpected)")

    print("\n=== Examples completed ===")
