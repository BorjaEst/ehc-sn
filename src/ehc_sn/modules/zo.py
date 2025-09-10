"""
Zero-Order Optimization Module for Memory-Efficient Gradient-Free Learning.

This module implements Zero-Order (ZO) optimization techniques, specifically the
Memory-efficient Zero-Order (MeZO) algorithm for gradient-free neural network training.
ZO methods are particularly useful for scenarios where gradients are expensive to compute
or when implementing biologically plausible learning algorithms.

The core component is a specialized Linear layer that estimates gradients using finite
differences with random perturbations, enabling optimization without backpropagation
while maintaining compatibility with standard PyTorch optimizers.

Key Features:
    - Gradient-free optimization using finite difference approximation
    - Memory-efficient implementation (no backward pass storage required)
    - Compatible with standard PyTorch optimizers (Adam, SGD, etc.)
    - Biologically plausible learning mechanism
    - Deterministic perturbation generation with seed control

Algorithm:
    The MeZO algorithm estimates gradients using:
    ∇θ ≈ (L(θ + εz) - L(θ - εz)) / (2ε) * z

    Where:
    - θ: model parameters
    - ε: perturbation magnitude (epsilon)
    - z: random perturbation vector
    - L: loss function

References:
    - MeZO: Fine-Tuning Language Models with Just Forward Passes (Malladi et al., 2023)
    - Zero-Order Optimization in Machine Learning (Chen et al., 2020)

Example:
    >>> # Create a model with ZO optimization
    >>> model = nn.Sequential(
    ...     Linear(10, 32, epsilon=1e-3),
    ...     nn.ReLU(),
    ...     Linear(32, 5, epsilon=1e-3)
    ... )
    >>>
    >>> # Standard forward pass (inference)
    >>> output = model(input_data)
    >>>
    >>> # Training with perturbations
    >>> output_1 = model(input_data, seed=42)  # First forward (+ε)
    >>> output_2 = model(input_data, seed=42)  # Second forward (-ε)
    >>>
    >>> # Compute finite difference and apply feedback
    >>> loss_diff = loss_1 - loss_2
    >>> for layer in model:
    ...     if hasattr(layer, 'feedback'):
    ...         layer.feedback(loss_diff.item())
"""

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# -------------------------------------------------------------------------------------------
class Linear(nn.Linear):
    """
    Zero-Order Linear layer with gradient estimation via finite differences.

    This layer implements the Memory-efficient Zero-Order (MeZO) optimization approach
    by estimating gradients using finite differences with random perturbations instead
    of traditional backpropagation. It maintains full compatibility with PyTorch's
    nn.Linear interface while enabling gradient-free learning.

    The layer operates in two modes:
    1. Inference mode (seed=None): Standard linear transformation
    2. Training mode (seed provided): Applies random perturbations for gradient estimation

    During training, the layer expects to be called twice with the same seed:
    - First call: applies +ε perturbation to weights
    - Second call: applies -ε perturbation to weights

    The difference in outputs is used to estimate gradients via finite differences.

    Args:
        *args: Positional arguments passed to nn.Linear (in_features, out_features, bias)
        epsilon: Perturbation magnitude for finite difference approximation.
            Smaller values provide more accurate gradients but may suffer from
            numerical precision issues. Typical range: 1e-4 to 1e-2. Default: 1e-3.
        **kwargs: Additional keyword arguments passed to nn.Linear

    Attributes:
        epsilon: The perturbation magnitude used for gradient estimation.
        _aux_seed: Internal state tracking the current perturbation seed.
        _direction: Cached weight perturbation vector for the current seed.
        _bias_direction: Cached bias perturbation vector for the current seed.
        _gen: Per-layer random number generator to avoid global RNG corruption.

    Example:
        >>> # Create a ZO linear layer
        >>> layer = Linear(784, 128, epsilon=1e-3)
        >>>
        >>> # Inference (standard behavior)
        >>> output = layer(input_data)
        >>>
        >>> # Training with perturbations
        >>> output_plus = layer(input_data, seed=42)   # +ε perturbation
        >>> output_minus = layer(input_data, seed=42)  # -ε perturbation
        >>>
        >>> # Apply gradient feedback
        >>> loss_diff = loss_plus - loss_minus
        >>> layer.feedback(loss_diff.item())

    Note:
        This implementation uses a deterministic random number generator seeded
        per-layer to ensure reproducible perturbations and avoid interference
        with other random operations in the model.
    """

    def __init__(self, *args, epsilon=1e-3, **kwargs):
        """
        Initialize the Zero-Order Linear layer.

        Args:
            *args: Positional arguments for nn.Linear (in_features, out_features, bias).
            epsilon: Perturbation magnitude for finite difference gradient estimation.
                Must be positive. Typical values range from 1e-4 to 1e-2. Default: 1e-3.
            **kwargs: Additional keyword arguments for nn.Linear.

        Raises:
            ValueError: If epsilon is not positive.
        """
        super().__init__(*args, **kwargs)
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.epsilon = epsilon
        self._aux_seed: Optional[int] = None
        self._direction: Optional[Tensor] = None
        self._bias_direction: Optional[Tensor] = None
        self._gen = torch.Generator(device=self.weight.device)
        self._gen.manual_seed(torch.initial_seed())

    def forward(self, input: Tensor, seed: Optional[int] = None) -> Tensor:
        """
        Forward pass with optional perturbation for gradient estimation.

        This method implements the core MeZO algorithm behavior:
        - If seed is None: Standard linear transformation (inference mode)
        - If seed is provided: Apply perturbations for gradient estimation (training mode)

        In training mode, the method expects to be called twice with the same seed:
        1. First call: applies +ε perturbation to weights and bias
        2. Second call: applies -ε perturbation to weights and bias

        The perturbation vectors are deterministically generated using the provided
        seed to ensure reproducibility and consistency between the two calls.

        Args:
            input: Input tensor with shape (batch_size, in_features).
            seed: Random seed for perturbation generation. If None, performs
                standard inference without perturbations. If provided, must be
                called twice with the same seed for proper gradient estimation.

        Returns:
            Output tensor with shape (batch_size, out_features).

        Raises:
            RuntimeError: If the perturbation state is inconsistent or invalid.

        Example:
            >>> layer = Linear(10, 5)
            >>>
            >>> # Inference mode
            >>> output = layer(input_data)
            >>>
            >>> # Training mode - two calls with same seed
            >>> output_1 = layer(input_data, seed=42)  # +ε perturbation
            >>> output_2 = layer(input_data, seed=42)  # -ε perturbation
        """

        # Standard inference forward
        if seed is None:  # We are in inference, use std forward
            return super().forward(input)

        # Calculate perturbation direction (gradient estimation)
        self._direction, self._bias_direction = self.perturbation_vector(seed)
        w_perturbation = self.epsilon * self._direction
        b_perturbation = self.epsilon * self._bias_direction if self._bias_direction is not None else None

        # First forward training with perturbation
        if seed != self._aux_seed:  # We are in the first forward training
            self._aux_seed = seed  # Store seed
            bias_aux = None if self.bias is None else self.bias + b_perturbation
            return F.linear(input, self.weight + w_perturbation, bias_aux)

        # Second forward training with opposite perturbation
        if seed == self._aux_seed:  # We are in the second forward training
            self._aux_seed = None  # Remove seed
            bias_aux = None if self.bias is None else self.bias - b_perturbation
            return F.linear(input, self.weight - w_perturbation, bias_aux)

        raise RuntimeError("Unexpected condition in ZOLinear forward")

    def feedback(self, projected_grad: float) -> None:
        """
        Apply gradient feedback using the finite difference estimate.

        This method computes and sets the gradients for the layer's parameters
        based on the projected gradient from the finite difference approximation.
        It implements the core gradient estimation formula:

        gradient = (projected_grad / (2 * epsilon)) * perturbation_direction

        The method must be called after completing a perturbation pair (two forward
        passes with the same seed) and before the optimizer step.

        Args:
            projected_grad: The finite difference value (loss_1 - loss_2) from
                the two perturbed forward passes. This represents the directional
                derivative along the random perturbation direction.

        Raises:
            RuntimeError: If called before completing a perturbation pair or if
                perturbation directions are not available.

        Example:
            >>> # After two forward passes
            >>> output_1 = layer(input_data, seed=42)  # +ε perturbation
            >>> output_2 = layer(input_data, seed=42)  # -ε perturbation
            >>>
            >>> # Compute losses and apply feedback
            >>> loss_1 = criterion(output_1, target)
            >>> loss_2 = criterion(output_2, target)
            >>> layer.feedback((loss_1 - loss_2).item())

        Note:
            This method directly modifies the .grad attributes of the layer's
            parameters, making them ready for optimizer.step().
        """
        if self._direction is None:
            raise RuntimeError("feedback called before completing a perturbation pair")

        scale = projected_grad / (2 * self.epsilon)
        self.weight.grad = scale * self._direction

        if self.bias is not None and self._bias_direction is not None:
            self.bias.grad = scale * self._bias_direction

        # Clear stored directions after feedback
        self._direction = None
        self._bias_direction = None

    def perturbation_vector(self, seed):
        """
        Generate deterministic perturbation vectors for weights and bias.

        This method creates random perturbation vectors that are used to estimate
        gradients via finite differences. The vectors are generated deterministically
        using the provided seed to ensure consistency between forward passes.

        The perturbation vectors have the same shape as the corresponding parameters
        and are sampled from a standard normal distribution. This choice provides
        unbiased gradient estimates while maintaining computational efficiency.

        Args:
            seed: Random seed for deterministic perturbation generation.
                Must be an integer to ensure reproducible results.

        Returns:
            Tuple containing:
            - w_direction (Tensor): Weight perturbation vector with shape
                matching self.weight.shape, sampled from N(0,1).
            - b_direction (Tensor or None): Bias perturbation vector with shape
                matching self.bias.shape if bias exists, otherwise None.

        Example:
            >>> layer = Linear(10, 5, bias=True)
            >>> w_dir, b_dir = layer.perturbation_vector(42)
            >>> print(w_dir.shape)  # torch.Size([5, 10])
            >>> print(b_dir.shape)  # torch.Size([5])

        Note:
            This method uses a per-layer random number generator to avoid
            corrupting the global PyTorch random state, ensuring that other
            random operations in the model remain unaffected.
        """
        # Use per-layer generator to avoid global RNG corruption
        self._gen.manual_seed(seed)
        w_direction = torch.randn(
            self.weight.shape, generator=self._gen, device=self.weight.device, dtype=self.weight.dtype
        )
        b_direction = None
        if self.bias is not None:
            b_direction = torch.randn(
                self.bias.shape, generator=self._gen, device=self.bias.device, dtype=self.bias.dtype
            )
        return w_direction, b_direction


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Demonstration of Zero-Order Linear layer usage with MeZO algorithm.

    This example showcases:
    1. Creating a multi-layer model with ZO optimization
    2. Standard inference without perturbations
    3. Training with perturbations and gradient estimation
    4. Applying feedback and performing optimization steps

    The example implements the complete MeZO training loop:
    - Two forward passes with opposite perturbations
    - Finite difference calculation
    - Gradient feedback to all layers
    - Standard optimizer step
    """
    print("=== ZOLinear Simplified MeZO Example ===")

    # Create model with ZOLinear layers
    class Model(nn.Module):
        """Simple neural network using Zero-Order Linear layers."""

        def __init__(self):
            super().__init__()
            self.fc1 = Linear(10, 32, epsilon=1e-3)
            self.fc2 = Linear(32, 16, epsilon=1e-3)
            self.fc3 = Linear(16, 5, epsilon=1e-3)

        def forward(self, x, seed=None):
            """Forward pass with optional seed for perturbation."""
            x = F.relu(self.fc1(x, seed))
            x = F.relu(self.fc2(x, seed))
            x = self.fc3(x, seed)
            return x

    model = Model()
    print(f"Created model:\n{model}")

    # Set optimizer
    optm = torch.optim.SGD(model.parameters(), lr=0.01)
    print("✓ Optimizer set successfully")

    # Test forward pass with perturbation
    input_data = torch.randn(4, 10)
    target = torch.randn(4, 5)

    # Normal forward pass
    output_normal = model(input_data)
    print(f"Normal forward pass output shape: {output_normal.shape}")
    print("✓ Normal forward pass successful")

    # Forward pass with perturbation
    output_1 = model(input_data, seed=42)  # First forward with perturbation
    output_2 = model(input_data, seed=42)  # Second forward runs opposite
    print(f"Forward pass with perturbation output_1 shape: {output_1.shape}")
    print(f"Forward pass with perturbation output_2 shape: {output_2.shape}")
    print("✓ Forward pass with perturbation successful")

    # Calculate feedback
    optm.zero_grad()
    losses = [torch.nn.MSELoss()(target, out) for out in (output_1, output_2)]
    projected_grad = losses[0] - losses[1]
    print("Projected gradient:", projected_grad.item())

    # Propagate projected gradient as feedback
    model.fc1.feedback(projected_grad)
    print("Gradient after feedback on fc1:", model.fc1.weight.grad.norm().item())
    model.fc2.feedback(projected_grad)
    print("Gradient after feedback on fc2:", model.fc2.weight.grad.norm().item())
    model.fc3.feedback(projected_grad)
    print("Gradient after feedback on fc3:", model.fc3.weight.grad.norm().item())
    print("✓ Feedback applied successfully")

    # Perform optimization step
    optm.step()
    print("✓ Optimization step completed successfully")
