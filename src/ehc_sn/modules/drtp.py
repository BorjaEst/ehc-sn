"""Direct Random Target Projection (DRTP) module for biologically plausible learning.

This module implements Direct Random Target Projection (DRTP), a biologically plausible
learning algorithm that uses fixed random projection matrices to propagate target signals
directly to hidden layers. Unlike standard backpropagation, DRTP eliminates the need for
symmetric weight transport by providing each layer with direct random projections of the
target signal.

DRTP addresses the weight transport problem in biological neural networks while maintaining
effective learning capabilities. The algorithm uses fixed random matrices to project target
signals to each hidden layer, enabling local learning without requiring knowledge of forward
path weights.

Key Features:
    - DRTPFunction: Custom autograd function implementing DRTP gradient computation
    - DRTPLayer: PyTorch module wrapper for easy integration into networks
    - Fixed random projection matrices that remain constant during training
    - Direct target signal propagation to all hidden layers

Classes:
    DRTPFunction: Core autograd function for DRTP gradient computation
    DRTPLayer: Neural network module implementing DRTP learning mechanism

Mathematical Formulation:
    Standard BP: grad = W^T @ upstream_grad
    DRTP: grad = B^T @ target_signal
    Where B is a fixed random matrix and target_signal is the desired output.

Examples:
    >>> # Create a network with DRTP learning
    >>> drtp_layer = DRTPLayer(target_dim=10, hidden_dim=128)
    >>> hidden = torch.randn(32, 128, requires_grad=True)
    >>> target = torch.randn(32, 10)
    >>>
    >>> # Forward pass includes target for gradient computation
    >>> output = drtp_layer(hidden, target)
    >>> loss.backward()  # Uses DRTP gradients automatically

Biological Motivation:
    DRTP provides a biologically plausible alternative to backpropagation by using
    random feedback connections that don't require symmetric weights. This addresses
    fundamental biological constraints while maintaining effective learning.

References:
    Nøkland, A. (2016). Direct feedback alignment provides learning in deep neural
    networks without loss gradients. arXiv preprint arXiv:1609.01596.
"""

from math import prod
from typing import Any, List, Tuple

import torch
from torch import Size, Tensor, autograd, nn


# -------------------------------------------------------------------------------------------
class DRTPFunction(autograd.Function):
    """
    Custom autograd function for Direct Random Target Projection (DRTP).
    The backward pass uses a fixed random projection matrix to propagate the error.
    """

    @staticmethod
    def forward(ctx, inputs: Tensor, fb_weights: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass: returns the input unchanged to maintain computational graph.
        """
        ctx.save_for_backward(fb_weights)
        ctx.target = target  # Save target signal for backward
        return inputs

    # -----------------------------------------------------------------------------------
    @staticmethod
    def backward(ctx, grad_output: Tensor, *gradients: Any) -> Tuple[Tensor, None, None]:
        """
        Backward pass: propagate the error using the fixed random matrix fb_weights.
        Instead of using grad_output, we use fb_weights^T · target as the gradient signal.
        """
        (fb_weights,) = ctx.saved_tensors
        batch_size = grad_output.shape[0]

        # DRTP: delta = fb_weights^T · target (not grad_output)
        # target shape: (batch, target_dim) @ fb_weights shape: (target_dim, hidden_dim),
        grad_est = torch.matmul(ctx.target.view(batch_size, -1), fb_weights)

        # Return gradients for input, fb_weights, target (None for non-learnable parameters)
        return grad_est.view(grad_output.shape), None, None


# -------------------------------------------------------------------------------------------
def drtp(input: Tensor, fb_weights: Tensor, target: Tensor) -> Tensor:
    """
    DRTP layer wrapper for use in nn.Module.

    Args:
        input: Hidden layer activations (batch_size, hidden_dim)
        fb_weights: Fixed random projection matrix (target_dim, hidden_dim)
        target: Target values (batch_size, target_dim)

    Returns:
        input: Unchanged input (to maintain computational graph)
    """
    return DRTPFunction.apply(input, fb_weights, target)


# -------------------------------------------------------------------------------------------
class DRTPLayer(nn.Module):
    """
    DRTP (Direct Random Target Projection) layer module.

    This layer implements the DRTP learning mechanism where gradients are computed
    using a fixed random projection of the target instead of standard backpropagation.
    The layer maintains a fixed random projection matrix fb_weights that maps target signals
    to the hidden layer dimensions.

    During forward pass, the layer returns its input unchanged to maintain the
    computational graph. During backward pass, it uses the custom DRTPFunction
    to provide modulatory signals based on the target projection.

    This implementation is biologically inspired and provides an alternative to
    standard backpropagation that doesn't require symmetric weight transport.

    Args:
        target_dim: Dimensionality of the target space (output dimension)
        hidden_dim: Dimensionality of the hidden layer where DRTP is applied

    Example:
        >>> drtp_layer = DRTPLayer(target_dim=10, hidden_dim=128)
        >>> hidden = torch.randn(32, 128)  # batch_size=32, hidden_dim=128
        >>> target = torch.randn(32, 10)   # batch_size=32, target_dim=10
        >>> output = drtp(hidden, target)  # Returns hidden unchanged
    """

    def __init__(self, target_dim: List[int] | int, hidden_dim: List[int] | int):
        """
        Initialize a DRTP layer with a fixed random projection matrix.

        Args:
            target_dim: Dimensionality of the target space (e.g., output classes)
            hidden_dim: Dimensionality of the hidden layer activations
        """
        super().__init__()

        # Store dimensions for reference
        self.target_dim = target_dim if isinstance(target_dim, int) else prod(target_dim)
        self.hidden_dim = hidden_dim if isinstance(hidden_dim, int) else prod(hidden_dim)

        # Shape: (target_dim, hidden_dim) to project target to hidden space
        fb_weights_shape = Size([self.target_dim, self.hidden_dim])

        # Convert to a non-trainable parameter to save (saves with the model)
        self.fb_weights = nn.Parameter(torch.Tensor(fb_weights_shape))
        self.reset_weights()  # Initis weights and requires_grad=False

    # -----------------------------------------------------------------------------------
    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass through the DRTP layer.

        During forward pass, the input is returned unchanged to maintain the
        computational graph. The DRTP mechanism is applied during the backward
        pass through the custom autograd function.

        Args:
            inputs: Hidden layer activations with shape (batch_size, *hidden_dim)
            target: Target values with shape (batch_size, *target_dim)

        Returns:
            Tensor: Input unchanged, maintaining computational graph for gradient flow

        Raises:
            ValueError: If input or target dimensions don't match expected shapes
        """

        # Apply DRTP function which handles the custom backward pass
        return DRTPFunction.apply(inputs, self.fb_weights, target)

    # -----------------------------------------------------------------------------------
    def extra_repr(self) -> str:
        """Return extra representation for better debugging and model inspection."""
        return f"target_dim={self.target_dim}, hidden_dim={self.hidden_dim}"

    # -----------------------------------------------------------------------------------
    def reset_weights(self):
        """
        Reinitialize the random projection matrix fb_weights.
        This can be useful for experimentation or resetting the DRTP behavior.
        """
        # Reinitialize the projection matrix with a new random matrix
        torch.nn.init.kaiming_uniform_(self.fb_weights)
        self.fb_weights.requires_grad = False  # Ensure it's non-trainable


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage of DRTP in a simple neural network
    print("=== DRTP Layer Example ===")

    # Network parameters
    input_dim, hidden1_dim, hidden2_dim, output_dim = 5, 4, 3, 2
    batch_size = 2

    # Create network layers
    layer1 = nn.Linear(input_dim, hidden1_dim)
    drtp1 = DRTPLayer(target_dim=output_dim, hidden_dim=hidden1_dim)

    layer2 = nn.Linear(hidden1_dim, hidden2_dim)
    drtp2 = DRTPLayer(target_dim=output_dim, hidden_dim=hidden2_dim)

    layer3 = nn.Linear(hidden2_dim, output_dim)  # Output layer (no DRTP)

    # Create input and target
    x = torch.randn(batch_size, input_dim, requires_grad=True)
    target = torch.randn(batch_size, output_dim)

    print(f"Input shape: {x.shape}")
    print(f"Target shape: {target.shape}")
    print(f"DRTP1 projection matrix shape: {drtp1.fb_weights.shape}")
    print(f"DRTP2 projection matrix shape: {drtp2.fb_weights.shape}")

    # Forward pass
    h1 = layer1(x)  # (batch_size, hidden1_dim)
    h1_drtp = drtp1(h1, target)  # Apply DRTP to hidden layer 1

    h2 = layer2(h1_drtp)  # (batch_size, hidden2_dim)
    h2_drtp = drtp2(h2, target)  # Apply DRTP to hidden layer 2

    output = layer3(h2_drtp)  # (batch_size, output_dim) - no DRTP on output

    # Compute loss
    loss = nn.MSELoss()(output, target)

    print(f"\nForward pass:")
    print(f"  Hidden 1 shape: {h1.shape}")
    print(f"  Hidden 2 shape: {h2.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Loss: {loss.item():.6f}")

    # Backward pass
    print(f"\nBefore backward:")
    print(f"  Layer1 grad: {layer1.weight.grad}")
    print(f"  Layer2 grad: {layer2.weight.grad}")
    print(f"  Layer3 grad: {layer3.weight.grad}")

    loss.backward()

    print(f"\nAfter backward:")
    print(f"  Layer1 grad shape: {layer1.weight.grad.shape if layer1.weight.grad is not None else None}")
    print(f"  Layer2 grad shape: {layer2.weight.grad.shape if layer2.weight.grad is not None else None}")
    print(f"  Layer3 grad shape: {layer3.weight.grad.shape if layer3.weight.grad is not None else None}")

    # Verify DRTP projections
    print(f"\nDRTP projections:")
    expected_grad1 = torch.matmul(target, drtp1.fb_weights)  # Should match h1 gradients
    expected_grad2 = torch.matmul(target, drtp2.fb_weights)  # Should match h2 gradients
    print(f"  Expected DRTP1 gradient shape: {expected_grad1.shape}")
    print(f"  Expected DRTP2 gradient shape: {expected_grad2.shape}")

    print(f"\nDRTP Layer example completed successfully!")
