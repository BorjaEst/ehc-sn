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
    """Custom autograd function implementing Direct Random Target Projection (DRTP).

    This function provides the core DRTP mechanism where hidden layers receive
    gradients computed using fixed random projections of the target signal,
    rather than through standard backpropagation. This eliminates the weight
    transport problem and provides a biologically plausible learning mechanism.

    The key insight of DRTP is that effective learning can occur when gradients
    are computed using random projections of the target signal directly, provided
    the projection matrices are appropriately scaled and maintained throughout training.

    Mathematical formulation:
        Standard BP: grad = W^T @ upstream_grad
        DRTP: grad = B^T @ target_signal
        Where B is a fixed random matrix and target_signal is the desired output.

    Biological motivation:
        Real neurons cannot implement backpropagation due to the weight transport
        problem - feedback connections don't have access to forward weights. DRTP
        provides a plausible alternative using fixed random feedback connections.

    Note:
        This function maintains the computational graph by returning inputs unchanged
        during forward pass, while implementing DRTP gradients during backward pass.
    """

    @staticmethod
    def forward(ctx, inputs: Tensor, fb_weights: Tensor, target: Tensor) -> Tensor:
        """Forward pass: store context and return input unchanged.

        Saves the feedback weights and target signal for use in the backward pass
        while returning the input tensor unchanged to maintain computational graph.

        Args:
            ctx: Context object for saving information for backward pass.
            inputs: Hidden layer activations of shape (batch_size, hidden_dim).
            fb_weights: Fixed random projection matrix of shape (target_dim, hidden_dim).
            target: Target signal of shape (batch_size, target_dim).

        Returns:
            Input tensor unchanged to preserve computational graph structure.
        """
        ctx.save_for_backward(fb_weights)
        ctx.target = target  # Save target signal for backward
        return inputs

    # -----------------------------------------------------------------------------------
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None, None]:
        """Backward pass: compute DRTP gradients using target projection.

        Computes gradients using the DRTP algorithm where the gradient signal
        is derived from the target projection rather than upstream gradients.
        This breaks the dependence on symmetric weight transport.

        Args:
            ctx: Context object containing saved tensors and target signal.
            grad_output: Upstream gradients (ignored in DRTP computation).

        Returns:
            Tuple containing:
                - DRTP gradient for inputs based on target projection
                - None for fb_weights (non-trainable)
                - None for target (non-trainable)

        Note:
            The DRTP gradient is computed as B^T @ target, where B is the fixed
            random projection matrix and target is the desired output signal.
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
    """Apply Direct Random Target Projection to input activations.

    Functional interface for applying DRTP learning mechanism to tensor inputs.
    This function wraps the DRTPFunction autograd function for convenient use
    in neural network forward passes.

    The function implements the DRTP algorithm where gradients are computed
    using fixed random projections of the target signal rather than standard
    backpropagation, providing a biologically plausible learning mechanism.

    Args:
        input: Hidden layer activations of shape (batch_size, hidden_dim).
            These are the activations that will receive DRTP gradients.
        fb_weights: Fixed random projection matrix of shape (target_dim, hidden_dim).
            Used to project target signal back to hidden layer dimensions.
        target: Target signal of shape (batch_size, target_dim).
            The desired output that drives the learning signal.

    Returns:
        Input tensor unchanged to maintain computational graph. The DRTP mechanism
        is applied during the backward pass through the custom autograd function.

    Examples:
        >>> # Apply DRTP to hidden layer activations
        >>> hidden = torch.randn(32, 128, requires_grad=True)
        >>> fb_matrix = torch.randn(10, 128)  # target_dim=10, hidden_dim=128
        >>> target_signal = torch.randn(32, 10)
        >>> output = drtp(hidden, fb_matrix, target_signal)
        >>> # output == hidden, but backward pass will use DRTP gradients
    """
    return DRTPFunction.apply(input, fb_weights, target)


# -------------------------------------------------------------------------------------------
class DRTPLayer(nn.Module):
    """Direct Random Target Projection (DRTP) layer for biologically plausible learning.

    This layer implements the DRTP learning mechanism where gradients are computed
    using a fixed random projection of the target signal instead of standard
    backpropagation. The layer maintains a fixed random projection matrix that
    maps target signals to hidden layer dimensions.

    During forward pass, the layer returns its input unchanged to maintain the
    computational graph. During backward pass, it uses the custom DRTPFunction
    to provide modulatory signals based on the target projection, eliminating
    the need for symmetric weight transport.

    The DRTP mechanism addresses the biological implausibility of backpropagation
    by using fixed random feedback connections that don't require knowledge of
    forward weights, while still enabling effective learning.

    Attributes:
        target_dim: Dimensionality of the target space (flattened if originally multi-dimensional)
        hidden_dim: Dimensionality of the hidden layer (flattened if originally multi-dimensional)
        fb_weights: Fixed random projection matrix of shape (target_dim, hidden_dim)

    Args:
        target_dim: Dimensionality of the target space. Can be int or list of ints.
            If list, will be flattened to total number of elements.
        hidden_dim: Dimensionality of the hidden layer. Can be int or list of ints.
            If list, will be flattened to total number of elements.

    Example:
        >>> # Create DRTP layer for projecting 10D targets to 128D hidden layer
        >>> drtp_layer = DRTPLayer(target_dim=10, hidden_dim=128)
        >>> hidden = torch.randn(32, 128)  # batch_size=32, hidden_dim=128
        >>> target = torch.randn(32, 10)   # batch_size=32, target_dim=10
        >>> output = drtp_layer(hidden, target)  # Returns hidden unchanged
        >>>
        >>> # Multi-dimensional example
        >>> drtp_layer = DRTPLayer(target_dim=[1, 32, 16], hidden_dim=256)
        >>> # target_dim becomes 512 (1*32*16), hidden_dim stays 256

    References:
        Nøkland, A. (2016). Direct feedback alignment provides learning in deep neural
        networks without loss gradients. arXiv preprint arXiv:1609.01596.
    """

    def __init__(self, target_dim: List[int] | int, hidden_dim: List[int] | int):
        """Initialize DRTP layer with fixed random projection matrix.

        Creates and initializes the fixed random projection matrix that will be
        used to map target signals to hidden layer gradients. The matrix is
        non-trainable and remains constant throughout training.

        Args:
            target_dim: Dimensionality of the target space. If list, total size
                is computed as the product of all dimensions.
            hidden_dim: Dimensionality of the hidden layer. If list, total size
                is computed as the product of all dimensions.

        Note:
            The projection matrix is initialized using Kaiming uniform initialization
            and is set as non-trainable to maintain the DRTP property of fixed
            random feedback connections.
        """
        super().__init__()

        # Store dimensions for reference
        self.target_dim = target_dim if isinstance(target_dim, int) else prod(target_dim)
        self.hidden_dim = hidden_dim if isinstance(hidden_dim, int) else prod(hidden_dim)

        # Shape: (target_dim, hidden_dim) to project target to hidden space
        fb_weights_shape = Size([self.target_dim, self.hidden_dim])

        # Convert to a non-trainable parameter to save (saves with the model)
        self.fb_weights = nn.Parameter(torch.Tensor(fb_weights_shape))
        self.reset_weights()  # Initialize weights and set requires_grad=False

    # -----------------------------------------------------------------------------------
    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        """Forward pass through the DRTP layer.

        Applies the DRTP mechanism to the input activations using the target signal.
        During forward pass, the input is returned unchanged to maintain the
        computational graph. The DRTP gradients are computed during the backward
        pass through the custom autograd function.

        Args:
            inputs: Hidden layer activations with shape (batch_size, hidden_dim).
                These activations will receive DRTP-based gradients during backprop.
            target: Target signal with shape (batch_size, target_dim).
                Used to compute the projection-based gradients in backward pass.

        Returns:
            Input tensor unchanged, maintaining computational graph for gradient flow.
            The DRTP mechanism modifies gradient computation during backward pass.

        Raises:
            RuntimeError: If tensor shapes are incompatible with expected dimensions.

        Note:
            The target tensor is used only for gradient computation during backward
            pass. The forward pass simply maintains the computational graph while
            setting up the context for DRTP gradient calculation.
        """

        # Apply DRTP function which handles the custom backward pass
        return DRTPFunction.apply(inputs, self.fb_weights, target)

    # -----------------------------------------------------------------------------------
    def extra_repr(self) -> str:
        """Return extra representation string for better debugging and model inspection.

        Returns:
            String representation showing target and hidden dimensions for debugging.
        """
        return f"target_dim={self.target_dim}, hidden_dim={self.hidden_dim}"

    # -----------------------------------------------------------------------------------
    def reset_weights(self):
        """Reinitialize the random projection matrix with new random values.

        Resets the fixed random projection matrix fb_weights using Kaiming uniform
        initialization. This can be useful for experimentation or resetting the
        DRTP behavior to different random feedback connections.

        Note:
            After reinitialization, the matrix is set as non-trainable to maintain
            the DRTP property of fixed feedback weights throughout training.
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
