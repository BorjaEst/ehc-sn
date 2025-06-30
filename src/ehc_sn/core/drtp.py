import torch
from torch import Tensor, autograd, nn


# -------------------------------------------------------------------------------------------
class DRTPFunction(autograd.Function):
    """
    Custom autograd function for Direct Random Target Projection (DRTP).
    The backward pass uses a fixed random projection matrix to propagate the error.
    """

    @staticmethod
    def forward(ctx, inputs: Tensor, B: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass: returns the input unchanged to maintain computational graph.
        """
        ctx.save_for_backward(B, target)
        return inputs

    # -----------------------------------------------------------------------------------
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """
        Backward pass: propagate the error using the fixed random matrix B.
        Instead of using grad_output, we use B^T · target as the gradient signal.
        """
        B, target = ctx.saved_tensors

        # DRTP: delta = B^T · target (not grad_output)
        # B shape: (target_dim, hidden_dim), target shape: (batch, target_dim)
        # Result should have same shape as input: (batch, hidden_dim)
        grad_input = torch.matmul(target, B)  # (batch, target_dim) @ (target_dim, hidden_dim)

        # Return gradients for input, B, target (None for non-learnable parameters)
        return grad_input, None, None


# -------------------------------------------------------------------------------------------
def drtp_layer(input: Tensor, B: Tensor, target: Tensor) -> Tensor:
    """
    DRTP layer wrapper for use in nn.Module.

    Args:
        input: Hidden layer activations (batch_size, hidden_dim)
        B: Fixed random projection matrix (target_dim, hidden_dim)
        target: Target values (batch_size, target_dim)

    Returns:
        input: Unchanged input (to maintain computational graph)
    """
    return DRTPFunction.apply(input, B, target)


# -------------------------------------------------------------------------------------------
class DRTPLayer(nn.Module):
    """
    DRTP (Direct Random Target Projection) layer module.

    This layer implements the DRTP learning mechanism where gradients are computed
    using a fixed random projection of the target instead of standard backpropagation.
    The layer maintains a fixed random projection matrix B that maps target signals
    to the hidden layer dimensions.

    During forward pass, the layer returns its input unchanged to maintain the
    computational graph. During backward pass, it uses the custom DRTPFunction
    to provide modulatory signals based on the target projection.

    This implementation is biologically inspired and provides an alternative to
    standard backpropagation that doesn't require symmetric weight transport.

    Args:
        target_dim: Dimensionality of the target space (output dimension)
        hidden_dim: Dimensionality of the hidden layer where DRTP is applied
        scale: Scaling factor for the random projection matrix initialization

    Example:
        >>> drtp_layer = DRTPLayer(target_dim=10, hidden_dim=128)
        >>> hidden = torch.randn(32, 128)  # batch_size=32, hidden_dim=128
        >>> target = torch.randn(32, 10)   # batch_size=32, target_dim=10
        >>> output = drtp_layer(hidden, target)  # Returns hidden unchanged
    """

    def __init__(self, target_dim: int, hidden_dim: int, scale: float = 0.1):
        """
        Initialize a DRTP layer with a fixed random projection matrix.

        Args:
            target_dim: Dimensionality of the target space (e.g., output classes)
            hidden_dim: Dimensionality of the hidden layer activations
            scale: Scaling factor for random matrix initialization (default: 0.1)
        """
        super().__init__()

        # Store dimensions for reference
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.scale = scale

        # Register the random projection matrix as a buffer (non-trainable)
        # Shape: (target_dim, hidden_dim) to allow target @ B multiplication
        self.register_buffer("B", torch.randn(target_dim, hidden_dim) * scale)

    # -----------------------------------------------------------------------------------
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass through the DRTP layer.

        During forward pass, the input is returned unchanged to maintain the
        computational graph. The DRTP mechanism is applied during the backward
        pass through the custom autograd function.

        Args:
            input: Hidden layer activations with shape (batch_size, hidden_dim)
            target: Target values with shape (batch_size, target_dim)

        Returns:
            Tensor: Input unchanged, maintaining computational graph for gradient flow

        Raises:
            ValueError: If input or target dimensions don't match expected shapes
        """
        # Validate input dimensions
        if input.size(-1) != self.hidden_dim:
            raise ValueError(
                f"Input last dimension {input.size(-1)} doesn't match "
                f"expected hidden_dim {self.hidden_dim}"
            )  # fmt: skip

        if target.size(-1) != self.target_dim:
            raise ValueError(
                f"Target last dimension {target.size(-1)} doesn't match "
                f"expected target_dim {self.target_dim}"
            )  # fmt: skip

        # Apply DRTP function which handles the custom backward pass
        return DRTPFunction.apply(input, self.B, target)

    # -----------------------------------------------------------------------------------
    def extra_repr(self) -> str:
        """Return extra representation for better debugging and model inspection."""
        return f"target_dim={self.target_dim}, hidden_dim={self.hidden_dim}, scale={self.scale}"

    # -----------------------------------------------------------------------------------
    def reinit_projection_matrix(self, scale: float = None):
        """
        Reinitialize the random projection matrix B.

        This can be useful for experimentation or resetting the DRTP behavior.

        Args:
            scale: New scaling factor. If None, uses the original scale.
        """
        if scale is None:
            scale = self.scale
        else:
            self.scale = scale

        # Reinitialize the projection matrix
        self.B.data = torch.randn(self.target_dim, self.hidden_dim) * scale


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
    print(f"DRTP1 projection matrix shape: {drtp1.B.shape}")
    print(f"DRTP2 projection matrix shape: {drtp2.B.shape}")

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
    expected_grad1 = torch.matmul(target, drtp1.B)  # Should match h1 gradients
    expected_grad2 = torch.matmul(target, drtp2.B)  # Should match h2 gradients
    print(f"  Expected DRTP1 gradient shape: {expected_grad1.shape}")
    print(f"  Expected DRTP2 gradient shape: {expected_grad2.shape}")

    print(f"\nDRTP Layer example completed successfully!")
