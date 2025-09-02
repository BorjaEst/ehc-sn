from math import prod
from typing import Any, List, Tuple

import torch
from torch import Size, Tensor, autograd, nn

dtp_loss = nn.MSELoss()  # DTP uses MSE loss for target comparison


# -------------------------------------------------------------------------------------------
class SRTPFunction(autograd.Function):
    """
    Custom autograd function for Selective Random Target Projection (SRTP).
    The backward pass uses a fixed random projection matrix to propagate the error.
    """

    @staticmethod
    def forward(ctx, inputs: Tensor, fb_weights: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass: returns the input unchanged to maintain computational graph.
        """
        ctx.save_for_backward(inputs, fb_weights)  # As wraps, inputs are the activations
        ctx.target = target  # Save target signal for backward
        return inputs

    # -----------------------------------------------------------------------------------
    @staticmethod
    def backward(ctx, grad_output: Tensor, *gradients: Any) -> Tuple[Tensor, None, None]:
        """
        Backward pass: propagate the error using the fixed random matrix fb_weights.
        Uses the local error signal (target - activations) projected through fb_weights.
        """
        activations, fb_weights = ctx.saved_tensors
        batch_size = grad_output.shape[0]

        # Calculate local error signal (target - current_output)
        error_signal = ctx.target - activations  # Shape: (batch_size, target_dim)

        # Project error through random feedback weights
        # error_signal: (batch_size, target_dim)
        # fb_weights: (target_dim, hidden_dim)
        grad_est = torch.matmul(error_signal, fb_weights)

        # Return gradients for input, fb_weights, target (None for non-learnable parameters)
        return grad_est, None, None


# -------------------------------------------------------------------------------------------
def srtp(input: Tensor, fb_weights: Tensor, target: Tensor) -> Tensor:
    """
    SRTP layer wrapper for use in nn.Module.

    Args:
        input: Hidden layer activations (batch_size, hidden_dim)
        fb_weights: Fixed random projection matrix (target_dim, hidden_dim)
        target: Target values (batch_size, target_dim)

    Returns:
        input: Unchanged input (to maintain computational graph)
    """
    return SRTPFunction.apply(input, fb_weights, target)


# -------------------------------------------------------------------------------------------
class SRTPLayer(nn.Module):
    """
    SRTP (Selective Random Target Projection) layer module.

    This layer implements the SRTP learning mechanism where gradients are computed
    using a fixed random projection of the target instead of standard backpropagation.
    The layer maintains a fixed random projection matrix fb_weights that maps target signals
    to the hidden layer dimensions.

    During forward pass, the layer returns its input unchanged to maintain the
    computational graph. During backward pass, it uses the custom SRTPFunction
    to provide modulatory signals based on the target projection.

    This implementation is biologically inspired and provides an alternative to
    standard backpropagation that doesn't require symmetric weight transport.

    Args:
        target_dim: Dimensionality of the target space (output dimension)
        hidden_dim: Dimensionality of the hidden layer where SRTP is applied

    Example:
        >>> srtp_layer = SRTPLayer(target_dim=10, hidden_dim=128)
        >>> hidden = torch.randn(32, 128)  # batch_size=32, hidden_dim=128
        >>> target = torch.randn(32, 10)   # batch_size=32, target_dim=10
        >>> output = srtp(hidden, target)  # Returns hidden unchanged
    """

    def __init__(self, target_dim: List[int] | int, hidden_dim: List[int] | int):
        """
        Initialize a SRTP layer with a fixed random projection matrix.

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
        Forward pass through the SRTP layer.

        During forward pass, the input is returned unchanged to maintain the
        computational graph. The SRTP mechanism is applied during the backward
        pass through the custom autograd function.

        Args:
            inputs: Hidden layer activations with shape (batch_size, *hidden_dim)
            target: Target values with shape (batch_size, *target_dim)

        Returns:
            Tensor: Input unchanged, maintaining computational graph for gradient flow

        Raises:
            ValueError: If input or target dimensions don't match expected shapes
        """

        # Apply SRTP function which handles the custom backward pass
        return SRTPFunction.apply(inputs, self.fb_weights, target)

    # -----------------------------------------------------------------------------------
    def extra_repr(self) -> str:
        """Return extra representation for better debugging and model inspection."""
        return f"target_dim={self.target_dim}, hidden_dim={self.hidden_dim}"

    # -----------------------------------------------------------------------------------
    def reset_weights(self):
        """
        Reinitialize the random projection matrix fb_weights.
        This can be useful for experimentation or resetting the SRTP behavior.
        """
        # Reinitialize the projection matrix with a new random matrix
        torch.nn.init.kaiming_uniform_(self.fb_weights)
        self.fb_weights.requires_grad = False  # Ensure it's non-trainable


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage of SRTP in a simple neural network
    print("=== SRTP Layer Example ===")

    # Network parameters
    input_dim, hidden1_dim, hidden2_dim, latent_dim = 5, 4, 3, 2
    batch_size = 2

    # Create network layers
    layer1 = nn.Linear(input_dim, hidden1_dim)
    layer2 = nn.Linear(hidden1_dim, hidden2_dim)
    layer3 = nn.Linear(hidden2_dim, latent_dim)
    layer4 = nn.Linear(latent_dim, hidden2_dim)
    srtp4 = SRTPLayer(target_dim=hidden2_dim, hidden_dim=hidden2_dim)

    # Create input and target
    x = torch.randn(batch_size, input_dim, requires_grad=True)
    label = torch.randn(batch_size, latent_dim)

    print(f"Input shape: {x.shape}")
    print(f"Target shape: {hidden2_dim}")
    print(f"SRTP4 projection matrix shape: {srtp4.fb_weights.shape}")

    # Forward pass
    h1 = layer1(x)  # (batch_size, hidden1_dim)
    h2 = layer2(h1)  # (batch_size, hidden2_dim)
    output = layer3(h2)  # (batch_size, latent_dim)
    h4 = layer4(output)  # (batch_size, hidden2_dim)
    h4_srtp = srtp4(h4, target=h2)  # Apply SRTP on h4

    # Compute loss
    loss = nn.MSELoss()(output, label)

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

    # Verify SRTP projections
    print(f"\nSRTP projections:")
    expected_grad = torch.matmul(h2, srtp4.fb_weights)  # Should match h2 gradients
    print(f"  Expected SRTP1 gradient shape: {expected_grad.shape}")

    print(f"\nSRTP Layer example completed successfully!")
