from math import prod
from typing import List, Tuple

import torch
from torch import Size, Tensor, autograd, nn

# -------------------------------------------------------------------------------------------
# Global storage for DFA error signals
# -------------------------------------------------------------------------------------------

_dfa_error: Tensor | None = None


def get_dfa_error() -> Tensor:
    """Get the current DFA error signal."""
    if _dfa_error is None:
        raise RuntimeError("No DFA error signal available. DFA hook not registered properly.")
    return _dfa_error


def set_dfa_error(error: Tensor) -> None:
    """Set the DFA error signal."""
    global _dfa_error
    _dfa_error = error


def clear_dfa_error() -> None:
    """Clear the DFA error signal."""
    global _dfa_error
    _dfa_error = None


def register_dfa_hook(output_tensor: Tensor) -> None:
    """
    Register a hook on the output tensor to capture gradients for DFA.

    This hook captures dL/dz_out during backpropagation and stores it
    for use by DFA layers throughout the network.

    Args:
        output_tensor: The final output tensor of the network (logits/predictions)
    """

    def capture_grad_hook(grad):
        if grad is not None:
            set_dfa_error(grad.detach())
        return grad

    output_tensor.register_hook(capture_grad_hook)


# -------------------------------------------------------------------------------------------
class DFAFunction(autograd.Function):
    """
    Custom autograd function for Direct Feedback Alignment (DFA).

    In DFA, each hidden layer receives gradients computed using a fixed random
    projection of the global output error, rather than through standard backpropagation.
    This eliminates the need for symmetric weight transport while maintaining
    effective learning.
    """

    @staticmethod
    def forward(ctx, inputs: Tensor, fb_weights: Tensor) -> Tensor:
        """
        Forward pass: returns the input unchanged to maintain computational graph.

        Args:
            inputs: Hidden layer activations
            fb_weights: Fixed random projection matrix (output_dim, hidden_dim)

        Returns:
            inputs: Unchanged to maintain computational graph
        """
        ctx.save_for_backward(fb_weights)
        return inputs

    # -----------------------------------------------------------------------------------
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        """
        Backward pass: Use global error signal with random feedback weights.

        The key insight of DFA is that instead of using the standard backpropagated
        gradient, we use: gradient = fb_weights^T @ global_error

        Args:
            grad_output: Standard backpropagated gradient (we ignore this in DFA)

        Returns:
            Tuple of gradients for (inputs, fb_weights)
        """
        (fb_weights,) = ctx.saved_tensors
        batch_size = grad_output.shape[0]

        # Get the global error signal - will raise if not available
        global_error = get_dfa_error()

        # DFA gradient computation: fb_weights^T @ global_error
        # global_error shape: (batch_size, output_dim)
        dfa_gradient = torch.matmul(global_error.view(batch_size, -1), fb_weights)

        # Reshape to match the input shape
        dfa_gradient = dfa_gradient.view(grad_output.shape)

        return dfa_gradient, None


# -------------------------------------------------------------------------------------------
def dfa(input: Tensor, fb_weights: Tensor) -> Tensor:
    """
    DFA layer wrapper for use in nn.Module.

    Args:
        input: Hidden layer activations (batch_size, hidden_dim)
        fb_weights: Fixed random projection matrix (output_dim, hidden_dim)

    Returns:
        input: Unchanged input (to maintain computational graph)
    """
    return DFAFunction.apply(input, fb_weights)


# -------------------------------------------------------------------------------------------
class DFALayer(nn.Module):
    """
    DFA (Direct Feedback Alignment) layer module.

    This layer implements the DFA learning mechanism where gradients are computed
    using a fixed random projection of the global output error instead of standard
    backpropagation. The layer maintains a fixed random projection matrix fb_weights
    that maps the global error signal to the hidden layer dimensions.

    During forward pass, the layer returns its input unchanged to maintain the
    computational graph. During backward pass, it uses the custom DFAFunction
    to compute gradients using the global error captured via hooks on the network output.

    This implementation is biologically inspired and provides an alternative to
    standard backpropagation that doesn't require symmetric weight transport.
    The global error is automatically captured when register_dfa_hook() is called
    on the network's final output tensor.

    Args:
        output_dim: Dimensionality of the output space (number of output units)
        hidden_dim: Dimensionality of the hidden layer where DFA is applied

    Example:
        >>> # Create network with DFA layers
        >>> dfa_layer = DFALayer(output_dim=10, hidden_dim=128)
        >>> hidden = torch.randn(32, 128, requires_grad=True)
        >>>
        >>> # Forward pass (returns input unchanged)
        >>> output = dfa_layer(hidden)
        >>>
        >>> # Register hook on final network output to capture global error
        >>> register_dfa_hook(final_output)
        >>>
        >>> # Backward pass automatically uses DFA gradients
        >>> loss.backward()
    """

    def __init__(self, output_dim: List[int] | int, hidden_dim: List[int] | int):
        """
        Initialize a DFA layer with a fixed random projection matrix.

        Args:
            output_dim: Dimensionality of the output space (e.g., number of classes)
            hidden_dim: Dimensionality of the hidden layer activations
        """
        super().__init__()

        # Store dimensions for reference
        self.output_dim = output_dim if isinstance(output_dim, int) else prod(output_dim)
        self.hidden_dim = hidden_dim if isinstance(hidden_dim, int) else prod(hidden_dim)

        # Shape: (output_dim, hidden_dim) to project error to hidden space
        fb_weights_shape = Size([self.output_dim, self.hidden_dim])

        # Convert to a non-trainable parameter to save (saves with the model)
        self.fb_weights = nn.Parameter(torch.Tensor(fb_weights_shape))
        self.reset_weights()  # Initis weights and requires_grad=False

    # -----------------------------------------------------------------------------------
    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass through the DFA layer.

        During forward pass, the input is returned unchanged to maintain the
        computational graph. The DFA mechanism is applied during the backward
        pass through the custom autograd function.

        Args:
            inputs: Hidden layer activations with shape (batch_size, *hidden_dim)

        Returns:
            Tensor: Input unchanged, maintaining computational graph for gradient flow

        Note:
            The global error signal is automatically captured during backpropagation
            through the hook registered on the network output. No manual error
            passing is required.
        """
        # Apply DFA function which handles the custom backward pass
        return DFAFunction.apply(inputs, self.fb_weights)

    # -----------------------------------------------------------------------------------
    def extra_repr(self) -> str:
        """Return extra representation for better debugging and model inspection."""
        return f"output_dim={self.output_dim}, hidden_dim={self.hidden_dim}"

    # -----------------------------------------------------------------------------------
    def reset_weights(self):
        """
        Reinitialize the random projection matrix fb_weights.
        This can be useful for experimentation or resetting the DFA behavior.
        """
        # Reinitialize the projection matrix with a new random matrix
        torch.nn.init.kaiming_uniform_(self.fb_weights)
        self.fb_weights.requires_grad = False  # Ensure it's non-trainable


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage of DFA in a simple neural network
    print("=== DFA Layer Example ===")

    # Network parameters
    input_dim, hidden1_dim, hidden2_dim, output_dim = 5, 4, 3, 2
    batch_size = 2

    # Create network layers with DFA
    layer1 = nn.Linear(input_dim, hidden1_dim)
    dfa1 = DFALayer(output_dim=output_dim, hidden_dim=hidden1_dim)

    layer2 = nn.Linear(hidden1_dim, hidden2_dim)
    dfa2 = DFALayer(output_dim=output_dim, hidden_dim=hidden2_dim)

    layer3 = nn.Linear(hidden2_dim, output_dim)  # Output layer (no DFA)

    # Create input and target
    x = torch.randn(batch_size, input_dim, requires_grad=True)
    target = torch.randn(batch_size, output_dim)

    print(f"Input shape: {x.shape}")
    print(f"Target shape: {target.shape}")
    print(f"DFA1 projection matrix shape: {dfa1.fb_weights.shape}")
    print(f"DFA2 projection matrix shape: {dfa2.fb_weights.shape}")

    # Forward pass with DFA layers integrated into the network
    h1 = layer1(x)  # (batch_size, hidden1_dim)
    h1_dfa = dfa1(h1)  # Apply DFA to hidden layer 1 (returns h1 unchanged)

    h2 = layer2(h1_dfa)  # (batch_size, hidden2_dim)
    h2_dfa = dfa2(h2)  # Apply DFA to hidden layer 2 (returns h2 unchanged)

    output = layer3(h2_dfa)  # (batch_size, output_dim)

    # IMPORTANT: Register DFA hook on output to capture global error
    register_dfa_hook(output)

    # Compute loss
    loss = nn.MSELoss()(output, target)

    print(f"\nForward pass:")
    print(f"  Hidden 1 shape: {h1.shape}")
    print(f"  Hidden 2 shape: {h2.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Loss: {loss.item():.6f}")

    # Backward pass - DFA gradients are automatically computed
    print(f"\nBefore backward:")
    print(f"  Layer1 grad: {layer1.weight.grad}")
    print(f"  Layer2 grad: {layer2.weight.grad}")
    print(f"  Layer3 grad: {layer3.weight.grad}")

    loss.backward()

    print(f"\nAfter backward:")
    print(f"  Layer1 grad shape: {layer1.weight.grad.shape if layer1.weight.grad is not None else None}")
    print(f"  Layer2 grad shape: {layer2.weight.grad.shape if layer2.weight.grad is not None else None}")
    print(f"  Layer3 grad shape: {layer3.weight.grad.shape if layer3.weight.grad is not None else None}")

    # Check if global error was captured
    try:
        global_error = get_dfa_error()
        print(f"\nGlobal error captured successfully:")
        print(f"  Global error shape: {global_error.shape}")

        # Verify DFA projections
        expected_grad1 = torch.matmul(global_error, dfa1.fb_weights)
        expected_grad2 = torch.matmul(global_error, dfa2.fb_weights)
        print(f"  Expected DFA1 gradient shape: {expected_grad1.shape}")
        print(f"  Expected DFA2 gradient shape: {expected_grad2.shape}")
    except RuntimeError as e:
        print(f"\nWarning: {e}")

    # Clean up
    clear_dfa_error()

    print(f"\nDFA Layer example completed successfully!")
