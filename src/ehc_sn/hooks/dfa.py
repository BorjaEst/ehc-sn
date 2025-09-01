from math import prod
from typing import Any, List, Tuple

import torch
from torch import Size, Tensor, autograd, nn


# -------------------------------------------------------------------------------------------
class DFAFunction(autograd.Function):
    """
    Custom autograd function for Direct Feedback Alignment (DFA).
    The backward pass uses a fixed random projection matrix to propagate the error.
    """

    @staticmethod
    def forward(ctx, inputs: Tensor, fb_weights: Tensor, grad_output: Tensor) -> Tensor:
        """
        Forward pass: returns the input unchanged to maintain computational graph.
        """
        ctx.save_for_backward(fb_weights, grad_output)
        return inputs

    # -----------------------------------------------------------------------------------
    @staticmethod
    def backward(ctx, grad_output: Tensor, *gradients: Any) -> Tuple[Tensor, None, None]:
        """
        Backward pass: propagate the error using the fixed random matrix fb_weights.
        DFA uses fb_weights^T · grad_output as the gradient signal.
        """
        fb_weights, saved_grad_output = ctx.saved_tensors
        nbatch = grad_output.shape[0]

        # DFA: delta = fb_weights^T · grad_output (error from output layer)
        # grad_output shape: (batch, output_dim) @ fb_weights shape: (output_dim, hidden_dim)
        grad_est = torch.matmul(saved_grad_output.view(nbatch, -1), fb_weights)

        # Return gradients for input, fb_weights, grad_output (None for non-learnable parameters)
        return grad_est.view(grad_output.shape), None, None


# -------------------------------------------------------------------------------------------
def dfa(input: Tensor, fb_weights: Tensor, grad_output: Tensor) -> Tensor:
    """
    DFA layer wrapper for use in nn.Module.

    Args:
        input: Hidden layer activations (batch_size, hidden_dim)
        fb_weights: Fixed random projection matrix (output_dim, hidden_dim)
        grad_output: Error signal from output layer (batch_size, output_dim)

    Returns:
        input: Unchanged input (to maintain computational graph)
    """
    return DFAFunction.apply(input, fb_weights, grad_output)


# -------------------------------------------------------------------------------------------
class DFALayer(nn.Module):
    """
    DFA (Direct Feedback Alignment) layer module.

    This layer implements the DFA learning mechanism where gradients are computed
    using a fixed random projection of the output error instead of standard backpropagation.
    The layer maintains a fixed random projection matrix fb_weights that maps error signals
    from the output layer to the hidden layer dimensions.

    During forward pass, the layer returns its input unchanged to maintain the
    computational graph. During backward pass, it uses the custom DFAFunction
    to provide modulatory signals based on the error projection.

    This implementation is biologically inspired and provides an alternative to
    standard backpropagation that doesn't require symmetric weight transport.

    Args:
        output_dim: Dimensionality of the output space (number of output units)
        hidden_dim: Dimensionality of the hidden layer where DFA is applied

    Example:
        >>> dfa_layer = DFALayer(output_dim=10, hidden_dim=128)
        >>> hidden = torch.randn(32, 128)  # batch_size=32, hidden_dim=128
        >>> grad_output = torch.randn(32, 10)   # batch_size=32, output_dim=10
        >>> output = dfa(hidden, grad_output)  # Returns hidden unchanged
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
    def forward(self, inputs: Tensor, grad_output: Tensor) -> Tensor:
        """
        Forward pass through the DFA layer.

        During forward pass, the input is returned unchanged to maintain the
        computational graph. The DFA mechanism is applied during the backward
        pass through the custom autograd function.

        Args:
            inputs: Hidden layer activations with shape (batch_size, *hidden_dim)
            grad_output: Error signal from output layer with shape (batch_size, *output_dim)

        Returns:
            Tensor: Input unchanged, maintaining computational graph for gradient flow

        Raises:
            ValueError: If input or grad_output dimensions don't match expected shapes
        """

        # Apply DFA function which handles the custom backward pass
        return DFAFunction.apply(inputs, self.fb_weights, grad_output)

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

    # Create network layers
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

    # Forward pass
    h1 = layer1(x)  # (batch_size, hidden1_dim)

    h2 = layer2(h1)  # (batch_size, hidden2_dim)

    output = layer3(h2)  # (batch_size, output_dim) - no DFA on output

    # Compute loss and get error signal
    loss = nn.MSELoss()(output, target)
    error = output - target  # Error signal for DFA

    # Apply DFA with error signal (in practice this would be integrated differently)
    h1_dfa = dfa1(h1, error)  # Apply DFA to hidden layer 1
    h2_dfa = dfa2(h2, error)  # Apply DFA to hidden layer 2

    print(f"\nForward pass:")
    print(f"  Hidden 1 shape: {h1.shape}")
    print(f"  Hidden 2 shape: {h2.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Error shape: {error.shape}")
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

    # Verify DFA projections
    print(f"\nDFA projections:")
    expected_grad1 = torch.matmul(error, dfa1.fb_weights)  # Should match h1 gradients
    expected_grad2 = torch.matmul(error, dfa2.fb_weights)  # Should match h2 gradients
    print(f"  Expected DFA1 gradient shape: {expected_grad1.shape}")
    print(f"  Expected DFA2 gradient shape: {expected_grad2.shape}")

    print(f"\nDFA Layer example completed successfully!")
