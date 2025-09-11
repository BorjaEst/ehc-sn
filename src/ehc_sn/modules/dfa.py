"""Direct Feedback Alignment (DFA) module for biologically plausible learning.

This module implements Direct Feedback Alignment (DFA), a biologically plausible
alternative to backpropagation that eliminates the need for symmetric weight transport.
In DFA, hidden layers receive direct feedback from the output error via fixed random
projection matrices, enabling local learning without full gradient information.

Key Features:
    - DFAFunction: Custom autograd function implementing DFA gradient computation
    - DFALayer: PyTorch module wrapper for easy integration into networks
    - Hook-based error signal management via centralized registry system
    - Biologically motivated learning without symmetric weight constraints

Classes:
    DFAFunction: Core autograd function for DFA gradient computation
    DFALayer: Neural network module implementing DFA learning mechanism

Functions:
    dfa: Functional interface for applying DFA to tensor inputs

Examples:
    >>> # Create a network with DFA learning
    >>> dfa_layer = DFALayer(output_dim=10, hidden_dim=128)
    >>> hidden = torch.randn(32, 128, requires_grad=True)
    >>> output = dfa_layer(hidden)  # Returns input unchanged
    >>>
    >>> # Register hook for error capture and perform backward pass
    >>> register_dfa_hook(final_output)
    >>> loss.backward()  # Uses DFA gradients automatically

References:
    Lillicrap, T. P., et al. (2016). Random synaptic feedback weights support
    error backpropagation for deep learning. Nature Communications, 7, 13276.
"""

from math import prod
from typing import List, Tuple

import torch
from torch import Size, Tensor, autograd, nn

from ehc_sn.hooks.registry import clear_dfa_error, get_dfa_error, register_dfa_hook


# -------------------------------------------------------------------------------------------
class DFAFunction(autograd.Function):
    """Custom autograd function implementing Direct Feedback Alignment (DFA).

    This function provides the core DFA mechanism where hidden layers receive
    gradients computed using fixed random projections of the global output error,
    rather than through standard backpropagation. This eliminates the weight
    transport problem and provides a biologically plausible learning mechanism.

    The key insight of DFA is that effective learning can occur even when gradients
    are computed using random projections of the output error, provided the feedback
    alignment property is satisfied through training dynamics.

    Mathematical formulation:
        Standard BP: grad = W^T @ upstream_grad
        DFA: grad = B^T @ global_error
        Where B is a fixed random matrix and global_error is the output error.

    Biological motivation:
        Real neurons cannot implement backpropagation due to the weight transport
        problem - feedback connections don't have access to forward weights. DFA
        provides a plausible alternative using random feedback connections.

    Note:
        This function maintains the computational graph by returning inputs unchanged
        during forward pass, while implementing DFA gradients during backward pass.
    """

    @staticmethod
    def forward(ctx, inputs: Tensor, fb_weights: Tensor) -> Tensor:
        """Forward pass maintaining computational graph.

        Returns the input unchanged to preserve the forward computation while
        storing the feedback weights for gradient computation during backward pass.

        Args:
            ctx: PyTorch autograd context for saving tensors between passes.
            inputs: Hidden layer activations of shape (batch_size, hidden_dim).
            fb_weights: Fixed random projection matrix of shape (output_dim, hidden_dim).

        Returns:
            Unchanged input tensor to maintain computational graph integrity.
        """
        ctx.save_for_backward(fb_weights)
        return inputs

    # -----------------------------------------------------------------------------------
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        """Backward pass implementing DFA gradient computation.

        Computes gradients using the global output error and fixed random feedback
        weights instead of standard backpropagation. This implements the core DFA
        mechanism where each hidden layer receives direct feedback from the output.

        Mathematical formulation:
            DFA gradient: grad = B^T @ global_error
            Where B is the fixed random feedback matrix and global_error is captured
            from the network output via the hook registry system.

        Args:
            ctx: PyTorch autograd context containing saved tensors from forward pass.
            grad_output: Standard backpropagated gradient (ignored in DFA).

        Returns:
            Tuple containing (input_gradient, feedback_weights_gradient).
            The feedback weights gradient is None since these are fixed.

        Raises:
            RuntimeError: If global error signal is not available from registry.
                This typically means register_dfa_hook() was not called on output.

        Note:
            The global error must be captured via register_dfa_hook() before
            backward pass. The standard grad_output is intentionally ignored.
        """
        (fb_weights,) = ctx.saved_tensors
        batch_size = grad_output.shape[0]

        # Get the global error signal - will raise if not available
        global_error = get_dfa_error()

        # DFA gradient computation: fb_weights^T @ global_error
        # global_error shape: (batch_size, output_dim)
        grad_est = torch.matmul(global_error.view(batch_size, -1), fb_weights)

        return grad_est.view(grad_output.shape), None


# -------------------------------------------------------------------------------------------
def dfa(input: Tensor, fb_weights: Tensor) -> Tensor:
    """Apply Direct Feedback Alignment to input activations.

    Functional interface for applying DFA learning mechanism to tensor inputs.
    This function wraps the DFAFunction autograd function for convenient use
    in neural network forward passes.

    Args:
        input: Hidden layer activations of shape (batch_size, hidden_dim).
            These are the activations that will receive DFA gradients.
        fb_weights: Fixed random projection matrix of shape (output_dim, hidden_dim).
            Used to project global error back to this layer's dimensions.

    Returns:
        Input tensor unchanged to maintain computational graph. The DFA mechanism
        is applied during the backward pass through the custom autograd function.

    Examples:
        >>> # Apply DFA to hidden layer activations
        >>> hidden = torch.randn(32, 128, requires_grad=True)
        >>> fb_matrix = torch.randn(10, 128)  # output_dim=10, hidden_dim=128
        >>> output = dfa(hidden, fb_matrix)
        >>> # output == hidden, but backward pass will use DFA gradients
    """
    return DFAFunction.apply(input, fb_weights)


# -------------------------------------------------------------------------------------------
class DFALayer(nn.Module):
    """Direct Feedback Alignment layer for biologically plausible learning.

    This layer implements the DFA learning mechanism where gradients are computed
    using a fixed random projection of the global output error instead of standard
    backpropagation. The layer maintains a fixed random projection matrix that
    maps the global error signal to the hidden layer dimensions.

    During forward pass, the layer returns its input unchanged to maintain the
    computational graph. During backward pass, it uses the custom DFAFunction
    to compute gradients using the global error captured via hooks on the network
    output tensor.

    This implementation is biologically inspired and provides an alternative to
    standard backpropagation that doesn't require symmetric weight transport,
    addressing the weight transport problem in biological neural networks.

    Attributes:
        fb_weights: Fixed random projection matrix of shape (output_dim, hidden_dim).
            These weights are initialized once and never updated during training.
            They provide the random feedback pathway for DFA learning.

    Examples:
        >>> # Create DFA layer for hidden layer with 128 units, output with 10 units
        >>> dfa_layer = DFALayer(output_dim=10, hidden_dim=128)
        >>> hidden = torch.randn(32, 128, requires_grad=True)
        >>>
        >>> # Forward pass returns input unchanged
        >>> output = dfa_layer(hidden)
        >>> assert torch.equal(output, hidden)
        >>>
        >>> # Register hook on final network output to capture global error
        >>> register_dfa_hook(final_network_output)
        >>> loss.backward()  # Uses DFA gradients automatically

    Note:
        The global error signal must be captured by calling register_dfa_hook()
        on the final network output before the backward pass. This error is then
        automatically used by all DFA layers in the network.
    """

    def __init__(self, output_dim: List[int] | int, hidden_dim: List[int] | int):
        """Initialize DFA layer with fixed random feedback weights.

        Creates a DFA layer that will apply Direct Feedback Alignment during
        the backward pass. The feedback weights are initialized randomly and
        remain fixed throughout training, providing the random feedback pathway
        characteristic of DFA learning.

        Args:
            output_dim: Dimensionality of the network output. Can be an integer
                for 1D outputs or a list of integers for multi-dimensional outputs.
                Used to determine the size of the global error signal.
            hidden_dim: Dimensionality of the hidden layer where DFA is applied.
                Can be an integer for 1D hidden layers or a list of integers for
                multi-dimensional hidden representations.

        Note:
            The feedback weights are initialized with standard normal distribution
            and are registered as non-trainable parameters to prevent updates.
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
        """Forward pass through the DFA layer.

        During forward pass, the input is returned unchanged to maintain the
        computational graph. The DFA mechanism is applied during the backward
        pass through the custom autograd function, which uses the global error
        signal captured via registered hooks.

        Args:
            inputs: Hidden layer activations with shape (batch_size, *hidden_dim).
                These are the activations that will receive DFA gradients during
                the backward pass.

        Returns:
            Input tensor unchanged, maintaining computational graph for gradient flow.
            The DFA gradients are computed automatically during backpropagation.

        Note:
            The global error signal is automatically captured during backpropagation
            through the hook registered on the network output. No manual error
            passing is required by the user.
        """
        # Apply DFA function which handles the custom backward pass
        return dfa(inputs, self.fb_weights)

    # -----------------------------------------------------------------------------------
    def extra_repr(self) -> str:
        """Return extra representation for model inspection and debugging.

        Provides a string representation of the layer's key parameters for
        use in model summaries and debugging output.

        Returns:
            String containing output_dim and hidden_dim for layer identification.
        """
        return f"output_dim={self.output_dim}, hidden_dim={self.hidden_dim}"

    # -----------------------------------------------------------------------------------
    def reset_weights(self):
        """Reinitialize the random projection matrix.

        Resets the fixed random feedback weights to new random values. This can
        be useful for experimentation, ablation studies, or resetting the DFA
        behavior without recreating the entire layer.

        Note:
            The weights remain non-trainable after reset to maintain DFA properties.
            Uses Kaiming uniform initialization for balanced gradient magnitudes.
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
