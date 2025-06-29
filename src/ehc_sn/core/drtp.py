import torch
from torch import Tensor


# -------------------------------------------------------------------------------------------
class DRTPFunction(torch.autograd.Function):
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
if __name__ == "__main__":
    # Example usage of DRTP in a simple neural network
    # This is a minimal example to demonstrate the DRTP functionality
    from torch import nn

    # Define layers and random projection matrices
    layer_1 = torch.nn.Linear(5, 4)  # 5 -> 4
    B1 = torch.randn(2, 4) * 0.1  # Random projection matrix (target_dim=2, hidden_dim=4)

    layer_2 = torch.nn.Linear(4, 3)  # 4 -> 3
    B2 = torch.randn(2, 3) * 0.1  # Random projection matrix (target_dim=2, hidden_dim=3)

    layer_3 = torch.nn.Linear(3, 2)  # 3 -> 2
    # No DRTP on the final output layer so we use standard backpropagation here

    x = torch.ones(1, 5, requires_grad=True)  # Input tensor with gradient tracking
    y_ = torch.ones(1, 2)  # Target tensor (h_*)

    # Full forward pass
    y1 = layer_1(x)  # Forward pass through layer 1: (1, 5) -> (1, 4)
    y1 = drtp_layer(y1, B1, y_)  # DRTP modulation on layer 1 output
    y2 = layer_2(y1)  # Forward pass through layer 2: (1, 4) -> (1, 3)
    y2 = drtp_layer(y2, B2, y_)  # DRTP modulation on layer 2 output
    y3 = layer_3(y2)  # Forward pass through layer 3: (1, 3) -> (1, 2)

    # To trigger backward pass, we need a scalar loss
    loss = nn.MSELoss()(y3, y_)  # Mean Squared Error loss

    # Backward pass
    print("Before backward pass:")
    print("  Layer 1 gradients:", layer_1.weight.grad)
    print("  Layer 2 gradients:", layer_2.weight.grad)
    print("  Layer 3 gradients:", layer_3.weight.grad)
    print("  Input gradients:", x.grad)

    loss.backward()  # This will use our custom DRTP backward pass

    print("\nAfter DRTP backward pass:")
    print("  Layer 1 weight gradients shape:", layer_1.weight.grad.shape if layer_1.weight.grad is not None else None)
    print("  Layer 1 weight gradients:", layer_1.weight.grad)
    print("  Layer 2 weight gradients shape:", layer_2.weight.grad.shape if layer_2.weight.grad is not None else None)
    print("  Layer 2 weight gradients:", layer_2.weight.grad)
    print("  Layer 3 weight gradients shape:", layer_3.weight.grad.shape if layer_3.weight.grad is not None else None)
    print("  Layer 3 weight gradients:", layer_3.weight.grad)
    print("  Input gradients:", x.grad)

    # Verify the DRTP computation manually
    print("\nManual DRTP verification:")
    print("  Network architecture: 5 -> 4 -> 3 -> 2")
    print("  B1 shape:", B1.shape)  # (2, 4) - projects target (2,) to layer1 output (4,)
    print("  B2 shape:", B2.shape)  # (2, 3) - projects target (2,) to layer2 output (3,)
    print("  y_ (target) shape:", y_.shape)  # (1, 2)
    print("  y1 shape:", y1.shape)  # (1, 4)
    print("  y2 shape:", y2.shape)  # (1, 3)
    print("  y3 (final output) shape:", y3.shape)  # (1, 2)

    # The gradients should be:
    # For layer 1: y_ @ B1 = (1, 2) @ (2, 4) = (1, 4)
    # For layer 2: y_ @ B2 = (1, 2) @ (2, 3) = (1, 3)
    manual_grad_1 = torch.matmul(y_, B1)
    manual_grad_2 = torch.matmul(y_, B2)
    print("  Manual DRTP gradient for layer 1:", manual_grad_1)
    print("  Manual DRTP gradient for layer 2:", manual_grad_2)
    print("  Final output y3:", y3)
