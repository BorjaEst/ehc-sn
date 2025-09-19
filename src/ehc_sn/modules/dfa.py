import math
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch import Tensor, nn


# -------------------------------------------------------------------------------------------
class Linear(nn.Linear):
    def __init__(self, *args, error_features: int, device=None, dtype=None, **kwargs) -> None:
        super().__init__(*args, device=device, dtype=dtype, **kwargs)
        self.error_features = error_features
        fb_weights = torch.zeros(error_features, self.out_features, device=device, dtype=dtype)
        self.register_buffer("fb_weight", fb_weights)
        self.reset_feedback()  # Initialize weights properly

    def forward(self, input: Tensor) -> Tensor:
        # Detach to enforce locality (no upstream gradient)
        return super().forward(input.detach())

    def feedback(self, error: Tensor, context: Tensor) -> None:
        delta = error.detach() @ self.fb_weight  # (batch_size, out_features)
        torch.autograd.backward(context, delta)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"error_features={self.error_features}"
        )

    def reset_feedback(self) -> None:
        limit = 1.0 / math.sqrt(self.error_features)
        torch.nn.init.uniform_(self.fb_weight, -limit, limit)


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage of DFA in a simple neural network
    print("=== DFA Layer Example ===")
    torch.manual_seed(0)

    # Network parameters
    input_dim, hidden1_dim, hidden2_dim, output_dim = 5, 4, 3, 2
    batch_size = 2

    # Create network layers with DFA
    layer1 = Linear(input_dim, hidden1_dim, error_features=output_dim)
    layer2 = Linear(hidden1_dim, hidden2_dim, error_features=output_dim)
    layer3 = nn.Linear(hidden2_dim, output_dim)  # Output layer (no DFA)

    # Optimizer
    parameters = list(layer1.parameters()) + list(layer2.parameters()) + list(layer3.parameters())
    optimizer = torch.optim.SGD(parameters, lr=1e-2)
    optimizer.zero_grad()

    # Create input and target
    x = torch.randn(batch_size, input_dim, requires_grad=True)
    target = torch.randn(batch_size, output_dim)

    print(f"Input shape: {x.shape}")
    print(f"Target shape: {target.shape}")
    print(f"DFA1 projection matrix shape: {layer1.fb_weight.shape}")
    print(f"DFA2 projection matrix shape: {layer2.fb_weight.shape}")

    # Forward pass
    h1 = torch.relu(layer1(x))  # (batch_size, hidden1_dim)
    h2 = torch.relu(layer2(h1))  # (batch_size, hidden2_dim)
    output = torch.sigmoid(layer3(h2.detach()))  # detach to isolate output path

    # Retain grads to verify DFA deltas
    h1.retain_grad()
    h2.retain_grad()

    print("\nForward pass:")
    print(f"  Hidden 1 shape: {h1.shape}")
    print(f"  Hidden 2 shape: {h2.shape}")
    print(f"  Output shape: {output.shape}")

    # Compute loss
    loss = nn.MSELoss()(output, target)
    print(f"  Loss: {loss.item():.6f}")

    # Global error for DFA
    global_error = (output - target).detach()  # (batch_size, output_dim)

    # DFA feedback (hidden layers)
    layer1.feedback(global_error, context=h1)
    layer2.feedback(global_error, context=h2)

    # Standard BP only for output layer
    loss.backward()

    print("\nAfter backward:")
    print(f"  Layer1 grad shape: {tuple(layer1.weight.grad.shape)}")
    print(f"  Layer2 grad shape: {tuple(layer2.weight.grad.shape)}")
    print(f"  Layer3 grad shape: {tuple(layer3.weight.grad.shape)}")

    # Verify DFA projections equal activation grads
    expected_grad1 = global_error @ layer1.fb_weight
    expected_grad2 = global_error @ layer2.fb_weight
    print("\nDFA projection vs activation grad (norm diffs):")
    print(f"  h1 grad diff: {(h1.grad - expected_grad1).norm().item():.4e}")
    print(f"  h2 grad diff: {(h2.grad - expected_grad2).norm().item():.4e}")

    # Optimize hidden weights
    optimizer.step()

    print("\nDFA Layer example completed successfully!")
