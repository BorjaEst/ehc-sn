import math
from typing import Any, List, Tuple

import torch
from torch import Size, Tensor, nn


# -------------------------------------------------------------------------------------------
class Linear(nn.Linear):
    def __init__(self, *args, target_features: int, device=None, dtype=None, **kwargs) -> None:
        super().__init__(*args, device=device, dtype=dtype, **kwargs)
        self.target_features = target_features
        fb_weights = torch.zeros(target_features, self.out_features, device=device, dtype=dtype)
        self.register_buffer("fb_weight", fb_weights)
        self.reset_feedback()  # Initialize weights properly

    def forward(self, input: Tensor) -> Tensor:
        # Detach to enforce locality (no upstream gradient)
        return super().forward(input.detach())

    def feedback(self, target: Tensor, context: Tensor) -> None:
        delta = target.detach() @ self.fb_weight  # (batch_size, out_features)
        torch.autograd.backward(context, delta)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"target_features={self.target_features}"
        )

    def reset_feedback(self) -> None:
        limit = 1.0 / math.sqrt(self.target_features)
        torch.nn.init.uniform_(self.fb_weight, -limit, limit)


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage of DRTP in a simple neural network
    print("=== DRTP Layer Example ===")
    torch.manual_seed(0)

    # Network parameters
    input_dim, hidden1_dim, hidden2_dim, output_dim = 5, 4, 3, 2
    batch_size = 2

    # Create network layers
    layer1 = Linear(input_dim, hidden1_dim, target_features=output_dim)
    layer2 = Linear(hidden1_dim, hidden2_dim, target_features=output_dim)
    layer3 = nn.Linear(hidden2_dim, output_dim)  # Output layer (no DRTP)

    # Optimizer
    parameters = list(layer1.parameters()) + list(layer2.parameters()) + list(layer3.parameters())
    optimizer = torch.optim.SGD(parameters, lr=1e-2)
    optimizer.zero_grad()

    # Create input and target
    x = torch.randn(batch_size, input_dim, requires_grad=True)
    target = torch.randn(batch_size, output_dim)

    print(f"Input shape: {x.shape}")
    print(f"Target shape: {target.shape}")
    print(f"DRTP1 projection matrix shape: {layer1.fb_weight.shape}")
    print(f"DRTP2 projection matrix shape: {layer2.fb_weight.shape}")

    # Forward pass
    h1 = torch.relu(layer1(x))  # (batch_size, hidden1_dim)
    h2 = torch.relu(layer2(h1))  # (batch_size, hidden2_dim)
    output = torch.sigmoid(layer3(h2.detach()))  # detach to isolate output path

    # Retain grads to verify DFA deltas
    h1.retain_grad()
    h2.retain_grad()

    print(f"\nForward pass:")
    print(f"  Hidden 1 shape: {h1.shape}")
    print(f"  Hidden 2 shape: {h2.shape}")
    print(f"  Output shape: {output.shape}")

    # Compute loss
    loss = nn.MSELoss()(output, target)
    print(f"  Loss: {loss.item():.6f}")

    # Feedback pass
    layer1.feedback(target, context=h1)
    layer2.feedback(target, context=h2)
    loss.backward()  # Backprop through output layer

    # Feedback pass
    print(f"\After backward:")
    print(f"  Layer1 grad: {layer1.weight.grad}")
    print(f"  Layer2 grad: {layer2.weight.grad}")
    print(f"  Layer3 grad: {layer3.weight.grad}")

    # Verify DRTP projections
    print(f"\nDRTP projections:")
    expected_grad1 = torch.matmul(target, layer1.fb_weight)  # Should match h1 gradients
    expected_grad2 = torch.matmul(target, layer2.fb_weight)  # Should match h2 gradients
    print(f"  Expected DRTP1 gradient shape: {expected_grad1.shape}")
    print(f"  Expected DRTP2 gradient shape: {expected_grad2.shape}")

    # Optimize weights
    optimizer.step()

    print(f"\nDRTP Layer example completed successfully!")
