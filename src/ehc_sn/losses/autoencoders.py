from typing import Tuple

import torch
from torch import Tensor, nn


class SparsityLoss(nn.Module):
    def __init__(self, sparsity_target: float = 0.05):
        super().__init__()
        self.sparsity_target = sparsity_target

    def forward(self, outputs: Tuple[Tensor, Tensor], targets: Tensor) -> Tensor:
        _, encodings = outputs
        # Calculate average activation across the batch
        avg_activation = torch.mean(encodings, dim=0)
        # Use L1 penalty for sparsity (simpler and more stable than KL divergence)
        sparsity_penalty = torch.mean(torch.abs(avg_activation - self.sparsity_target))
        return sparsity_penalty


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, outputs, targets):
        reconstruction, _ = outputs
        return self.mse(reconstruction, targets)
