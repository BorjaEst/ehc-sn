from typing import Tuple

import torch
from torch import Tensor, nn


class SparsityLoss(nn.Module):
    def __init__(self, sparsity_target: float = 0.05):
        super().__init__()
        self.sparsity_target = sparsity_target

    def forward(self, outputs: Tuple[Tensor, Tensor], targets: Tensor) -> Tensor:
        _, encodings = outputs
        kl_div = self.kl_divergence(encodings.mean(dim=0), self.sparsity_target)
        return torch.sum(kl_div)

    @staticmethod
    def kl_divergence(input: Tensor, target: float, eps=1e-10) -> Tensor:
        kl = input * torch.log((input + eps) / (target + eps))
        kl += (1 - input) * torch.log((1 - input + eps) / (1 - target + eps))
        return torch.sum(kl)


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, outputs, targets):
        reconstruction, _ = outputs
        return self.mse(reconstruction, targets)
