from typing import Optional, Tuple

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn


class SparseLossParams(BaseModel):

    model_config = {"extra": "forbid"}  # Forbid extra fields not defined in the model

    sparsity_target: float = Field(default=0.05, description="Target activation rate for sparsity")
    sparsity_weight: float = Field(default=0.1, description="Weight for sparsity loss term (beta)")


class SparsityLoss(nn.Module):
    def __init__(self, params: Optional[SparseLossParams] = None):
        super().__init__()
        self.params = SparseLossParams() if params is None else params

    def forward(self, outputs: Tuple[Tensor, Tensor], targets: Tensor) -> Tensor:
        _, encodings = outputs
        kl_div = self.kl_divergence(encodings.mean(dim=0), self.params.sparsity_target)
        return self.params.sparsity_weight * torch.sum(kl_div)

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
