import dataclasses
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor, nn


@dataclasses.dataclass
class SparseLoss:
    reconstruction: Tensor = torch.tensor(0.0)
    sparsity: Tensor = torch.tensor(0.0)

    @property
    def total(self) -> float:
        return self.reconstruction + self.sparsity


def sparse_loss(y: Tensor, y_: Tensor, activations: List[Tensor], beta: float, target: float) -> SparseLoss:
    """Calculate sparse loss for autoencoders"""
    return SparseLoss(
        reconstruction=reconstruction(y, y_),
        sparsity=sparsity(activations, beta, target),
    )


def reconstruction(y: Tensor, y_: Tensor) -> Tensor:
    """Calculate reconstruction loss using Mean Squared Error"""
    return nn.functional.mse_loss(y, y_)


def sparsity(activations: List[Tensor], beta: float, target: float) -> Tensor:
    """Calculate sparsity loss using KL divergence"""
    return beta * sum(kl_divergence(target, x.mean(dim=0)) for x in activations)


def kl_divergence(input: float, target: Tensor, eps=1e-10) -> Tensor:
    kl = input * torch.log((input + eps) / (target + eps))
    kl += (1 - input) * torch.log((1 - input + eps) / (1 - target + eps))
    return torch.sum(kl)
