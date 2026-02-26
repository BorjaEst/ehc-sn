"""Halting head modules for ACT controllers."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LinearHaltingHead(nn.Module):
    """Linear halting head that maps features to (halt, continue) logits."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self._head = nn.Linear(hidden_size, 2, bias=True)

        with torch.no_grad():
            self._head.weight.zero_()
            self._head.bias.fill_(-5)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        logits = self._head(features.to(torch.float32))
        return logits[..., 0], logits[..., 1]


__all__ = ["LinearHaltingHead"]
