from typing import Any

import torch
from torch import Tensor, nn


# =============================================================================
def bounded_positive_scale(
    x: Tensor,
    *,
    min_sigma: float = 1e-4,
    max_sigma: float = 1e2,
) -> Tensor:
    sigma = nn.functional.softplus(x) + min_sigma
    return torch.clamp(sigma, min=min_sigma, max=max_sigma)


# =============================================================================
def positive_sigma(
    x: Tensor,
    *,
    eps: float = 1e-4,
) -> Tensor:
    return nn.functional.softplus(x) + eps


# =============================================================================
__all__ = [
    "bounded_positive_scale",
    "positive_sigma",
]
