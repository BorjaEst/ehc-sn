from typing import Optional

import torch
from torch import Tensor

__all__ = ["rademacher_like"]


# -------------------------------------------------------------------------------------------
def rademacher_like(t: Tensor) -> Tensor:
    """Return a {-1, +1} Rademacher tensor with the same shape/device/dtype as t."""
    return torch.empty_like(t).bernoulli_(0.5).mul_(2).sub_(1)
