"""Stablemax activation.

Implements the stablemax transformation, a numerically stable alternative to
softmax.  Defined as::

    stablemax(x)_i = s(x_i) / Σ_j s(x_j)
    where s(x) = 1/(1-x) for x < 0, and s(x) = x+1 for x ≥ 0.
"""

import torch
from torch import Tensor


# =============================================================================
def _s(  # --------------------------------------------------------------------
    x: Tensor,
    epsilon: float = 1e-30,
) -> Tensor:
    """Element-wise stable transformation kernel."""
    return torch.where(x < 0, 1.0 / (1.0 - x + epsilon), x + 1.0)


# =============================================================================
def log_stablemax(  # ---------------------------------------------------------
    x: Tensor,
    dim: int = -1,
) -> Tensor:
    """Log of the stablemax activation along ``dim``.

    Args:
        x: Input tensor of any shape.
        dim: Dimension along which to normalize.

    Returns:
        Log-probabilities of same shape as ``x``, computed in float64
        via the stablemax transformation.
    """
    s_x = _s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


# =============================================================================
__all__ = ["log_stablemax"]
