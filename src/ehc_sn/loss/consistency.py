"""Representation-consistency loss primitives.

This module provides flat-tensor consistency penalties for variational and
energy-based losses. Inputs are model-agnostic tensors of shape ``(B, D)`` and
outputs are per-example loss values of shape ``(B,)``.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Normal


# =================================================================================================
def mse_consistency(  # --------------------------------------------------------------------------
    pred: Tensor, target: Tensor,
) -> Tensor:  # fmt: skip
    """Return half-squared-error consistency for flat latent codes.

    Args:
        pred: Predicted or prior code of shape ``(B, D)``.
        target: Target or posterior code of shape ``(B, D)``.

    Returns:
        Per-example squared-error consistency with shape ``(B,)``.
    """
    return 0.5 * (pred - target).pow(2).sum(dim=-1)


# =================================================================================================
def nll_consistency(  # --------------------------------------------------------------------------
    pred: Tensor, mean: Tensor, std: Tensor, *, min_std: float = 1e-6,
) -> Tensor:  # fmt: skip
    """Return Gaussian negative log-likelihood consistency for flat latent codes.

    Args:
        pred: Target code evaluated under the Gaussian prior, shape ``(B, D)``.
        mean: Prior mean of shape ``(B, D)``.
        std: Prior standard deviation of shape ``(B, D)``.
        min_std: Minimum standard deviation for numerical stability.

    Returns:
        Per-example negative log-likelihood with shape ``(B,)``.
    """
    safe_std = torch.clamp(std, min=min_std)
    prior = Normal(loc=mean, scale=safe_std)
    return -prior.log_prob(pred).sum(dim=-1)


# =================================================================================================
__all__ = ["mse_consistency", "nll_consistency"]
