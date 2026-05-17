"""Representation-consistency loss primitives.

This module provides flat-tensor consistency penalties and latent-code helpers
for rollout heads. Inputs are model-agnostic tensors of shape ``(B, D)`` and
outputs are per-example loss values of shape ``(B,)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch
from torch import Tensor
from torch.distributions import Normal

LatentCode: TypeAlias = Tensor | Sequence[Tensor]


# =============================================================================
@dataclass(frozen=True)
class LatentRelation:
    """Named binary relation between two semantic latent-code sides.

    Each side may itself be a single tensor or a multi-block latent code. The
    relation object stays agnostic to whether the two sides represent a
    posterior/prior pair, two transition states, or another semantic
    comparison.
    """

    lhs: LatentCode
    rhs: LatentCode


# =============================================================================
def iter_latent_codes(  # -----------------------------------------------------
    code: LatentCode,
) -> tuple[Tensor, ...]:
    """Return a tuple view over one or more latent-code blocks."""
    if isinstance(code, Tensor):
        return (code,)
    return tuple(code)


# =============================================================================
def sum_latent_terms(  # ------------------------------------------------------
    loss_fn: Any,
    pred: LatentCode,
    target: LatentCode,
) -> Tensor:
    """Apply a flat-tensor loss over one or more latent-code blocks.

    Args:
        loss_fn: Callable returning a per-example tensor of shape ``(B,)``.
        pred: Predicted code tensor or sequence of tensors.
        target: Target code tensor or sequence of tensors.

    Returns:
        Per-example loss values of shape ``(B,)``.
    """
    pred_codes = iter_latent_codes(pred)
    target_codes = iter_latent_codes(target)
    if len(pred_codes) != len(target_codes):
        raise ValueError(
            "Latent code groups must have the same number of blocks."
        )

    total: Tensor | None = None
    for pred_code, target_code in zip(pred_codes, target_codes, strict=True):
        term = loss_fn(pred_code, target_code)
        total = term if total is None else total + term

    if total is None:
        raise ValueError("Latent code groups must not be empty.")
    return total


# =============================================================================
def mean_latent_norm(  # ------------------------------------------------------
    code: LatentCode,
) -> Tensor:
    """Return the mean block-wise activation norm for diagnostics."""
    block_means = [
        block.detach().norm(dim=-1).mean() for block in iter_latent_codes(code)
    ]
    return torch.stack(block_means).mean()


# =============================================================================
def mse_consistency(  # -------------------------------------------------------
    pred: Tensor,
    target: Tensor,
) -> Tensor:
    """Return half-squared-error consistency for flat latent codes.

    Args:
        pred: Predicted or prior code of shape ``(B, D)``.
        target: Target or posterior code of shape ``(B, D)``.

    Returns:
        Per-example squared-error consistency with shape ``(B,)``.
    """
    return 0.5 * (pred - target).pow(2).sum(dim=-1)


# =============================================================================
def nll_consistency(  # -------------------------------------------------------
    pred: Tensor,
    mean: Tensor,
    std: Tensor,
    *,
    min_std: float = 1e-6,
) -> Tensor:
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


# =============================================================================
__all__ = [
    "LatentCode",
    "LatentRelation",
    "iter_latent_codes",
    "mean_latent_norm",
    "mse_consistency",
    "nll_consistency",
    "sum_latent_terms",
]
