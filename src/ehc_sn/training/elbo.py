"""Model-agnostic ELBO utilities.

This module provides flat and multi-block Gaussian KL helpers plus simple beta
weight scheduling for ELBO-style objectives. Inputs are tensors of shape
``(B, D)`` or sequences of such tensors, and outputs are per-example values of
shape ``(B,)`` unless otherwise documented.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor

LatentCode = Tensor | Sequence[Tensor]
BetaSchedule = Literal["constant", "linear_warmup"]


# =================================================================================================
def _iter_blocks(code: LatentCode) -> tuple[Tensor, ...]:
    """Return a tuple view over one or more latent blocks."""
    if isinstance(code, Tensor):
        return (code,)
    return tuple(code)


# =================================================================================================
def gaussian_kl_divergence(  # -------------------------------------------------------------------
    posterior_mean: Tensor, posterior_std: Tensor, prior_mean: Tensor, prior_std: Tensor, *,
    min_std: float = 1e-6,
) -> Tensor:  # fmt: skip
    """Return per-example KL divergence between diagonal Gaussian.

    Computes ``KL(q || p)`` where both ``q`` and ``p`` are diagonal Gaussian
    parameterized by means and standard deviations.
    """
    safe_posterior_std = torch.clamp(posterior_std, min=min_std)
    safe_prior_std = torch.clamp(prior_std, min=min_std)

    posterior_var = safe_posterior_std.pow(2)
    prior_var = safe_prior_std.pow(2)
    mean_delta_sq = (posterior_mean - prior_mean).pow(2)
    log_std_ratio = torch.log(safe_prior_std) - torch.log(safe_posterior_std)

    kl = log_std_ratio + (posterior_var + mean_delta_sq) / (2.0 * prior_var) - 0.5
    return kl.sum(dim=-1)


# =================================================================================================
def sum_gaussian_kl_divergence(  # ---------------------------------------------------------------
    posterior_mean: LatentCode, posterior_std: LatentCode, prior_mean: LatentCode,
    prior_std: LatentCode, *,
    min_std: float = 1e-6,
) -> Tensor:  # fmt: skip
    """Return per-example KL divergence summed over one or more latent blocks."""
    posterior_mean_blocks = _iter_blocks(posterior_mean)
    posterior_std_blocks = _iter_blocks(posterior_std)
    prior_mean_blocks = _iter_blocks(prior_mean)
    prior_std_blocks = _iter_blocks(prior_std)

    n_blocks = len(posterior_mean_blocks)
    if not (n_blocks == len(posterior_std_blocks) == len(prior_mean_blocks) == len(prior_std_blocks)):
        raise ValueError("Posterior and prior moments must expose the same number of latent blocks.")

    total: Tensor | None = None
    for post_mean, post_std, prev_mean, prev_std in zip(
        posterior_mean_blocks,
        posterior_std_blocks,
        prior_mean_blocks,
        prior_std_blocks,
        strict=True,
    ):
        term = gaussian_kl_divergence(post_mean, post_std, prev_mean, prev_std, min_std=min_std)
        total = term if total is None else total + term

    if total is None:
        raise ValueError("Latent moment groups must not be empty.")
    return total


# =================================================================================================
def compute_elbo_loss(  # ------------------------------------------------------------------------
    loss_obs_nll_sum: Tensor, loss_kl_sum: Tensor,
    loss_reg_sum: Tensor | None = None,
) -> Tensor:  # fmt: skip
    """Return the scalar negative ELBO objective for a step."""
    if loss_reg_sum is None:
        loss_reg_sum = loss_obs_nll_sum.new_zeros(())
    return loss_obs_nll_sum + loss_kl_sum + loss_reg_sum


# =================================================================================================
__all__ = [ 
    "BetaSchedule", "LatentCode", "compute_elbo_loss", "gaussian_kl_divergence",
    "sum_gaussian_kl_divergence",
]  # fmt: skip
