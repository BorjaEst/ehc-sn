"""Activation-regularization loss primitives.

This module provides flat-tensor penalties for variational and rollout heads.
Inputs are tensors of shape ``(B, D)`` and outputs are per-example values of
shape ``(B,)``.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from torch import Tensor

from ehc_sn.loss.consistency import LatentCode, iter_latent_codes

RegularizationNorm: TypeAlias = Literal["none", "l1", "l2"]


# =============================================================================
def l1_penalty(  # ------------------------------------------------------------
    code: Tensor,
) -> Tensor:
    """Return per-example L1 penalty for a flat code tensor."""
    return code.abs().sum(dim=-1)


# =============================================================================
def l2_penalty(  # ------------------------------------------------------------
    code: Tensor,
) -> Tensor:
    """Return per-example squared L2 penalty for a flat code tensor."""
    return code.pow(2).sum(dim=-1)


# =============================================================================
def sum_regularization_terms(  # ----------------------------------------------
    code: LatentCode,
    norm: RegularizationNorm,
) -> Tensor:
    """Return per-example regularization over one or more latent-code blocks."""
    code_blocks = iter_latent_codes(code)
    if not code_blocks:
        raise ValueError("Latent code groups must not be empty.")

    if norm == "none":
        return code_blocks[0].new_zeros((code_blocks[0].shape[0],))

    penalty_fn = l1_penalty if norm == "l1" else l2_penalty
    total: Tensor | None = None
    for code_block in code_blocks:
        term = penalty_fn(code_block)
        total = term if total is None else total + term

    if total is None:
        raise ValueError("Latent code groups must not be empty.")
    return total


# =============================================================================
__all__ = [
    "RegularizationNorm",
    "l1_penalty",
    "l2_penalty",
    "sum_regularization_terms",
]
