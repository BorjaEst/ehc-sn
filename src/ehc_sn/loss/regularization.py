"""Activation-regularization loss primitives.

This module provides flat-tensor penalties for variational and rollout heads.
Inputs are tensors of shape ``(B, D)`` and outputs are per-example values of
shape ``(B,)``.
"""

from __future__ import annotations

from torch import Tensor


# =================================================================================================
def l1_penalty(  # --------------------------------------------------------------------------------
    code: Tensor,
) -> Tensor:  # fmt: skip
    """Return per-example L1 penalty for a flat code tensor."""
    return code.abs().sum(dim=-1)


# =================================================================================================
def l2_penalty(  # --------------------------------------------------------------------------------
    code: Tensor,
) -> Tensor:  # fmt: skip
    """Return per-example squared L2 penalty for a flat code tensor."""
    return code.pow(2).sum(dim=-1)


# =================================================================================================
__all__ = ["l1_penalty", "l2_penalty"]
