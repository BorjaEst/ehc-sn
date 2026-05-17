"""
Cross-entropy loss functions for EHC-SN.
This module provides cross-entropy loss functions for variational and rollout
heads. Inputs are model-agnostic tensors of shape ``(B, D)`` and outputs are
per-example loss values of shape ``(B,)``.
"""

from typing import Literal, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor

from ehc_sn.activations.stablemax import log_stablemax

LossType: TypeAlias = Literal[
    "stablemax_cross_entropy", "softmax_cross_entropy"
]


# =============================================================================
def stablemax_cross_entropy(  # -----------------------------------------------
    logits,
    labels,
    ignore_index: int = -100,
) -> Tensor:
    """ """
    valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
    prediction = torch.gather(
        logprobs,
        index=transformed_labels.to(torch.long).unsqueeze(-1),
        dim=-1,
    ).squeeze(-1)
    return -torch.where(valid_mask, prediction, 0)


# =============================================================================
def softmax_cross_entropy(  # -------------------------------------------------
    logits,
    labels,
    ignore_index: int = -100,
) -> Tensor:
    """ """
    return F.cross_entropy(
        logits.to(torch.float32).view(-1, logits.shape[-1]),
        labels.to(torch.long).view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(labels.shape)


# =============================================================================
__all__ = ["LossType", "stablemax_cross_entropy", "softmax_cross_entropy"]
