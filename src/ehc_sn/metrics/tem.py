"""Task-agnostic TEM observation accuracy metrics.

This module provides a standalone function for computing TEM observation
accuracy from logits and labels using argmax equality.  It is task-agnostic
and preserves the exact metric key set and numerator/denominator semantics
of the original ``build_arena_tem_observation_metrics``.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from ehc_sn.metrics.keys import (
    TEM_ACC_OBS_PATH_ALL,
    TEM_ACC_OBS_PATH_REVISIT,
    TEM_ACC_OBS_POST_ALL,
    TEM_ACC_OBS_POST_REVISIT,
    TEM_ACC_OBS_RECALL_ALL,
    TEM_ACC_OBS_RECALL_REVISIT,
)
from ehc_sn.metrics.step_metrics import RatioStat


# =============================================================================
def compute_tem_observation_accuracy(
    *,
    logits_post: Tensor,
    logits_recall: Tensor,
    logits_path: Tensor,
    labels: Tensor,
    protocol_mask: Tensor,
) -> dict[str, RatioStat]:
    """Compute TEM observation accuracy metrics from logits and labels.

    Uses argmax equality to determine correctness.  Produces the same six
    metric keys as the original task-owned helper, preserving denominator
    and numerator semantics exactly.

    Args:
        logits_post: Posterior observation logits, shape ``(B, V)``.
        logits_recall: Recall observation logits, shape ``(B, V)``.
        logits_path: Path observation logits, shape ``(B, V)``.
        labels: Ground-truth observation ids, shape ``(B,)`` int64.
        protocol_mask: Revisit eligibility mask, shape ``(B,)`` bool.

    Returns:
        Dict mapping six ``TEM_ACC_OBS_*`` metric keys to ``RatioStat``.
    """
    batch_count = labels.new_tensor(float(labels.shape[0]), dtype=torch.float32)
    revisit_count = protocol_mask.sum().float().clamp_min(1.0)

    def _correct_all(logits: Tensor) -> Tensor:
        return logits.argmax(dim=-1).eq(labels).sum().float()

    def _correct_revisit(logits: Tensor) -> Tensor:
        correct = logits.argmax(dim=-1).eq(labels)
        return (correct & protocol_mask).sum().float()

    return {
        TEM_ACC_OBS_POST_REVISIT: RatioStat(
            numerator_sum=_correct_revisit(logits_post),
            denominator_sum=revisit_count,
        ),
        TEM_ACC_OBS_RECALL_REVISIT: RatioStat(
            numerator_sum=_correct_revisit(logits_recall),
            denominator_sum=revisit_count,
        ),
        TEM_ACC_OBS_PATH_REVISIT: RatioStat(
            numerator_sum=_correct_revisit(logits_path),
            denominator_sum=revisit_count,
        ),
        TEM_ACC_OBS_POST_ALL: RatioStat(
            numerator_sum=_correct_all(logits_post),
            denominator_sum=batch_count,
        ),
        TEM_ACC_OBS_RECALL_ALL: RatioStat(
            numerator_sum=_correct_all(logits_recall),
            denominator_sum=batch_count,
        ),
        TEM_ACC_OBS_PATH_ALL: RatioStat(
            numerator_sum=_correct_all(logits_path),
            denominator_sum=batch_count,
        ),
    }


# =============================================================================
__all__ = ["compute_tem_observation_accuracy"]
