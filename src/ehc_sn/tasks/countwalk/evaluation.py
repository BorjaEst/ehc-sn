"""Countwalk task-owned evaluation helpers.

Provides masked digit accuracy, exact digit-sequence accuracy, exact numeric
value accuracy, and mean absolute error.  V1 evaluation operates on fixed-slot
masked digit outputs — no autoregressive generation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .contracts import COUNTWALK_IGNORE_DIGIT, DIGIT_WIDTH, N_BUCKETS


# =============================================================================
@dataclass(frozen=True)
class CountwalkScoreReport:
    """Aggregate evaluation report for one Countwalk batch or epoch."""

    digit_accuracy: float
    """Fraction of active digit slots predicted correctly."""

    sequence_accuracy: float
    """Fraction of episodes where all active digit slots are correct."""

    value_accuracy: float
    """Fraction of episodes where the predicted numeric value is exactly correct."""

    mean_absolute_error: float
    """Mean absolute error between predicted and ground-truth numeric values."""

    n_episodes: int
    """Number of evaluated episodes."""


# =============================================================================
def _logits_to_digits(digit_logits: Tensor) -> Tensor:
    """Convert ``(B, DIGIT_WIDTH, 10)`` logits to ``(B, DIGIT_WIDTH)`` predicted digits."""
    return digit_logits.argmax(dim=-1)


def _digits_to_value(digits: Tensor) -> Tensor:
    """Convert ``(B, DIGIT_WIDTH)`` digit tensors to ``(B,)`` integer values.

    Treats the digit tensor as right-aligned decimal digits (position 0 = most
    significant).  E.g. digits=[1, 2, 7] → value 127.
    """
    B, W = digits.shape
    powers = torch.tensor(
        [10 ** (W - 1 - d) for d in range(W)],
        dtype=digits.dtype,
        device=digits.device,
    )
    return (digits * powers).sum(dim=-1)


# =============================================================================
def digit_accuracy(
    digit_logits: Tensor,
    target_digits: Tensor,
    target_digit_mask: Tensor,
) -> Tensor:
    """Fraction of active digit slots predicted correctly.

    Args:
        digit_logits: Shape ``(B, DIGIT_WIDTH, 10)`` float.
        target_digits: Shape ``(B, DIGIT_WIDTH)`` int — raw digit labels 0..9.
        target_digit_mask: Shape ``(B, DIGIT_WIDTH)`` bool — active slots.

    Returns:
        Scalar float accuracy over active slots.
    """
    pred = _logits_to_digits(digit_logits)
    correct = (pred == target_digits) & target_digit_mask
    n_active = target_digit_mask.sum().clamp(min=1)
    return correct.sum().float() / n_active


def sequence_accuracy(
    digit_logits: Tensor,
    target_digits: Tensor,
    target_digit_mask: Tensor,
) -> Tensor:
    """Fraction of episodes where all active digit slots are correct.

    Args:
        digit_logits: Shape ``(B, DIGIT_WIDTH, 10)`` float.
        target_digits: Shape ``(B, DIGIT_WIDTH)`` int — raw digit labels 0..9.
        target_digit_mask: Shape ``(B, DIGIT_WIDTH)`` bool — active slots.

    Returns:
        Scalar float accuracy over episodes.
    """
    pred = _logits_to_digits(digit_logits)
    correct = (pred == target_digits) | (~target_digit_mask)
    all_correct = correct.all(dim=-1)
    return all_correct.float().mean()


def value_accuracy(
    digit_logits: Tensor,
    target_value: Tensor,
) -> Tensor:
    """Fraction of episodes where the predicted numeric value is exactly correct.

    Args:
        digit_logits: Shape ``(B, DIGIT_WIDTH, 10)`` float.
        target_value: Shape ``(B,)`` int — ground-truth numeric value.

    Returns:
        Scalar float accuracy over episodes.
    """
    pred_digits = _logits_to_digits(digit_logits)
    pred_value = _digits_to_value(pred_digits)
    return (pred_value == target_value).float().mean()


def mean_absolute_error(
    digit_logits: Tensor,
    target_value: Tensor,
) -> Tensor:
    """Mean absolute error between predicted and ground-truth numeric values.

    Args:
        digit_logits: Shape ``(B, DIGIT_WIDTH, 10)`` float.
        target_value: Shape ``(B,)`` int — ground-truth numeric value.

    Returns:
        Scalar float MAE.
    """
    pred_digits = _logits_to_digits(digit_logits)
    pred_value = _digits_to_value(pred_digits).long()
    return (pred_value - target_value.long()).abs().float().mean()


def score_countwalk_batch(
    digit_logits: Tensor,
    target_digits: Tensor,
    target_digit_mask: Tensor,
    target_value: Tensor,
) -> CountwalkScoreReport:
    """Compute a full :class:`CountwalkScoreReport` for one batch.

    Args:
        digit_logits: Shape ``(B, DIGIT_WIDTH, 10)`` float.
        target_digits: Shape ``(B, DIGIT_WIDTH)`` int — raw digit labels 0..9.
        target_digit_mask: Shape ``(B, DIGIT_WIDTH)`` bool — active slots.
        target_value: Shape ``(B,)`` int — ground-truth numeric value.

    Returns:
        :class:`CountwalkScoreReport` with all scalar metrics.
    """
    return CountwalkScoreReport(
        digit_accuracy=float(digit_accuracy(digit_logits, target_digits, target_digit_mask)),
        sequence_accuracy=float(sequence_accuracy(digit_logits, target_digits, target_digit_mask)),
        value_accuracy=float(value_accuracy(digit_logits, target_value)),
        mean_absolute_error=float(mean_absolute_error(digit_logits, target_value)),
        n_episodes=int(digit_logits.shape[0]),
    )


# =============================================================================
def score_countwalk_stratified(
    digit_logits: Tensor,
    target_digits: Tensor,
    target_digit_mask: Tensor,
    target_value: Tensor,
    *,
    cue_surface_ids: Tensor,
    anchor_regime_ids: Tensor,
    eval_bucket_ids: Tensor,
) -> dict[str, dict[str, CountwalkScoreReport]]:
    """Compute per-stratum :class:`CountwalkScoreReport` for one batch.

    Breaks down the same metrics as :func:`score_countwalk_batch` by:
    - cue surface (CUE_DIGIT=0, CUE_SET=1)
    - anchor regime (ANCHOR_ONLY=0, SPARSE_REANCHOR=1)
    - evaluation bucket (BUCKET_ID=0 .. BUCKET_STRETCH_OOD=4)

    Returns a nested mapping::

        {
            "by_cue_surface": {str(cue_id): CountwalkScoreReport, ...},
            "by_anchor_regime": {str(regime_id): CountwalkScoreReport, ...},
            "by_bucket": {str(bucket_id): CountwalkScoreReport, ...},
        }

    Strata with zero episodes are omitted.

    Args:
        digit_logits: Shape ``(B, DIGIT_WIDTH, 10)`` float.
        target_digits: Shape ``(B, DIGIT_WIDTH)`` int.
        target_digit_mask: Shape ``(B, DIGIT_WIDTH)`` bool.
        target_value: Shape ``(B,)`` int.
        cue_surface_ids: Shape ``(B,)`` int — per-episode cue surface id.
        anchor_regime_ids: Shape ``(B,)`` int — per-episode anchor regime id.
        eval_bucket_ids: Shape ``(B,)`` int — per-episode bucket id.

    Returns:
        Nested mapping of stratum label → :class:`CountwalkScoreReport`.
    """

    def _score_mask(mask: Tensor) -> CountwalkScoreReport | None:
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return None
        return score_countwalk_batch(
            digit_logits[idx],
            target_digits[idx],
            target_digit_mask[idx],
            target_value[idx],
        )

    from ehc_sn.tasks.countwalk.contracts import N_ANCHOR_REGIMES, N_CUE_SURFACES

    by_cue: dict[str, CountwalkScoreReport] = {}
    for cid in range(N_CUE_SURFACES):
        r = _score_mask(cue_surface_ids == cid)
        if r is not None:
            by_cue[str(cid)] = r

    by_regime: dict[str, CountwalkScoreReport] = {}
    for rid in range(N_ANCHOR_REGIMES):
        r = _score_mask(anchor_regime_ids == rid)
        if r is not None:
            by_regime[str(rid)] = r

    by_bucket: dict[str, CountwalkScoreReport] = {}
    for bid in range(N_BUCKETS):
        r = _score_mask(eval_bucket_ids == bid)
        if r is not None:
            by_bucket[str(bid)] = r

    return {
        "by_cue_surface": by_cue,
        "by_anchor_regime": by_regime,
        "by_bucket": by_bucket,
    }


# =============================================================================
__all__ = [
    "CountwalkScoreReport",
    "digit_accuracy",
    "sequence_accuracy",
    "value_accuracy",
    "mean_absolute_error",
    "score_countwalk_batch",
    "score_countwalk_stratified",
]
