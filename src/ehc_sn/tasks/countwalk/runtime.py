"""Countwalk task-level runtime helpers.

Generic task coercion helpers and replay batch schema constants.  These are
referenced by data-generation scripts, replay capabilities, and any Lightning
runtimes; they must not depend on execution-binding internals.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor

from ehc_sn.types import Batch

from .builder import (
    CHANNEL_ANCHOR_REGIME_ID,
    CHANNEL_CUE_SURFACE_ID,
    CHANNEL_EVAL_BUCKET_ID,
    CHANNEL_QUERY_MASK,
    CHANNEL_TARGET_DIGIT_MASK,
    CHANNEL_TARGET_DIGITS,
    CHANNEL_TARGET_VALUE,
    CHANNEL_TRAJECTORY_ANCHOR_VISIBLE,
    CHANNEL_TRAJECTORY_CUE_MASK,
    CHANNEL_TRAJECTORY_CUE_TOKENS,
    CHANNEL_TRAJECTORY_EPISODE_START,
    CHANNEL_TRAJECTORY_LENGTH,
    CHANNEL_TRAJECTORY_PREVIOUS_ACTION,
    CHANNEL_TRAJECTORY_VALID_STEP,
    CHANNEL_WORLD_ID,
)
from .contracts import CountwalkTargets

# =============================================================================
# Replay batch schema constants
# =============================================================================

COUNTWALK_REPLAY_REQUIRED_KEYS: Final[tuple[str, ...]] = (
    CHANNEL_TRAJECTORY_PREVIOUS_ACTION,
    CHANNEL_TRAJECTORY_ANCHOR_VISIBLE,
    CHANNEL_CUE_SURFACE_ID,
    CHANNEL_TRAJECTORY_CUE_TOKENS,
    CHANNEL_TRAJECTORY_CUE_MASK,
    CHANNEL_TRAJECTORY_EPISODE_START,
    CHANNEL_TRAJECTORY_VALID_STEP,
    CHANNEL_TRAJECTORY_LENGTH,
    CHANNEL_QUERY_MASK,
    CHANNEL_TARGET_DIGITS,
    CHANNEL_TARGET_DIGIT_MASK,
    CHANNEL_TARGET_VALUE,
)

COUNTWALK_REPLAY_OPTIONAL_KEYS: Final[tuple[str, ...]] = (
    CHANNEL_WORLD_ID,
    CHANNEL_ANCHOR_REGIME_ID,
    CHANNEL_EVAL_BUCKET_ID,
)

COUNTWALK_STEP_KEYS: Final[tuple[str, ...]] = (
    "previous_action",
    "cue_visible",
    "cue_surface_id",
    "cue_tokens",
    "cue_token_mask",
    "step_count",
    "episode_start",
    "is_query",
)


# =============================================================================
def batch_size_from_countwalk_batch(batch: Batch) -> int:
    """Return the leading batch dimension from a Countwalk replay batch."""
    return int(batch[CHANNEL_TRAJECTORY_PREVIOUS_ACTION].shape[0])


# =============================================================================
def infer_countwalk_replay_batch_keys(batch: Batch) -> tuple[str, ...]:
    """Return the replay batch keys present in *batch*, validating required keys."""
    missing = [k for k in COUNTWALK_REPLAY_REQUIRED_KEYS if k not in batch]
    if missing:
        raise KeyError("Countwalk replay batch missing required keys: " + ", ".join(missing))
    return COUNTWALK_REPLAY_REQUIRED_KEYS + tuple(k for k in COUNTWALK_REPLAY_OPTIONAL_KEYS if k in batch)


# =============================================================================
def batch_extract_countwalk_targets(batch: Batch, cursor: Tensor) -> CountwalkTargets:
    """Extract :class:`CountwalkTargets` from a replay batch at the given cursor.

    This is a batch-and-cursor-aware helper intended for objectives that need
    to score predictions against ground-truth targets at the query step.

    The targets (target_digits, target_digit_mask, target_value) are per-episode
    fields stored in the batch; the cursor is used to index the query step where
    the supervision applies.

    Args:
        batch: Countwalk replay batch containing target channels.
        cursor: Per-slot cursor positions. Shape ``(B,)`` int64.
            Used to validate that the cursor is at a query step.

    Returns:
        :class:`CountwalkTargets` with typed Tensor fields.

    Note:
        Target fields are per-episode scalars, not per-step arrays.
        Caller is responsible for ensuring only query steps are scored.
    """
    return CountwalkTargets(
        target_digits=batch[CHANNEL_TARGET_DIGITS].long(),
        target_digit_mask=batch[CHANNEL_TARGET_DIGIT_MASK].bool(),
        target_value=batch[CHANNEL_TARGET_VALUE].long(),
    )


# =============================================================================
__all__ = [
    "COUNTWALK_REPLAY_REQUIRED_KEYS",
    "COUNTWALK_REPLAY_OPTIONAL_KEYS",
    "COUNTWALK_STEP_KEYS",
    "batch_size_from_countwalk_batch",
    "infer_countwalk_replay_batch_keys",
    "batch_extract_countwalk_targets",
]
