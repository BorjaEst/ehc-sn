"""Countwalk task-owned contracts, constants, and semantic types.

Countwalk is a replay-based task over a bounded integer line (NumberLine substrate).
The agent navigates the hidden integer state; cues provide sporadic positional
information; the task queries the agent to report the current value via digit outputs.

V1 scope: legal-only replay corpus, digit and exact-set cue surfaces, anchor-only
and sparse-reanchor anchor regimes, fixed-slot masked digit targets.  No reward
surface, no adapter, no Lightning binding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from torch import Tensor

# =============================================================================
# Action constants (must match ehc_sn.envs.numberline)
# =============================================================================

ACTION_STAY: Final[int] = 0
"""STAY action — remain at the current state."""

ACTION_PREV: Final[int] = 1
"""PREV action — move to state − 1 (illegal at state 0)."""

ACTION_NEXT: Final[int] = 2
"""NEXT action — move to state + 1 (illegal at state n_states − 1)."""

N_ACTIONS: Final[int] = 3
"""Total number of actions in the Countwalk action space."""

# =============================================================================
# Cue surface constants
# =============================================================================

CUE_DIGIT: Final[int] = 0
"""Digit cue surface: the cue tokens encode the current value as right-aligned digits."""

CUE_SET: Final[int] = 1
"""Exact-set cue surface: the cue tokens encode cardinality as ITEM tokens."""

N_CUE_SURFACES: Final[int] = 2
"""Total number of cue surfaces in V1."""

# =============================================================================
# Anchor regime constants
# =============================================================================

ANCHOR_ONLY: Final[int] = 0
"""Anchor-only regime: anchor visible only at step 0."""

SPARSE_REANCHOR: Final[int] = 1
"""Sparse-reanchor regime: anchor at step 0 plus one interior non-terminal step."""

N_ANCHOR_REGIMES: Final[int] = 2
"""Total number of anchor regimes in V1."""

# =============================================================================
# Token vocabulary
# =============================================================================

TOKEN_PAD: Final[int] = 0
"""Padding token (no information)."""

TOKEN_DIGIT_0: Final[int] = 1
"""Token for digit 0.  TOKEN_DIGIT_d = TOKEN_DIGIT_0 + d for d in 0..9."""

TOKEN_DIGIT_9: Final[int] = 10
"""Token for digit 9."""

TOKEN_ITEM: Final[int] = 11
"""ITEM token used in exact-set cue encoding."""

COUNTWALK_TOKEN_VOCAB_SIZE: Final[int] = 12
"""Total token vocabulary size (TOKEN_PAD=0 through TOKEN_ITEM=11)."""

# =============================================================================
# Digit slot configuration
# =============================================================================

DIGIT_WIDTH: Final[int] = 3
"""Fixed digit slot width for query targets and digit cue tokens.

Width 3 accommodates values 0..999, covering primary (0..n_states-1) and
stretch OOD (0..199) splits without interface changes.
"""

COUNTWALK_IGNORE_DIGIT: Final[int] = -100
"""Ignore label for masked digit supervision (matches PyTorch cross-entropy convention)."""

# =============================================================================
# Task family constant
# =============================================================================

TASK_FAMILY: Final[str] = "countwalk"
"""Canonical task family name."""

# =============================================================================
# Evaluation bucket constants
# =============================================================================

BUCKET_ID: Final[int] = 0
"""In-distribution bucket: ID value range and ID horizon."""

BUCKET_RANGE_OOD: Final[int] = 1
"""Range OOD bucket: out-of-ID value range, ID horizon."""

BUCKET_HORIZON_OOD: Final[int] = 2
"""Horizon OOD bucket: ID value range, out-of-ID horizon."""

BUCKET_JOINT_OOD: Final[int] = 3
"""Joint OOD bucket: out-of-ID value range AND out-of-ID horizon."""

BUCKET_STRETCH_OOD: Final[int] = 4
"""Stretch OOD bucket: far out-of-distribution values (100..199), any horizon."""

N_BUCKETS: Final[int] = 5
"""Total number of evaluation buckets."""

# Value-range boundaries
BUCKET_ID_VALUE_MAX: Final[int] = 63
"""Maximum value (inclusive) for the ID and horizon-OOD buckets."""

BUCKET_RANGE_OOD_VALUE_MIN: Final[int] = 64
"""Minimum value (inclusive) for range-OOD and joint-OOD buckets."""

BUCKET_RANGE_OOD_VALUE_MAX: Final[int] = 99
"""Maximum value (inclusive) for range-OOD and joint-OOD buckets."""

BUCKET_STRETCH_OOD_VALUE_MIN: Final[int] = 100
"""Minimum value (inclusive) for the stretch-OOD bucket."""

BUCKET_STRETCH_OOD_VALUE_MAX: Final[int] = 199
"""Maximum value (inclusive) for the stretch-OOD bucket."""

# Horizon boundaries
BUCKET_ID_HORIZON_MAX: Final[int] = 8
"""Maximum horizon (max_steps, inclusive) for ID and range-OOD buckets."""

BUCKET_HORIZON_OOD_MIN: Final[int] = 9
"""Minimum horizon (max_steps, inclusive) for horizon-OOD, joint-OOD, and stretch-OOD."""

BUCKET_HORIZON_OOD_MAX: Final[int] = 16
"""Maximum horizon (max_steps, inclusive) for any OOD bucket."""


# =============================================================================
# Typed task contracts
# =============================================================================


@dataclass(frozen=True)
class CountwalkTaskInput:
    """Task-owned Countwalk step inputs delivered to the model."""

    previous_action: Tensor
    """Per-slot previous action token.  Shape ``(B,)`` int64."""

    cue_visible: Tensor
    """Per-slot anchor visibility flag.  Shape ``(B,)`` bool."""

    cue_surface_id: Tensor
    """Per-slot cue surface identifier.  Shape ``(B,)`` int64."""

    cue_tokens: Tensor
    """Per-slot cue token sequence.  Shape ``(B, cue_token_width)`` int64."""

    cue_token_mask: Tensor
    """Per-slot cue token validity mask.  Shape ``(B, cue_token_width)`` bool."""

    step_count: Tensor
    """Per-slot step count.  Shape ``(B,)`` int64."""

    episode_start: Tensor
    """Per-slot episode-start flag.  Shape ``(B,)`` bool."""

    is_query: Tensor
    """Per-slot terminal query flag.  Shape ``(B,)`` bool."""


@dataclass(frozen=True)
class CountwalkTargets:
    """Supervision targets for Countwalk digit prediction."""

    target_digits: Tensor
    """Fixed-slot digit targets.  Shape ``(B, DIGIT_WIDTH)`` int64.

    Right-aligned digit class labels (0..9).  Leading inactive slots use
    ``COUNTWALK_IGNORE_DIGIT`` to match PyTorch cross-entropy masking.
    """

    target_digit_mask: Tensor
    """Active digit slot mask.  Shape ``(B, DIGIT_WIDTH)`` bool."""

    target_value: Tensor
    """Ground-truth numeric value.  Shape ``(B,)`` int64."""


@dataclass(frozen=True)
class CountwalkTaskOutput:
    """Task-owned Countwalk prediction payload.

    V1 only supports fixed-slot digit logits; no autoregressive decoding.
    """

    digit_logits: Tensor
    """Fixed-slot digit prediction logits.  Shape ``(B, DIGIT_WIDTH, 10)`` float32.

    Axis 2 is over digit classes 0..9 (not the full token vocabulary).
    """


# =============================================================================
__all__ = [
    # Action constants
    "ACTION_STAY",
    "ACTION_PREV",
    "ACTION_NEXT",
    "N_ACTIONS",
    # Cue surface constants
    "CUE_DIGIT",
    "CUE_SET",
    "N_CUE_SURFACES",
    # Anchor regime constants
    "ANCHOR_ONLY",
    "SPARSE_REANCHOR",
    "N_ANCHOR_REGIMES",
    # Token vocabulary
    "TOKEN_PAD",
    "TOKEN_DIGIT_0",
    "TOKEN_DIGIT_9",
    "TOKEN_ITEM",
    "COUNTWALK_TOKEN_VOCAB_SIZE",
    # Digit configuration
    "DIGIT_WIDTH",
    "COUNTWALK_IGNORE_DIGIT",
    # Family
    "TASK_FAMILY",
    # Evaluation bucket constants
    "BUCKET_ID",
    "BUCKET_RANGE_OOD",
    "BUCKET_HORIZON_OOD",
    "BUCKET_JOINT_OOD",
    "BUCKET_STRETCH_OOD",
    "N_BUCKETS",
    "BUCKET_ID_VALUE_MAX",
    "BUCKET_RANGE_OOD_VALUE_MIN",
    "BUCKET_RANGE_OOD_VALUE_MAX",
    "BUCKET_STRETCH_OOD_VALUE_MIN",
    "BUCKET_STRETCH_OOD_VALUE_MAX",
    "BUCKET_ID_HORIZON_MAX",
    "BUCKET_HORIZON_OOD_MIN",
    "BUCKET_HORIZON_OOD_MAX",
    # Typed contracts
    "CountwalkTaskInput",
    "CountwalkTargets",
    "CountwalkTaskOutput",
]
