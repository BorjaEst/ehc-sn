"""Arena task reward configuration and opt-in reward wrapper.

Arena reward is opt-in and disabled by default (REQ-004).  Task reward
configs express canonical semantic score-to-reward mappings only; any extra
shaping beyond these canonical mappings belongs to the adapter layer.
"""

from __future__ import annotations

from enum import Enum
from typing import Final

import torch
from pydantic import BaseModel
from torch import Tensor

from .evaluation import ArenaStructuralScore


# =============================================================================
class ArenaRewardMode(str, Enum):
    """Supported semantic score-to-reward mappings for arena replay.

    ``DISABLED`` is the canonical default.  Only add a mode here when it
    represents a distinct semantic claim about what the score means as a
    training signal.  Generic shaping parameters belong in the adapter layer.
    """

    DISABLED = "disabled"
    """No reward is computed.  Benchmark evaluation uses the score directly."""
    SCORE_DELTA_DENSE = "score_delta_dense"
    """Dense reward: accuracy_all at step t minus accuracy_all at step t-1."""
    TERMINAL_ONLY = "terminal_only"
    """Sparse reward: accuracy_all on the terminal step, zero elsewhere."""


# =============================================================================
class ArenaRewardConfig(BaseModel, extra="forbid"):
    """Opt-in arena reward configuration.

    ``mode`` defaults to :attr:`ArenaRewardMode.DISABLED` so that benchmark
    evaluation never accidentally introduces a reward signal.  Callers that
    want reward-augmented arena training must explicitly opt in.
    """

    mode: ArenaRewardMode = ArenaRewardMode.DISABLED


# =============================================================================
def compute_arena_reward(
    score: ArenaStructuralScore,
    prev_score: ArenaStructuralScore | None,
    *,
    config: ArenaRewardConfig,
    is_terminal: Tensor,
) -> Tensor | None:
    """Compute the arena reward tensor, or return ``None`` when disabled.

    Args:
        score: Structural score for the current step.
        prev_score: Structural score for the previous step.  Required when
            ``config.mode`` is :attr:`ArenaRewardMode.SCORE_DELTA_DENSE`;
            ignored otherwise.  When ``None`` in delta mode the current
            accuracy is used as the reward (first step has no prior to diff).
        config: Reward configuration specifying the semantic mapping.
        is_terminal: Boolean mask indicating terminal steps, shape ``(B,)``.

    Returns:
        Reward tensor of shape ``(B,)`` float, or ``None`` when disabled.
    """
    if config.mode is ArenaRewardMode.DISABLED:
        return None

    accuracy = score.accuracy_all
    if config.mode is ArenaRewardMode.TERMINAL_ONLY:
        zeros = torch.zeros_like(accuracy)
        return torch.where(is_terminal, accuracy, zeros)

    if config.mode is ArenaRewardMode.SCORE_DELTA_DENSE:
        if prev_score is None:
            return accuracy.clone()
        return accuracy - prev_score.accuracy_all

    raise ValueError(f"Unsupported ArenaRewardMode: {config.mode!r}")


# =============================================================================
__all__ = [
    "ArenaRewardConfig",
    "ArenaRewardMode",
    "compute_arena_reward",
]
