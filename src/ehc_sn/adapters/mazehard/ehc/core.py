"""MazeHard+EHC bridge family core — family-stable settings and constants.

Holds task-side adapter settings shared by all EHC bridge versions.
Uses the task-owned vocabulary constant directly; no HRM-family import.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ehc_sn.tasks.mazehard.runtime import MAZE_HARD_VOCAB_SIZE

# =============================================================================
DEFAULT_MAZE_HARD_EHC_VOCAB_SIZE: int = MAZE_HARD_VOCAB_SIZE
"""Default vocabulary size for MazeHard+EHC bridges (task-owned canonical value)."""


# =============================================================================
class MazeHardEHCAdapterSettings(BaseModel, extra="forbid"):
    """Task-side MazeHard settings for the EHC bridge family."""

    vocab_size: int = Field(
        default=DEFAULT_MAZE_HARD_EHC_VOCAB_SIZE,
        ge=1,
        description=(
            "MazeHard token vocabulary size used by encoder and decoder heads. "
            "Defaults to the canonical vocabulary including the solution-overlay token."
        ),
    )


# =============================================================================
__all__ = [
    "DEFAULT_MAZE_HARD_EHC_VOCAB_SIZE",
    "MazeHardEHCAdapterSettings",
]
