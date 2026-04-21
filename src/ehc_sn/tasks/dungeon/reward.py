"""Dungeon task reward configuration for online control training.

Task reward configs express canonical semantic score-to-reward mappings only.
Generic shaping parameters (distance coefficients, time discounts, etc.)
belong in the adapter layer, not here (REQ-003, REQ-004).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# =============================================================================
class DungeonRewardMode(str, Enum):
    """Supported semantic reward modes for dungeon online control."""

    GOAL_TERMINAL = "goal_terminal"
    """Sparse reward: +goal_reward on goal reach, zero otherwise."""
    GOAL_DENSE = "goal_dense"
    """Dense reward: inverse-distance shaping toward the goal at each step."""
    STEP_PENALTY = "step_penalty"
    """Constant step cost encouraging episode efficiency."""


# =============================================================================
class DungeonRewardConfig(BaseModel, extra="forbid"):
    """Task-owned dungeon reward wiring.

    Canonical semantic score-to-reward mappings only.
    Generic shaping parameters belong to the adapter layer.
    """

    mode: DungeonRewardMode = DungeonRewardMode.GOAL_TERMINAL
    """Reward mode selecting the semantic mapping."""
    goal_reward: float = Field(default=1.0, ge=0.0)
    """Scalar reward magnitude awarded on goal reach (used in GOAL_TERMINAL)."""
    step_cost: float = Field(default=0.0, ge=0.0)
    """Per-step cost subtracted from reward (used in STEP_PENALTY)."""


# =============================================================================
__all__ = [
    "DungeonRewardConfig",
    "DungeonRewardMode",
]
