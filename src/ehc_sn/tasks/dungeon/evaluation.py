"""Dungeon task evaluation helpers.

Owns episode-level correctness primitives and the benchmark-facing
:class:`DungeonScoreReport` aggregate score surface.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


# =============================================================================
@dataclass(frozen=True)
class DungeonScoreReport:
    """Canonical aggregate benchmark-facing score report for a batch of dungeon episodes.

    This is the task-owned aggregate surface; benchmarks consume these fields.
    Computed by :func:`build_dungeon_score_report` from a batch of
    :class:`DungeonStepScore` units.
    """

    success_rate: Tensor
    """Mean episode success rate, scalar float."""
    mean_score: Tensor
    """Mean path-efficiency score per episode, scalar float."""
    mean_steps: Tensor
    """Mean episode length in steps, scalar float."""


# =============================================================================
@dataclass(frozen=True)
class DungeonStepScore:
    """Per-slot local semantic state for one dungeon episode step.

    Carries task-observable quantities only.  The pre-computed efficiency
    score (a reward signal) was removed; it is derived in
    :func:`build_dungeon_score_report` and :class:`~ehc_sn.tasks.dungeon.reward.DungeonRewardProjector`
    when needed.
    """

    success: Tensor
    """Goal reached within episode horizon, shape ``(B,)`` bool."""
    episode_steps: Tensor
    """Steps taken in the episode, shape ``(B,)`` int64."""


# =============================================================================
def build_dungeon_step_score(
    success: Tensor,
    episode_steps: Tensor,
) -> DungeonStepScore:
    """Return a :class:`DungeonStepScore` from raw episode-end tensors.

    Args:
        success: Goal-reached indicator per episode, shape ``(B,)`` bool.
        episode_steps: Steps taken per episode, shape ``(B,)`` int64.

    Returns:
        :class:`DungeonStepScore` with task-observable local state.
    """
    return DungeonStepScore(success=success, episode_steps=episode_steps)


# =============================================================================
def build_dungeon_score_report(
    step_scores: DungeonStepScore,
) -> DungeonScoreReport:
    """Return aggregate evaluation metrics from a batch of dungeon step scores.

    Args:
        step_scores: Episode-level primitive scores for a completed batch.

    Returns:
        :class:`DungeonScoreReport` with batch-mean scalar metrics.
    """
    return DungeonScoreReport(
        success_rate=step_scores.success.float().mean(),
        mean_score=(step_scores.success.float() / step_scores.episode_steps.float().clamp_min(1.0)).mean(),
        mean_steps=step_scores.episode_steps.float().mean(),
    )


# =============================================================================
__all__ = [
    "DungeonScoreReport",
    "DungeonStepScore",
    "build_dungeon_score_report",
    "build_dungeon_step_score",
]
