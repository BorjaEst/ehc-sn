"""Dungeon task evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .contracts import DungeonScoreReport as _DungeonScoreReport


# =============================================================================
@dataclass(frozen=True)
class DungeonEpisodeResult:
    """Aggregate evaluation result over a batch of dungeon episodes."""

    success_rate: Tensor
    """Mean episode success rate, scalar float."""
    mean_score: Tensor
    """Mean episode score per episode, scalar float."""
    mean_steps: Tensor
    """Mean episode length in steps, scalar float."""


# =============================================================================
def build_dungeon_score_report(
    success: Tensor,
    episode_steps: Tensor,
) -> _DungeonScoreReport:
    """Return a :class:`~ehc_sn.tasks.dungeon.contracts.DungeonScoreReport` from raw episode-end tensors.

    The canonical task-owned score is the sparse path-efficiency signal:
    ``1 / episode_steps`` for successful episodes and 0 for failures.  This
    makes the score a pure function of task-observable quantities with no
    dependence on reward shaping or adapter-layer conventions.

    Args:
        success: Goal-reached indicator per episode, shape ``(B,)`` bool.
        episode_steps: Steps taken per episode, shape ``(B,)`` int64.

    Returns:
        :class:`~ehc_sn.tasks.dungeon.contracts.DungeonScoreReport` with the computed score field.
    """
    steps_f = episode_steps.float().clamp_min(1.0)
    score = success.float() / steps_f
    return _DungeonScoreReport(
        success=success,
        score=score,
        episode_steps=episode_steps,
    )


# =============================================================================
def compute_dungeon_episode_score(
    reports: _DungeonScoreReport,
) -> DungeonEpisodeResult:
    """Return aggregate evaluation metrics from a batch of dungeon score reports.

    Args:
        reports: Score reports for a batch of completed dungeon episodes.

    Returns:
        :class:`DungeonEpisodeResult` with batch-mean scalar metrics.
    """
    return DungeonEpisodeResult(
        success_rate=reports.success.float().mean(),
        mean_score=reports.score.float().mean(),
        mean_steps=reports.episode_steps.float().mean(),
    )


# =============================================================================
__all__ = [
    "DungeonEpisodeResult",
    "build_dungeon_score_report",
    "compute_dungeon_episode_score",
]
