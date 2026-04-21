"""Dungeon task evaluation helpers for online control benchmarking."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .contracts import DungeonScoreReport


# =============================================================================
@dataclass(frozen=True)
class DungeonEpisodeResult:
    """Aggregate evaluation result over a batch of dungeon episodes."""

    success_rate: Tensor
    """Mean episode success rate, scalar float."""
    mean_return: Tensor
    """Mean cumulative return per episode, scalar float."""
    mean_steps: Tensor
    """Mean episode length in steps, scalar float."""


# =============================================================================
def compute_dungeon_episode_score(
    reports: DungeonScoreReport,
) -> DungeonEpisodeResult:
    """Return aggregate evaluation metrics from a batch of dungeon score reports.

    Args:
        reports: Score reports for a batch of completed dungeon episodes.

    Returns:
        :class:`DungeonEpisodeResult` with batch-mean scalar metrics.
    """
    return DungeonEpisodeResult(
        success_rate=reports.success.float().mean(),
        mean_return=reports.total_return.float().mean(),
        mean_steps=reports.episode_steps.float().mean(),
    )


# =============================================================================
__all__ = [
    "DungeonEpisodeResult",
    "compute_dungeon_episode_score",
]
