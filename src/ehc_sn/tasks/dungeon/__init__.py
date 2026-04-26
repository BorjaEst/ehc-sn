"""Dungeon task family — goal-directed navigation in a dungeon world.

Dungeon defines the observation/action ontology, episode semantics, and
episode score for goal-directed dungeon-world navigation.  These are
task-owned semantics; they do not depend on any particular execution mode.

Canonical task surfaces:

- :mod:`~ehc_sn.tasks.dungeon.contracts` — observation, action, and score contracts
  (including :class:`DungeonScoreReport`, the benchmark-facing score type).
- :mod:`~ehc_sn.tasks.dungeon.evaluation` — episode score builder and aggregate helpers.
- :mod:`~ehc_sn.tasks.dungeon.batch` — task-level batch coercion helpers.

Benchmark evaluation constructs :class:`DungeonScoreReport` via
:func:`~ehc_sn.tasks.dungeon.evaluation.build_dungeon_score_report` and
aggregates with :func:`~ehc_sn.tasks.dungeon.evaluation.compute_dungeon_episode_score`.
Token-supervision targets are not defined in dungeon v1.
"""

from . import batch, contracts, evaluation  # noqa: F401
from .contracts import DUNGEON_ACTION_COUNT, DungeonAction, DungeonScoreReport, DungeonTaskInput
from .evaluation import DungeonEpisodeResult, build_dungeon_score_report, compute_dungeon_episode_score

__all__ = [
    "DUNGEON_ACTION_COUNT",
    "DungeonAction",
    "DungeonEpisodeResult",
    "DungeonScoreReport",
    "DungeonTaskInput",
    "batch",
    "build_dungeon_score_report",
    "compute_dungeon_episode_score",
    "contracts",
    "evaluation",
]
