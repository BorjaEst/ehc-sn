"""Dungeon task family — goal-directed navigation in a dungeon world.

Dungeon defines the observation/action ontology, episode semantics, and
episode score for goal-directed dungeon-world navigation.  These are
task-owned semantics; they do not depend on any particular execution-binding
capability.

Stable task surface:

- :mod:`~ehc_sn.tasks.dungeon.contracts` — observation, action, and score contracts.
- :mod:`~ehc_sn.tasks.dungeon.evaluation` — step score builder and aggregate report.
- :mod:`~ehc_sn.tasks.dungeon.reward` — task-owned reward projection.
- :mod:`~ehc_sn.tasks.dungeon.providers` — task-owned evaluation case providers
  (:class:`~ehc_sn.tasks.dungeon.providers.DungeonReplayDiagnosticProvider`,
  :class:`~ehc_sn.tasks.dungeon.providers.DungeonFixedProbeProvider`).

Dungeon v1 models episode-terminal task facts only; token-supervision targets
are not defined.  No online-control execution-binding capability is implemented:
the environment kernel is not yet written.

"""

from .builder import (
    DUNGEON_TASK_CHANNELS,
    DUNGEON_TRAJECTORY_CHANNELS,
    TASK_FAMILY,
    build_dungeon_task_corpus,
    validate_dungeon_task_root,
    validate_dungeon_task_sample,
)
from .contracts import DUNGEON_ACTION_COUNT, DungeonAction, DungeonTaskInput, DungeonTaskOutput
from .evaluation import DungeonScoreReport, DungeonStepScore
from .reward import DungeonRewardConfig, DungeonRewardProjector

__all__ = [
    "DUNGEON_ACTION_COUNT",
    "DungeonAction",
    "DUNGEON_TASK_CHANNELS",
    "DUNGEON_TRAJECTORY_CHANNELS",
    "DungeonRewardConfig",
    "DungeonRewardProjector",
    "DungeonScoreReport",
    "DungeonStepScore",
    "DungeonTaskInput",
    "DungeonTaskOutput",
    "TASK_FAMILY",
    "build_dungeon_task_corpus",
    "validate_dungeon_task_root",
    "validate_dungeon_task_sample",
]
