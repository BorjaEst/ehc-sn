"""MazeHard task family — token-prediction over maze layouts.

MazeHard defines the token-prediction protocol, sequence evaluation semantics,
and dense improvement reward semantics.  These are task-owned semantics; they
do not depend on any particular execution binding.

Stable task surface:

- :mod:`~ehc_sn.tasks.mazehard.contracts` — task contracts and constants.
- :mod:`~ehc_sn.tasks.mazehard.evaluation` — sequence evaluation and aggregate report.
- :mod:`~ehc_sn.tasks.mazehard.reward` — task-owned reward semantics.

Execution-binding capability:

- :class:`MazeHardDeliberationCapability` — deliberation actor-critic capability.
  Full module: :mod:`ehc_sn.tasks.mazehard.capabilities.deliberation`.

Raw channel-to-batch coercion lives in :mod:`ehc_sn.adapters.mazehard.hrm.core`.
"""

from .builder import (
    MAZEHARD_TASK_CHANNELS,
    TASK_FAMILY,
    build_mazehard_task_corpus,
    validate_mazehard_task_root,
    validate_mazehard_task_sample,
)
from .capabilities.deliberation import MazeHardDeliberationCapability
from .contracts import MAZE_HARD_IGNORE_LABEL_ID, MazeHardTargets, MazeHardTaskInput, MazeHardTaskOutput
from .evaluation import MazeHardScoreReport, MazeHardStepScore
from .reward import MazeHardRewardConfig, MazeHardRewardProjector

__all__ = [
    "MAZE_HARD_IGNORE_LABEL_ID",
    "MAZEHARD_TASK_CHANNELS",
    "MazeHardDeliberationCapability",
    "MazeHardRewardConfig",
    "MazeHardRewardProjector",
    "MazeHardScoreReport",
    "MazeHardStepScore",
    "MazeHardTargets",
    "MazeHardTaskInput",
    "MazeHardTaskOutput",
    "TASK_FAMILY",
    "build_mazehard_task_corpus",
    "validate_mazehard_task_root",
    "validate_mazehard_task_sample",
]
