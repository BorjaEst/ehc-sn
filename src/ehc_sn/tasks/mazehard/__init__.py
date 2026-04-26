"""MazeHard task public export surface.

Canonical task-first surfaces:

- :mod:`~ehc_sn.tasks.mazehard.contracts` — task contracts and constants
  (including :class:`MazeHardAggregateReport`, the benchmark-facing score type).
- :mod:`~ehc_sn.tasks.mazehard.evaluation` — sequence evaluation and aggregate report.
- :mod:`~ehc_sn.tasks.mazehard.batch` — task-level batch coercion helpers.

Mode bindings (execution-mode, not task identity):

- :mod:`~ehc_sn.tasks.mazehard.modes` — deliberation and other mode bindings.
"""

from . import batch, contracts, evaluation, modes  # noqa: F401
from .contracts import MAZE_HARD_IGNORE_LABEL_ID, MazeHardAggregateReport, MazeHardTargets, MazeHardTaskInput, MazeHardTaskOutput
from .evaluation import (
    MazeHardSequenceMetrics,
    build_maze_hard_report,
    evaluate_maze_hard_sequences,
    is_maze_hard_sequence_correct,
)

__all__ = [
    "MAZE_HARD_IGNORE_LABEL_ID",
    "MazeHardAggregateReport",
    "MazeHardSequenceMetrics",
    "MazeHardTargets",
    "MazeHardTaskInput",
    "MazeHardTaskOutput",
    "batch",
    "build_maze_hard_report",
    "contracts",
    "evaluate_maze_hard_sequences",
    "evaluation",
    "is_maze_hard_sequence_correct",
    "modes",
]
