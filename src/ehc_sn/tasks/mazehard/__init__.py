"""MazeHard task family — token-prediction over maze layouts.

MazeHard defines the token-prediction protocol, sequence evaluation semantics,
and dense improvement reward semantics.  These are task-owned semantics; they
do not depend on any particular execution binding.

Stable task surface:

- :mod:`~ehc_sn.tasks.mazehard.contracts` — task contracts and constants.
- :mod:`~ehc_sn.tasks.mazehard.evaluation` — sequence evaluation and aggregate report.
- :mod:`~ehc_sn.tasks.mazehard.reward` — task-owned reward semantics.
- :mod:`~ehc_sn.tasks.mazehard.providers` — task-owned evaluation case providers
  (:class:`~ehc_sn.tasks.mazehard.providers.MazeHardReplayDiagnosticProvider`,
  :class:`~ehc_sn.tasks.mazehard.providers.MazeHardFixedProbeProvider`).

Execution-binding capability:

- :class:`MazeHardDeliberationCapability` — deliberation actor-critic capability.
  Full module: :mod:`ehc_sn.tasks.mazehard.capabilities.deliberation`.

Raw channel-to-batch coercion lives in :mod:`ehc_sn.tasks.mazehard.runtime`.
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
from .traces import (
    MazeHardEvaluationSourceContext,
    MazeHardTraceSupplements,
    apply_mazehard_trace_supplements,
    build_mazehard_trace_supplements,
)

__all__ = [
    "MAZE_HARD_IGNORE_LABEL_ID",
    "MAZEHARD_TASK_CHANNELS",
    "MazeHardDeliberationCapability",
    "MazeHardEvaluationSourceContext",
    "MazeHardRewardConfig",
    "MazeHardRewardProjector",
    "MazeHardScoreReport",
    "MazeHardStepScore",
    "MazeHardTargets",
    "MazeHardTaskInput",
    "MazeHardTaskOutput",
    "MazeHardTraceSupplements",
    "TASK_FAMILY",
    "apply_mazehard_trace_supplements",
    "build_mazehard_task_corpus",
    "build_mazehard_trace_supplements",
    "validate_mazehard_task_root",
    "validate_mazehard_task_sample",
]
