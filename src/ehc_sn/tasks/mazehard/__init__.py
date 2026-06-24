"""MazeHard task family — token-prediction over maze layouts.

MazeHard defines the token-prediction protocol, sequence evaluation semantics,
and dense improvement reward semantics.  These are task-owned semantics; they
do not depend on any particular execution binding.

Stable task surface:

- :mod:`~ehc_sn.tasks.mazehard.contracts` — task contracts and constants.
- :mod:`~ehc_sn.tasks.mazehard.evaluation` — sequence evaluation and aggregate report.
- :mod:`~ehc_sn.tasks.mazehard.reward` — task-owned reward semantics.
- :mod:`~ehc_sn.tasks.mazehard.providers` — task-owned evaluation case providers
  (:class:`~ehc_sn.tasks.mazehard.providers.MazeHardReplayProvider`,
  :class:`~ehc_sn.tasks.mazehard.providers.MazeHardFixedProbeProvider`).

Step evaluator (RL-feedback seam):

- :class:`MazeHardStepEvaluator` — task-owned ``TaskStepEvaluator`` implementation.
  Full module: :mod:`ehc_sn.tasks.mazehard.evaluators.step`.

Raw channel-to-batch coercion lives in :mod:`ehc_sn.tasks.mazehard.runtime`.
"""

from .builder import (
    MAZEHARD_TASK_CHANNELS,
    TASK_FAMILY,
    build_mazehard_task_corpus,
    validate_mazehard_root,
    validate_mazehard_sample,
    validate_mazehard_task_root,
    validate_mazehard_task_sample,
)
from .contracts import (
    MAZE_HARD_IGNORE_LABEL_ID,
    MazeHardTargets,
    MazeHardTaskInput,
    MazeHardTaskOutput,
)
from .corpus import load_sample, load_split_arrays, load_split_manifest
from .diagnostics import compute_corpus_statistics, select_samples
from .evaluation import MazeHardScoreReport, MazeHardStepScore
from .evaluators.step import MazeHardStepEvaluator
from .inspection import MazeHardSampleInspection, prepare_sample_inspection
from .reward import MazeHardRewardConfig, MazeHardRewardProjector
from .traces import (
    MazeHardEvaluationSourceContext,
    MazeHardTraceSupplements,
    apply_mazehard_trace_supplements,
    build_mazehard_trace_supplements,
)
from .validation import (
    MazeHardValidationIssue,
    validate_all_samples,
    validate_stored_sample,
)

__all__ = [
    "MAZE_HARD_IGNORE_LABEL_ID",
    "MAZEHARD_TASK_CHANNELS",
    "MazeHardEvaluationSourceContext",
    "MazeHardRewardConfig",
    "MazeHardRewardProjector",
    "MazeHardSampleInspection",
    "MazeHardScoreReport",
    "MazeHardStepEvaluator",
    "MazeHardStepScore",
    "MazeHardTargets",
    "MazeHardTaskInput",
    "MazeHardTaskOutput",
    "MazeHardTraceSupplements",
    "MazeHardValidationIssue",
    "TASK_FAMILY",
    "apply_mazehard_trace_supplements",
    "build_mazehard_task_corpus",
    "build_mazehard_trace_supplements",
    "compute_corpus_statistics",
    "load_sample",
    "load_split_arrays",
    "load_split_manifest",
    "prepare_sample_inspection",
    "select_samples",
    "validate_all_samples",
    "validate_mazehard_root",
    "validate_mazehard_sample",
    "validate_mazehard_task_root",
    "validate_mazehard_task_sample",
    "validate_stored_sample",
]
