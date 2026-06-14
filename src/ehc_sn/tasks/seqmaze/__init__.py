"""SeqMaze task family — transition-graph edge-lookup probe and path-prediction.

SeqMaze tests whether a model can reason about sequences and transitions
without any spatial grounding.  Phase 0 is the edge-lookup probe that
validates successor-index embedding viability.  Phase 1 (v1) is the
path-prediction task that tests whether the model can infer shortest paths.

Phase 0 surface:

- :mod:`~ehc_sn.tasks.seqmaze.contracts` — task input/output contracts.
- :mod:`~ehc_sn.tasks.seqmaze.evaluation` — edge-prediction and path-prediction score reports.
- :mod:`~ehc_sn.tasks.seqmaze.runtime` — batch key constants and extraction helpers.
- :mod:`~ehc_sn.tasks.seqmaze._graph_utils` — DAG generation and permutation utilities.
- :mod:`~ehc_sn.tasks.seqmaze._data` — on-the-fly probe dataset.
"""

from .contracts import (
    SEQMAZE_IGNORE_LABEL_ID,
    SeqMazeProbeInput,
    SeqMazeProbeOutput,
    SeqMazeProbeTargets,
    SeqMazeTargets,
    SeqMazeTaskInput,
    SeqMazeTaskOutput,
)
from .evaluation import (
    SEQMAZE_V1_METRIC_SPECS,
    SEQMAZE_V1_SCORING_SPEC,
    SeqMazeProbeScoreReport,
    SeqMazeScoreReport,
    SeqMazeStepScore,
    SeqMazeValidationScorer,
    build_seqmaze_step_score,
)
from .reward import SeqMazeRewardConfig, SeqMazeRewardProjector
from .runtime import (
    SEQUENCE_BATCH_KEYS,
    SEQUENCE_MAX_V1_BATCH_KEYS,
    SeqMazeRuntime,
    SeqMazeRuntimeConfig,
    extract_seqmaze_probe_input,
    extract_seqmaze_probe_targets,
    extract_seqmaze_targets,
    extract_seqmaze_task_input,
)

__all__ = [
    "SEQMAZE_IGNORE_LABEL_ID",
    "SeqMazeProbeInput",
    "SeqMazeProbeTargets",
    "SeqMazeProbeOutput",
    "SeqMazeProbeScoreReport",
    "SeqMazeValidationScorer",
    "SeqMazeRewardConfig",
    "SeqMazeRewardProjector",
    "SeqMazeRuntime",
    "SeqMazeRuntimeConfig",
    "SeqMazeStepScore",
    "build_seqmaze_step_score",
    "SEQUENCE_BATCH_KEYS",
    "extract_seqmaze_probe_input",
    "extract_seqmaze_probe_targets",
    "SeqMazeTaskInput",
    "SeqMazeTaskOutput",
    "SeqMazeTargets",
    "SeqMazeScoreReport",
    "SEQMAZE_V1_METRIC_SPECS",
    "SEQMAZE_V1_SCORING_SPEC",
    "SEQUENCE_MAX_V1_BATCH_KEYS",
    "extract_seqmaze_targets",
    "extract_seqmaze_task_input",
]
