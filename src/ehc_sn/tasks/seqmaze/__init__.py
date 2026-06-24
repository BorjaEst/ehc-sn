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

Corpus surface:

- :mod:`~ehc_sn.tasks.seqmaze.builder` — corpus materialization and validation.
- :mod:`~ehc_sn.tasks.seqmaze.corpus` — persisted array reader.
- :mod:`~ehc_sn.tasks.seqmaze.validation` — artifact invariant checking.
- :mod:`~ehc_sn.tasks.seqmaze.diagnostics` — corpus statistics.
- :mod:`~ehc_sn.tasks.seqmaze.inspection` — sample inspection.
"""

from .builder import (
    SEQMAZE_TASK_CHANNELS,
    TASK_FAMILY,
    build_seqmaze_task_corpus,
    validate_seqmaze_root,
    validate_seqmaze_sample,
)
from .contracts import (
    SEQMAZE_IGNORE_LABEL_ID,
    SeqMazeProbeInput,
    SeqMazeProbeOutput,
    SeqMazeProbeTargets,
    SeqMazeTargets,
    SeqMazeTaskInput,
    SeqMazeTaskOutput,
)
from .corpus import (
    load_sample,
    load_split_arrays,
    load_split_manifest,
)
from .diagnostics import compute_corpus_statistics
from .evaluation import (
    SEQMAZE_V1_METRIC_SPECS,
    SEQMAZE_V1_SCORING_SPEC,
    SeqMazeProbeScoreReport,
    SeqMazeScoreReport,
    SeqMazeStepScore,
    SeqMazeValidationScorer,
    build_seqmaze_step_score,
)
from .inspection import (
    SeqMazeSampleInspection,
    prepare_sample_inspection,
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
from .validation import (
    SeqMazeValidationIssue,
    validate_all_samples,
    validate_stored_sample,
)

__all__ = [
    "SEQMAZE_IGNORE_LABEL_ID",
    "SEQMAZE_TASK_CHANNELS",
    "SeqMazeProbeInput",
    "SeqMazeProbeTargets",
    "SeqMazeProbeOutput",
    "SeqMazeProbeScoreReport",
    "SeqMazeSampleInspection",
    "SeqMazeValidationIssue",
    "SeqMazeValidationScorer",
    "SeqMazeRewardConfig",
    "SeqMazeRewardProjector",
    "SeqMazeRuntime",
    "SeqMazeRuntimeConfig",
    "SeqMazeStepScore",
    "TASK_FAMILY",
    "build_seqmaze_step_score",
    "build_seqmaze_task_corpus",
    "compute_corpus_statistics",
    "load_sample",
    "load_split_arrays",
    "load_split_manifest",
    "prepare_sample_inspection",
    "validate_all_samples",
    "validate_seqmaze_root",
    "validate_seqmaze_sample",
    "validate_stored_sample",
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
