"""Countwalk task family — replay over a bounded integer number line.

Countwalk defines a 1-D episodic task: the agent navigates a hidden integer
position via PREV/NEXT/STAY actions and is occasionally cued with its current
position.  The task queries the agent to report the final position as fixed-slot
digit outputs.

V1 scope: legal-only replay corpus, digit and exact-set cue surfaces, anchor-only
and sparse-reanchor anchor regimes, fixed-slot masked digit targets, five-bucket
evaluation taxonomy (ID / range-OOD / horizon-OOD / joint-OOD / stretch-OOD).

Stable task surface:

- :mod:`~ehc_sn.tasks.countwalk.contracts` — action/cue/regime/token/bucket
  constants and typed task contracts.
- :mod:`~ehc_sn.tasks.countwalk.builder` — channel constants, sample validator, and
  corpus builder.
- :mod:`~ehc_sn.tasks.countwalk.evaluation` — digit accuracy, sequence accuracy,
  value accuracy, MAE, and stratified scoring.
- :mod:`~ehc_sn.tasks.countwalk.runtime` — replay batch schema constants and
  batch-cursor-aware target extraction.
- :mod:`~ehc_sn.tasks.countwalk.providers` — task-owned evaluation case providers
  (:class:`~ehc_sn.tasks.countwalk.providers.CountwalkReplayDiagnosticProvider`,
  :class:`~ehc_sn.tasks.countwalk.providers.CountwalkFixedProbeProvider`).

Execution-binding capability:

- :class:`CountwalkReplayCapability` — replay trajectory controller capability.
  Full module: :mod:`ehc_sn.tasks.countwalk.capabilities.replay`.
"""

from .builder import (
    CHANNEL_ANCHOR_REGIME_ID,
    CHANNEL_CUE_SURFACE_ID,
    CHANNEL_EVAL_BUCKET_ID,
    CHANNEL_QUERY_MASK,
    CHANNEL_TARGET_DIGIT_MASK,
    CHANNEL_TARGET_DIGITS,
    CHANNEL_TARGET_VALUE,
    CHANNEL_TRAJECTORY_ANCHOR_VISIBLE,
    CHANNEL_TRAJECTORY_CUE_MASK,
    CHANNEL_TRAJECTORY_CUE_TOKENS,
    CHANNEL_TRAJECTORY_EPISODE_START,
    CHANNEL_TRAJECTORY_LENGTH,
    CHANNEL_TRAJECTORY_PREVIOUS_ACTION,
    CHANNEL_TRAJECTORY_VALID_STEP,
    CHANNEL_TRAJECTORY_VALUE,
    CHANNEL_WORLD_ID,
    COUNTWALK_TASK_CHANNELS,
    TASK_FAMILY,
    build_countwalk_task_corpus,
    validate_countwalk_task_root,
    validate_countwalk_task_sample,
)
from .capabilities.replay import CountwalkReplayCapability
from .contracts import (
    ACTION_NEXT,
    ACTION_PREV,
    ACTION_STAY,
    ANCHOR_ONLY,
    BUCKET_HORIZON_OOD,
    BUCKET_HORIZON_OOD_MAX,
    BUCKET_HORIZON_OOD_MIN,
    BUCKET_ID,
    BUCKET_ID_HORIZON_MAX,
    BUCKET_ID_VALUE_MAX,
    BUCKET_JOINT_OOD,
    BUCKET_RANGE_OOD,
    BUCKET_RANGE_OOD_VALUE_MAX,
    BUCKET_RANGE_OOD_VALUE_MIN,
    BUCKET_STRETCH_OOD,
    BUCKET_STRETCH_OOD_VALUE_MAX,
    BUCKET_STRETCH_OOD_VALUE_MIN,
    COUNTWALK_IGNORE_DIGIT,
    COUNTWALK_TOKEN_VOCAB_SIZE,
    CUE_DIGIT,
    CUE_SET,
    DIGIT_WIDTH,
    N_ACTIONS,
    N_ANCHOR_REGIMES,
    N_BUCKETS,
    N_CUE_SURFACES,
    SPARSE_REANCHOR,
    TOKEN_DIGIT_0,
    TOKEN_DIGIT_9,
    TOKEN_ITEM,
    TOKEN_PAD,
    CountwalkTargets,
    CountwalkTaskInput,
    CountwalkTaskOutput,
)
from .evaluation import (
    CountwalkScoreReport,
    digit_accuracy,
    mean_absolute_error,
    score_countwalk_batch,
    score_countwalk_stratified,
    sequence_accuracy,
    value_accuracy,
)
from .runtime import (
    COUNTWALK_REPLAY_OPTIONAL_KEYS,
    COUNTWALK_REPLAY_REQUIRED_KEYS,
    COUNTWALK_STEP_KEYS,
    batch_extract_countwalk_targets,
    batch_size_from_countwalk_batch,
    infer_countwalk_replay_batch_keys,
)
from .traces import (
    CountwalkEvaluationSourceContext,
    CountwalkTraceSupplements,
    apply_countwalk_trace_supplements,
    build_countwalk_trace_supplements,
)

__all__ = [
    # Family
    "TASK_FAMILY",
    # Action constants
    "ACTION_STAY",
    "ACTION_PREV",
    "ACTION_NEXT",
    "N_ACTIONS",
    # Cue surface constants
    "CUE_DIGIT",
    "CUE_SET",
    "N_CUE_SURFACES",
    # Anchor regime constants
    "ANCHOR_ONLY",
    "SPARSE_REANCHOR",
    "N_ANCHOR_REGIMES",
    # Token vocabulary
    "TOKEN_PAD",
    "TOKEN_DIGIT_0",
    "TOKEN_DIGIT_9",
    "TOKEN_ITEM",
    "COUNTWALK_TOKEN_VOCAB_SIZE",
    # Digit configuration
    "DIGIT_WIDTH",
    "COUNTWALK_IGNORE_DIGIT",
    # Evaluation bucket constants
    "BUCKET_ID",
    "BUCKET_RANGE_OOD",
    "BUCKET_HORIZON_OOD",
    "BUCKET_JOINT_OOD",
    "BUCKET_STRETCH_OOD",
    "N_BUCKETS",
    "BUCKET_ID_VALUE_MAX",
    "BUCKET_RANGE_OOD_VALUE_MIN",
    "BUCKET_RANGE_OOD_VALUE_MAX",
    "BUCKET_STRETCH_OOD_VALUE_MIN",
    "BUCKET_STRETCH_OOD_VALUE_MAX",
    "BUCKET_ID_HORIZON_MAX",
    "BUCKET_HORIZON_OOD_MIN",
    "BUCKET_HORIZON_OOD_MAX",
    # Channel names
    "COUNTWALK_TASK_CHANNELS",
    "CHANNEL_WORLD_ID",
    "CHANNEL_CUE_SURFACE_ID",
    "CHANNEL_ANCHOR_REGIME_ID",
    "CHANNEL_EVAL_BUCKET_ID",
    "CHANNEL_TRAJECTORY_VALUE",
    "CHANNEL_TRAJECTORY_PREVIOUS_ACTION",
    "CHANNEL_TRAJECTORY_ANCHOR_VISIBLE",
    "CHANNEL_TRAJECTORY_CUE_TOKENS",
    "CHANNEL_TRAJECTORY_CUE_MASK",
    "CHANNEL_TRAJECTORY_EPISODE_START",
    "CHANNEL_TRAJECTORY_VALID_STEP",
    "CHANNEL_TRAJECTORY_LENGTH",
    "CHANNEL_QUERY_MASK",
    "CHANNEL_TARGET_DIGITS",
    "CHANNEL_TARGET_DIGIT_MASK",
    "CHANNEL_TARGET_VALUE",
    # Typed contracts
    "CountwalkTaskInput",
    "CountwalkTargets",
    "CountwalkTaskOutput",
    # Evaluation
    "CountwalkScoreReport",
    "digit_accuracy",
    "sequence_accuracy",
    "value_accuracy",
    "mean_absolute_error",
    "score_countwalk_batch",
    "score_countwalk_stratified",
    # Runtime
    "COUNTWALK_REPLAY_REQUIRED_KEYS",
    "COUNTWALK_REPLAY_OPTIONAL_KEYS",
    "COUNTWALK_STEP_KEYS",
    "batch_size_from_countwalk_batch",
    "infer_countwalk_replay_batch_keys",
    "batch_extract_countwalk_targets",
    # Builders
    "build_countwalk_task_corpus",
    "validate_countwalk_task_root",
    "validate_countwalk_task_sample",
    # Capabilities
    "CountwalkReplayCapability",
    # Trace supplements
    "CountwalkEvaluationSourceContext",
    "CountwalkTraceSupplements",
    "build_countwalk_trace_supplements",
    "apply_countwalk_trace_supplements",
]
