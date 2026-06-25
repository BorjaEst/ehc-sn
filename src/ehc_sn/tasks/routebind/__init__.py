"""Routebind task family — goal-conditioned spatial prospective-field prediction.

Routebind tests whether HRM can separate and recombine abstract learned
transition structure over observation identities with concrete spatial
arrangements.  The model receives a 2-D layout with walls, observation
identities, a start position, and a semantic goal cue, and must predict
a discounted trajectory field and a semantic waypoint field over the 900
spatial positions.

The task combines two forms of structure — visible spatial (grid layout,
walls, start/goal positions) and hidden semantic (a fixed DAG over
observation identities not supplied in the input) — that must be jointly
reasoned over.  The hidden observation-transition graph is learned
parametrically and reused across unseen spatial instantiations.

Stable task surface:

- :mod:`~ehc_sn.tasks.routebind.contracts` — vocabulary, typed contracts, corpus schema.
- :mod:`~ehc_sn.tasks.routebind.oracle` — product-state shortest-path oracle.
- :mod:`~ehc_sn.tasks.routebind.targets` — oracle-result-to-target-field encoding.
- :mod:`~ehc_sn.tasks.routebind.decoding` — prediction-to-route decoding.
- :mod:`~ehc_sn.tasks.routebind.validation` — artifact invariant checking.
- :mod:`~ehc_sn.tasks.routebind.evaluation` — metrics and score report.
- :mod:`~ehc_sn.tasks.routebind.builder` — corpus orchestration.
- :mod:`~ehc_sn.tasks.routebind.runtime` — batch key constants and extraction helpers.
- :mod:`~ehc_sn.tasks.routebind.traces` — evaluation source context and trace supplements.
- :mod:`~ehc_sn.tasks.routebind.supervision` — supervision struct and coercion.

Routebind v1 supports single-step field prediction with fixed-depth recurrent
deliberation.  No replay controller, no trajectory, and no ACT deliberation
are needed at the task level — deliberation belongs in the adapter or
training regime.
"""

from .builder import (
    ROUTEBIND_CORPUS_CHANNELS,
    ROUTEBIND_MODEL_INPUT_CHANNELS,
    ROUTEBIND_PRESETS,
    ROUTEBIND_TARGET_CHANNELS,
    TASK_FAMILY,
    CompletionPolicy,
    GenerationFunnel,
    QuerySelectionProfile,
    RealizedAdmissionPolicy,
    RoutebindPreset,
    build_routebind_task_corpus,
    resolve_preset,
    validate_routebind_root,
)
from .contracts import (
    CELL_FREE,
    CELL_OBSERVATION,
    CELL_WALL,
    DELTA_TO_DIRECTION,
    DIRECTION_DELTA,
    ROUTEBIND_IGNORE_LABEL_ID,
    ROUTEBIND_SCHEMA,
    Direction,
    PolicyTransition,
    RoutebindCorpusSchema,
    RoutebindTargets,
    RoutebindTaskInput,
    RoutebindTaskOutput,
)
from .corpus import (
    load_sample,
    load_split_arrays,
    load_split_manifest,
)
from .decoding import (
    decode_next_direction,
    extract_route_from_trajectory_field,
    extract_waypoint_sequence,
    extract_waypoints_from_field,
)
from .diagnostics import (
    compute_corpus_statistics,
    select_samples,
)
from .evaluation import (
    ROUTEBIND_METRIC_SPECS,
    ROUTEBIND_SCORING_SPEC,
    RoutebindStepScore,
)
from .inspection import (
    RoutebindSampleInspection,
    prepare_sample_inspection,
)
from .oracle import (
    OracleResult,
    compute_goal_distance_table,
    reconstruct_from_policy,
    solve_product_state_route,
)
from .runtime import (
    ROUTEBIND_BATCH_KEYS,
    extract_routebind_targets,
    extract_routebind_task_input,
)
from .supervision import (
    RoutebindSupervision,
    build_routebind_supervision,
)
from .targets import (
    encode_trajectory_field,
    encode_waypoint_field,
    validate_decay_consistency,
)
from .traces import (
    RoutebindEvaluationSourceContext,
    RoutebindTraceSupplements,
    apply_routebind_trace_supplements,
    build_routebind_trace_supplements,
)
from .validation import (
    OracleValidationContext,
    ValidationIssue,
    check_oracle_optimal_subgraph,
    validate_corpus_root,
    validate_generated_sample,
    validate_stored_sample,
    validate_support_channels,
)

__all__ = [
    "CELL_FREE",
    "CELL_OBSERVATION",
    "CELL_WALL",
    "CompletionPolicy",
    "DELTA_TO_DIRECTION",
    "DIRECTION_DELTA",
    "Direction",
    "GenerationFunnel",
    "OracleResult",
    "PolicyTransition",
    "QuerySelectionProfile",
    "ROUTEBIND_BATCH_KEYS",
    "ROUTEBIND_CORPUS_CHANNELS",
    "ROUTEBIND_IGNORE_LABEL_ID",
    "ROUTEBIND_METRIC_SPECS",
    "ROUTEBIND_MODEL_INPUT_CHANNELS",
    "ROUTEBIND_PRESETS",
    "ROUTEBIND_SCHEMA",
    "ROUTEBIND_SCORING_SPEC",
    "ROUTEBIND_TARGET_CHANNELS",
    "RealizedAdmissionPolicy",
    "RoutebindCorpusSchema",
    "RoutebindEvaluationSourceContext",
    "RoutebindSupervision",
    "RoutebindPreset",
    "RoutebindSampleInspection",
    "RoutebindStepScore",
    "RoutebindTargets",
    "RoutebindTaskInput",
    "RoutebindTaskOutput",
    "RoutebindTraceSupplements",
    "TASK_FAMILY",
    "OracleValidationContext",
    "ValidationIssue",
    "apply_routebind_trace_supplements",
    "build_routebind_supervision",
    "build_routebind_task_corpus",
    "build_routebind_trace_supplements",
    "check_oracle_optimal_subgraph",
    "compute_corpus_statistics",
    "compute_goal_distance_table",
    "decode_next_direction",
    "encode_trajectory_field",
    "encode_waypoint_field",
    "extract_route_from_trajectory_field",
    "extract_routebind_targets",
    "extract_routebind_task_input",
    "extract_waypoint_sequence",
    "extract_waypoints_from_field",
    "load_sample",
    "load_split_arrays",
    "load_split_manifest",
    "prepare_sample_inspection",
    "reconstruct_from_policy",
    "resolve_preset",
    "select_samples",
    "solve_product_state_route",
    "validate_corpus_root",
    "validate_decay_consistency",
    "validate_generated_sample",
    "validate_routebind_root",
    "validate_stored_sample",
    "validate_support_channels",
]
