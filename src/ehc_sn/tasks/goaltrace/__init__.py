"""Goaltrace task family — goal-conditioned prospective field prediction.

Goaltrace is the isolated HRM/PFC training task for goal-conditioned
prospective field prediction.  It trains the model to transform a current
location, a goal observation, and state-dependent relational weights into a
goal-conditioned prospective firing field over a fixed learned DAG.

The task isolates the PFC computation by providing — as oracle task inputs —
the signals that TEM/HPC would eventually supply in the integrated EHP
architecture.  HRM does not retrieve a map, infer its current location, or
reconstruct episodic bindings.  It learns only the transformation:

    current state + goal + current relational field -> goal-directed prospective field

Stable task surface:

- :mod:`~ehc_sn.tasks.goaltrace.contracts` — task input/output/targets contracts.
- :mod:`~ehc_sn.tasks.goaltrace.evaluation` — step score and aggregate report.
- :mod:`~ehc_sn.tasks.goaltrace.builder` — corpus materialization and validation.
- :mod:`~ehc_sn.tasks.goaltrace.runtime` — batch key constants and extraction helpers.
- :mod:`~ehc_sn.tasks.goaltrace.traces` — evaluation source context and trace supplements.
- :mod:`~ehc_sn.tasks.goaltrace.environment` — TorchRL EnvBase stub (contract symmetry).

Goaltrace v1 supports single-step field prediction with oracle-derived
targets.  No replay controller, no trajectory, and no ACT deliberation
are needed at the task level — deliberation belongs in the adapter or
training regime.
"""

from .builder import (
    GOALTRACE_CORPUS_CHANNELS,
    GOALTRACE_EVALUATION_META_CHANNELS,
    GOALTRACE_MODEL_INPUT_CHANNELS,
    GOALTRACE_TARGET_CHANNELS,
    GOALTRACE_TASK_CHANNELS,
    TASK_FAMILY,
    build_goaltrace_task_corpus,
    validate_goaltrace_root,
    validate_goaltrace_sample,
)
from .contracts import (
    GoaltraceTargets,
    GoaltraceTaskInput,
    GoaltraceTaskOutput,
)
from .evaluation import (
    GOALTRACE_METRIC_SPECS,
    GOALTRACE_SCORING_SPEC,
    GoaltraceStepScore,
    build_goaltrace_step_score,
    compute_goaltrace_step_score,
)
from .runtime import (
    GOALTRACE_BATCH_KEYS,
    extract_goaltrace_targets,
    extract_goaltrace_task_input,
)
from .traces import (
    GoaltraceEvaluationSourceContext,
    GoaltraceTraceSupplements,
    apply_goaltrace_trace_supplements,
    build_goaltrace_trace_supplements,
)

__all__ = [
    "GOALTRACE_BATCH_KEYS",
    "GOALTRACE_CORPUS_CHANNELS",
    "GOALTRACE_EVALUATION_META_CHANNELS",
    "GOALTRACE_METRIC_SPECS",
    "GOALTRACE_MODEL_INPUT_CHANNELS",
    "GOALTRACE_SCORING_SPEC",
    "GOALTRACE_TARGET_CHANNELS",
    "GOALTRACE_TASK_CHANNELS",
    "GoaltraceEvaluationSourceContext",
    "GoaltraceStepScore",
    "GoaltraceTargets",
    "GoaltraceTaskInput",
    "GoaltraceTaskOutput",
    "GoaltraceTraceSupplements",
    "TASK_FAMILY",
    "apply_goaltrace_trace_supplements",
    "build_goaltrace_step_score",
    "build_goaltrace_task_corpus",
    "build_goaltrace_trace_supplements",
    "compute_goaltrace_step_score",
    "extract_goaltrace_task_input",
    "extract_goaltrace_targets",
    "validate_goaltrace_root",
    "validate_goaltrace_sample",
]
