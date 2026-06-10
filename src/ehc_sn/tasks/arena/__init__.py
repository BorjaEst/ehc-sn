"""Arena task family — structural navigation over a maze world (v1, topology-free).

Arena defines the observation/action ontology, revisit semantics, and additive
structural score for maze-world navigation.  These are task-owned semantics;
they do not depend on any particular execution binding.

Arena replay v1 is topology-free: the parent dungeongen substrate is the sole
owner of spatial geometry.  One stored sample equals one episode.  Training
uses a fixed offline corpus only.

Stable task surface:

- :mod:`~ehc_sn.tasks.arena.contracts` — observation, action, and score contracts.
- :mod:`~ehc_sn.tasks.arena.evaluation` — structural score and evaluation helpers.
- :mod:`~ehc_sn.tasks.arena.reward` — task-owned reward projection.
- :mod:`~ehc_sn.tasks.arena.providers` — task-owned evaluation case providers
  (:class:`~ehc_sn.tasks.arena.providers.ArenaReplayProvider`,
  :class:`~ehc_sn.tasks.arena.providers.ArenaFixedProbeProvider`).

Optional execution-binding capability:

- :class:`ArenaReplayCapability` — replay execution-binding capability.
  Full module: :mod:`ehc_sn.tasks.arena.capabilities.replay`.

Benchmark evaluation consumes :class:`~ehc_sn.tasks.arena.evaluation.ArenaScoreReport` directly from the
task-owned contracts surface.  Whole-trajectory model inputs are forbidden
(REQ-001, SCI-001).
"""

from .builder import (
    ARENA_SPATIAL_CHANNELS,
    ARENA_TASK_CHANNELS,
    TASK_FAMILY,
    build_arena_task_corpus,
    validate_arena_task_root,
    validate_arena_task_sample,
)
from .capabilities.replay import ArenaReplayCapability
from .contracts import (
    ArenaAction,
    ArenaTargets,
    ArenaTaskInput,
    ArenaTaskOutput,
)
from .evaluation import ArenaScoreReport, ArenaStepScore
from .reward import ArenaRewardConfig, ArenaRewardProjector
from .traces import (
    ArenaEvaluationSourceContext,
    ArenaTraceSupplements,
    apply_arena_trace_supplements,
    build_arena_trace_supplements,
)

__all__ = [
    "ArenaAction",
    "ARENA_SPATIAL_CHANNELS",
    "ARENA_TASK_CHANNELS",
    "ArenaEvaluationSourceContext",
    "ArenaReplayCapability",
    "ArenaRewardConfig",
    "ArenaRewardProjector",
    "ArenaScoreReport",
    "ArenaStepScore",
    "ArenaTargets",
    "ArenaTaskInput",
    "ArenaTaskOutput",
    "ArenaTraceSupplements",
    "TASK_FAMILY",
    "apply_arena_trace_supplements",
    "build_arena_task_corpus",
    "build_arena_trace_supplements",
    "validate_arena_task_root",
    "validate_arena_task_sample",
]
