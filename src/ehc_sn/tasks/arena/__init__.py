"""Arena task family — structural navigation over a maze world.

Arena defines the observation/action ontology, revisit semantics, and additive
structural score for maze-world navigation.  These are task-owned semantics;
they do not depend on any particular execution mode.

Canonical task surfaces:

- :mod:`~ehc_sn.tasks.arena.contracts` — observation, action, and score contracts
  (including :class:`ArenaStructuralScore`, the benchmark-facing score type).
- :mod:`~ehc_sn.tasks.arena.evaluation` — structural score and evaluation helpers.
- :mod:`~ehc_sn.tasks.arena.batch` — task-level batch coercion helpers.

Mode bindings (execution-mode, not task identity):

- :mod:`~ehc_sn.tasks.arena.modes` — replay and other mode bindings.

Benchmark evaluation consumes :class:`ArenaStructuralScore` directly from the
task-owned contracts surface.  Whole-trajectory model inputs are forbidden
(REQ-001, SCI-001).
"""

from . import batch, contracts, evaluation, modes  # noqa: F401
from .contracts import ARENA_ACTION_COUNT, ArenaAction, ArenaStructuralScore, ArenaTargets, ArenaTaskInput, ArenaTaskOutput
from .evaluation import (
    ArenaEpisodeSemantics,
    ArenaPathwayMetrics,
    coerce_observation_ids,
    coerce_revisit_mask,
    compute_arena_structural_score,
    evaluate_observation_logits,
    extract_arena_episode_semantics,
)

__all__ = [
    "ARENA_ACTION_COUNT",
    "ArenaAction",
    "ArenaEpisodeSemantics",
    "ArenaPathwayMetrics",
    "ArenaStructuralScore",
    "ArenaTargets",
    "ArenaTaskInput",
    "ArenaTaskOutput",
    "batch",
    "contracts",
    "coerce_observation_ids",
    "coerce_revisit_mask",
    "compute_arena_structural_score",
    "evaluate_observation_logits",
    "evaluation",
    "extract_arena_episode_semantics",
    "modes",
]
