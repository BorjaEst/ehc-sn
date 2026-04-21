"""Arena task family — stepwise teacher-forced replay for structural-knowledge claims.

Arena is the canonical task/runtime family backing B1-style structural-knowledge
benchmark reports.  It is stepwise, teacher-forced, and recurrent.
Whole-trajectory model inputs are forbidden (REQ-001, SCI-001).

Benchmark evaluation consumes :class:`ArenaStructuralScore`.
RL training may opt in to :class:`ArenaRewardConfig` (disabled by default, REQ-004).
"""

from .contracts import ARENA_ACTION_COUNT, ArenaAction, ArenaTargets, ArenaTaskInput, ArenaTaskOutput
from .evaluation import (
    ArenaEpisodeSemantics,
    ArenaPathwayMetrics,
    ArenaStructuralScore,
    coerce_observation_ids,
    coerce_revisit_mask,
    compute_arena_structural_score,
    evaluate_observation_logits,
    extract_arena_episode_semantics,
)
from .reward import ArenaRewardConfig, ArenaRewardMode, compute_arena_reward
from .runtime import (
    ARENA_REPLAY_OPTIONAL_KEYS,
    ARENA_REPLAY_REQUIRED_KEYS,
    ARENA_STEP_KEYS,
    ArenaReplayTrajectoryRuntime,
    annotate_arena_revisit_state,
    batch_size_from_arena_batch,
    coerce_arena_step_input,
    coerce_arena_targets,
    extract_arena_step_tensors,
    infer_arena_replay_batch_keys,
    new_arena_visit_counts,
    record_arena_visit,
)

__all__ = [
    "ARENA_ACTION_COUNT",
    "ARENA_REPLAY_OPTIONAL_KEYS",
    "ARENA_REPLAY_REQUIRED_KEYS",
    "ARENA_STEP_KEYS",
    "ArenaAction",
    "ArenaEpisodeSemantics",
    "ArenaPathwayMetrics",
    "ArenaReplayTrajectoryRuntime",
    "ArenaRewardConfig",
    "ArenaRewardMode",
    "ArenaStructuralScore",
    "ArenaTargets",
    "ArenaTaskInput",
    "ArenaTaskOutput",
    "annotate_arena_revisit_state",
    "batch_size_from_arena_batch",
    "coerce_arena_step_input",
    "coerce_arena_targets",
    "coerce_observation_ids",
    "coerce_revisit_mask",
    "compute_arena_reward",
    "compute_arena_structural_score",
    "evaluate_observation_logits",
    "extract_arena_episode_semantics",
    "extract_arena_step_tensors",
    "infer_arena_replay_batch_keys",
    "new_arena_visit_counts",
    "record_arena_visit",
]
