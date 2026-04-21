"""Dungeon task family — reward-first online control.

Dungeon is the canonical task/runtime family backing B1-B3-style online
control benchmark reports.  Success semantics are goal/return-driven (REQ-002).
Token-supervision targets are not defined in dungeon v1.
"""

from .contracts import DUNGEON_ACTION_COUNT, DungeonAction, DungeonObservation, DungeonScoreReport, DungeonTaskInput
from .evaluation import DungeonEpisodeResult, compute_dungeon_episode_score
from .reward import DungeonRewardConfig, DungeonRewardMode
from .runtime import (
    DUNGEON_RESET_OPTIONAL_KEYS,
    DUNGEON_RESET_REQUIRED_KEYS,
    DUNGEON_STEP_KEYS,
    DungeonControllerRuntime,
    annotate_dungeon_revisit_state,
    batch_size_from_dungeon_batch,
    build_dungeon_reset_td,
    coerce_dungeon_step_input,
    extract_dungeon_step_data,
    infer_dungeon_static_batch_keys,
    make_dungeon_policy_input,
    new_dungeon_visit_counts,
    record_dungeon_visit,
)

__all__ = [
    "DUNGEON_ACTION_COUNT",
    "DUNGEON_RESET_OPTIONAL_KEYS",
    "DUNGEON_RESET_REQUIRED_KEYS",
    "DUNGEON_STEP_KEYS",
    "DungeonAction",
    "DungeonControllerRuntime",
    "DungeonEpisodeResult",
    "DungeonObservation",
    "DungeonRewardConfig",
    "DungeonRewardMode",
    "DungeonScoreReport",
    "DungeonTaskInput",
    "annotate_dungeon_revisit_state",
    "batch_size_from_dungeon_batch",
    "build_dungeon_reset_td",
    "coerce_dungeon_step_input",
    "compute_dungeon_episode_score",
    "extract_dungeon_step_data",
    "infer_dungeon_static_batch_keys",
    "make_dungeon_policy_input",
    "new_dungeon_visit_counts",
    "record_dungeon_visit",
]
