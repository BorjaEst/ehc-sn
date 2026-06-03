"""Canonical trace-artifact path and meta-key constants.

Every trace/meta path string used across figure selectors, templates,
and reporting code is defined exactly once here.  Selectors, templates,
and the figure registry import from this module rather than duplicating
literal strings.

Consolidated constants (identical string values shared by multiple
regions) are exposed under both a canonical name and region-specific
aliases for backward compatibility.
"""

# ── Shared traces ────────────────────────────────────────────────────────────
TRACE_KEY_Q_VALUES = "value/q_values"
TRACE_KEY_Q_LOGITS = "value/q_logits"

# ── Shared meta ──────────────────────────────────────────────────────────────
META_KEY_ENVIRONMENTS = "environments"

# ── Arena task (environment layout) ──────────────────────────────────────────
ARENA_TRACE_KEY_WALL_MASK = "arena/wall_mask"
ARENA_TRACE_KEY_OBSERVATION_IDS = "arena/observation_ids"
ARENA_TRACE_KEY_TRAJECTORY_LOCATIONS = "arena/trajectory_locations"
ARENA_TRACE_KEY_REVISIT_MASK = "arena/revisit_mask"
ARENA_TRACE_KEY_ACTIONS = "arena/actions"
ARENA_TRACE_KEY_VALID_MASK = "arena/valid_mask"

# ── Arena TEM (prediction streams) ───────────────────────────────────────────
TEM_TRACE_KEY_PRED_INFERENCE = "pred/observation_id/inference"
TEM_TRACE_KEY_PRED_RETRIEVED = "pred/observation_id/retrieved"
TEM_TRACE_KEY_PRED_ANCESTRAL = "pred/observation_id/ancestral"
TEM_META_KEY_TARGET_OBS_ID = "target/observation_id"

# ── MazeHard ─────────────────────────────────────────────────────────────────
MAZEHARD_TRACE_KEY_HALTED = "act/halted"
MAZEHARD_TRACE_KEY_PRED_OVERLAY = "pred/solution_overlay"
MAZEHARD_META_KEY_INPUT_IDS = "input_ids"
MAZEHARD_META_KEY_GT_OVERLAY = "target/solution_overlay"

# ── Shared spatial (world_step/*) ────────────────────────────────────────────
WORLD_TRACE_KEY_LOCATION_IDS = "world_step/location_ids"
WORLD_TRACE_KEY_OBSERVATION = "world_step/observation"


# ── LEC ──────────────────────────────────────────────────────────────────────
LEC_TRACE_KEY_CELLS = "diagnostic/lec/cells"
LEC_TRACE_KEY_FILTERED = "diagnostic/lec/filtered"
LEC_META_KEY_ALPHA = "lec/filter/alpha_sigmoid"
LEC_META_KEY_WF = "lec/w_f_sigmoid"

# ── MEC ──────────────────────────────────────────────────────────────────────
MEC_TRACE_KEY_CELLS = "diagnostic/mec/location_mean"

# ── HPC ──────────────────────────────────────────────────────────────────────
HPC_TRACE_KEY_CELLS = "diagnostic/hpc/location_mean"
HPC_TRACE_KEY_MEMORY = "diagnostic/hpc/memory"
