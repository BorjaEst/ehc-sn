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
TEM_TRACE_KEY_PRED_POST = "pred/observation_id/post"
TEM_TRACE_KEY_PRED_RECALL = "pred/observation_id/recall"
TEM_TRACE_KEY_PRED_PATH = "pred/observation_id/path"
TEM_META_KEY_TARGET_OBS_ID = "target/observation_id"

# ── MazeHard ─────────────────────────────────────────────────────────────────
MAZEHARD_TRACE_KEY_HALTED = "act/halted"
MAZEHARD_TRACE_KEY_PRED_OVERLAY = "pred/solution_overlay"
MAZEHARD_META_KEY_INPUT_IDS = "input_ids"
MAZEHARD_META_KEY_GT_OVERLAY = "target/solution_overlay"
MAZEHARD_META_KEY_CASE_ID = "mazehard/case_id"

# ── PFC / HRM hidden-state diagnostics ───────────────────────────────────────
PFC_TRACE_KEY_Z_H = "pfc/z_H"
PFC_TRACE_KEY_Z_L = "pfc/z_L"

# ── Shared spatial (world_step/*) ────────────────────────────────────────────
WORLD_TRACE_KEY_LOCATION_IDS = "world_step/location_ids"
WORLD_TRACE_KEY_OBSERVATION = "world_step/observation"

# ── LEC ──────────────────────────────────────────────────────────────────────
LEC_TRACE_KEY_CELLS = "diagnostic/lec/cells"
LEC_TRACE_KEY_FILTERED = "diagnostic/lec/filtered"
LEC_TRACE_KEY_SENSORY_CODE = "diagnostic/lec/sensory_code"
LEC_META_KEY_ALPHA = "lec/filter/alpha_sigmoid"
LEC_META_KEY_WF = "lec/w_f_sigmoid"

# ── MEC ──────────────────────────────────────────────────────────────────────
MEC_TRACE_KEY_CELLS = "diagnostic/mec/location_mean"

# ── HPC ──────────────────────────────────────────────────────────────────────
HPC_TRACE_KEY_CELLS = "diagnostic/hpc/location_mean"
HPC_TRACE_KEY_MEMORY = "diagnostic/hpc/memory"

# ── SeqMaze ────────────────────────────────────────────────────────────────
SEQMAZE_META_KEY_TARGET_PATH = "target_path"
SEQMAZE_META_KEY_PATH_MASK = "path_mask"
SEQMAZE_META_KEY_PATH_LENGTH = "path_length"
SEQMAZE_META_KEY_NODE_VALID_MASK = "node_valid_mask"
SEQMAZE_META_KEY_NODE_START_FLAG = "node_start_flag"
SEQMAZE_META_KEY_NODE_GOAL_FLAG = "node_goal_flag"
SEQMAZE_META_KEY_N_NODES = "n_nodes"
SEQMAZE_META_KEY_TARGET_PATH_LEN = "target_path_len"

# ── Goaltrace ────────────────────────────────────────────────────────────────
GOALTRACE_TRACE_KEY_FIRING_FIELD = "goaltrace/firing_field"
GOALTRACE_META_KEY_TARGET_FIELD = "goaltrace/target_field"
GOALTRACE_META_KEY_WEIGHT = "goaltrace/weight"
GOALTRACE_META_KEY_NODE_MASK = "goaltrace/node_mask"
GOALTRACE_META_KEY_OBSERVATION_ID = "goaltrace/observation_id"
GOALTRACE_META_KEY_CURRENT_FLAG = "goaltrace/current_flag"
GOALTRACE_META_KEY_GOAL_FLAG = "goaltrace/goal_flag"
GOALTRACE_META_KEY_SUCCESSOR_INDICES = "goaltrace/successor_indices"
GOALTRACE_META_KEY_SUCCESSOR_MASK = "goaltrace/successor_mask"

# ── Routebind ────────────────────────────────────────────────────────────────
ROUTEBIND_META_KEY_CELL_TYPE = "routebind/cell_type"
ROUTEBIND_META_KEY_OBSERVATION_ID = "routebind/observation_id"
ROUTEBIND_META_KEY_START_FLAG = "routebind/start_flag"
ROUTEBIND_META_KEY_GOAL_FLAG = "routebind/goal_flag"
ROUTEBIND_META_KEY_CELL_MASK = "routebind/cell_mask"
ROUTEBIND_META_KEY_TARGET_TRAJECTORY = "routebind/target_trajectory"
ROUTEBIND_META_KEY_TARGET_WAYPOINT = "routebind/target_waypoint"
ROUTEBIND_META_KEY_N_OBSERVATIONS = "routebind/n_observations"
ROUTEBIND_META_KEY_CANVAS_WIDTH = "routebind/canvas_width"
ROUTEBIND_META_KEY_CANVAS_HEIGHT = "routebind/canvas_height"
