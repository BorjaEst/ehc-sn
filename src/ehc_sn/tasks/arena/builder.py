"""Arena task corpus materialization — v1 (spatial arrays embedded).

Arena replay v1 embeds spatial geometry (topology, observations, mask_valid)
directly in the task corpus alongside trajectory channels.  These arrays are
reconstructed from layout data at build time and stored per-episode so that
evaluation providers and trace builders never need external parent resolution.

Task corpus channels (exact frozen order):

- ``trajectory_row``              — (N, T_max) int32, -1 sentinel on padded steps
- ``trajectory_col``              — (N, T_max) int32, -1 sentinel on padded steps
- ``trajectory_observation_id``   — (N, T_max) int32, precomputed from parent
- ``trajectory_previous_action``  — (N, T_max) int32, step 0 = STAY, -1 on padded
- ``trajectory_landmark_id``      — (N, T_max) int32, -1 when absent or padded
- ``trajectory_is_revisit``       — (N, T_max) bool,  False on padded steps
- ``trajectory_episode_start``    — (N, T_max) bool,  True only at step 0
- ``trajectory_valid_step``       — (N, T_max) bool,  = (t < trajectory_length)
- ``trajectory_length``           — (N,)       int32
- ``topology``                    — (N, H, W) bool,   wall mask
- ``observations``                — (N, H, W) int32,  observation id per cell
- ``mask_valid``                  — (N, H, W) bool,   valid cell mask

Provenance lives in root metadata and per-sample index task_metadata.
ARENA_SPATIAL_CHANNELS is non-empty (topology, observations, mask_valid).

Path written: ``data/processed/arena/<corpus>/v<version>/``

Freeze rule: after this reset (task_protocol_version = 1), any semantic change
to stored channels, shapes, dtypes, sentinels, or to the registered Arena v1
start/walk policy ids is a new protocol version.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Final, Literal, TypeAlias

import numpy as np

from ehc_sn.data.index import read_index
from ehc_sn.data.layout import SpatialLayout, validate_spatial_layout
from ehc_sn.data.lifecycle import (
    extract_version,
    staging_root,
    validate_version_root,
    write_index_at_root,
    write_split,
)
from ehc_sn.data.manifest import write_manifest
from ehc_sn.tasks._replay_build import (
    first_true_cell,
    random_valid_cell,
    random_walk,
    random_walk_no_backtrack,
    random_walk_straight_bias,
)

# =============================================================================
# Protocol constants (frozen for task_protocol_version = 1)
# =============================================================================

TASK_FAMILY: Final[str] = "arena"
TASK_SCHEMA_VERSION: Final[int] = 1
TASK_PROTOCOL_VERSION: Final[int] = 1

ArenaStartPolicy: TypeAlias = Literal["random_valid", "canonical_entrance"]
ArenaWalkPolicy: TypeAlias = Literal[
    "no_immediate_backtrack", "uniform", "legacy_angle_bias"
]
ArenaWalkFn: TypeAlias = Callable[
    ..., tuple[np.ndarray, np.ndarray, np.ndarray]
]

DEFAULT_START_POLICY: Final[ArenaStartPolicy] = "random_valid"
DEFAULT_WALK_POLICY: Final[ArenaWalkPolicy] = "no_immediate_backtrack"
DEFAULT_MAX_STEPS: Final[int] = 250

START_POLICY_RANDOM_VALID_ID: Final[str] = "random_valid_cell_v1"
START_POLICY_CANONICAL_ENTRANCE_ID: Final[str] = (
    "dungeongen_canonical_entrance_v1"
)
WALK_POLICY_NO_IMMEDIATE_BACKTRACK_ID: Final[str] = (
    "random_walk_no_immediate_backtrack_v1"
)
WALK_POLICY_UNIFORM_ID: Final[str] = "random_walk_uniform_v1"

_START_POLICY_ID_BY_NAME: Final[dict[str, str]] = {
    "random_valid": START_POLICY_RANDOM_VALID_ID,
    "canonical_entrance": START_POLICY_CANONICAL_ENTRANCE_ID,
}
WALK_POLICY_LEGACY_ANGLE_BIAS_ID: Final[str] = (
    "random_walk_legacy_angle_bias_v1"
)

_WALK_POLICY_ID_BY_NAME: Final[dict[str, str]] = {
    "no_immediate_backtrack": WALK_POLICY_NO_IMMEDIATE_BACKTRACK_ID,
    "uniform": WALK_POLICY_UNIFORM_ID,
    "legacy_angle_bias": WALK_POLICY_LEGACY_ANGLE_BIAS_ID,
}
_WALK_FUNCTION_BY_NAME: Final[dict[str, ArenaWalkFn]] = {
    "no_immediate_backtrack": random_walk_no_backtrack,
    "uniform": random_walk,
    "legacy_angle_bias": random_walk_straight_bias,
}
_ALLOWED_START_POLICY_IDS: Final[frozenset[str]] = frozenset(
    _START_POLICY_ID_BY_NAME.values()
)
_ALLOWED_WALK_POLICY_IDS: Final[frozenset[str]] = frozenset(
    _WALK_POLICY_ID_BY_NAME.values()
)

_ACTION_DELTAS: Final[tuple[tuple[int, int], ...]] = (
    (0, 0),  # STAY
    (-1, 0),  # UP
    (0, 1),  # RIGHT
    (1, 0),  # DOWN
    (0, -1),  # LEFT
)

# =============================================================================
# Channel registry (exact frozen order — must not be reordered)
# =============================================================================

CHANNEL_TRAJECTORY_ROW: Final[str] = "trajectory_row"
CHANNEL_TRAJECTORY_COL: Final[str] = "trajectory_col"
CHANNEL_TRAJECTORY_OBSERVATION_ID: Final[str] = "trajectory_observation_id"
CHANNEL_TRAJECTORY_PREVIOUS_ACTION: Final[str] = "trajectory_previous_action"
CHANNEL_TRAJECTORY_LANDMARK_ID: Final[str] = "trajectory_landmark_id"
CHANNEL_TRAJECTORY_IS_REVISIT: Final[str] = "trajectory_is_revisit"
CHANNEL_TRAJECTORY_EPISODE_START: Final[str] = "trajectory_episode_start"
CHANNEL_TRAJECTORY_VALID_STEP: Final[str] = "trajectory_valid_step"
CHANNEL_TRAJECTORY_LENGTH: Final[str] = "trajectory_length"

CHANNEL_TOPOLOGY: Final[str] = "topology"
CHANNEL_OBSERVATIONS: Final[str] = "observations"
CHANNEL_MASK_VALID: Final[str] = "mask_valid"

ARENA_TASK_CHANNELS: Final[list[str]] = [
    CHANNEL_TRAJECTORY_ROW,
    CHANNEL_TRAJECTORY_COL,
    CHANNEL_TRAJECTORY_OBSERVATION_ID,
    CHANNEL_TRAJECTORY_PREVIOUS_ACTION,
    CHANNEL_TRAJECTORY_LANDMARK_ID,
    CHANNEL_TRAJECTORY_IS_REVISIT,
    CHANNEL_TRAJECTORY_EPISODE_START,
    CHANNEL_TRAJECTORY_VALID_STEP,
    CHANNEL_TRAJECTORY_LENGTH,
    CHANNEL_TOPOLOGY,
    CHANNEL_OBSERVATIONS,
    CHANNEL_MASK_VALID,
]

ARENA_SPATIAL_CHANNELS: Final[list[str]] = [
    CHANNEL_TOPOLOGY,
    CHANNEL_OBSERVATIONS,
    CHANNEL_MASK_VALID,
]
"""Arena spatial channels embedded in the task corpus for self-contained evaluation."""

_ARENA_TRAJECTORY_DTYPES: Final[dict[str, np.dtype]] = {
    CHANNEL_TRAJECTORY_ROW: np.dtype(np.int32),
    CHANNEL_TRAJECTORY_COL: np.dtype(np.int32),
    CHANNEL_TRAJECTORY_OBSERVATION_ID: np.dtype(np.int32),
    CHANNEL_TRAJECTORY_PREVIOUS_ACTION: np.dtype(np.int32),
    CHANNEL_TRAJECTORY_LANDMARK_ID: np.dtype(np.int32),
    CHANNEL_TRAJECTORY_IS_REVISIT: np.dtype(bool),
    CHANNEL_TRAJECTORY_EPISODE_START: np.dtype(bool),
    CHANNEL_TRAJECTORY_VALID_STEP: np.dtype(bool),
    CHANNEL_TRAJECTORY_LENGTH: np.dtype(np.int32),
    CHANNEL_TOPOLOGY: np.dtype(bool),
    CHANNEL_OBSERVATIONS: np.dtype(np.int32),
    CHANNEL_MASK_VALID: np.dtype(bool),
}

_SPLITS: Final[tuple[str, ...]] = ("train", "val", "test")
_DUNGEONGEN_SPLIT_SEED_OFFSET: Final[dict[str, int]] = {
    "train": 0,
    "val": 100_000,
    "test": 200_000,
}


# =============================================================================
# Walk-seed derivation (frozen)
# =============================================================================


def derive_walk_seed(
    *,
    seed: int,
    split: str,
    parent_sample_id: str,
    episode_index: int,
    start_policy_id: str,
    walk_policy_id: str,
    max_steps: int,
) -> int:
    """Derive a deterministic walk seed from episode provenance.

    Computes SHA-256 over the canonical JSON of the provenance dict (keys
    sorted), takes the first 8 bytes, and interprets them as unsigned
    little-endian int64.

    Returns:
        Unsigned 64-bit integer seed.
    """
    canonical = json.dumps(
        {
            "episode_index": episode_index,
            "max_steps": max_steps,
            "parent_sample_id": parent_sample_id,
            "seed": seed,
            "split": split,
            "start_policy_id": start_policy_id,
            "walk_policy_id": walk_policy_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(canonical).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


# =============================================================================
# Single-episode builder
# =============================================================================


def _build_episode_from_layout(
    layout: SpatialLayout,
    max_steps: int,
    walk_seed: int,
    *,
    start_cell: tuple[int, int] | None = None,
    walk_fn=random_walk_no_backtrack,
    target_shape: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """Build one Arena episode from an openfield :class:`SpatialLayout`.

    Constructs a 2-D ``mask_valid`` from the layout's row/col mapping and
    ``valid_state_mask``, then samples a walk using the same protocol as
    dungeongen-based :func:`_build_episode`.

    Args:
        layout: Spatial layout record (validated).
        max_steps: Number of steps in the trajectory.
        walk_seed: Deterministic RNG seed for this episode.
        target_shape: Optional ``(H, W)`` to pad spatial arrays to.
            When ``None``, spatial arrays use the layout's native size.

    Returns:
        Dict of Arena channel arrays (no padding — caller stacks later).
    """
    rng = np.random.default_rng(np.uint64(walk_seed))
    N = layout["graph_state_count"]
    row_col = layout["state_to_row_col"]
    valid_mask = layout["valid_state_mask"]
    obs_ids_src = layout["observation_id"]
    as_ = layout["action_space"]
    deltas = as_["action_deltas"]
    stay_a = as_["stay_action"] if as_["stay_action"] is not None else -1

    # Build a 2-D mask_valid for the walk function.
    max_row = int(row_col[:, 0].max()) + 1
    max_col = int(row_col[:, 1].max()) + 1
    mask_valid_2d = np.zeros((max_row, max_col), dtype=bool)
    for s in range(N):
        if valid_mask[s]:
            r, c = int(row_col[s, 0]), int(row_col[s, 1])
            mask_valid_2d[r, c] = True

    if start_cell is None:
        start_r, start_c = random_valid_cell(mask_valid_2d, rng)
    else:
        start_r, start_c = start_cell

    rows, cols, prev_actions = walk_fn(
        mask_valid_2d,
        (start_r, start_c),
        max_steps,
        rng=rng,
        action_deltas=tuple(deltas),
        stay_action=stay_a,
    )

    # Map (row, col) → observation_id via layout's state index.
    obs_ids = np.zeros(max_steps, dtype=np.int32)
    lm_ids = np.full(
        max_steps, -1, dtype=np.int32
    )  # openfield has no landmarks
    for t in range(max_steps):
        r, c = int(rows[t]), int(cols[t])
        # Find the state index for this (row, col).
        match = np.where((row_col[:, 0] == r) & (row_col[:, 1] == c))[0]
        if len(match) > 0:
            obs_ids[t] = obs_ids_src[match[0]]
        else:
            obs_ids[t] = -1  # sentinel (should not happen for valid walks)

    # Revisit flags.
    visited: set[tuple[int, int]] = set()
    is_revisit = np.zeros(max_steps, dtype=bool)
    for t in range(max_steps):
        cell = (int(rows[t]), int(cols[t]))
        is_revisit[t] = cell in visited
        visited.add(cell)

    episode_start = np.zeros(max_steps, dtype=bool)
    episode_start[0] = True
    valid_step = np.ones(max_steps, dtype=bool)
    traj_length = np.int32(max_steps)

    # Build grid-space spatial arrays from layout data.
    # topology: (H, W) bool — passable cells = True.
    # observations: (H, W) int32 — observation id per cell, -1 for walls.
    # mask_valid: (H, W) bool — valid-state mask (same as mask_valid_2d).
    if target_shape is not None:
        th, tw = target_shape
        topology = np.zeros((th, tw), dtype=bool)
        observations = np.full((th, tw), -1, dtype=np.int32)
        topology[:max_row, :max_col] = mask_valid_2d
        observations[:max_row, :max_col] = -1
        m_valid_pad = np.zeros((th, tw), dtype=bool)
        m_valid_pad[:max_row, :max_col] = mask_valid_2d
    else:
        topology = np.zeros((max_row, max_col), dtype=bool)
        observations = np.full((max_row, max_col), -1, dtype=np.int32)
        m_valid_pad = mask_valid_2d
    for s in range(N):
        r, c = int(row_col[s, 0]), int(row_col[s, 1])
        topology[r, c] = valid_mask[s]
        observations[r, c] = obs_ids_src[s]

    return {
        CHANNEL_TRAJECTORY_ROW: rows,
        CHANNEL_TRAJECTORY_COL: cols,
        CHANNEL_TRAJECTORY_OBSERVATION_ID: obs_ids,
        CHANNEL_TRAJECTORY_PREVIOUS_ACTION: prev_actions,
        CHANNEL_TRAJECTORY_LANDMARK_ID: lm_ids,
        CHANNEL_TRAJECTORY_IS_REVISIT: is_revisit,
        CHANNEL_TRAJECTORY_EPISODE_START: episode_start,
        CHANNEL_TRAJECTORY_VALID_STEP: valid_step,
        CHANNEL_TRAJECTORY_LENGTH: np.array(traj_length, dtype=np.int32),
        CHANNEL_TOPOLOGY: topology,
        CHANNEL_OBSERVATIONS: observations,
        CHANNEL_MASK_VALID: m_valid_pad,
    }


def _resolve_start_policy_id(start_policy: str) -> str:
    """Resolve one public start-policy name to its frozen manifest id."""
    policy_id = _START_POLICY_ID_BY_NAME.get(start_policy)
    if policy_id is None:
        raise ValueError(
            f"Unsupported Arena start_policy {start_policy!r}. "
            f"Expected one of {sorted(_START_POLICY_ID_BY_NAME)}."
        )
    return policy_id


def _resolve_walk_policy_id(walk_policy: str) -> str:
    """Resolve one public walk-policy name to its frozen manifest id."""
    policy_id = _WALK_POLICY_ID_BY_NAME.get(walk_policy)
    if policy_id is None:
        raise ValueError(
            f"Unsupported Arena walk_policy {walk_policy!r}. "
            f"Expected one of {sorted(_WALK_POLICY_ID_BY_NAME)}."
        )
    return policy_id


def _resolve_walk_function(walk_policy: str) -> ArenaWalkFn:
    """Resolve one public walk-policy name to its trajectory sampler."""
    walk_fn = _WALK_FUNCTION_BY_NAME.get(walk_policy)
    if walk_fn is None:
        raise ValueError(
            f"Unsupported Arena walk_policy {walk_policy!r}. "
            f"Expected one of {sorted(_WALK_FUNCTION_BY_NAME)}."
        )
    return walk_fn


# =============================================================================
# Validation
# =============================================================================

# STAY action id (used by the per-sample validator below).
_ACTION_STAY: int = 0


def _find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` to find the repo root (contains pyproject.toml)."""
    current = start.resolve()
    for _ in range(20):
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()


def validate_arena_task_sample(data: dict[str, np.ndarray]) -> None:
    """Validate an Arena task corpus sample.

    Checks:
    - All required channels present.
    - trajectory_valid_step == (t < trajectory_length).
    - trajectory_episode_start True only at step 0.
    - trajectory_previous_action[0] == STAY.
    - Invalid padded positions use frozen sentinels.

    Raises:
        ValueError: On any contract violation.
    """
    missing = set(ARENA_TASK_CHANNELS) - data.keys()
    if missing:
        raise ValueError(
            f"Arena task sample missing channels: {sorted(missing)}"
        )

    length = int(np.asarray(data[CHANNEL_TRAJECTORY_LENGTH]).flat[0])
    valid_step = data[CHANNEL_TRAJECTORY_VALID_STEP]
    T = valid_step.shape[-1]
    expected_valid = np.arange(T) < length
    if not np.array_equal(valid_step, expected_valid):
        raise ValueError(
            "trajectory_valid_step must equal (t < trajectory_length)."
        )

    episode_start = data[CHANNEL_TRAJECTORY_EPISODE_START]
    if not bool(episode_start.flat[0]):
        raise ValueError("trajectory_episode_start[0] must be True.")
    if T > 1 and episode_start[1:].any():
        raise ValueError(
            "trajectory_episode_start must be False at all steps except 0."
        )

    prev_action = data[CHANNEL_TRAJECTORY_PREVIOUS_ACTION]
    if int(prev_action.flat[0]) != _ACTION_STAY:
        raise ValueError(
            f"trajectory_previous_action[0] must be STAY ({_ACTION_STAY})."
        )

    # Check padded sentinels.
    if length < T:
        for ch, sentinel in [
            (CHANNEL_TRAJECTORY_ROW, -1),
            (CHANNEL_TRAJECTORY_COL, -1),
            (CHANNEL_TRAJECTORY_OBSERVATION_ID, -1),
            (CHANNEL_TRAJECTORY_PREVIOUS_ACTION, -1),
            (CHANNEL_TRAJECTORY_LANDMARK_ID, -1),
        ]:
            arr = data[ch]
            if not np.all(arr[length:] == sentinel):
                raise ValueError(
                    f"{ch}: padded positions (t >= {length}) must be {sentinel}."
                )
        for ch in (
            CHANNEL_TRAJECTORY_IS_REVISIT,
            CHANNEL_TRAJECTORY_EPISODE_START,
            CHANNEL_TRAJECTORY_VALID_STEP,
        ):
            arr = data[ch]
            if arr[length:].any():
                raise ValueError(
                    f"{ch}: padded positions (t >= {length}) must be False."
                )


def validate_arena_task_root(
    root: Path, *, _repo_root: Path | None = None
) -> dict:
    """Validate an Arena task corpus root against task-owned semantics.

    Checks required manifest constants, channel array shapes/dtypes, per-sample
    invariants, and parent substrate lineage.

    Args:
        root: Arena task corpus version root.
        _repo_root: Override repo root used to resolve parent_substrate paths.
            Defaults to auto-detection via pyproject.toml walk. Primarily for
            testing with temporary directories.

    Returns:
        Parsed manifest dict.

    Raises:
        ValueError: On any contract violation.
        FileNotFoundError: When a required file or parent index is absent.
    """
    manifest = validate_version_root(root)
    if manifest.get("dataset_class") != "task_corpus":
        raise ValueError("Root is not a task_corpus.")
    if manifest.get("task") != TASK_FAMILY:
        raise ValueError(
            f"Root task is {manifest.get('task')!r}, expected {TASK_FAMILY!r}."
        )
    # Remove the guard that rejected spatial channels (they are now populated).

    # Validate required manifest constants.
    # action_count and parent_family are layout-dependent, not global constants.
    _required_manifest_constants = {
        "task_schema_version": TASK_SCHEMA_VERSION,
        "task_protocol_version": TASK_PROTOCOL_VERSION,
        "store_row_col": True,
    }
    for field, expected in _required_manifest_constants.items():
        actual = manifest.get(field)
        if actual != expected:
            raise ValueError(
                f"Manifest field {field!r}: expected {expected!r}, got {actual!r}."
            )
    # store_landmark_id must be present (layout-dependent).
    sli = manifest.get("store_landmark_id")
    if not isinstance(sli, bool):
        raise ValueError(
            f"Manifest 'store_landmark_id' must be a bool, got {sli!r}."
        )
    # action_count must be present and positive (value is layout-dependent).
    act_count = manifest.get("action_count")
    if not isinstance(act_count, int) or act_count < 1:
        raise ValueError(
            f"Manifest 'action_count' must be a positive int, got {act_count!r}."
        )
    # parent_family must be present (value is layout-dependent).
    parent_family = manifest.get("parent_family")
    if not isinstance(parent_family, str) or not parent_family:
        raise ValueError(
            f"Manifest 'parent_family' must be a non-empty string, got {parent_family!r}."
        )
    start_policy_id = manifest.get("start_policy_id")
    if start_policy_id not in _ALLOWED_START_POLICY_IDS:
        raise ValueError(
            "Manifest field 'start_policy_id': expected one of "
            f"{sorted(_ALLOWED_START_POLICY_IDS)!r}, got {start_policy_id!r}."
        )
    walk_policy_id = manifest.get("walk_policy_id")
    if walk_policy_id not in _ALLOWED_WALK_POLICY_IDS:
        raise ValueError(
            "Manifest field 'walk_policy_id': expected one of "
            f"{sorted(_ALLOWED_WALK_POLICY_IDS)!r}, got {walk_policy_id!r}."
        )
    if "observation_vocab_size" not in manifest:
        raise ValueError(
            "Manifest missing required field 'observation_vocab_size'."
        )
    obs_vocab = manifest["observation_vocab_size"]
    if not isinstance(obs_vocab, int) or obs_vocab < 1:
        raise ValueError(
            f"Manifest 'observation_vocab_size' must be a positive int, got {obs_vocab!r}."
        )

    # Spatial channels are (N, H, W); all others are (N, T) or (N,) for length.
    _SPATIAL_CHANNELS = {
        CHANNEL_TOPOLOGY,
        CHANNEL_OBSERVATIONS,
        CHANNEL_MASK_VALID,
    }

    # Validate per-split channel arrays.
    for split, n in manifest["n_samples"].items():
        split_dir = root / split
        arrays: dict[str, np.ndarray] = {}
        for ch in ARENA_TASK_CHANNELS:
            ch_file = split_dir / f"{ch}.npy"
            if not ch_file.exists():
                raise FileNotFoundError(
                    f"Missing task channel '{ch}' in {split_dir}."
                )
            arrays[ch] = np.load(ch_file, mmap_mode="r")
            expected_dtype = _ARENA_TRAJECTORY_DTYPES[ch]
            if arrays[ch].dtype != expected_dtype:
                raise ValueError(
                    f"Channel '{ch}' in split '{split}' has dtype {arrays[ch].dtype}, "
                    f"expected {expected_dtype}."
                )
            if ch in _SPATIAL_CHANNELS:
                expected_ndim = 3
            else:
                expected_ndim = 1 if ch == CHANNEL_TRAJECTORY_LENGTH else 2
            if arrays[ch].ndim != expected_ndim:
                raise ValueError(
                    f"Channel '{ch}' in split '{split}' has rank {arrays[ch].ndim}, "
                    f"expected {expected_ndim}."
                )
            if arrays[ch].shape[0] != n:
                raise ValueError(
                    f"Channel '{ch}' in split '{split}' has {arrays[ch].shape[0]} samples, "
                    f"manifest declares {n}."
                )

        for i in range(n):
            validate_arena_task_sample(
                {ch: arrays[ch][i] for ch in ARENA_TASK_CHANNELS}
            )

    # Validate parent substrate lineage via per-sample index metadata.
    all_entries = read_index(root / "index.jsonl")
    parent_substrate_rel = manifest.get("parent_substrate")
    if not parent_substrate_rel:
        raise ValueError("Manifest missing required field 'parent_substrate'.")

    resolved_repo_root = (
        _repo_root if _repo_root is not None else _find_repo_root(root)
    )
    parent_index_path = (
        resolved_repo_root / parent_substrate_rel / "index.jsonl"
    )

    if not parent_index_path.exists():
        # Layout sources without a parent shared substrate (e.g. openfield)
        # do not have a parent index to validate against.  parent_substrate
        # is provenance metadata only — skip lineage checks.
        return manifest

    parent_all_entries = read_index(parent_index_path)
    # Build a split-keyed lookup so validation is split-matched (a train Arena
    # sample must only reference a train parent id, etc.).
    parent_ids_by_split: dict[str, set[str]] = {}
    for e in parent_all_entries:
        parent_ids_by_split.setdefault(e.split, set()).add(e.id)

    for split in manifest["n_samples"]:
        split_entries = [e for e in all_entries if e.split == split]
        valid_parent_ids = parent_ids_by_split.get(split, set())
        episode_indices_by_parent: dict[str, set[int]] = {}
        walk_seeds_by_parent: dict[str, set[int]] = {}
        for entry in split_entries:
            meta = entry.task_metadata
            if meta is None:
                raise ValueError(
                    f"Arena index entry {entry.id!r} missing task_metadata."
                )
            pid = meta["parent_sample_id"]
            ep_idx = meta["episode_index"]
            wseed = meta["walk_seed"]
            if pid not in valid_parent_ids:
                raise ValueError(
                    f"Arena entry {entry.id!r}: parent_sample_id {pid!r} not found "
                    f"in parent index split '{split}'."
                )
            ep_set = episode_indices_by_parent.setdefault(pid, set())
            if ep_idx in ep_set:
                raise ValueError(
                    f"Duplicate episode_index {ep_idx} for parent_sample_id {pid!r} in split {split!r}."
                )
            ep_set.add(ep_idx)
            ws_set = walk_seeds_by_parent.setdefault(pid, set())
            if wseed in ws_set:
                raise ValueError(
                    f"Duplicate walk_seed {wseed} for parent_sample_id {pid!r} in split {split!r}."
                )
            ws_set.add(wseed)

    return manifest


# =============================================================================
# Unified arena builder (consumes any list[SpatialLayout])
# =============================================================================


def build_arena_task_corpus(
    version_root: Path,
    *,
    layouts: list[SpatialLayout],
    corpus: str = "default",
    walk_policy: ArenaWalkPolicy = "legacy_angle_bias",
    n_episodes_per_layout: int = 10,
    max_steps: int = 250,
    seed: int = 42,
) -> None:
    """Build an Arena task corpus from :class:`SpatialLayout` records.

    Three-seed hierarchy: each layout (topology + sensory assignment) produces
    ``n_episodes_per_layout`` walk episodes, each with a different ``walk_seed``.

    This is the single unified entry point for any layout source.  Source-
    specific builders (dungeongen, openfield) should produce
    ``list[SpatialLayout]`` and then call this function.

    Args:
        version_root: Destination versioned root.
        layouts: Validated spatial layout records.
        corpus: Corpus label (e.g. ``"dungeons"``, ``"openfield-square"``).
        walk_policy: Walk policy name (default ``\"legacy_angle_bias\"``).
        n_episodes_per_layout: Walk episodes per layout instance.
        max_steps: Trajectory length.
        seed: Base RNG seed for walk-seed derivation.

    Raises:
        FileExistsError: When *version_root* already exists.
    """
    walk_policy_id = _resolve_walk_policy_id(walk_policy)
    walk_fn = _resolve_walk_function(walk_policy)

    version = extract_version(version_root)
    for layout in layouts:
        validate_spatial_layout(layout)

    action_count = layouts[0]["action_space"]["action_count"]
    action_id_space_str = ",".join(
        f"{n}={i}"
        for i, n in enumerate(layouts[0]["action_space"]["action_names"])
    )
    sensory_vocab_size = layouts[0]["sensory_vocab_size"]
    topology_type = layouts[0]["topology_type"]
    layout_family = layouts[0]["layout_family"]

    # Group layouts by split (default to "train" when absent).
    layout_splits: dict[str, list[tuple[int, SpatialLayout]]] = {}
    for li, layout in enumerate(layouts):
        split = layout.get("split", "train")
        layout_splits.setdefault(split, []).append((li, layout))

    total_episodes = len(layouts) * n_episodes_per_layout
    split_counts = {
        split: len(group) * n_episodes_per_layout
        for split, group in layout_splits.items()
    }

    stage_params = {
        "corpus": corpus,
        "max_steps": max_steps,
        "seed": seed,
        "walk_policy": walk_policy,
        "n_episodes_per_layout": n_episodes_per_layout,
        "n_layouts": len(layouts),
        "walk_policy_id": walk_policy_id,
    }

    with staging_root(version_root) as tmp:
        # Compute the maximum grid extent across all layouts for padding
        # spatial arrays to a uniform shape.
        max_grid_h = 0
        max_grid_w = 0
        for layout in layouts:
            rc = layout["state_to_row_col"]
            h = int(rc[:, 0].max()) + 1
            w = int(rc[:, 1].max()) + 1
            max_grid_h = max(max_grid_h, h)
            max_grid_w = max(max_grid_w, w)
        target_shape = (max_grid_h, max_grid_w)

        all_entries: list = []

        for split in sorted(layout_splits):
            samples: list[dict[str, np.ndarray]] = []
            per_sample_ids: list[str] = []
            per_sample_extra: list[dict] = []
            split_layouts = layout_splits[split]

            for li, layout in split_layouts:
                start_cell = None  # random uniformly from valid states
                for ep_idx in range(n_episodes_per_layout):
                    walk_seed = derive_walk_seed(
                        seed=seed,
                        split=split,
                        parent_sample_id=layout["layout_id"],
                        episode_index=ep_idx,
                        start_policy_id=START_POLICY_RANDOM_VALID_ID,
                        walk_policy_id=walk_policy_id,
                        max_steps=max_steps,
                    )
                    episode = _build_episode_from_layout(
                        layout,
                        max_steps,
                        walk_seed,
                        start_cell=start_cell,
                        walk_fn=walk_fn,
                        target_shape=target_shape,
                    )
                    samples.append(episode)
                    per_sample_ids.append(
                        f"arena-{corpus}-{li:04d}-ep{ep_idx:04d}"
                    )
                    per_sample_extra.append(
                        {
                            "task_metadata": {
                                "layout_instance_id": layout["layout_id"],
                                "topology_type": topology_type,
                                "topology_seed": layout["topology_seed"],
                                "sensory_seed": layout["sensory_seed"],
                                "episode_index": ep_idx,
                                "walk_seed": walk_seed,
                            }
                        }
                    )

            entries = write_split(
                tmp,
                split,
                samples,
                source=TASK_FAMILY,
                channels=ARENA_TASK_CHANNELS,
                topology_kind=topology_type,
                n_states=layouts[0]["graph_state_count"],
                extent=[max_grid_h],
                index_kwargs={},
                per_sample_ids=per_sample_ids,
                per_sample_extra=per_sample_extra,
                sample_validator=validate_arena_task_sample,
            )
            all_entries.extend(entries)

        write_index_at_root(all_entries, tmp)

        write_manifest(
            tmp,
            dataset_class="task_corpus",
            family=TASK_FAMILY,
            version=version,
            channels=ARENA_TASK_CHANNELS,
            topology_kind=topology_type,
            n_states=layouts[0]["graph_state_count"],
            extent=[int(layouts[0]["state_to_row_col"][:, 0].max()) + 1],
            n_samples=split_counts,
            source_id=layout_family,
            builder="ehc_sn.tasks.arena.build_arena_task_corpus",
            seed=seed,
            stage_params=stage_params,
            parent_family=layout_family,
            parent_version="1",
            task_schema_version=TASK_SCHEMA_VERSION,
            task_protocol_version=TASK_PROTOCOL_VERSION,
            task=TASK_FAMILY,
            corpus=corpus,
            parent_substrate=f"data/processed/{layout_family}/v1",
            start_policy_id=START_POLICY_RANDOM_VALID_ID,
            walk_policy_id=walk_policy_id,
            action_count=action_count,
            action_id_space=action_id_space_str,
            store_row_col=True,
            store_landmark_id=False,
            observation_vocab_size=sensory_vocab_size,
        )

    n_total = sum(split_counts.values())
    print(
        f"Arena corpus written to {version_root}  "
        f"({n_total} samples, {len(layouts)} layouts, "
        f"{n_episodes_per_layout} episodes/layout)."
    )
