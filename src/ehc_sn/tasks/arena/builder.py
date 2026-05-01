"""Arena task corpus materialization — v1 (topology-free, frozen).

Arena replay v1 is topology-free inside the Arena task corpus.  The parent
dungeongen substrate is the sole owner of spatial geometry.  One stored Arena
sample equals one episode.  Training uses a fixed offline corpus only.

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

Provenance lives in root metadata and per-sample index task_metadata.
ARENA_SPATIAL_CHANNELS is empty.

Path written: ``data/processed/arena/<corpus>/v<version>/``

Freeze rule: after this reset (task_protocol_version = 1), any semantic change
to stored channels, shapes, dtypes, sentinels, or walk/start policies is a new
protocol version.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

import numpy as np

from ehc_sn.data.index import read_index
from ehc_sn.data.lifecycle import extract_version, staging_root, validate_version_root, write_index_at_root, write_split
from ehc_sn.data.manifest import write_manifest
from ehc_sn.data.substrate.dungeongen import SHARED_CHANNELS as DUNGEON_SUBSTRATE_CHANNELS
from ehc_sn.data.substrate.dungeongen import SHARED_FAMILY as DUNGEON_SHARED_FAMILY
from ehc_sn.data.substrate.reader import iter_substrate_entries_and_samples, load_substrate_manifest
from ehc_sn.tasks._replay_build import random_valid_cell, random_walk_no_backtrack

# =============================================================================
# Protocol constants (frozen for task_protocol_version = 1)
# =============================================================================

TASK_FAMILY: Final[str] = "arena"
TASK_SCHEMA_VERSION: Final[int] = 1
TASK_PROTOCOL_VERSION: Final[int] = 1

START_POLICY_ID: Final[str] = "random_valid_cell_v1"
WALK_POLICY_ID: Final[str] = "random_walk_no_immediate_backtrack_v1"

ACTION_COUNT: Final[int] = 5
ACTION_ID_SPACE: Final[str] = "STAY=0,UP=1,RIGHT=2,DOWN=3,LEFT=4"

_ACTION_STAY: Final[int] = 0
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
]

ARENA_SPATIAL_CHANNELS: Final[list[str]] = []
"""Arena replay v1 carries no spatial channels. Parent substrate owns all geometry."""

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
}

_SPLITS: Final[tuple[str, ...]] = ("train", "val", "test")


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


def _build_episode(
    parent_sample: dict[str, np.ndarray],
    max_steps: int,
    walk_seed: int,
) -> dict[str, np.ndarray]:
    """Build one Arena episode from a parent substrate sample.

    Args:
        parent_sample: Dict with at least ``mask_valid``, ``observations``,
            and ``landmarks`` from the dungeongen substrate.
        max_steps: Number of steps in the trajectory.
        walk_seed: Deterministic RNG seed for this episode.

    Returns:
        Dict of Arena channel arrays (no padding — caller stacks later).

    Raises:
        RuntimeError: When ``mask_valid`` has no passable cells.
    """
    rng = np.random.default_rng(np.uint64(walk_seed))
    mask_valid: np.ndarray = parent_sample["mask_valid"]
    observations: np.ndarray = parent_sample["observations"]
    landmarks: np.ndarray = parent_sample["landmarks"]

    start_r, start_c = random_valid_cell(mask_valid, rng)

    rows, cols, prev_actions = random_walk_no_backtrack(
        mask_valid,
        (start_r, start_c),
        max_steps,
        rng=rng,
        action_deltas=_ACTION_DELTAS,
        stay_action=_ACTION_STAY,
    )

    # Precompute model-visible ids from parent spatial maps.
    obs_ids = np.array([observations[r, c] for r, c in zip(rows, cols)], dtype=np.int32)
    lm_ids = np.array([landmarks[r, c] for r, c in zip(rows, cols)], dtype=np.int32)
    # Normalize landmark sentinel: dungeongen uses 0 for "no landmark"; Arena v1 uses -1.
    lm_ids[lm_ids == 0] = -1

    # Precompute revisit flags (True iff state at step t appeared earlier).
    visited: set[tuple[int, int]] = set()
    is_revisit = np.zeros(max_steps, dtype=bool)
    for t in range(max_steps):
        cell = (int(rows[t]), int(cols[t]))
        is_revisit[t] = cell in visited
        visited.add(cell)

    episode_start = np.zeros(max_steps, dtype=bool)
    episode_start[0] = True
    valid_step = np.ones(max_steps, dtype=bool)  # no padding; caller handles T_max
    traj_length = np.int32(max_steps)

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
    }


# =============================================================================
# Validation
# =============================================================================


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
        raise ValueError(f"Arena task sample missing channels: {sorted(missing)}")

    length = int(np.asarray(data[CHANNEL_TRAJECTORY_LENGTH]).flat[0])
    valid_step = data[CHANNEL_TRAJECTORY_VALID_STEP]
    T = valid_step.shape[-1]
    expected_valid = np.arange(T) < length
    if not np.array_equal(valid_step, expected_valid):
        raise ValueError("trajectory_valid_step must equal (t < trajectory_length).")

    episode_start = data[CHANNEL_TRAJECTORY_EPISODE_START]
    if not bool(episode_start.flat[0]):
        raise ValueError("trajectory_episode_start[0] must be True.")
    if T > 1 and episode_start[1:].any():
        raise ValueError("trajectory_episode_start must be False at all steps except 0.")

    prev_action = data[CHANNEL_TRAJECTORY_PREVIOUS_ACTION]
    if int(prev_action.flat[0]) != _ACTION_STAY:
        raise ValueError(f"trajectory_previous_action[0] must be STAY ({_ACTION_STAY}).")

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
                raise ValueError(f"{ch}: padded positions (t >= {length}) must be {sentinel}.")
        for ch in (CHANNEL_TRAJECTORY_IS_REVISIT, CHANNEL_TRAJECTORY_EPISODE_START, CHANNEL_TRAJECTORY_VALID_STEP):
            arr = data[ch]
            if arr[length:].any():
                raise ValueError(f"{ch}: padded positions (t >= {length}) must be False.")


def validate_arena_task_root(root: Path, *, _repo_root: Path | None = None) -> dict:
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
        raise ValueError(f"Root task is {manifest.get('task')!r}, expected {TASK_FAMILY!r}.")
    if ARENA_SPATIAL_CHANNELS:
        raise ValueError("ARENA_SPATIAL_CHANNELS must be empty for Arena v1.")

    # Validate required manifest constants.
    _required_manifest_constants = {
        "task_schema_version": TASK_SCHEMA_VERSION,
        "task_protocol_version": TASK_PROTOCOL_VERSION,
        "parent_family": DUNGEON_SHARED_FAMILY,
        "start_policy_id": START_POLICY_ID,
        "walk_policy_id": WALK_POLICY_ID,
        "action_count": ACTION_COUNT,
        "action_id_space": ACTION_ID_SPACE,
        "store_row_col": True,
        "store_landmark_id": True,
    }
    for field, expected in _required_manifest_constants.items():
        actual = manifest.get(field)
        if actual != expected:
            raise ValueError(f"Manifest field {field!r}: expected {expected!r}, got {actual!r}.")
    if "observation_vocab_size" not in manifest:
        raise ValueError("Manifest missing required field 'observation_vocab_size'.")
    obs_vocab = manifest["observation_vocab_size"]
    if not isinstance(obs_vocab, int) or obs_vocab < 1:
        raise ValueError(f"Manifest 'observation_vocab_size' must be a positive int, got {obs_vocab!r}.")

    # Validate per-split channel arrays.
    for split, n in manifest["n_samples"].items():
        split_dir = root / split
        arrays: dict[str, np.ndarray] = {}
        for ch in ARENA_TASK_CHANNELS:
            ch_file = split_dir / f"{ch}.npy"
            if not ch_file.exists():
                raise FileNotFoundError(f"Missing task channel '{ch}' in {split_dir}.")
            arrays[ch] = np.load(ch_file, mmap_mode="r")
            expected_dtype = _ARENA_TRAJECTORY_DTYPES[ch]
            if arrays[ch].dtype != expected_dtype:
                raise ValueError(f"Channel '{ch}' in split '{split}' has dtype {arrays[ch].dtype}, " f"expected {expected_dtype}.")
            expected_ndim = 1 if ch == CHANNEL_TRAJECTORY_LENGTH else 2
            if arrays[ch].ndim != expected_ndim:
                raise ValueError(f"Channel '{ch}' in split '{split}' has rank {arrays[ch].ndim}, " f"expected {expected_ndim}.")
            if arrays[ch].shape[0] != n:
                raise ValueError(f"Channel '{ch}' in split '{split}' has {arrays[ch].shape[0]} samples, " f"manifest declares {n}.")

        for i in range(n):
            validate_arena_task_sample({ch: arrays[ch][i] for ch in ARENA_TASK_CHANNELS})

    # Validate parent substrate lineage via per-sample index metadata.
    all_entries = read_index(root / "index.jsonl")
    parent_substrate_rel = manifest.get("parent_substrate")
    if not parent_substrate_rel:
        raise ValueError("Manifest missing required field 'parent_substrate'.")

    resolved_repo_root = _repo_root if _repo_root is not None else _find_repo_root(root)
    parent_index_path = resolved_repo_root / parent_substrate_rel / "index.jsonl"

    if not parent_index_path.exists():
        raise FileNotFoundError(
            f"Parent substrate index not found at {parent_index_path}. "
            "Build the parent substrate first or check 'parent_substrate' in the manifest."
        )

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
                raise ValueError(f"Arena index entry {entry.id!r} missing task_metadata.")
            pid = meta["parent_sample_id"]
            ep_idx = meta["episode_index"]
            wseed = meta["walk_seed"]
            if pid not in valid_parent_ids:
                raise ValueError(f"Arena entry {entry.id!r}: parent_sample_id {pid!r} not found " f"in parent index split '{split}'.")
            ep_set = episode_indices_by_parent.setdefault(pid, set())
            if ep_idx in ep_set:
                raise ValueError(f"Duplicate episode_index {ep_idx} for parent_sample_id {pid!r} in split {split!r}.")
            ep_set.add(ep_idx)
            ws_set = walk_seeds_by_parent.setdefault(pid, set())
            if wseed in ws_set:
                raise ValueError(f"Duplicate walk_seed {wseed} for parent_sample_id {pid!r} in split {split!r}.")
            ws_set.add(wseed)

    return manifest


def _find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` to find the repo root (contains pyproject.toml)."""
    current = start.resolve()
    for _ in range(20):
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # Fallback: return cwd.
    return Path.cwd()


# =============================================================================
# Builder
# =============================================================================


def build_arena_task_corpus(
    version_root: Path,
    *,
    parent_substrate: Path,
    corpus: str = "default",
    train_parent_maps: int = 200,
    val_parent_maps: int = 40,
    test_parent_maps: int = 40,
    train_episodes_per_parent: int = 100,
    val_episodes_per_parent: int = 4,
    test_episodes_per_parent: int = 1,
    max_steps: int = 250,
    seed: int = 42,
) -> None:
    """Build the Arena task corpus at *version_root* (topology-free, frozen v1).

    Layout: map-major.  For each selected parent map, ``K`` episodes are
    generated in sequence, so sample index ``i * K + k`` is episode ``k`` of
    parent map ``i``.  Total samples per split = ``parent_maps * episodes_per_parent``.

    Parent-map selection is deterministic: the first ``parent_maps`` entries
    in canonical parent index order for each split.

    Args:
        version_root: Destination versioned root (e.g. ``data/processed/arena/default/v1``).
        parent_substrate: Path to the parent dungeongen shared substrate root.
        corpus: Corpus label (e.g. ``"default"``).
        train_parent_maps: Number of parent maps to use for training.
        val_parent_maps: Number of parent maps to use for validation.
        test_parent_maps: Number of parent maps to use for testing.
        train_episodes_per_parent: Episodes per parent map in the training split.
        val_episodes_per_parent: Episodes per parent map in the validation split.
        test_episodes_per_parent: Episodes per parent map in the test split.
        max_steps: Trajectory length (same for all splits).
        seed: Base RNG seed used in walk-seed derivation.

    Raises:
        FileExistsError: When *version_root* already exists.
        ValueError: When the parent is not a dungeongen shared substrate, or
            the parent split has fewer entries than requested.
    """
    version = extract_version(version_root)
    parent_manifest = load_substrate_manifest(parent_substrate)

    if parent_manifest.get("family") != DUNGEON_SHARED_FAMILY:
        raise ValueError(
            f"Arena task corpus requires a {DUNGEON_SHARED_FAMILY!r} shared substrate, " f"got family={parent_manifest.get('family')!r}."
        )

    split_parent_maps = {
        "train": train_parent_maps,
        "val": val_parent_maps,
        "test": test_parent_maps,
    }
    split_episodes_per_parent = {
        "train": train_episodes_per_parent,
        "val": val_episodes_per_parent,
        "test": test_episodes_per_parent,
    }
    parent_n = parent_manifest.get("n_samples", {})
    for split in _SPLITS:
        avail = parent_n.get(split, 0)
        requested = split_parent_maps[split]
        if requested > avail:
            raise ValueError(
                f"Requested {requested} parent maps for split {split!r} but parent "
                f"substrate only has {avail} entries. Reduce --{split}-parent-maps."
            )

    split_counts = {split: split_parent_maps[split] * split_episodes_per_parent[split] for split in _SPLITS}

    # Derive observation_vocab_size from parent manifest; compute from data if absent.
    if "n_observations" in parent_manifest:
        observation_vocab_size = int(parent_manifest["n_observations"])
    else:
        observation_vocab_size = _compute_observation_vocab_size(parent_substrate)

    canonical_parent = f"data/processed/{parent_manifest['family']}/v{parent_manifest['version']}"

    stage_params = {
        "corpus": corpus,
        "max_steps": max_steps,
        "seed": seed,
        "test_episodes_per_parent": test_episodes_per_parent,
        "test_parent_maps": test_parent_maps,
        "train_episodes_per_parent": train_episodes_per_parent,
        "train_parent_maps": train_parent_maps,
        "val_episodes_per_parent": val_episodes_per_parent,
        "val_parent_maps": val_parent_maps,
        "parent_version": parent_manifest["version"],
        "start_policy_id": START_POLICY_ID,
        "walk_policy_id": WALK_POLICY_ID,
    }

    # We need parent topology_kind/n_states/extent for dataset.json provenance.
    parent_topology_kind: str = parent_manifest["topology_kind"]
    parent_n_states: int = parent_manifest["n_states"]
    parent_extent: list[int] = parent_manifest["extent"]

    # Channels to load from parent substrate per sample.
    _PARENT_CHANNELS = ["mask_valid", "observations", "landmarks"]

    with staging_root(version_root) as tmp:
        all_entries = []

        for split in _SPLITS:
            n_parent = split_parent_maps[split]
            k_ep = split_episodes_per_parent[split]

            samples: list[dict[str, np.ndarray]] = []
            per_sample_ids: list[str] = []
            per_sample_extra: list[dict] = []

            parent_iter = iter_substrate_entries_and_samples(parent_substrate, split, _PARENT_CHANNELS)

            for parent_idx, (parent_entry, parent_sample) in enumerate(parent_iter):
                if parent_idx >= n_parent:
                    break

                for ep_idx in range(k_ep):
                    walk_seed = derive_walk_seed(
                        seed=seed,
                        split=split,
                        parent_sample_id=parent_entry.id,
                        episode_index=ep_idx,
                        start_policy_id=START_POLICY_ID,
                        walk_policy_id=WALK_POLICY_ID,
                        max_steps=max_steps,
                    )
                    episode = _build_episode(parent_sample, max_steps, walk_seed)
                    samples.append(episode)
                    per_sample_ids.append(f"arena-{split}-{parent_entry.id}-ep{ep_idx:04d}")
                    per_sample_extra.append(
                        {
                            "task_metadata": {
                                "parent_sample_id": parent_entry.id,
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
                topology_kind=parent_topology_kind,
                n_states=parent_n_states,
                extent=parent_extent,
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
            topology_kind=parent_topology_kind,
            n_states=parent_n_states,
            extent=parent_extent,
            n_samples=split_counts,
            source_id=DUNGEON_SHARED_FAMILY,
            builder="ehc_sn.tasks.arena.build_arena_task_corpus",
            seed=seed,
            stage_params=stage_params,
            parent_family=parent_manifest["family"],
            parent_version=parent_manifest["version"],
            task_schema_version=TASK_SCHEMA_VERSION,
            task_protocol_version=TASK_PROTOCOL_VERSION,
            task=TASK_FAMILY,
            corpus=corpus,
            parent_substrate=canonical_parent,
            start_policy_id=START_POLICY_ID,
            walk_policy_id=WALK_POLICY_ID,
            action_count=ACTION_COUNT,
            action_id_space=ACTION_ID_SPACE,
            store_row_col=True,
            store_landmark_id=True,
            observation_vocab_size=observation_vocab_size,
        )

    n_total = sum(split_counts.values())
    print(f"Arena task corpus written to {version_root}  ({n_total} samples).")


def _compute_observation_vocab_size(substrate_root: Path) -> int:
    """Compute observation vocabulary size by scanning all splits."""
    max_id = -1
    for split in _SPLITS:
        obs_file = substrate_root / split / "observations.npy"
        if obs_file.exists():
            arr = np.load(obs_file, mmap_mode="r")
            split_max = int(arr.max())
            max_id = max(max_id, split_max)
    if max_id < 0:
        raise RuntimeError("Could not compute observation_vocab_size: no observations.npy found.")
    return max_id + 1
