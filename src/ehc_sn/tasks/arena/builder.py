"""Arena task corpus materialization.

Owns Arena-specific task channel schema, validation, and the builder that
produces the Arena task corpus over a parent dungeongen shared substrate.

Task corpus channels extend shared substrate channels with trajectory semantics:

- ``topology``, ``observations``, ``mask_valid``, ``regions``, ``landmarks``:
  inherited from the dungeongen shared substrate.
- ``trajectory_row``: Per-step agent row index.  ``(T,)`` int32.
- ``trajectory_col``: Per-step agent column index.  ``(T,)`` int32.
- ``trajectory_previous_action``: Per-step previous action token.  ``(T,)`` int32.
- ``trajectory_episode_start``: Episode-start flag.  ``(T,)`` bool.
- ``trajectory_valid_step``: Validity mask (``t < length``).  ``(T,)`` bool.
- ``trajectory_length``: Effective trajectory length scalar.  int32.

Path written: ``data/processed/arena/<corpus>/v<version>/``
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

from ehc_sn.data.lifecycle import extract_version, staging_root, validate_version_root, write_index_at_root, write_split
from ehc_sn.data.manifest import write_manifest
from ehc_sn.data.substrate.dungeongen import SHARED_CHANNELS as DUNGEON_SUBSTRATE_CHANNELS
from ehc_sn.data.substrate.dungeongen import SHARED_FAMILY as DUNGEON_SHARED_FAMILY
from ehc_sn.data.substrate.reader import iter_substrate_samples, load_substrate_manifest
from ehc_sn.tasks._replay_build import first_true_cell, random_walk

# =============================================================================
CHANNEL_TRAJECTORY_ROW: Final[str] = "trajectory_row"
CHANNEL_TRAJECTORY_COL: Final[str] = "trajectory_col"
CHANNEL_TRAJECTORY_PREVIOUS_ACTION: Final[str] = "trajectory_previous_action"
CHANNEL_TRAJECTORY_EPISODE_START: Final[str] = "trajectory_episode_start"
CHANNEL_TRAJECTORY_VALID_STEP: Final[str] = "trajectory_valid_step"
CHANNEL_TRAJECTORY_LENGTH: Final[str] = "trajectory_length"

ARENA_TRAJECTORY_CHANNELS: Final[list[str]] = [
    CHANNEL_TRAJECTORY_ROW,
    CHANNEL_TRAJECTORY_COL,
    CHANNEL_TRAJECTORY_PREVIOUS_ACTION,
    CHANNEL_TRAJECTORY_EPISODE_START,
    CHANNEL_TRAJECTORY_VALID_STEP,
    CHANNEL_TRAJECTORY_LENGTH,
]

ARENA_TASK_CHANNELS: Final[list[str]] = DUNGEON_SUBSTRATE_CHANNELS + ARENA_TRAJECTORY_CHANNELS
"""All channels in the Arena task corpus (dungeongen substrate + trajectory channels)."""

ARENA_SPATIAL_CHANNELS: Final[list[str]] = DUNGEON_SUBSTRATE_CHANNELS

_ARENA_TRAJECTORY_DTYPES: dict[str, np.dtype] = {
    CHANNEL_TRAJECTORY_ROW: np.dtype(np.int32),
    CHANNEL_TRAJECTORY_COL: np.dtype(np.int32),
    CHANNEL_TRAJECTORY_PREVIOUS_ACTION: np.dtype(np.int32),
    CHANNEL_TRAJECTORY_EPISODE_START: np.dtype(bool),
    CHANNEL_TRAJECTORY_VALID_STEP: np.dtype(bool),
    CHANNEL_TRAJECTORY_LENGTH: np.dtype(np.int32),
}

TASK_FAMILY: Final[str] = "arena"

_ACTION_DELTAS: Final[tuple[tuple[int, int], ...]] = (
    (0, 0),  # STAY
    (-1, 0),  # UP
    (0, 1),  # RIGHT
    (1, 0),  # DOWN
    (0, -1),  # LEFT
)
_ACTION_STAY: Final[int] = 0

_SPLITS: tuple[str, ...] = ("train", "val", "test")
_SPLIT_SEED_OFFSET: dict[str, int] = {"train": 0, "val": 100_000, "test": 200_000}


# =============================================================================
def validate_arena_task_sample(data: dict[str, np.ndarray]) -> None:
    """Validate an Arena task corpus sample.

    Raises:
        ValueError: On any contract violation.
    """
    missing = set(ARENA_TASK_CHANNELS) - data.keys()
    if missing:
        raise ValueError(f"Arena task sample missing channels: {sorted(missing)}")

    valid_step = data[CHANNEL_TRAJECTORY_VALID_STEP]
    length = data[CHANNEL_TRAJECTORY_LENGTH]
    T = valid_step.shape[-1]
    expected = np.arange(T) < int(np.asarray(length).flat[0])
    if not np.array_equal(valid_step, expected):
        raise ValueError("Prefix invariant violated: trajectory_valid_step must equal (t < trajectory_length).")


def validate_arena_task_root(root: Path) -> dict:
    """Validate an Arena task corpus root against task-owned semantics.

    Args:
        root: Resolved versioned Arena task corpus root.

    Returns:
        Parsed manifest dict.

    Raises:
        ValueError: On any contract violation.
        FileNotFoundError: When a required file is absent.
    """
    manifest = validate_version_root(root)
    if manifest.get("dataset_class") != "task_corpus":
        raise ValueError("Root is not a task_corpus.")
    if manifest.get("task") != TASK_FAMILY:
        raise ValueError(f"Root task is {manifest.get('task')!r}, expected {TASK_FAMILY!r}.")

    for split, n in manifest["n_samples"].items():
        split_dir = root / split
        arrays: dict[str, np.ndarray] = {}
        for ch in ARENA_TASK_CHANNELS:
            ch_file = split_dir / f"{ch}.npy"
            if not ch_file.exists():
                raise FileNotFoundError(f"Missing task channel '{ch}' in {split_dir}.")
            arrays[ch] = np.load(ch_file, mmap_mode="r")

        for ch in ARENA_TRAJECTORY_CHANNELS:
            arr = arrays[ch]
            expected_dtype = _ARENA_TRAJECTORY_DTYPES[ch]
            if arr.dtype != expected_dtype:
                raise ValueError(f"Trajectory channel '{ch}' in split '{split}' has dtype {arr.dtype}, " f"expected {expected_dtype}.")
            expected_ndim = 1 if ch == CHANNEL_TRAJECTORY_LENGTH else 2
            if arr.ndim != expected_ndim:
                raise ValueError(f"Trajectory channel '{ch}' in split '{split}' has rank {arr.ndim}, " f"expected {expected_ndim}.")
            if arr.shape[0] != n:
                raise ValueError(f"Trajectory channel '{ch}' in split '{split}' has {arr.shape[0]} samples, " f"manifest declares {n}.")

        for i in range(n):
            sample = {ch: arrays[ch][i] for ch in ARENA_TASK_CHANNELS}
            validate_arena_task_sample(sample)

    return manifest


# =============================================================================
def _add_trajectory(
    substrate_sample: dict[str, np.ndarray],
    max_steps: int,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    """Add trajectory channels to a substrate sample dict."""
    rng = np.random.default_rng(seed)
    mask_valid = substrate_sample["mask_valid"]
    start_cell = first_true_cell(mask_valid)
    if start_cell is None:
        raise RuntimeError(f"Empty valid mask for seed={seed}.")

    traj_rng = np.random.default_rng(int(rng.integers(2**31)))
    rows, cols, prev_actions = random_walk(
        mask_valid,
        start_cell,
        max_steps,
        rng=traj_rng,
        action_deltas=_ACTION_DELTAS,
        stay_action=_ACTION_STAY,
    )

    traj_length = np.int32(max_steps)
    episode_start = np.zeros(max_steps, dtype=bool)
    episode_start[0] = True
    valid_step = np.arange(max_steps, dtype=np.int32) < traj_length

    return {
        **substrate_sample,
        CHANNEL_TRAJECTORY_ROW: rows,
        CHANNEL_TRAJECTORY_COL: cols,
        CHANNEL_TRAJECTORY_PREVIOUS_ACTION: prev_actions,
        CHANNEL_TRAJECTORY_EPISODE_START: episode_start,
        CHANNEL_TRAJECTORY_VALID_STEP: valid_step,
        CHANNEL_TRAJECTORY_LENGTH: np.array(traj_length, dtype=np.int32),
    }


# =============================================================================
def build_arena_task_corpus(
    version_root: Path,
    *,
    parent_substrate: Path,
    corpus: str = "default",
    n_train: int = 200,
    n_val: int = 40,
    n_test: int = 40,
    max_steps: int = 50,
    seed: int = 42,
) -> None:
    """Build the Arena task corpus at *version_root*.

    The version integer is derived from the ``v<N>`` leaf of *version_root*;
    there is no separate ``version`` parameter.

    Args:
        version_root: Destination versioned root
            (e.g. ``data/processed/arena/default/v1``).  Must not exist.
        parent_substrate: Path to the parent dungeongen shared substrate root.
        corpus: Corpus label (e.g. ``"default"``).
        n_train: Number of training samples.
        n_val: Number of validation samples.
        n_test: Number of test samples.
        max_steps: Trajectory length.
        seed: Base RNG seed.

    Raises:
        FileExistsError: When *version_root* already exists (immutable root).
        ValueError: When the parent is not a ``shared_substrate``.
        RuntimeError: When a sample has an empty valid mask.
    """
    version = extract_version(version_root)
    parent_manifest = load_substrate_manifest(parent_substrate)

    if parent_manifest.get("family") != DUNGEON_SHARED_FAMILY:
        raise ValueError(
            f"Arena task corpus requires a {DUNGEON_SHARED_FAMILY!r} shared substrate, " f"got family={parent_manifest.get('family')!r}."
        )

    split_counts = {"train": n_train, "val": n_val, "test": n_test}
    parent_n = parent_manifest.get("n_samples", {})
    for split, n in split_counts.items():
        avail = parent_n.get(split, 0)
        if n > avail:
            raise ValueError(f"Requested {n} {split!r} samples but parent substrate only has {avail}.")

    shape: tuple[int, int] = tuple(parent_manifest["shape"])  # type: ignore[assignment]

    stage_params = {
        "corpus": corpus,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "max_steps": max_steps,
        "seed": seed,
        "parent_version": parent_manifest["version"],
    }
    canonical_parent = f"data/processed/{parent_manifest['family']}/v{parent_manifest['version']}"

    with staging_root(version_root) as tmp:
        all_entries = []
        for split in _SPLITS:
            n = split_counts[split]
            samples = [
                _add_trajectory(s, max_steps, seed=seed + _SPLIT_SEED_OFFSET[split] + i)
                for i, s in enumerate(iter_substrate_samples(parent_substrate, split, DUNGEON_SUBSTRATE_CHANNELS))
                if i < n
            ]

            entries = write_split(
                tmp,
                split,
                samples,
                source=TASK_FAMILY,
                shape=shape,
                channels=ARENA_TASK_CHANNELS,
                spatial_channels=ARENA_SPATIAL_CHANNELS,
                index_kwargs={"n_observations": parent_manifest.get("n_observations", 0), "n_goals": 0, "difficulty": "medium"},
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
            shape=shape,
            n_samples=split_counts,
            source_id=DUNGEON_SHARED_FAMILY,
            builder="ehc_sn.tasks.arena.build_arena_task_corpus",
            seed=seed,
            stage_params=stage_params,
            parent_family=parent_manifest["family"],
            parent_version=parent_manifest["version"],
            task_schema_version=1,
            task_protocol_version=1,
            task=TASK_FAMILY,
            corpus=corpus,
            parent_substrate=canonical_parent,
        )

    n_total = n_train + n_val + n_test
    print(f"Arena task corpus written to {version_root}  ({n_total} samples.)")


__all__ = [
    "TASK_FAMILY",
    "ARENA_TRAJECTORY_CHANNELS",
    "ARENA_TASK_CHANNELS",
    "ARENA_SPATIAL_CHANNELS",
    "validate_arena_task_sample",
    "validate_arena_task_root",
    "build_arena_task_corpus",
]
