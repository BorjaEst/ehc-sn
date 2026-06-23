"""Dungeon task corpus materialization.

Owns Dungeon-specific task channel schema, validation, and the builder that
produces the Dungeon task corpus over a parent dungeongen shared substrate.

Task corpus channels extend the shared substrate channels (topology,
observations, mask_valid, regions, landmarks) with replay protocol:

- ``trajectory_row``: Per-step agent row index.  Shape ``(T,)`` per sample.
- ``trajectory_col``: Per-step agent column index.  Shape ``(T,)`` per sample.
- ``trajectory_previous_action``: Per-step previous action token. ``(T,)``.
- ``trajectory_episode_start``: Episode-start flag per step. ``(T,)`` bool.
- ``trajectory_valid_step``: Validity mask (``t < length``). ``(T,)`` bool.
- ``trajectory_length``: Effective trajectory length scalar.

Path written: ``data/processed/dungeon/<corpus>/v<version>/``
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

from ehc_sn.data.lifecycle import (
    extract_version,
    staging_root,
    validate_version_root,
    write_index_at_root,
    write_split,
)
from ehc_sn.data.manifest import write_manifest
from ehc_sn.data.substrate.dungeongen import (
    SHARED_CHANNELS as DUNGEON_SUBSTRATE_CHANNELS,
)
from ehc_sn.data.substrate.dungeongen import (
    SHARED_FAMILY as DUNGEON_SHARED_FAMILY,
)
from ehc_sn.data.substrate.reader import (
    iter_substrate_samples,
    load_substrate_manifest,
)
from ehc_sn.tasks._replay_build import first_true_cell, random_walk

# =============================================================================
CHANNEL_TRAJECTORY_ROW: Final[str] = "trajectory_row"
CHANNEL_TRAJECTORY_COL: Final[str] = "trajectory_col"
CHANNEL_TRAJECTORY_PREVIOUS_ACTION: Final[str] = "trajectory_previous_action"
CHANNEL_TRAJECTORY_EPISODE_START: Final[str] = "trajectory_episode_start"
CHANNEL_TRAJECTORY_VALID_STEP: Final[str] = "trajectory_valid_step"
CHANNEL_TRAJECTORY_LENGTH: Final[str] = "trajectory_length"
CHANNEL_TRAJECTORY_GOAL_ROW: Final[str] = "trajectory_goal_row"
CHANNEL_TRAJECTORY_GOAL_COL: Final[str] = "trajectory_goal_col"

DUNGEON_TRAJECTORY_CHANNELS: Final[list[str]] = [
    CHANNEL_TRAJECTORY_ROW,
    CHANNEL_TRAJECTORY_COL,
    CHANNEL_TRAJECTORY_PREVIOUS_ACTION,
    CHANNEL_TRAJECTORY_EPISODE_START,
    CHANNEL_TRAJECTORY_VALID_STEP,
    CHANNEL_TRAJECTORY_LENGTH,
    CHANNEL_TRAJECTORY_GOAL_ROW,
    CHANNEL_TRAJECTORY_GOAL_COL,
]

DUNGEON_TASK_CHANNELS: Final[list[str]] = (
    DUNGEON_SUBSTRATE_CHANNELS + DUNGEON_TRAJECTORY_CHANNELS
)
"""All channels in the Dungeon task corpus (shared channels + trajectory channels)."""

_DUNGEON_TRAJECTORY_DTYPES: dict[str, np.dtype] = {
    CHANNEL_TRAJECTORY_ROW: np.dtype(np.int32),
    CHANNEL_TRAJECTORY_COL: np.dtype(np.int32),
    CHANNEL_TRAJECTORY_PREVIOUS_ACTION: np.dtype(np.int32),
    CHANNEL_TRAJECTORY_EPISODE_START: np.dtype(bool),
    CHANNEL_TRAJECTORY_VALID_STEP: np.dtype(bool),
    CHANNEL_TRAJECTORY_LENGTH: np.dtype(np.int32),
    CHANNEL_TRAJECTORY_GOAL_ROW: np.dtype(np.int32),
    CHANNEL_TRAJECTORY_GOAL_COL: np.dtype(np.int32),
}

TASK_FAMILY: Final[str] = "dungeon"

_ACTION_DELTAS: Final[tuple[tuple[int, int], ...]] = (
    (0, 0),  # STAY
    (-1, 0),  # UP
    (0, 1),  # RIGHT
    (1, 0),  # DOWN
    (0, -1),  # LEFT
)
_ACTION_STAY: Final[int] = 0

_SPLITS: tuple[str, ...] = ("train", "val", "test")
_SPLIT_SEED_OFFSET: dict[str, int] = {
    "train": 0,
    "val": 100_000,
    "test": 200_000,
}


# =============================================================================
def validate_dungeon_task_sample(  # ------------------------------------------
    data: dict[str, np.ndarray],
) -> None:
    """Validate a Dungeon task corpus sample.

    Raises:
        ValueError: On any contract violation.
    """
    missing = set(DUNGEON_TASK_CHANNELS) - data.keys()
    if missing:
        raise ValueError(
            f"Dungeon task sample missing channels: {sorted(missing)}",
        )

    if (
        CHANNEL_TRAJECTORY_VALID_STEP in data
        and CHANNEL_TRAJECTORY_LENGTH in data
    ):
        valid_step = data[CHANNEL_TRAJECTORY_VALID_STEP]
        length = data[CHANNEL_TRAJECTORY_LENGTH]
        T = valid_step.shape[-1]
        expected = np.arange(T) < int(np.asarray(length).flat[0])
        if not np.array_equal(valid_step, expected):
            raise ValueError(
                "Prefix invariant violated: trajectory_valid_step must equal (t < trajectory_length).",
            )


# =============================================================================
def validate_dungeon_task_root(  # --------------------------------------------
    root: Path,
) -> dict:
    """Validate a Dungeon task corpus root against task-owned semantics.

    Args:
        root: Resolved versioned Dungeon task corpus root.

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
        raise ValueError(
            f"Root task is {manifest.get('task')!r}, expected {TASK_FAMILY!r}."
        )

    for split, n in manifest["n_samples"].items():
        split_dir = root / split
        arrays: dict[str, np.ndarray] = {}
        for ch in DUNGEON_TASK_CHANNELS:
            ch_file = split_dir / f"{ch}.npy"
            if not ch_file.exists():
                raise FileNotFoundError(
                    f"Missing task channel '{ch}' in {split_dir}."
                )
            arrays[ch] = np.load(ch_file, mmap_mode="r")

        for ch in DUNGEON_TRAJECTORY_CHANNELS:
            arr = arrays[ch]
            expected_dtype = _DUNGEON_TRAJECTORY_DTYPES[ch]
            if arr.dtype != expected_dtype:
                raise ValueError(
                    f"Trajectory channel '{ch}' in split '{split}' has dtype {arr.dtype}, "
                    f"expected {expected_dtype}."
                )
            expected_ndim = (
                1
                if ch
                in (
                    CHANNEL_TRAJECTORY_LENGTH,
                    CHANNEL_TRAJECTORY_GOAL_ROW,
                    CHANNEL_TRAJECTORY_GOAL_COL,
                )
                else 2
            )
            if arr.ndim != expected_ndim:
                raise ValueError(
                    f"Trajectory channel '{ch}' in split '{split}' has rank {arr.ndim}, "
                    f"expected {expected_ndim}."
                )
            if arr.shape[0] != n:
                raise ValueError(
                    f"Trajectory channel '{ch}' in split '{split}' has {arr.shape[0]} samples, "
                    f"manifest declares {n}."
                )

        for i in range(n):
            sample = {ch: arrays[ch][i] for ch in DUNGEON_TASK_CHANNELS}
            validate_dungeon_task_sample(sample)

    return manifest


# =============================================================================
def _add_trajectory(  # -------------------------------------------------------
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

    # Sample a goal cell: a valid cell distinct from start_cell when possible.
    valid_cells = np.argwhere(mask_valid.astype(bool))
    non_start = valid_cells[
        (valid_cells[:, 0] != start_cell[0])
        | (valid_cells[:, 1] != start_cell[1])
    ]
    if len(non_start) > 0:
        goal_idx = int(rng.integers(0, len(non_start)))
        goal_row, goal_col = int(non_start[goal_idx, 0]), int(
            non_start[goal_idx, 1]
        )
    else:
        goal_row, goal_col = (
            start_cell[0],
            start_cell[1],
        )  # degenerate single-cell layout

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
        CHANNEL_TRAJECTORY_GOAL_ROW: np.array(goal_row, dtype=np.int32),
        CHANNEL_TRAJECTORY_GOAL_COL: np.array(goal_col, dtype=np.int32),
    }


# =============================================================================
def build_dungeon_task_corpus(  # ---------------------------------------------
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
    """Build the Dungeon task corpus at *version_root*.

    The version integer is derived from the ``v<N>`` leaf of *version_root*;
    there is no separate ``version`` parameter.

    Args:
        version_root: Destination versioned root
            (e.g. ``data/processed/dungeon/default/v1``).  Must not exist.
        parent_substrate: Path to the parent dungeongen shared substrate root.
        corpus: Corpus label (e.g. ``"default"``).
        n_train: Number of training samples.
        n_val: Number of validation samples.
        n_test: Number of test samples.
        max_steps: Trajectory length.
        seed: Base RNG seed.

    Raises:
        FileExistsError: When *version_root* already exists (immutable root).
        FileNotFoundError: When *parent_substrate* has no manifest.
        ValueError: When the parent is not a ``shared_substrate``.
        RuntimeError: When a sample has an empty valid mask.
    """
    version = extract_version(version_root)
    parent_manifest = load_substrate_manifest(parent_substrate)

    if parent_manifest.get("family") != DUNGEON_SHARED_FAMILY:
        raise ValueError(
            f"Dungeon task corpus requires a {DUNGEON_SHARED_FAMILY!r} shared substrate, "
            f"got family={parent_manifest.get('family')!r}."
        )

    split_counts = {"train": n_train, "val": n_val, "test": n_test}
    parent_n = parent_manifest.get("n_samples", {})
    for split, n in split_counts.items():
        avail = parent_n.get(split, 0)
        if n > avail:
            raise ValueError(
                f"Requested {n} {split!r} samples but parent substrate only has {avail}."
            )

    parent_extent: list[int] = parent_manifest["extent"]
    parent_topology_kind: str = parent_manifest["topology_kind"]
    parent_n_states: int = parent_manifest["n_states"]

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
                _add_trajectory(
                    s, max_steps, seed=seed + _SPLIT_SEED_OFFSET[split] + i
                )
                for i, s in enumerate(
                    iter_substrate_samples(
                        parent_substrate, split, DUNGEON_SUBSTRATE_CHANNELS
                    )
                )
                if i < n
            ]

            entries = write_split(
                tmp,
                split,
                samples,
                source=TASK_FAMILY,
                channels=DUNGEON_TASK_CHANNELS,
                topology_kind=parent_topology_kind,
                n_states=parent_n_states,
                extent=parent_extent,
                index_kwargs={},
                sample_validator=validate_dungeon_task_sample,
            )
            all_entries.extend(entries)

        write_index_at_root(all_entries, tmp)

        write_manifest(
            tmp,
            dataset_class="task_corpus",
            family=TASK_FAMILY,
            version=version,
            channels=DUNGEON_TASK_CHANNELS,
            topology_kind=parent_topology_kind,
            n_states=parent_n_states,
            extent=parent_extent,
            n_samples=split_counts,
            source_id=DUNGEON_SHARED_FAMILY,
            builder="ehc_sn.tasks.dungeon.build_dungeon_task_corpus",
            seed=seed,
            stage_params=stage_params,
            task_schema_version=1,
            task_protocol_version=1,
            task=TASK_FAMILY,
            corpus=corpus,
            manifest_schema_version=1,
            parents={
                "shared_substrate": {
                    "family": parent_manifest["family"],
                    "root": canonical_parent,
                    "version": parent_manifest["version"],
                },
            },
        )

    n_total = n_train + n_val + n_test
    print(
        f"Dungeon task corpus written to {version_root}  ({n_total} samples.)"
    )


# =============================================================================
__all__ = [
    "TASK_FAMILY",
    "DUNGEON_TRAJECTORY_CHANNELS",
    "DUNGEON_TASK_CHANNELS",
    "validate_dungeon_task_sample",
    "validate_dungeon_task_root",
    "build_dungeon_task_corpus",
]
