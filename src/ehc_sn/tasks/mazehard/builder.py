"""MazeHard task corpus materialization.

Owns MazeHard-specific task channel schema, validation, and the builder that
produces the MazeHard task corpus over a maze-nd shared substrate.

All channels (topology, mask_valid, start, goals, solution) are read from
the maze-nd shared substrate, which preserves source problem annotations from
artifact_schema_version 1 onward.  The task builder does not depend on interim
or raw records.

Path written: ``data/processed/mazehard/<corpus>/v<version>/``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import numpy as np

from ehc_sn.data.lifecycle import (
    extract_version,
    staging_root,
    validate_version_root,
    write_index_at_root,
    write_split,
)
from ehc_sn.data.manifest import write_manifest
from ehc_sn.data.substrate.maze_nd import SHARED_FAMILY as MAZE_ND_SHARED_FAMILY
from ehc_sn.data.substrate.reader import (
    iter_substrate_entries_and_samples,
    load_substrate_manifest,
)

# =============================================================================
TASK_FAMILY: Final[str] = "mazehard"
"""Task namespace for the MazeHard task corpus."""

MAZEHARD_TASK_CHANNELS: Final[list[str]] = [
    "topology",
    "mask_valid",
    "start",
    "goals",
    "solution",
]
"""All channels in the MazeHard task corpus.

All channels are read from the parent maze-nd shared substrate, which
preserves source problem annotations from
:attr:`artifact_schema_version` 1 onward.  The task builder does not
depend on interim or raw records."""

_MAZEHARD_REQUIRED_PARENT_CHANNELS: tuple[str, ...] = (
    "topology",
    "mask_valid",
    "start",
    "goals",
    "solution",
)
"""Channels the mazehard task builder requires in the parent maze-nd substrate."""

MAZEHARD_TASK_CHANNEL_DTYPES: dict[str, np.dtype] = {
    "topology": np.dtype(bool),
    "mask_valid": np.dtype(bool),
    "start": np.dtype(bool),
    "goals": np.dtype(bool),
    "solution": np.dtype(np.int32),
}
"""Expected numpy dtypes for MazeHard task corpus channels."""

_SPLITS: tuple[str, ...] = ("train", "val", "test")


# =============================================================================
def validate_mazehard_sample(data: dict[str, np.ndarray]) -> None:
    """Validate a MazeHard task corpus sample against the task channel schema.

    Raises:
        ValueError: On any contract violation.
    """
    missing = set(MAZEHARD_TASK_CHANNELS) - data.keys()
    if missing:
        raise ValueError(
            f"MazeHard task sample missing channels: {sorted(missing)}"
        )

    shapes: dict[str, tuple[int, ...]] = {}
    for name, arr in data.items():
        if (
            name in MAZEHARD_TASK_CHANNEL_DTYPES
            and arr.dtype != MAZEHARD_TASK_CHANNEL_DTYPES[name]
        ):
            raise ValueError(
                f"Channel '{name}' has dtype {arr.dtype}, expected "
                f"{MAZEHARD_TASK_CHANNEL_DTYPES[name]}."
            )
        if name in MAZEHARD_TASK_CHANNEL_DTYPES and arr.ndim >= 2:
            shapes[name] = arr.shape[-2:]

    unique = set(shapes.values())
    if len(unique) > 1:
        detail = ", ".join(f"'{k}': {v}" for k, v in shapes.items())
        raise ValueError(
            "All MazeHard task channels must share (H, W) shape. "
            f"Got: {detail}."
        )


# Backward-compat alias
validate_mazehard_task_sample = validate_mazehard_sample


def validate_mazehard_root(root: Path) -> dict:
    """Validate a MazeHard task corpus root against task-owned semantics.

    Args:
        root: Resolved versioned MazeHard task corpus root.

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
        for ch in MAZEHARD_TASK_CHANNELS:
            ch_file = split_dir / f"{ch}.npy"
            if not ch_file.exists():
                raise FileNotFoundError(
                    f"Missing task channel '{ch}' in {split_dir}."
                )
            arrays[ch] = np.load(ch_file, mmap_mode="r")
            if arrays[ch].shape[0] != n:
                raise ValueError(
                    f"Task channel '{ch}' in split '{split}' "
                    f"has {arrays[ch].shape[0]} samples, manifest declares {n}."
                )

        for i in range(n):
            sample = {ch: arrays[ch][i] for ch in MAZEHARD_TASK_CHANNELS}
            validate_mazehard_sample(sample)

    return manifest


# Backward-compat alias
validate_mazehard_task_root = validate_mazehard_root


def _sample_entry_pairs(
    pairs: list[tuple[Any, dict[str, np.ndarray]]],
    n: int,
    rng: np.random.Generator,
) -> list[tuple[Any, dict[str, np.ndarray]]]:
    if n == 0:
        return []
    if n == len(pairs):
        return list(pairs)
    indices = rng.choice(len(pairs), size=n, replace=False)
    return [pairs[int(i)] for i in indices]


# =============================================================================
def build_mazehard_task_corpus(
    version_root: Path,
    *,
    substrate_root: Path,
    corpus: str = "default",
    n_train: int = 1000,
    n_val: int = 40,
    n_test: int = 40,
    seed: int = 42,
) -> None:
    """Build the MazeHard task corpus at *version_root*.

    All channels (topology, mask_valid, start, goals, solution) are read
    from the maze-nd shared substrate.  The task builder does not
    depend on interim or raw records.

    The version integer is derived from the ``v<N>`` leaf of *version_root*;
    there is no separate ``version`` parameter.

    Args:
        version_root: Destination versioned root
            (e.g. ``data/processed/mazehard/default/v1``).  Must not exist.
        substrate_root: Path to the maze-nd shared substrate version
            root.  Must contain a ``manifest.json`` with all required
            channels (topology, mask_valid, start, goals, solution).
        corpus: Corpus label (e.g. ``"default"``).
        n_train: Number of training samples (capped by substrate split size).
        n_val: Number of validation samples (capped by substrate split size).
        n_test: Number of test samples (capped by substrate split size).
        seed: Deterministic sampling seed for task corpus membership.

    Raises:
        FileExistsError: When *version_root* already exists (immutable root).
        FileNotFoundError: When *substrate_root* has no manifest.
        ValueError: When the parent is not a ``maze-nd`` shared_substrate, or
            the parent lacks required source annotation channels (start,
            goals, solution), or requested split counts exceed availability.
    """
    version = extract_version(version_root)
    parent_manifest = load_substrate_manifest(substrate_root)

    if parent_manifest.get("family") != MAZE_ND_SHARED_FAMILY:
        raise ValueError(
            f"MazeHard task corpus requires a {MAZE_ND_SHARED_FAMILY!r} shared "
            f"substrate, got family={parent_manifest.get('family')!r}."
        )

    # Validate parent substrate has the expected content-schema version.
    parent_artifact_version = parent_manifest.get("artifact_schema_version")
    if parent_artifact_version != 1:
        raise ValueError(
            f"Parent shared_substrate has unsupported "
            f"artifact_schema_version={parent_artifact_version}. "
            f"Expected 1.\n"
            f"Rebuild the maze-nd substrate:\n"
            f"    python scripts/data-gen/build-maze-nd.py build"
        )

    # Validate parent substrate has all required channels.
    parent_channels = set(parent_manifest.get("channels", []))
    missing = [
        ch
        for ch in _MAZEHARD_REQUIRED_PARENT_CHANNELS
        if ch not in parent_channels
    ]
    if missing:
        raise ValueError(
            f"Parent substrate is maze-nd but lacks required source annotation "
            f"channels: {', '.join(missing)}. "
            f"Rebuild the maze-nd substrate with the current builder:\n"
            f"    python scripts/data-gen/build-maze-nd.py build"
        )

    split_counts = {"train": n_train, "val": n_val, "test": n_test}
    parent_n = parent_manifest.get("n_samples", {})
    for split, n in split_counts.items():
        avail = parent_n.get(split, 0)
        if n > avail:
            raise ValueError(
                f"Requested {n} {split!r} samples "
                f"but parent substrate only has {avail}."
            )
    parent_extent: list[int] = parent_manifest["extent"]
    parent_topology_kind: str = parent_manifest["topology_kind"]
    parent_n_states: int = parent_manifest["n_states"]

    stage_params = {
        "corpus": corpus,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "seed": seed,
        "parent_version": parent_manifest["version"],
    }
    canonical_parent = (
        f"data/processed/{parent_manifest['family']}/"
        f"v{parent_manifest['version']}"
    )

    seed_seq = np.random.SeedSequence(seed)
    split_rngs = dict(
        zip(
            _SPLITS,
            [np.random.default_rng(s) for s in seed_seq.spawn(len(_SPLITS))],
        )
    )

    with staging_root(version_root) as tmp:
        all_entries = []
        for split in _SPLITS:
            n = split_counts[split]
            entry_sample_pairs = list(
                iter_substrate_entries_and_samples(
                    substrate_root,
                    split,
                    list(_MAZEHARD_REQUIRED_PARENT_CHANNELS),
                )
            )
            entry_sample_pairs = _sample_entry_pairs(
                entry_sample_pairs,
                n,
                split_rngs[split],
            )

            samples = []
            per_sample_extra = []
            for entry, substrate_sample in entry_sample_pairs:
                if not entry.source_record_id:
                    raise ValueError(
                        f"Substrate entry {entry.id!r} has no source_record_id. "
                        "Rebuild the parent substrate."
                    )
                samples.append(substrate_sample)
                group_index = (
                    entry.task_metadata.get("group_index")
                    if entry.task_metadata
                    else None
                )
                per_sample_extra.append(
                    {
                        "source_record_id": entry.source_record_id,
                        "task_metadata": {
                            "puzzle_index": int(
                                entry.source_record_id.split(":")[1]
                            ),
                            "group_index": group_index,
                            "raw_split": entry.source_record_id.split(":")[0],
                        },
                    }
                )

            entries = write_split(
                tmp,
                split,
                samples,
                source=TASK_FAMILY,
                channels=MAZEHARD_TASK_CHANNELS,
                topology_kind=parent_topology_kind,
                n_states=parent_n_states,
                extent=parent_extent,
                index_kwargs={},
                per_sample_extra=per_sample_extra,
                sample_validator=validate_mazehard_task_sample,
            )
            all_entries.extend(entries)

        write_index_at_root(all_entries, tmp)

        write_manifest(
            tmp,
            dataset_class="task_corpus",
            family=TASK_FAMILY,
            version=version,
            channels=MAZEHARD_TASK_CHANNELS,
            topology_kind=parent_topology_kind,
            n_states=parent_n_states,
            extent=parent_extent,
            n_samples=split_counts,
            source_id="huggingface/maze_hard_augmented",
            builder="ehp_sn.tasks.mazehard.build_mazehard_task_corpus",
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
        f"MazeHard task corpus written to {version_root}  ({n_total} samples.)"
    )


__all__ = [
    "TASK_FAMILY",
    "MAZEHARD_TASK_CHANNELS",
    "MAZEHARD_TASK_CHANNEL_DTYPES",
    "validate_mazehard_sample",
    "validate_mazehard_root",
    "build_mazehard_task_corpus",
]
