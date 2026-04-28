"""MazeHard task corpus materialization.

Owns MazeHard-specific task channel schema, validation, and the builder that
produces the MazeHard task corpus over a parent maze-nd shared substrate.

Task corpus channels extend the shared substrate (topology, mask_valid) with
task-owned semantics:

- ``start``: Agent start position(s) -- from source encoding 'S'.
- ``goals``: Target goal position(s) -- from source encoding 'G'.
- ``solution``: Supervised shortest-path labels -- from source labels field.

The task builder joins by stable source identity: each substrate index entry
carries a ``source_record_id`` (e.g. ``"train:12345"``) that maps back to the
raw puzzle_index.  The raw corpus is indexed on first use and looked up by
identity, not by split-local position.

Path written: ``data/processed/mazehard/<corpus>/v<version>/``
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

from ehc_sn.data._validator import _validate_structure
from ehc_sn.data._writer import _extract_version, _staging_root, write_index_at_root, write_split
from ehc_sn.data.manifest import write_manifest
from ehc_sn.data.mazehard_builder import iter_interim_records
from ehc_sn.data.mazehard_raw import normalize_raw_record
from ehc_sn.data.substrate_reader import iter_substrate_entries_and_samples, load_substrate_manifest

# =============================================================================
# Task corpus channel schema
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

Shared channels (topology, mask_valid) come from the parent maze-nd substrate.
Task-owned channels (start, goals, solution) encode the navigation goal and
supervised path, read from the raw source by stable source identity.
"""

MAZEHARD_TASK_CHANNEL_DTYPES: dict[str, np.dtype] = {
    "topology": np.dtype(bool),
    "mask_valid": np.dtype(bool),
    "start": np.dtype(bool),
    "goals": np.dtype(bool),
    "solution": np.dtype(np.int32),
}
"""Expected numpy dtypes for MazeHard task corpus channels."""

_SPLITS: tuple[str, ...] = ("train", "val", "test")
_SUBSTRATE_CHANNELS = ["topology", "mask_valid"]
_TASK_CHANNELS = ["start", "goals", "solution"]


# =============================================================================
def validate_mazehard_task_sample(data: dict[str, np.ndarray]) -> None:
    """Validate a MazeHard task corpus sample against the task channel schema.

    Checks all task channels are present, all have the expected dtype, and all
    spatial channels share the same ``(H, W)`` shape.

    Raises:
        ValueError: On any contract violation.
    """
    missing = set(MAZEHARD_TASK_CHANNELS) - data.keys()
    if missing:
        raise ValueError(f"MazeHard task sample missing channels: {sorted(missing)}")

    shapes: dict[str, tuple[int, ...]] = {}
    for name, arr in data.items():
        if name in MAZEHARD_TASK_CHANNEL_DTYPES and arr.dtype != MAZEHARD_TASK_CHANNEL_DTYPES[name]:
            raise ValueError(f"Channel '{name}' has dtype {arr.dtype}, expected {MAZEHARD_TASK_CHANNEL_DTYPES[name]}.")
        if name in MAZEHARD_TASK_CHANNEL_DTYPES and arr.ndim >= 2:
            shapes[name] = arr.shape[-2:]

    unique = set(shapes.values())
    if len(unique) > 1:
        detail = ", ".join(f"'{k}': {v}" for k, v in shapes.items())
        raise ValueError(f"All MazeHard task channels must share (H, W) shape. Got: {detail}.")


# =============================================================================
def validate_mazehard_task_root(root: Path) -> dict:
    """Validate a MazeHard task corpus root against task-owned semantics.

    Calls the structural validator then enforces task-owned channel presence,
    dtype, and per-sample spatial shape consistency for every split declared in
    the manifest, by delegating to :func:`validate_mazehard_task_sample` for
    each persisted sample.

    Args:
        root: Resolved versioned MazeHard task corpus root.

    Returns:
        Parsed manifest dict.

    Raises:
        ValueError: On any contract violation.
        FileNotFoundError: When a required file is absent.
    """
    manifest = _validate_structure(root)
    if manifest.get("dataset_class") != "task_corpus":
        raise ValueError("Root is not a task_corpus.")
    if manifest.get("task") != TASK_FAMILY:
        raise ValueError(f"Root task is {manifest.get('task')!r}, expected {TASK_FAMILY!r}.")

    for split, n in manifest["n_samples"].items():
        split_dir = root / split
        # Load all task channels for this split.
        arrays: dict[str, np.ndarray] = {}
        for ch in MAZEHARD_TASK_CHANNELS:
            ch_file = split_dir / f"{ch}.npy"
            if not ch_file.exists():
                raise FileNotFoundError(f"Missing task channel '{ch}' in {split_dir}.")
            arrays[ch] = np.load(ch_file, mmap_mode="r")
            if arrays[ch].shape[0] != n:
                raise ValueError(f"Task channel '{ch}' in split '{split}' has {arrays[ch].shape[0]} samples, " f"manifest declares {n}.")

        # Validate per-sample semantics (dtype + spatial shape consistency via sample validator).
        for i in range(n):
            sample = {ch: arrays[ch][i] for ch in MAZEHARD_TASK_CHANNELS}
            validate_mazehard_task_sample(sample)

    return manifest


# =============================================================================
def _build_raw_index(interim_root: Path) -> dict[str, dict]:
    """Return a dict mapping source_record_id -> raw record for all splits.

    Reads both the train and test interim splits eagerly.  For MazeHard the
    interim corpus has two splits (train, test); val substrate entries reference
    train records via their source_record_id.

    Args:
        interim_root: Directory containing the interim ``.jsonl`` files.

    Returns:
        ``{"train:N": record, "test:N": record, ...}``
    """
    raw_by_id: dict[str, dict] = {}
    for split in ("train", "test"):
        for record in iter_interim_records(interim_root, split):
            rid = f"{split}:{record['puzzle_index']}"
            raw_by_id[rid] = record
    return raw_by_id


# =============================================================================
def build_mazehard_task_corpus(
    version_root: Path,
    *,
    parent_substrate: Path,
    interim_root: Path,
    corpus: str = "default",
    n_train: int = 200,
    n_val: int = 40,
    n_test: int = 40,
    seed: int = 42,
) -> None:
    """Build the MazeHard task corpus at *version_root*.

    Reads topology and mask_valid from the parent maze-nd shared substrate by
    stable source identity (``source_record_id`` in the substrate index), then
    joins with the corresponding interim records to recover start, goals, and
    solution channels.  Writes a versioned, immutable task corpus.

    The version integer is derived from the ``v<N>`` leaf of *version_root*;
    there is no separate ``version`` parameter.

    Args:
        version_root: Destination versioned root
            (e.g. ``data/processed/mazehard/default/v1``).  Must not exist.
        parent_substrate: Path to the parent maze-nd shared substrate version
            root.  Must contain a ``manifest.json``.
        interim_root: Path to the interim MazeHard directory (containing
            ``train.jsonl`` and ``test.jsonl``).
        corpus: Corpus label (e.g. ``"default"``).
        n_train: Number of training samples (capped by substrate split size).
        n_val: Number of validation samples (capped by substrate split size).
        n_test: Number of test samples (capped by substrate split size).
        seed: Reserved for downstream compatibility.

    Raises:
        FileExistsError: When *version_root* already exists (immutable root).
        FileNotFoundError: When *parent_substrate* has no manifest, or raw files
            are missing.
        ValueError: When the parent is not a ``shared_substrate``, or a
            substrate entry has no ``source_record_id``, or a source_record_id
            is not found in the raw corpus.
    """
    version = _extract_version(version_root)
    parent_manifest = load_substrate_manifest(parent_substrate)

    if parent_manifest.get("family") != "maze-nd":
        raise ValueError(f"MazeHard task corpus requires a 'maze-nd' shared substrate, " f"got family={parent_manifest.get('family')!r}.")

    split_counts = {"train": n_train, "val": n_val, "test": n_test}
    parent_n = parent_manifest.get("n_samples", {})
    for split, n in split_counts.items():
        avail = parent_n.get(split, 0)
        if n > avail:
            raise ValueError(f"Requested {n} {split!r} samples but parent substrate only has {avail}.")
    shape: tuple[int, int] = tuple(parent_manifest["shape"])  # type: ignore[assignment]

    # Build complete raw index keyed by source_record_id.
    raw_by_id = _build_raw_index(interim_root)

    stage_params = {
        "corpus": corpus,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "seed": seed,
        "parent_version": parent_manifest["version"],
    }
    canonical_parent = f"data/processed/{parent_manifest['family']}/v{parent_manifest['version']}"

    with _staging_root(version_root) as tmp:
        all_entries = []
        for split in _SPLITS:
            n = split_counts[split]
            entry_sample_pairs = list(iter_substrate_entries_and_samples(parent_substrate, split, _SUBSTRATE_CHANNELS))[:n]

            samples = []
            for entry, substrate_sample in entry_sample_pairs:
                if not entry.source_record_id:
                    raise ValueError(
                        f"Substrate entry {entry.id!r} has no source_record_id. "
                        "Rebuild the parent substrate with a version of mazehard_builder that "
                        "populates source_record_id."
                    )
                raw_record = raw_by_id.get(entry.source_record_id)
                if raw_record is None:
                    raise ValueError(f"source_record_id {entry.source_record_id!r} not found in interim at " f"{interim_root}.")
                normalized = normalize_raw_record(raw_record)
                sample = {**substrate_sample, **{ch: normalized[ch] for ch in _TASK_CHANNELS}}
                samples.append(sample)

            entries = write_split(
                tmp,
                split,
                samples,
                source=TASK_FAMILY,
                shape=shape,
                channels=MAZEHARD_TASK_CHANNELS,
                spatial_channels=MAZEHARD_TASK_CHANNELS,
                index_kwargs={"n_observations": 0, "n_goals": 1, "difficulty": "medium"},
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
            shape=shape,
            n_samples=split_counts,
            source_id="huggingface/maze_hard_augmented",
            builder="ehc_sn.tasks.mazehard.data.build_mazehard_task_corpus",
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
    print(f"MazeHard task corpus written to {version_root}  ({n_total} samples.")


# =============================================================================
__all__ = [
    "TASK_FAMILY",
    "MAZEHARD_TASK_CHANNELS",
    "MAZEHARD_TASK_CHANNEL_DTYPES",
    "validate_mazehard_task_sample",
    "build_mazehard_task_corpus",
]
