"""Shared-substrate builder for the maze-nd dataset family.

Orchestrates the shared-substrate pipeline for the maze-nd source family:

1. :func:`ensure_raw` — download HuggingFace raw files.
2. :func:`prepare_interim` — validate and normalize raw records to interim.
3. :func:`build_shared_substrate` — build the versioned immutable substrate root.

Public surface: MazeNdSourceRecord, SHARED_FAMILY, SHARED_CHANNELS, ensure_raw,
prepare_interim, build_shared_substrate.

Shared-substrate channels include source problem annotations (start,
goals, solution) in addition to structural channels (topology,
mask_valid).  These are reusable source facts needed by downstream task
builders, not task-owned labels.  Task protocol (episode encoding,
target format) belongs in the respective task corpus.

Interim layer: ``data/interim/maze-nd/`` — one JSONL file per raw split.
Shared substrate: ``data/processed/maze-nd/v<version>/``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import numpy as np

from ehc_sn.data.lifecycle import (
    extract_version,
    staging_root,
    write_index_at_root,
    write_split,
)
from ehc_sn.data.manifest import write_manifest
from ehc_sn.data.substrate._maze_nd_raw import (
    ensure_raw_corpus,
    iter_raw_records,
    normalize_raw_record,
)
from ehc_sn.data.substrate.grid2d import TOPOLOGY_KIND as _GRID2D_KIND
from ehc_sn.data.substrate.grid2d import validate_grid2d_sample

SHARED_FAMILY: str = "maze-nd"
"""Shared-substrate family name for the HuggingFace maze-nd source."""

SHARED_CHANNELS: list[str] = [
    "topology",
    "mask_valid",
    "start",
    "goals",
    "solution",
]
"""Shared-substrate channels including source problem annotations.

Structural channels (topology, mask_valid) describe the maze layout.
Source annotation channels (start, goals, solution) are reusable
problem-instance facts from the upstream source, preserved so that
downstream task builders do not depend on interim or raw records.
"""

_SPLITS: tuple[str, ...] = ("train", "val", "test")
_SOURCE_ID: str = "huggingface/maze_hard_augmented"


# ---------------------------------------------------------------------------
class MazeNdSourceRecord(TypedDict):
    """Stable source-family record from the maze-nd raw corpus.

    Fields reflect the raw source data for one maze puzzle. This is the
    canonical identity type for joining substrate entries with source records.
    """

    inputs: list
    """2-D grid of character strings encoding the maze layout."""

    labels: list
    """2-D grid of character strings encoding the solution path."""

    puzzle_index: int
    """Stable puzzle index within the raw split."""

    group_index: int
    """Group index within the raw corpus."""

    set: str
    """Raw split name (``"train"`` or ``"test"``)."""


# ---------------------------------------------------------------------------
def ensure_raw(
    raw_root: Path, *, repo_id: str = "flaitenberger/maze_hard_augmented"
) -> None:
    """Download the maze-nd raw corpus into *raw_root* if not already present.

    Args:
        raw_root: Destination directory for raw files.
        repo_id: HuggingFace dataset repo id.
    """
    ensure_raw_corpus(raw_root, repo_id=repo_id)


def prepare_interim(raw_root: Path, interim_root: Path) -> None:
    """Validate and write raw maze-nd records to the interim leaf.

    Reads both raw splits (``"train"`` and ``"test"``), validates they are
    non-empty, then writes one uncompressed JSONL file per split under
    *interim_root*.

    Args:
        raw_root: Directory containing the raw HuggingFace corpus files.
        interim_root: Destination interim leaf (e.g. ``data/interim/maze-nd``).

    Raises:
        RuntimeError: When a raw split is empty.
    """
    interim_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        records = list(iter_raw_records(raw_root, split))
        if not records:
            raise RuntimeError(
                f"No raw records for split '{split}' in {raw_root}."
            )
        out = interim_root / f"{split}.jsonl"
        with out.open("w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")


def _iter_interim_records(interim_root: Path, split: str):
    """Yield records from the maze-nd interim leaf."""
    path = interim_root / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Interim file not found: {path}.  Run prepare-interim first."
        )
    with path.open() as fh:
        for line in fh:
            yield json.loads(line)


def _sample_records(
    records: list[dict],
    n: int,
    rng: np.random.Generator,
) -> list[dict]:
    if n == 0:
        return []
    if n == len(records):
        return list(records)
    indices = rng.choice(len(records), size=n, replace=False)
    return [records[int(i)] for i in indices]


def build_shared_substrate(
    version_root: Path,
    *,
    interim_root: Path,
    n_train: int = 200,
    n_val: int = 40,
    n_test: int = 40,
    seed: int = 42,
) -> None:
    """Build the maze-nd shared substrate at *version_root*.

    Reads normalized records from *interim_root* (produced by
    :func:`prepare_interim`) and writes a versioned, immutable dataset root
    containing structural channels (``topology``, ``mask_valid``) and source
    problem annotations (``start``, ``goals``, ``solution``).

    The raw source provides ``train`` and ``test`` splits only. ``n_train``
    records are sampled deterministically from the training population.
    ``n_val`` and ``n_test`` are sampled deterministically from the raw test
    population as non-overlapping partitions.

    The version integer is derived from the ``v<N>`` leaf of *version_root*;
    there is no separate ``version`` parameter.

    Args:
        version_root: Destination versioned root
            (e.g. ``data/processed/maze-nd/v1``).  Must not already exist.
        interim_root: Interim leaf (e.g. ``data/interim/maze-nd``).
        n_train: Number of training samples.
        n_val: Number of validation samples.
        n_test: Number of test samples.
        seed: Deterministic base seed.

    Raises:
        FileExistsError: When *version_root* already exists (immutable root).
        ValueError: When the version leaf name is not ``v<integer>``.
        RuntimeError: When the interim does not have enough records.
    """
    version = extract_version(version_root)
    train_population = list(_iter_interim_records(interim_root, "train"))
    test_population = list(_iter_interim_records(interim_root, "test"))

    if len(train_population) < n_train:
        raise RuntimeError(
            f"Interim maze-nd train split has only {len(train_population)} records, "
            f"need {n_train} (n_train={n_train})."
        )
    raw_test_needed = n_val + n_test
    if len(test_population) < raw_test_needed:
        raise RuntimeError(
            f"Interim maze-nd test split has only {len(test_population)} records, "
            f"need {raw_test_needed} (n_val={n_val} + n_test={n_test})."
        )

    seed_seq = np.random.SeedSequence(seed)
    train_seq, test_seq = seed_seq.spawn(2)
    train_rng = np.random.default_rng(train_seq)
    test_rng = np.random.default_rng(test_seq)

    raw_train_records = _sample_records(train_population, n_train, train_rng)
    val_test_records = _sample_records(
        test_population, raw_test_needed, test_rng
    )
    raw_val_records = val_test_records[:n_val]
    raw_test_records = val_test_records[n_val:]

    raw_by_split: dict[str, list[dict]] = {
        "train": raw_train_records,
        "val": raw_val_records,
        "test": raw_test_records,
    }

    first_normalized = normalize_raw_record(raw_train_records[0])
    topo_shape: tuple[int, int] = first_normalized["topology"].shape
    n_states = topo_shape[0] * topo_shape[1]

    stage_params = {
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "seed": seed,
    }

    with staging_root(version_root) as tmp:
        all_entries = []
        for split, records in raw_by_split.items():
            samples = [
                {ch: normalize_raw_record(r)[ch] for ch in SHARED_CHANNELS}
                for r in records
            ]
            raw_split = "test" if split in ("val", "test") else "train"
            per_sample_extra = [
                {
                    "source_record_id": f"{raw_split}:{r['puzzle_index']}",
                    "task_metadata": {"group_index": r["group_index"]},
                }
                for r in records
            ]
            entries = write_split(
                tmp,
                split,
                samples,
                source=SHARED_FAMILY,
                channels=SHARED_CHANNELS,
                topology_kind=_GRID2D_KIND,
                n_states=n_states,
                extent=[topo_shape[0], topo_shape[1]],
                index_kwargs={},
                per_sample_extra=per_sample_extra,
                sample_validator=validate_grid2d_sample,
            )
            all_entries.extend(entries)

        write_index_at_root(all_entries, tmp)

        write_manifest(
            tmp,
            dataset_class="shared_substrate",
            family=SHARED_FAMILY,
            version=version,
            channels=SHARED_CHANNELS,
            topology_kind=_GRID2D_KIND,
            n_states=n_states,
            extent=[topo_shape[0], topo_shape[1]],
            n_samples={s: len(raw_by_split[s]) for s in _SPLITS},
            source_id=_SOURCE_ID,
            builder="ehc_sn.data.substrate.maze_nd.build_shared_substrate",
            seed=seed,
            stage_params=stage_params,
            shared_schema_version=2,
        )

    n_total = n_train + n_val + n_test
    print(
        f"maze-nd shared substrate written to {version_root}  ({n_total} samples)."
    )


def validate_maze_nd_shared_root(root: Path) -> dict:
    """Validate a maze-nd shared substrate root against generic and family-owned rules.

    Args:
        root: Resolved versioned maze-nd shared substrate root.

    Returns:
        Parsed manifest dict.

    Raises:
        ValueError: On any contract violation.
        FileNotFoundError: When a required file is absent.
    """
    from ehc_sn.data.lifecycle import validate_version_root

    manifest = validate_version_root(root)
    if manifest.get("family") != SHARED_FAMILY:
        raise ValueError(
            f"Root family is {manifest.get('family')!r}, expected {SHARED_FAMILY!r}."
        )
    if manifest.get("topology_kind") != _GRID2D_KIND:
        raise ValueError(
            f"Root topology_kind is {manifest.get('topology_kind')!r}, expected {_GRID2D_KIND!r}."
        )
    missing_ch = set(SHARED_CHANNELS) - set(manifest.get("channels", []))
    if missing_ch:
        raise ValueError(
            f"Manifest missing required maze-nd channels: {sorted(missing_ch)}"
        )  # noqa: E501

    return manifest


__all__ = [
    "MazeNdSourceRecord",
    "SHARED_FAMILY",
    "SHARED_CHANNELS",
    "ensure_raw",
    "prepare_interim",
    "build_shared_substrate",
    "validate_maze_nd_shared_root",
]
