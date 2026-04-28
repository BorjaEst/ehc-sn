"""Shared-substrate builder for the maze-nd dataset family.

Orchestrates the shared-substrate pipeline for MazeHard:

1. ``mazehard_raw.ensure_raw_corpus`` — download HuggingFace raw files.
2. ``prepare_mazehard_interim`` — validate and normalize raw records to interim.
3. ``iter_interim_records`` — stream normalized interim records.
4. ``mazehard_raw.normalize_raw_record`` — convert each record to channel dict.
5. ``_writer.create_version_root`` — create the immutable version leaf.
6. ``_writer.write_split`` / ``write_index_at_root`` — write processed output.
7. ``manifest.write_manifest`` — write the authoritative root manifest.

Shared-substrate channels (topology, mask_valid) are task-neutral.
Task-owned channels (start, goals, solution) belong in the MazeHard task
corpus; see ``ehc_sn.tasks.mazehard.data``.

Interim layer: ``data/interim/maze-nd/`` — one JSONL file per raw split.
Shared substrate: ``data/processed/maze-nd/v<version>/``
"""

from __future__ import annotations

import json
from pathlib import Path

from ehc_sn.data._writer import _extract_version, _staging_root, write_index_at_root, write_split
from ehc_sn.data.manifest import write_manifest
from ehc_sn.data.mazehard_raw import ensure_raw_corpus, iter_raw_records, normalize_raw_record

SHARED_FAMILY: str = "maze-nd"
"""Shared-substrate family name for the HuggingFace maze-nd source."""

MAZEHARD_SUBSTRATE_CHANNELS: list[str] = ["topology", "mask_valid"]
"""Shared-substrate channels.  Task-owned channels live in the task corpus."""

_SPLITS: tuple[str, ...] = ("train", "val", "test")
_SOURCE_ID: str = "huggingface/maze_hard_augmented"
"""Stable upstream source identifier."""


# ---------------------------------------------------------------------------
def prepare_mazehard_interim(raw_root: Path, interim_root: Path) -> None:
    """Validate and write raw maze-nd records to the interim leaf.

    Reads both raw splits (``"train"`` and ``"test"``), validates they are
    non-empty, then writes one uncompressed JSONL file per split under
    *interim_root*.  The interim is a deterministic normalized passthrough of
    the raw source in a canonical location.

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
            raise RuntimeError(f"No raw records for split '{split}' in {raw_root}.")
        out = interim_root / f"{split}.jsonl"
        with out.open("w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")


def iter_interim_records(interim_root: Path, split: str):
    """Yield records from the maze-nd interim leaf.

    Args:
        interim_root: Interim root (e.g. ``data/interim/maze-nd``).
        split: ``"train"`` or ``"test"``.

    Yields:
        Raw record dicts as stored in the interim JSONL file.

    Raises:
        FileNotFoundError: When the interim file is missing.
    """
    path = interim_root / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Interim file not found: {path}.  Run prepare-interim first."
        )
    with path.open() as fh:
        for line in fh:
            yield json.loads(line)


# ---------------------------------------------------------------------------
def build_mazehard_substrate(
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
    :func:`prepare_mazehard_interim`) and writes a versioned, immutable
    dataset root containing only the shared-substrate channels
    (``topology``, ``mask_valid``).

    The raw source provides ``train`` and ``test`` splits only.  ``n_val``
    records are carved from the tail of the first ``n_train + n_val`` interim
    training records.

    The version integer is derived from the ``v<N>`` leaf of *version_root*
    and must match the path; there is no separate ``version`` parameter.

    Args:
        version_root: Destination versioned root
            (e.g. ``data/processed/maze-nd/v1``).  Must not already exist.
            Leaf must be ``v<integer>``.
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
    version = _extract_version(version_root)
    raw_train_needed = n_train + n_val
    raw_train_records: list[dict] = []
    for record in iter_interim_records(interim_root, "train"):
        raw_train_records.append(record)
        if len(raw_train_records) >= raw_train_needed:
            break

    if len(raw_train_records) < raw_train_needed:
        raise RuntimeError(
            f"Interim maze-nd train split has only {len(raw_train_records)} records, "
            f"need {raw_train_needed} (n_train={n_train} + n_val={n_val})."
        )

    raw_val_records = raw_train_records[n_train:]
    raw_train_records = raw_train_records[:n_train]

    raw_test_records: list[dict] = []
    for record in iter_interim_records(interim_root, "test"):
        raw_test_records.append(record)
        if len(raw_test_records) >= n_test:
            break

    if len(raw_test_records) < n_test:
        raise RuntimeError(
            f"Interim maze-nd test split has only {len(raw_test_records)} records, need {n_test}."
        )

    raw_by_split: dict[str, list[dict]] = {
        "train": raw_train_records,
        "val": raw_val_records,
        "test": raw_test_records[:n_test],
    }

    first_normalized = normalize_raw_record(raw_train_records[0])
    shape: tuple[int, int] = first_normalized["topology"].shape

    stage_params = {"n_train": n_train, "n_val": n_val, "n_test": n_test, "seed": seed}

    with _staging_root(version_root) as tmp:
        all_entries = []
        for split, records in raw_by_split.items():
            samples = [
                {ch: normalize_raw_record(r)[ch] for ch in MAZEHARD_SUBSTRATE_CHANNELS}
                for r in records
            ]
            raw_split = "test" if split == "test" else "train"
            per_sample_extra = [{"source_record_id": f"{raw_split}:{r['puzzle_index']}"} for r in records]
            entries = write_split(
                tmp,
                split,
                samples,
                source=SHARED_FAMILY,
                shape=shape,
                channels=MAZEHARD_SUBSTRATE_CHANNELS,
                spatial_channels=MAZEHARD_SUBSTRATE_CHANNELS,
                index_kwargs={"n_observations": 0, "n_goals": 0, "difficulty": "medium"},
                per_sample_extra=per_sample_extra,
            )
            all_entries.extend(entries)

        write_index_at_root(all_entries, tmp)

        write_manifest(
            tmp,
            dataset_class="shared_substrate",
            family=SHARED_FAMILY,
            version=version,
            channels=MAZEHARD_SUBSTRATE_CHANNELS,
            shape=shape,
            n_samples={s: len(raw_by_split[s]) for s in _SPLITS},
            source_id=_SOURCE_ID,
            builder="ehc_sn.data.mazehard_builder.build_mazehard_substrate",
            seed=seed,
            stage_params=stage_params,
        )

    n_total = n_train + n_val + n_test
    print(f"maze-nd shared substrate written to {version_root}  ({n_total} samples).")


__all__ = [
    "SHARED_FAMILY",
    "MAZEHARD_SUBSTRATE_CHANNELS",
    "prepare_mazehard_interim",
    "iter_interim_records",
    "build_mazehard_substrate",
]
