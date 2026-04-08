"""Benchmark-manifest schemas, builders, and I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field, model_validator

from ehc_sn.data.index import read_index

BENCHMARKS_DIRNAME = "benchmarks"
B0_BENCHMARK_DIRNAME = "b0"
BENCHMARK_MANIFEST_FILENAME = "manifest.json"


class B0SubsetEntry(BaseModel, extra="forbid"):
    """One selected MazeHard sample inside the persisted B0 subset."""

    sample_id: str = Field(..., description="Canonical maze id in the hard subset.")
    difficulty_value: int = Field(..., description="Parsed MazeHard difficulty value for this sample.")


class MazeHardSubsetManifest(BaseModel, extra="forbid"):
    """Persisted B0 hard-subset selection for one MazeHard split."""

    benchmark_id: str = Field(default="b0", description="Canonical benchmark identifier.")
    dataset_root: str = Field(..., description="Processed MazeHard dataset root.")
    source: str = Field(..., description="Source dataset identifier.")
    split: str = Field(..., description="Dataset split used to derive the subset.")
    selection_rule: str = Field(..., description="Human-readable subset selection rule.")
    entries: list[B0SubsetEntry] = Field(..., min_length=1, description="Selected B0 subset entries in deterministic order.")

    @model_validator(mode="after")
    def validate_entries(self) -> "MazeHardSubsetManifest":
        sample_ids = [entry.sample_id for entry in self.entries]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("B0 hard-subset manifest sample ids must be unique.")
        if sample_ids != sorted(sample_ids, key=_stable_id_key):
            raise ValueError("B0 hard-subset manifest entries must be ordered deterministically by sample id.")
        return self

    @property
    def sample_ids(self) -> list[str]:
        """Return the selected sample ids in canonical order."""
        return [entry.sample_id for entry in self.entries]

    @property
    def difficulty_values(self) -> list[int]:
        """Return the selected difficulty values aligned with ``sample_ids``."""
        return [entry.difficulty_value for entry in self.entries]

    @property
    def n_selected(self) -> int:
        """Return the number of selected subset entries."""
        return len(self.entries)


def build_b0_hard_subset_manifest(
    dataset_root: Path,
    *,
    split: str = "test",
    quantile: float = 0.9,
) -> MazeHardSubsetManifest:
    """Build the persisted B0 hard-subset manifest from a processed MazeHard root."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must lie in [0.0, 1.0], got {quantile}.")

    index_path = dataset_root / "index.jsonl"
    entries = [entry for entry in read_index(index_path) if entry.split == split]
    if not entries:
        raise ValueError(f"No MazeHard index entries found for split {split!r} in {index_path}.")

    difficulties = [_parse_difficulty_value(entry.difficulty, entry_id=entry.id) for entry in entries]
    threshold = int(np.quantile(np.asarray(difficulties, dtype=np.int32), quantile, method="higher"))

    selected = sorted(
        ((entry.id, difficulty) for entry, difficulty in zip(entries, difficulties, strict=True) if difficulty >= threshold),
        key=lambda item: _stable_id_key(item[0]),
    )
    if not selected:
        raise ValueError(f"No MazeHard entries met the requested hard-subset threshold {threshold} for split {split!r}.")

    return MazeHardSubsetManifest(
        benchmark_id="b0",
        dataset_root=str(dataset_root),
        source=entries[0].source,
        split=split,
        selection_rule=f"difficulty >= split quantile {quantile:.3f} (threshold={threshold})",
        entries=[B0SubsetEntry(sample_id=sample_id, difficulty_value=difficulty) for sample_id, difficulty in selected],
    )


def b0_hard_subset_path(dataset_root: Path) -> Path:
    """Return the canonical B0 hard-subset manifest path for ``dataset_root``."""
    return dataset_root / BENCHMARKS_DIRNAME / B0_BENCHMARK_DIRNAME / BENCHMARK_MANIFEST_FILENAME


def write_mazehard_subset_manifest(manifest: MazeHardSubsetManifest, path: Path) -> None:
    """Serialize a MazeHard hard-subset manifest to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def read_mazehard_subset_manifest(path: Path) -> MazeHardSubsetManifest:
    """Load a MazeHard hard-subset manifest from JSON."""
    return MazeHardSubsetManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _parse_difficulty_value(difficulty: str, *, entry_id: str) -> int:
    """Return one integer MazeHard difficulty value from the processed index."""
    try:
        return int(difficulty)
    except ValueError as exc:
        raise ValueError(f"MazeHard entry {entry_id} has a non-integer difficulty value: {difficulty!r}.") from exc


def _stable_id_key(value: str) -> tuple[int, str]:
    """Return a deterministic sort key for string sample ids."""
    return (0, f"{int(value):020d}") if value.isdigit() else (1, value)


__all__ = [
    "B0SubsetEntry",
    "B0_BENCHMARK_DIRNAME",
    "BENCHMARK_MANIFEST_FILENAME",
    "BENCHMARKS_DIRNAME",
    "build_b0_hard_subset_manifest",
    "MazeHardSubsetManifest",
    "b0_hard_subset_path",
    "read_mazehard_subset_manifest",
    "write_mazehard_subset_manifest",
]
