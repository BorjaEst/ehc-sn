"""Build the persisted B2-B3 dungeon benchmark contract manifest."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from typer import Option, Typer, echo

from ehc_sn.data._canonical import shortest_path_distances
from ehc_sn.data.benchmark_manifest import (
    HELD_OUT_ONE_SHOT_ROLE,
    TRAIN_ROLE,
    DungeonBenchmarkEntry,
    DungeonBenchmarkManifest,
    GoalSpec,
    GridPoint,
    StartSpec,
    b23_contracts_path,
    write_dungeon_benchmark_manifest,
)
from ehc_sn.data.contracts import build_layout_benchmark_contract, component_from_channels, resolve_preferred_start
from ehc_sn.data.index import MazeIndexEntry, read_index
from ehc_sn.data.schema import CHANNEL_MASK_VALID, CHANNEL_START, CHANNEL_TOPOLOGY

# =================================================================================================
# Configuration
# -------------------------------------------------------------------------------------------------

app = Typer(pretty_exceptions_enable=False)
DEFAULT_DATASET_ROOT = Path("data/processed/dungeons")
DEFAULT_SPLITS = ("train", "val", "test")


# =================================================================================================
# CLI Commands
# -------------------------------------------------------------------------------------------------


# =================================================================================================
@app.command()
def build_contracts(  # ---------------------------------------------------------------------------
    dataset_root: Path = Option(DEFAULT_DATASET_ROOT, "--dataset-root", help="Processed dungeon dataset root."),
    out_path: Path | None = Option(None, "--out-path", help="Optional override for the B23 contract manifest path."),
    splits: list[str] = Option(list(DEFAULT_SPLITS), "--splits", help="Dataset splits to include in the persisted manifest."),
    seed: int = Option(4700, "--seed", help="Persisted generation seed recorded in every contract entry."),
) -> None:  # fmt: skip
    """Build the canonical persisted B2-B3 benchmark contracts from a processed dungeon root."""
    manifest = generate_benchmark_manifest(dataset_root=dataset_root, splits=splits, seed=seed)
    resolved_out_path = b23_contracts_path(dataset_root) if out_path is None else out_path
    write_dungeon_benchmark_manifest(manifest, resolved_out_path)
    echo(f"Wrote {len(manifest.entries)} benchmark entries to {resolved_out_path}")


# =================================================================================================
# Manifest Pipeline
# -------------------------------------------------------------------------------------------------


def generate_benchmark_manifest(  # ---------------------------------------------------------------
    *, dataset_root: Path, splits: Sequence[str] = DEFAULT_SPLITS, seed: int,
) -> DungeonBenchmarkManifest:  # fmt: skip
    """Return the persisted B23 benchmark manifest for the selected processed splits."""
    index_entries = read_index(dataset_root / "index.jsonl")
    requested_splits = tuple(dict.fromkeys(splits))
    missing = [split for split in requested_splits if split not in {entry.split for entry in index_entries}]
    if missing:
        raise ValueError(f"Unknown split(s) requested for benchmark-contract generation: {missing}")

    entries_by_split: dict[str, list[MazeIndexEntry]] = {split: [] for split in requested_splits}
    for entry in index_entries:
        if entry.split in requested_splits:
            entries_by_split[entry.split].append(entry)

    manifest_entries: list[DungeonBenchmarkEntry] = []
    for split in requested_splits:
        split_entries = entries_by_split[split]
        arrays = _load_split_arrays(dataset_root=dataset_root, split=split)
        for row_index, index_entry in enumerate(split_entries):
            manifest_entries.append(
                _build_contract_entry(
                    dataset_root=dataset_root,
                    split=split,
                    entry=index_entry,
                    row_index=row_index,
                    arrays=arrays,
                    seed=seed,
                )
            )

    manifest_entries.sort(
        key=lambda entry: int(entry.layout_id) if entry.layout_id.isdigit() else entry.layout_id,
    )
    return DungeonBenchmarkManifest(entries=manifest_entries)


# =================================================================================================
# Helpers
# -------------------------------------------------------------------------------------------------


# =================================================================================================
def _load_split_arrays(  # ------------------------------------------------------------------------
    *, dataset_root: Path, split: str,
) -> dict[str, np.ndarray]:  # fmt: skip
    """Load the processed per-split arrays required to build benchmark contracts."""
    split_dir = dataset_root / split
    arrays: dict[str, np.ndarray] = {}
    for channel in (CHANNEL_MASK_VALID, CHANNEL_START, CHANNEL_TOPOLOGY):
        path = split_dir / f"{channel}.npy"
        if path.exists():
            arrays[channel] = np.load(path, mmap_mode="r")
    if CHANNEL_MASK_VALID not in arrays and CHANNEL_TOPOLOGY not in arrays:
        raise ValueError(f"Split directory {split_dir} must contain either '{CHANNEL_MASK_VALID}.npy' or '{CHANNEL_TOPOLOGY}.npy'.")
    return arrays


# =================================================================================================
def _build_contract_entry(  # ---------------------------------------------------------------------
    *, dataset_root: Path, split: str, entry: MazeIndexEntry, row_index: int,
    arrays: dict[str, np.ndarray], seed: int,
) -> DungeonBenchmarkEntry:  # fmt: skip
    """Build one per-layout B23 contract entry."""
    mask_valid = np.asarray(arrays[CHANNEL_MASK_VALID][row_index], dtype=bool) if CHANNEL_MASK_VALID in arrays else None
    topology = np.asarray(arrays[CHANNEL_TOPOLOGY][row_index], dtype=bool) if CHANNEL_TOPOLOGY in arrays else None
    start_mask = np.asarray(arrays[CHANNEL_START][row_index], dtype=bool) if CHANNEL_START in arrays else None

    component = component_from_channels(
        mask_valid=mask_valid,
        topology=topology,
    )
    preferred_start = resolve_preferred_start(
        component=component,
        start_mask=start_mask,
    )
    contract = build_layout_benchmark_contract(
        component=component,
        entry_id=entry.id,
        split=entry.split,
        preferred_start=preferred_start,
    )

    def goal_spec(id, row, col):
        return GoalSpec(
            goal_id=id, row=row, col=col,
            distance_from_canonical_start=int(contract.start_distances[(row, col)]),
            role=TRAIN_ROLE if id <= 4 else HELD_OUT_ONE_SHOT_ROLE,
        )  # fmt: skip

    def goal_dist(goal, row, col):
        return shortest_path_distances(component, (row, col))[(goal.row, goal.col)]

    def probe_spec(id, row, col):
        return StartSpec(
            start_id=id, row=row, col=col,
            distances_to_goals={str(goal.goal_id): int(goal_dist(goal, row, col)) for goal in goals},
        )  # fmt: skip

    goals = [goal_spec(goal_id, row, col) for goal_id, (row, col) in enumerate(contract.goal_cells, start=1)]
    probe_starts = [probe_spec(start_id, row, col) for start_id, (row, col) in enumerate(contract.probe_start_cells, start=2)]
    distance_bins = _build_distance_bins(goals)

    return DungeonBenchmarkEntry(
        layout_id=entry.id,
        split=split,
        dataset_root=str(dataset_root),
        canonical_start=GridPoint(row=contract.canonical_start[0], col=contract.canonical_start[1]),
        goals=goals,
        probe_starts=probe_starts,
        largest_component_size=int(component.sum()),
        distance_bins=distance_bins,
        generation_seed=seed,
    )


# =================================================================================================
def _build_distance_bins(  # ----------------------------------------------------------------------
    goals: Sequence[GoalSpec],
) -> dict[str, list[int]]:  # fmt: skip
    """Return the B1-ready distance-bin summary keyed by canonical-start distance."""
    distance_bins: dict[str, list[int]] = defaultdict(list)
    for goal in goals:
        if goal.role != TRAIN_ROLE:
            continue
        distance_bins[str(goal.distance_from_canonical_start)].append(goal.goal_id)
    return {key: sorted(value) for key, value in sorted(distance_bins.items(), key=lambda item: int(item[0]))}


# =================================================================================================
# Entry Point
# -------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    app()
