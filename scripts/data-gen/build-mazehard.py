"""Download and process MazeHard splits plus derived benchmark subsets."""

from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RemoteEntryNotFoundError
from typer import Option, Typer, echo

from ehc_sn.data._canonical import (
    binary_structural_landmarks,
    canonical_cell_from_mask,
    farthest_reachable_cell,
    largest_component_mask,
    sample_observations,
    singleton_mask,
    stable_text_seed,
)
from ehc_sn.data.benchmarks.mazehard_b0 import MazeHardSubsetManifest, b0_hard_subset_path, write_mazehard_subset_manifest
from ehc_sn.data.datasets import MazeMetadata
from ehc_sn.data.index import MazeIndexEntry, read_index, write_index
from ehc_sn.data.schema import (
    CHANNEL_GOALS,
    CHANNEL_LANDMARKS,
    CHANNEL_MASK_VALID,
    CHANNEL_OBSERVATIONS,
    CHANNEL_SOLUTION,
    CHANNEL_START,
    CHANNEL_TOPOLOGY,
)

# =================================================================================================
# Configuration
# -------------------------------------------------------------------------------------------------

app = Typer(pretty_exceptions_enable=False)
MAZEHARD_REPO = "sapientinc/maze-30x30-hard-1k"
RAW_PATH = "data/raw/maze-30x30-hard-1k"
PROCESSED_PATH = "data/processed/maze-30x30-hard-1k"


# =================================================================================================
# CLI Commands
# -------------------------------------------------------------------------------------------------


# =================================================================================================
@app.command()
def process_huggingface(  # -----------------------------------------------------------------------
    raw_dir: Path = Option(Path(RAW_PATH), "--raw-dir", help="Directory to store downloaded raw CSV files."),
    out_dir: Path = Option(Path(PROCESSED_PATH), "--out-dir", help="Output directory for processed data."),
    repo: str = Option(MAZEHARD_REPO, "--repo", help="HuggingFace dataset repo ID."),
    splits: list[str] = Option(["train", "val", "test"], "--splits", help="Dataset splits to download and process."),
    n_observations: int = Option(45, "--n-obs", help="Observation vocabulary size for deterministic assignment."),
) -> None:  # fmt: skip
    """Download mazes from HuggingFace, save raw CSVs, and build per-channel .npy files + JSONL index."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"

    # Remove existing index so we rebuild from scratch on each run.
    if index_path.exists():
        index_path.unlink()
        echo(f"Removed existing index: {index_path}")

    maze_id = 0
    for split in splits:
        echo(f"Resolving split '{split}' …")
        raw_csv, provenance = _resolve_split_csv(raw_dir=raw_dir, repo=repo, split=split)
        if raw_csv is None:
            continue
        echo(f"Processing split '{split}' …")
        entries = _process_csv(
            raw_csv,
            split=split,
            out_dir=out_dir,
            source=repo,
            provenance=provenance,
            start_id=maze_id,
            n_observations=n_observations,
        )
        write_index(entries, index_path, append=True)
        maze_id += len(entries)
        echo(f"  → {len(entries)} mazes written (total so far: {maze_id})")

    echo(f"\nDone. Index written to {index_path}")


@app.command("build-b0-hard-subset")
def build_b0_hard_subset(  # ----------------------------------------------------------------------
    dataset_root: Path = Option(Path(PROCESSED_PATH), "--dataset-root", help="Processed MazeHard dataset root."),
    split: str = Option("test", "--split", help="Dataset split used to derive the benchmark hard subset."),
    quantile: float = Option(0.9, "--quantile", min=0.0, max=1.0, help="Inclusive difficulty quantile threshold."),
    out_path: Path | None = Option(None, "--out-path", help="Optional override for the hard-subset manifest path."),
) -> None:  # fmt: skip
    """Materialize the preregistered B0 hard-subset manifest from the processed difficulty index."""
    index_path = dataset_root / "index.jsonl"
    entries = [entry for entry in read_index(index_path) if entry.split == split]
    if not entries:
        raise ValueError(f"No MazeHard index entries found for split '{split}' in {index_path}.")

    difficulties = [_parse_difficulty_value(entry.difficulty, entry_id=entry.id) for entry in entries]
    threshold = int(np.quantile(np.asarray(difficulties, dtype=np.int32), quantile, method="higher"))

    selected = sorted(
        ((entry.id, difficulty) for entry, difficulty in zip(entries, difficulties, strict=True) if difficulty >= threshold),
        key=lambda item: int(item[0]) if item[0].isdigit() else item[0],
    )
    if not selected:
        raise ValueError(f"No MazeHard entries met the requested hard-subset threshold {threshold} for split '{split}'.")

    manifest = MazeHardSubsetManifest(
        dataset_root=str(dataset_root),
        source=entries[0].source,
        split=split,
        selection_rule=f"difficulty >= split quantile {quantile:.3f} (threshold={threshold})",
        sample_ids=[sample_id for sample_id, _ in selected],
        difficulty_values=[difficulty for _, difficulty in selected],
        n_selected=len(selected),
    )
    resolved_out_path = b0_hard_subset_path(dataset_root) if out_path is None else out_path
    write_mazehard_subset_manifest(manifest, resolved_out_path)
    echo(f"Wrote B0 hard-subset manifest with {manifest.n_selected} ids to {resolved_out_path}")


# =================================================================================================
# Source Resolution
# -------------------------------------------------------------------------------------------------


# =================================================================================================
def _resolve_split_csv(
    *, raw_dir: Path, repo: str, split: str,
) -> tuple[Path | None, dict[str, str]]:  # fmt: skip
    """Resolve one raw CSV split, preferring local overrides over remote downloads."""
    raw_csv = raw_dir / f"{split}.csv"
    if raw_csv.exists():
        echo(f"  → Using local override: {raw_csv}")
        return raw_csv, _build_provenance(split=split, source=repo, raw_csv=raw_csv, origin="local_override")

    hf_path = _download_remote_split(repo=repo, split=split)
    if hf_path is None:
        echo(f"  → Split '{split}' not found locally or in {repo}, skipping.")
        return None, _build_provenance(split=split, source=repo, raw_csv=raw_csv, origin="missing")

    shutil.copy2(hf_path, raw_csv)
    echo(f"  → Downloaded remote split to {raw_csv}")
    return raw_csv, _build_provenance(split=split, source=repo, raw_csv=raw_csv, origin="remote_download")


# =================================================================================================
def _download_remote_split(
    *, repo: str, split: str,
) -> Path | None:  # fmt: skip
    """Download one split CSV from HuggingFace and return the cached file path."""
    try:
        return Path(hf_hub_download(repo_id=repo, filename=f"{split}.csv", repo_type="dataset"))
    except RemoteEntryNotFoundError:
        return None


# =================================================================================================
def _build_provenance(
    *, split: str, source: str, raw_csv: Path, origin: str,
) -> dict[str, str]:  # fmt: skip
    """Build per-split provenance metadata written into dataset.json."""
    return {
        "origin": origin,
        "raw_path": str(raw_csv),
        "remote_repo": source,
        "remote_file": f"{split}.csv",
    }


# =================================================================================================
# CSV Processing
# -------------------------------------------------------------------------------------------------


# =================================================================================================
def _process_csv(  # ------------------------------------------------------------------------------
    csv_path: Path, split: str, out_dir: Path, source: str, provenance: dict[str, str], start_id: int,
    n_observations: int,
) -> list[MazeIndexEntry]:  # fmt: skip
    """Parse one CSV split, write per-channel .npy files, dataset.json, and return index entries."""
    all_channels: list[dict[str, np.ndarray]] = []
    difficulties: list[str] = []

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            channels, difficulty = _process_csv_row(row, n_observations=n_observations)
            all_channels.append(channels)
            difficulties.append(difficulty)

    # Stack per channel → (N, H, W) and save as individual .npy files.
    meta = _build_metadata(source, split, channels=all_channels, provenance=provenance)
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    for ch in meta.channels:
        arr = np.stack([c[ch] for c in all_channels])
        np.save(split_dir / f"{ch}.npy", arr)

    # Write per-split dataset.json metadata.
    (split_dir / "dataset.json").write_text(meta.model_dump_json(indent=2))

    # Build index entries referencing the split directory and row indices.
    entries: list[MazeIndexEntry] = []
    for i in range(meta.n_samples):
        entry = _build_idx_entry(meta, all_channels, difficulties, i=i, start_id=start_id)
        entries.append(entry)

    return entries


def _parse_difficulty_value(difficulty: str, *, entry_id: str) -> int:
    """Return one integer MazeHard difficulty value from the processed index."""
    try:
        return int(difficulty)
    except ValueError as exc:
        raise ValueError(f"MazeHard entry {entry_id} has a non-integer difficulty value: {difficulty!r}.") from exc


# =================================================================================================
def _process_csv_row(  # --------------------------------------------------------------------------
    row: dict[str, str], *, n_observations: int,
) -> tuple[dict[str, np.ndarray], str]:  # fmt: skip
    """Parse one CSV row into channel arrays and difficulty string."""
    q_grid, a_grid = _grid_to_array(row["question"]), _grid_to_array(row["answer"])
    topology = q_grid != "#"
    mask_valid = largest_component_mask(topology)
    start_cell = canonical_cell_from_mask(q_grid == "S", mask_valid)
    if start_cell is None:
        fallback = canonical_cell_from_mask(mask_valid, mask_valid)
        if fallback is None:
            raise ValueError("MazeHard row has no valid passable cells for a canonical start.")
        start_cell = fallback

    goal_cell = canonical_cell_from_mask(q_grid == "G", mask_valid, reference=start_cell)
    if goal_cell is None:
        goal_cell = farthest_reachable_cell(mask_valid, start_cell)

    channels = {
        CHANNEL_TOPOLOGY: topology,
        CHANNEL_MASK_VALID: mask_valid,
        CHANNEL_START: singleton_mask(topology.shape, start_cell),
        CHANNEL_GOALS: singleton_mask(topology.shape, goal_cell),
        CHANNEL_SOLUTION: np.where(a_grid == "o", 1, 0).astype(np.int32),
        CHANNEL_OBSERVATIONS: sample_observations(
            mask_valid,
            n_observations,
            seed=stable_text_seed(row["question"]),
        ),
        CHANNEL_LANDMARKS: binary_structural_landmarks(mask_valid),
    }
    return channels, row.get("rating", "")


# =================================================================================================
def _grid_to_array(  # ----------------------------------------------------------------------------
    flat: str,
) -> np.ndarray:  # fmt: skip
    """Convert a flat character string to a 2-D character array.

    Supports both newline-delimited rows and a pure flat string (assumed square).
    """
    flat = flat.strip("\n\r")
    if "\n" in flat:
        rows = [list(line) for line in flat.split("\n")]
    else:
        side = int(math.isqrt(len(flat)))
        if side * side != len(flat):
            raise ValueError(f"Flat grid string length {len(flat)} is not a perfect square.")
        rows = [list(flat[i * side : (i + 1) * side]) for i in range(side)]
    return np.array(rows, dtype="U1")


# =================================================================================================
# Processed Metadata
# -------------------------------------------------------------------------------------------------


# =================================================================================================
def _build_metadata(
    source: str, split: str, *, channels: list[dict[str, np.ndarray]], provenance: dict[str, str],
) -> MazeMetadata:  # fmt: skip
    """Extract metadata from channel dict for dataset.json."""
    return MazeMetadata(
        source=source,
        split=split,
        n_samples=len(channels),
        shape=list(channels[0][CHANNEL_TOPOLOGY].shape),
        channels=list(channels[0].keys()),
        **provenance,
    )


# =================================================================================================
def _build_idx_entry(  # --------------------------------------------------------------------------
    meta: MazeMetadata, channels: list[dict[str, np.ndarray]], difficulties: list[str], *,
    i: int, start_id: int = 0,
) -> MazeIndexEntry:  # fmt: skip
    """Build one MazeIndexEntry for the i-th maze in the split."""
    obs_arr = channels[i][CHANNEL_OBSERVATIONS]
    valid = channels[i][CHANNEL_MASK_VALID]
    n_obs = int(obs_arr[valid].max()) + 1 if valid.any() else 0
    return MazeIndexEntry(
        id=str(start_id + i),
        source=meta.source,
        split=meta.split,
        shape=meta.shape,
        channels=meta.channels,
        n_observations=n_obs,
        n_goals=int(channels[i][CHANNEL_GOALS].sum()),
        difficulty=difficulties[i],
    )


# =================================================================================================
# Entry Point
# -------------------------------------------------------------------------------------------------


# =================================================================================================
if __name__ == "__main__":
    app()
