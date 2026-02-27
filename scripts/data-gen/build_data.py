from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from typer import Option, Typer, echo

from ehc_sn.data.index import MazeIndexEntry, write_index
from ehc_sn.data.schema import CHANNEL_GOALS, CHANNEL_SOLUTION, CHANNEL_START, CHANNEL_TOPOLOGY

app = Typer(pretty_exceptions_enable=False)
MAZEHARD_REPO = "sapientinc/maze-30x30-hard-1k"
RAW_PATH = "data/raw/maze-30x30-hard-1k"
PROCESSED_PATH = "data/processed/maze-30x30-hard-1k"


# =================================================================================================
# CLI command
# -------------------------------------------------------------------------------------------------


# =================================================================================================
@app.command()
def process_huggingface(  # -----------------------------------------------------------------------
    raw_dir: Path = Option(Path(RAW_PATH), "--raw-dir", help="Directory to store downloaded raw CSV files."),
    out_dir: Path = Option(Path(PROCESSED_PATH), "--out-dir", help="Output directory for processed data."),
    repo: str = Option(MAZEHARD_REPO, "--repo", help="HuggingFace dataset repo ID."),
    splits: list[str] = Option(["train", "val", "test"], "--splits", help="Dataset splits to download and process."),
) -> None:  # fmt: skip
    """Download mazes from HuggingFace, save raw CSVs, and build canonical NPZ + JSONL index."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"

    # Remove existing index so we rebuild from scratch on each run.
    if index_path.exists():
        index_path.unlink()
        echo(f"Removed existing index: {index_path}")

    maze_id = 0
    for split in splits:
        echo(f"Downloading {repo} / {split}.csv …")
        hf_path = Path(hf_hub_download(repo_id=repo, filename=f"{split}.csv", repo_type="dataset"))
        raw_csv = raw_dir / f"{split}.csv"
        shutil.copy2(hf_path, raw_csv)
        echo(f"  → Raw CSV saved to {raw_csv}")
        echo(f"Processing split '{split}' from {raw_csv} …")
        entries = _process_csv(raw_csv, split=split, out_dir=out_dir, source=repo, start_id=maze_id)
        write_index(entries, index_path, append=True)
        maze_id += len(entries)
        echo(f"  → {len(entries)} mazes written (total so far: {maze_id})")

    echo(f"\nDone. Index written to {index_path}")


# =================================================================================================
# Internal helpers
# -------------------------------------------------------------------------------------------------


# =================================================================================================
def _process_csv(  # ------------------------------------------------------------------------------
    csv_path: Path, split: str, out_dir: Path, source: str, start_id: int,
) -> list[MazeIndexEntry]:  # fmt: skip
    """Parse one CSV split file, write NPZ files, and return index entries."""
    out_split = out_dir / split
    out_split.mkdir(parents=True, exist_ok=True)

    entries: list[MazeIndexEntry] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for local_idx, row in enumerate(reader):
            maze_id = start_id + local_idx
            idx_entry, channels = _gen_maze(maze_id, row, split, source)
            np.savez_compressed(out_dir / idx_entry.file, **channels)
            entries.append(idx_entry)

    return entries


# =================================================================================================
def _gen_maze(  # ---------------------------------------------------------------------------------
    maze_id: int, row: dict[str, str], split: str, source: str,
) -> tuple[MazeIndexEntry, dict[str, np.ndarray]]:  # fmt: skip
    """Generate NPZ channels and index entry for one maze."""
    q_grid, a_grid = _grid_to_array(row["question"]), _grid_to_array(row["answer"])
    channels = {
        CHANNEL_TOPOLOGY: (q_grid != "#"),
        CHANNEL_START: (q_grid == "S"),
        CHANNEL_GOALS: (q_grid == "G"),
        CHANNEL_SOLUTION: np.where(a_grid == "o", 1, 0).astype(np.int32),
    }
    idx_entry = MazeIndexEntry(
        id=str(maze_id),
        file=f"{split}/maze_{maze_id:05d}.npz",
        source=source,
        split=split,
        shape=q_grid.shape,
        channels=list(channels.keys()),
        difficulty=row.get("rating", ""),
    )
    return idx_entry, channels

# =================================================================================================
def _grid_to_array(  # ----------------------------------------------------------------------------
    flat: str,
) -> np.ndarray:  # fmt: skip
    """Convert a flat character string to a 2-D character array.

    Supports both newline-delimited rows and a pure flat string (assumed square).
    """
    flat = flat.strip()
    if "\n" in flat:
        rows = [list(line) for line in flat.split("\n")]
    else:
        side = int(math.isqrt(len(flat)))
        if side * side != len(flat):
            raise ValueError(f"Flat grid string length {len(flat)} is not a perfect square.")
        rows = [list(flat[i * side : (i + 1) * side]) for i in range(side)]
    return np.array(rows, dtype="U1")


# =================================================================================================
if __name__ == "__main__":
    app()
# =================================================================================================
if __name__ == "__main__":
    app()
