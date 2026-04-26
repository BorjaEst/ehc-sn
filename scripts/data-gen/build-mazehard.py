"""Build a canonical processed MazeHard dataset from scratch.

Generates deterministic static spatial samples (topology, mask_valid, start,
goals, solution) and writes them as stacked ``.npy`` files under a structured
output root that the existing ``MazeDataset`` / ``Datamodule`` pipeline can
load directly.

Usage examples
--------------
Quick local build (default output root, small sample counts)::

    python build-mazehard.py

Larger overridden build::

    python build-mazehard.py \\
        --output-root /scratch/data/processed/mazehard \\
        --n-train 4000 --n-val 500 --n-test 500 \\
        --height 17 --width 17 --seed 1234 --overwrite
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import deque
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

# ---------------------------------------------------------------------------
# Make the package importable when running the script directly.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from ehc_sn.data._canonical import farthest_reachable_cell, first_true_cell, largest_component_mask, singleton_mask
from ehc_sn.data.index import MazeIndexEntry, write_index
from ehc_sn.data.schema import validate_npz

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SOURCE = "mazehard"
CHANNELS = ["topology", "mask_valid", "start", "goals", "solution"]
SPLITS = ("train", "val", "test")
_DIRS4 = ((-1, 0), (0, -1), (0, 1), (1, 0))
_SPLIT_SEED_OFFSET: dict[str, int] = {"train": 0, "val": 100_000, "test": 200_000}


# ---------------------------------------------------------------------------
# Topology generation
# ---------------------------------------------------------------------------


def _make_topology(H: int, W: int, *, rng: np.random.Generator) -> np.ndarray:
    """Return a random passable mask with outer walls and ~20 % interior blocks."""
    topology = np.zeros((H, W), dtype=bool)
    topology[1:-1, 1:-1] = True
    interior = np.argwhere(topology)
    n_block = max(0, int(0.20 * len(interior)))
    if n_block:
        idxs = rng.choice(len(interior), size=n_block, replace=False)
        for i in idxs:
            r, c = map(int, interior[i])
            topology[r, c] = False
    return topology


def _path_mask(mask_valid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> np.ndarray:
    """Return int32 mask with 1 on BFS-shortest-path cells, 0 elsewhere."""
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue: deque[tuple[int, int]] = deque([start])
    found = False
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            found = True
            break
        for dr, dc in _DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask_valid.shape[0] and 0 <= nc < mask_valid.shape[1]:
                if mask_valid[nr, nc] and (nr, nc) not in parent:
                    parent[(nr, nc)] = (r, c)
                    queue.append((nr, nc))

    out = np.zeros(mask_valid.shape, dtype=np.int32)
    if not found:
        return out
    cell: tuple[int, int] | None = goal
    while cell is not None:
        out[cell] = 1
        cell = parent[cell]
    return out


# ---------------------------------------------------------------------------
# Per-sample generation
# ---------------------------------------------------------------------------


def _generate_sample(
    H: int,
    W: int,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    """Generate one deterministic MazeHard sample."""
    rng = np.random.default_rng(seed)

    topology = _make_topology(H, W, rng=rng)
    mask_valid = largest_component_mask(topology)

    start_cell = first_true_cell(mask_valid)
    if start_cell is None:
        raise RuntimeError(f"Empty valid mask for seed={seed}.")
    goal_cell = farthest_reachable_cell(mask_valid, start_cell)

    start = singleton_mask((H, W), start_cell)
    goals = singleton_mask((H, W), goal_cell)
    solution = _path_mask(mask_valid, start_cell, goal_cell)

    return {
        "topology": topology,
        "mask_valid": mask_valid,
        "start": start,
        "goals": goals,
        "solution": solution,
    }


# ---------------------------------------------------------------------------
# Pure builder function
# ---------------------------------------------------------------------------


def build_mazehard_dataset(
    output_root: Path,
    *,
    n_train: int = 200,
    n_val: int = 40,
    n_test: int = 40,
    height: int = 11,
    width: int = 11,
    seed: int = 42,
    overwrite: bool = False,
) -> None:
    """Build a canonical MazeHard processed dataset.

    Writes stacked ``.npy`` files and an ``index.jsonl`` under ``output_root``
    in the layout expected by :class:`~ehc_sn.data.datasets.MazeDataset`.

    Args:
        output_root: Destination directory (created if absent).
        n_train: Number of training samples.
        n_val: Number of validation samples.
        n_test: Number of test samples.
        height: Grid height in cells (including outer walls).
        width: Grid width in cells (including outer walls).
        seed: Base RNG seed; per-sample seeds are derived deterministically.
        overwrite: If ``True``, delete and recreate ``output_root``; if
            ``False`` and ``output_root`` already exists, raise ``FileExistsError``.

    Raises:
        FileExistsError: When ``output_root`` exists and ``overwrite`` is
            ``False``.
        RuntimeError: When a generated sample has an empty valid mask.
    """
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}. Use --overwrite to rebuild.")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True)

    split_counts = {"train": n_train, "val": n_val, "test": n_test}
    all_entries: list[MazeIndexEntry] = []

    for split in SPLITS:
        n = split_counts[split]
        split_dir = output_root / split
        split_dir.mkdir()

        stacked: dict[str, list[np.ndarray]] = {ch: [] for ch in CHANNELS}

        for idx in range(n):
            sample_seed = seed + _SPLIT_SEED_OFFSET[split] + idx
            sample = _generate_sample(height, width, seed=sample_seed)
            for ch in CHANNELS:
                stacked[ch].append(sample[ch])

        # Validate spatial channels before writing.
        spatial_sample = {ch: stacked[ch][0] for ch in CHANNELS}
        validate_npz(spatial_sample)

        # Stack and save.
        arrays: dict[str, np.ndarray] = {ch: np.stack(stacked[ch], axis=0) for ch in CHANNELS}
        for ch, arr in arrays.items():
            np.save(split_dir / f"{ch}.npy", arr)

        # Write per-split dataset.json.
        (split_dir / "dataset.json").write_text(
            json.dumps(
                {
                    "source": SOURCE,
                    "split": split,
                    "n_samples": n,
                    "shape": [height, width],
                    "channels": CHANNELS,
                },
                indent=2,
            )
        )

        # Collect index entries.
        for idx in range(n):
            sample_id = f"{SOURCE}-{split}-{idx + 1:06d}"
            all_entries.append(
                MazeIndexEntry(
                    id=sample_id,
                    source=SOURCE,
                    split=split,
                    shape=(height, width),
                    channels=CHANNELS,
                    n_observations=0,
                    n_goals=1,
                    difficulty="medium",
                )
            )

    write_index(all_entries, output_root / "index.jsonl")
    print(f"MazeHard dataset written to {output_root}  ({sum(split_counts.values())} samples).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    output_root: Annotated[
        Path,
        typer.Option("--output-root", help="Dataset root directory."),
    ] = Path("data/processed/mazehard"),
    n_train: Annotated[int, typer.Option("--n-train", help="Training samples.")] = 200,
    n_val: Annotated[int, typer.Option("--n-val", help="Validation samples.")] = 40,
    n_test: Annotated[int, typer.Option("--n-test", help="Test samples.")] = 40,
    height: Annotated[int, typer.Option("--height", help="Grid height (cells).")] = 11,
    width: Annotated[int, typer.Option("--width", help="Grid width (cells).")] = 11,
    seed: Annotated[int, typer.Option("--seed", help="Base RNG seed.")] = 42,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite/--no-overwrite", help="Delete output root and rebuild."),
    ] = False,
) -> None:
    """Build a canonical MazeHard processed dataset."""
    build_mazehard_dataset(
        output_root.resolve(),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        height=height,
        width=width,
        seed=seed,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    app()
