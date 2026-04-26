"""Build a canonical processed Dungeon dataset from scratch.

Generates deterministic replay-style samples (topology, observations,
mask_valid, regions, landmarks, trajectory_*) and writes them as stacked
``.npy`` files under a structured output root compatible with the existing
Arena replay consumer in ``replay.py``.

The TEM configs (``training.tem-v1.toml`` and ``training.tem-v2.toml``) both
point ``dataset_path`` at ``data/processed/dungeon``.

Usage examples
--------------
Quick local build (default output root, small sample counts)::

    python build-dungeon.py

Larger overridden build::

    python build-dungeon.py \\
        --output-root /scratch/data/processed/dungeon \\
        --n-train 2000 --n-val 200 --n-test 200 \\
        --height 13 --width 13 --n-observations 8 \\
        --max-steps 80 --seed 7 --overwrite
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

from ehc_sn.data._canonical import binary_structural_landmarks, first_true_cell, largest_component_mask, sample_observations
from ehc_sn.data.index import MazeIndexEntry, write_index
from ehc_sn.data.schema import validate_npz

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SOURCE = "dungeon"
SPATIAL_CHANNELS = [
    "topology",
    "observations",
    "mask_valid",
    "regions",
    "landmarks",
]
TRAJECTORY_CHANNELS = [
    "trajectory_row",
    "trajectory_col",
    "trajectory_previous_action",
    "trajectory_episode_start",
    "trajectory_valid_step",
    "trajectory_length",
]
CHANNELS = SPATIAL_CHANNELS + TRAJECTORY_CHANNELS

SPLITS = ("train", "val", "test")
_DIRS4 = ((-1, 0), (0, -1), (0, 1), (1, 0))
# Action ids matching ehc_sn/tasks/_movement.py
ACTION_STAY = 0
ACTION_UP = 1
ACTION_RIGHT = 2
ACTION_DOWN = 3
ACTION_LEFT = 4
_ACTION_DELTAS = ((0, 0), (-1, 0), (0, 1), (1, 0), (0, -1))

_SPLIT_SEED_OFFSET: dict[str, int] = {"train": 0, "val": 100_000, "test": 200_000}


# ---------------------------------------------------------------------------
# Topology + spatial channel generation
# ---------------------------------------------------------------------------


def _make_topology(H: int, W: int, *, rng: np.random.Generator) -> np.ndarray:
    """Random passable mask with outer walls and ~25 % interior blocks."""
    topology = np.zeros((H, W), dtype=bool)
    topology[1:-1, 1:-1] = True
    interior = np.argwhere(topology)
    n_block = max(0, int(0.25 * len(interior)))
    if n_block:
        idxs = rng.choice(len(interior), size=n_block, replace=False)
        for i in idxs:
            r, c = map(int, interior[i])
            topology[r, c] = False
    return topology


def _room_regions(H: int, W: int, mask_valid: np.ndarray) -> np.ndarray:
    """Assign each valid cell to one of four quadrant rooms (0–3)."""
    regions = np.full((H, W), -1, dtype=np.int32)
    mid_r, mid_c = H // 2, W // 2
    for r, c in np.argwhere(mask_valid):
        room = (int(r) >= mid_r) * 2 + (int(c) >= mid_c)
        regions[r, c] = room
    return regions


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------


def _random_walk(
    mask_valid: np.ndarray,
    start: tuple[int, int],
    n_steps: int,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random walk over passable cells; returns (rows, cols, prev_actions)."""
    rows = np.zeros(n_steps, dtype=np.int32)
    cols = np.zeros(n_steps, dtype=np.int32)
    prev_actions = np.zeros(n_steps, dtype=np.int32)

    r, c = start
    rows[0], cols[0] = r, c
    prev_actions[0] = ACTION_STAY  # no action taken before first position

    for t in range(1, n_steps):
        valid_moves: list[tuple[int, int, int]] = []
        for action, (dr, dc) in enumerate(_ACTION_DELTAS):
            if action == ACTION_STAY:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask_valid.shape[0] and 0 <= nc < mask_valid.shape[1]:
                if mask_valid[nr, nc]:
                    valid_moves.append((action, nr, nc))

        if valid_moves:
            action, nr, nc = valid_moves[int(rng.integers(len(valid_moves)))]
        else:
            action, nr, nc = ACTION_STAY, r, c

        r, c = nr, nc
        rows[t] = r
        cols[t] = c
        prev_actions[t] = action

    return rows, cols, prev_actions


# ---------------------------------------------------------------------------
# Per-sample generation
# ---------------------------------------------------------------------------


def _generate_sample(
    H: int,
    W: int,
    n_observations: int,
    max_steps: int,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    """Generate one deterministic Dungeon replay sample."""
    rng = np.random.default_rng(seed)

    topology = _make_topology(H, W, rng=rng)
    mask_valid = largest_component_mask(topology)

    start_cell = first_true_cell(mask_valid)
    if start_cell is None:
        raise RuntimeError(f"Empty valid mask for seed={seed}.")

    observations = sample_observations(mask_valid, n_observations, seed=int(rng.integers(2**31)))
    regions = _room_regions(H, W, mask_valid)
    landmarks = binary_structural_landmarks(mask_valid)

    # Trajectory: random walk of exactly max_steps valid steps.
    traj_seed = int(rng.integers(2**31))
    traj_rng = np.random.default_rng(traj_seed)
    rows, cols, prev_actions = _random_walk(mask_valid, start_cell, max_steps, rng=traj_rng)

    traj_length = np.int32(max_steps)
    episode_start = np.zeros(max_steps, dtype=bool)
    episode_start[0] = True
    valid_step = np.arange(max_steps, dtype=np.int32) < traj_length

    # Assertions: shape, bounds, passable, prefix invariant.
    assert rows.shape == (max_steps,) and cols.shape == (max_steps,)
    assert np.all((rows >= 0) & (rows < H)), "row out of bounds"
    assert np.all((cols >= 0) & (cols < W)), "col out of bounds"
    for t in range(max_steps):
        assert mask_valid[rows[t], cols[t]], f"step {t} not on passable cell"
    assert np.all(valid_step == (np.arange(max_steps) < traj_length)), "prefix invariant"

    return {
        "topology": topology,
        "observations": observations,
        "mask_valid": mask_valid,
        "regions": regions,
        "landmarks": landmarks,
        "trajectory_row": rows,
        "trajectory_col": cols,
        "trajectory_previous_action": prev_actions,
        "trajectory_episode_start": episode_start,
        "trajectory_valid_step": valid_step,
        "trajectory_length": np.array(traj_length, dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# Pure builder function
# ---------------------------------------------------------------------------


def build_dungeon_dataset(
    output_root: Path,
    *,
    n_train: int = 200,
    n_val: int = 40,
    n_test: int = 40,
    height: int = 11,
    width: int = 11,
    n_observations: int = 6,
    max_steps: int = 50,
    seed: int = 42,
    overwrite: bool = False,
) -> None:
    """Build a canonical Dungeon processed dataset.

    Writes stacked ``.npy`` files and an ``index.jsonl`` under ``output_root``
    in the layout expected by :class:`~ehc_sn.data.datasets.MazeDataset`.

    Args:
        output_root: Destination directory (created if absent).
        n_train: Number of training samples.
        n_val: Number of validation samples.
        n_test: Number of test samples.
        height: Grid height in cells (including outer walls).
        width: Grid width in cells (including outer walls).
        n_observations: Number of distinct observation ids to assign.
        max_steps: Trajectory length (all trajectories are this long).
        seed: Base RNG seed; per-sample seeds are derived deterministically.
        overwrite: If ``True``, delete and recreate ``output_root``; if
            ``False`` and ``output_root`` already exists, raise
            ``FileExistsError``.

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
            sample = _generate_sample(height, width, n_observations, max_steps, seed=sample_seed)
            for ch in CHANNELS:
                stacked[ch].append(sample[ch])

        # Validate only the spatial channels.
        spatial_sample = {ch: stacked[ch][0] for ch in SPATIAL_CHANNELS}
        validate_npz(spatial_sample)

        # Stack and save all channels.
        for ch in SPATIAL_CHANNELS:
            arr = np.stack(stacked[ch], axis=0)
            np.save(split_dir / f"{ch}.npy", arr)

        # Trajectory: row/col/prev_action/valid_step/episode_start are (N, T).
        for ch in ["trajectory_row", "trajectory_col", "trajectory_previous_action", "trajectory_episode_start", "trajectory_valid_step"]:
            arr = np.stack(stacked[ch], axis=0)
            np.save(split_dir / f"{ch}.npy", arr)

        # trajectory_length is (N,).
        traj_len_arr = np.array(stacked["trajectory_length"], dtype=np.int32)
        np.save(split_dir / "trajectory_length.npy", traj_len_arr)

        # Per-split dataset.json.
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

        # Index entries.
        for idx in range(n):
            sample_id = f"{SOURCE}-{split}-{idx + 1:06d}"
            all_entries.append(
                MazeIndexEntry(
                    id=sample_id,
                    source=SOURCE,
                    split=split,
                    shape=(height, width),
                    channels=CHANNELS,
                    n_observations=n_observations,
                    n_goals=0,
                    difficulty="medium",
                )
            )

    write_index(all_entries, output_root / "index.jsonl")
    print(f"Dungeon dataset written to {output_root}  ({sum(split_counts.values())} samples).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    output_root: Annotated[
        Path,
        typer.Option("--output-root", help="Dataset root directory."),
    ] = Path("data/processed/dungeon"),
    n_train: Annotated[int, typer.Option("--n-train", help="Training samples.")] = 200,
    n_val: Annotated[int, typer.Option("--n-val", help="Validation samples.")] = 40,
    n_test: Annotated[int, typer.Option("--n-test", help="Test samples.")] = 40,
    height: Annotated[int, typer.Option("--height", help="Grid height (cells).")] = 11,
    width: Annotated[int, typer.Option("--width", help="Grid width (cells).")] = 11,
    n_observations: Annotated[int, typer.Option("--n-observations", help="Distinct observation ids.")] = 6,
    max_steps: Annotated[int, typer.Option("--max-steps", help="Trajectory length per sample.")] = 50,
    seed: Annotated[int, typer.Option("--seed", help="Base RNG seed.")] = 42,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite/--no-overwrite", help="Delete output root and rebuild."),
    ] = False,
) -> None:
    """Build a canonical Dungeon processed dataset."""
    build_dungeon_dataset(
        output_root.resolve(),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        height=height,
        width=width,
        n_observations=n_observations,
        max_steps=max_steps,
        seed=seed,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    app()
