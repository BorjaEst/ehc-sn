from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
from dungeongen.layout.generator import DungeonGenerator
from dungeongen.layout.models import Dungeon
from dungeongen.layout.occupancy import CellType
from dungeongen.layout.params import DungeonArchetype, DungeonSize, GenerationParams
from typer import Option, Typer, echo

from ehc_sn.data.datasets import MazeMetadata
from ehc_sn.data.index import MazeIndexEntry, write_index
from ehc_sn.data.schema import CHANNEL_MASK_VALID, CHANNEL_OBSERVATIONS, CHANNEL_REGIONS, CHANNEL_TOPOLOGY

app = Typer(pretty_exceptions_enable=False)
RAW_PATH = "data/raw/dungeons"
PROCESSED_PATH = "data/processed/dungeons"
PASSABLE_TYPES = frozenset({CellType.ROOM, CellType.PASSAGE, CellType.DOOR})


# =================================================================================================
# CLI command
# -------------------------------------------------------------------------------------------------


# =================================================================================================
@app.command()
def process_dungeongen(  # ------------------------------------------------------------------------
    out_dir: Path = Option(Path(PROCESSED_PATH), "--out-dir", help="Output directory for processed data."),
    raw_dir: Path = Option(Path(RAW_PATH), "--raw-dir", help="Directory for serialised raw dungeons."),
    n_train: int = Option(800, "--n-train", help="Number of training dungeons to generate."),
    n_val: int = Option(100, "--n-val", help="Number of validation dungeons to generate."),
    n_test: int = Option(100, "--n-test", help="Number of test dungeons to generate."),
    size: str = Option("small", "--size", help="Dungeon size (tiny, small, medium, large, xlarge)."),
    archetype: str = Option("classic", "--archetype", help="Dungeon archetype (classic, warren, temple, crypt, cavern, fortress, lair)."),
    n_observations: int = Option(45, "--n-obs", help="Observation vocabulary size for random assignment."),
    density: float = Option(0.6, "--density", help="Room packing density (0.0 sparse … 1.0 tight)."),
    seed: int = Option(42, "--seed", help="Base RNG seed (incremented per dungeon)."),
    margin: int = Option(1, "--margin", help="Grid margin (wall border) around the dungeon bounding box."),
) -> None:  # fmt: skip
    """Generate dungeons with dungeongen, rasterize to grids, and write per-channel .npy files + JSONL index."""
    dg_size = DungeonSize[size.upper()]
    dg_arch = DungeonArchetype[archetype.upper()]
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    index_path = out_dir / "index.jsonl"
    if index_path.exists():
        index_path.unlink()
        echo(f"Removed existing index: {index_path}")

    splits: list[tuple[str, int]] = [("train", n_train), ("val", n_val), ("test", n_test)]
    maze_id = 0

    for split, count in splits:
        echo(f"Generating {count} {split} dungeons (size={size}, archetype={archetype}) …")
        all_channels: list[dict[str, np.ndarray]] = []
        raw_records: list[dict] = []

        for i in range(count):
            dungeon_seed = seed + maze_id + i
            params = GenerationParams(size=dg_size, archetype=dg_arch, density=density, seed=dungeon_seed)
            generator = DungeonGenerator(params)
            dungeon = generator.generate(seed=dungeon_seed)

            # Collect raw dungeon record in memory (written as compressed archive later).
            raw_records.append(_dungeon_to_dict(dungeon))

            # Rasterize dungeon → channel arrays.
            channels = _rasterize_dungeon(generator, dungeon, n_observations, margin, rng_seed=dungeon_seed)
            all_channels.append(channels)

        # Flush all raw records to a single compressed JSONL file.
        _save_raw_split(raw_records, raw_dir / f"{split}.jsonl.gz")

        # Pad all dungeons in this split to the same (H, W) so arrays can be stacked.
        all_channels = _pad_channels_to_common_shape(all_channels)

        # Build per-split metadata and write stacked .npy files.
        meta = _build_metadata(split=split, channels=all_channels)
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for ch in meta.channels:
            arr = np.stack([c[ch] for c in all_channels])
            np.save(split_dir / f"{ch}.npy", arr)
        (split_dir / "dataset.json").write_text(meta.model_dump_json(indent=2))

        # Build JSONL index entries.
        entries = [_build_idx_entry(meta, all_channels, i=i, start_id=maze_id) for i in range(count)]
        write_index(entries, index_path, append=True)
        maze_id += count
        echo(f"  → {count} dungeons written (total so far: {maze_id})")

    echo(f"\nDone. Index written to {index_path}")


# =================================================================================================
# Internal helpers
# -------------------------------------------------------------------------------------------------


# =================================================================================================
def _rasterize_dungeon(  # ------------------------------------------------------------------------
    generator: DungeonGenerator, dungeon: Dungeon, n_observations: int, margin: int, rng_seed: int,
) -> dict[str, np.ndarray]:  # fmt: skip
    """Convert a dungeongen ``Dungeon`` + its generator's occupancy grid into canonical channel arrays.

    The rasterization proceeds as:
      1. Compute tight bounding box from the dungeon bounds.
      2. Add *margin* wall tiles on all sides.
      3. For each grid cell classify passable vs wall from the occupancy grid.
      4. Assign room/region IDs and random observation IDs to passable cells.
    """
    bx1, by1, bx2, by2 = dungeon.bounds  # exclusive upper bounds
    # Grid dimensions including margin on all sides.
    H = (by2 - by1) + 2 * margin
    W = (bx2 - bx1) + 2 * margin
    # Origin offset: maps dungeon (x, y) → grid (row, col).
    ox, oy = bx1 - margin, by1 - margin

    occupancy = generator.occupancy

    # -- Topology: passable cells -------------------------------------------------
    topology = np.zeros((H, W), dtype=bool)
    for dy in range(H):
        for dx in range(W):
            cell_type = occupancy.get_type(ox + dx, oy + dy)
            if cell_type in PASSABLE_TYPES:
                topology[dy, dx] = True

    # -- Regions: room ID per tile ------------------------------------------------
    regions = np.zeros((H, W), dtype=np.int32)
    room_id_map: dict[str, int] = {}
    for rid, cells in occupancy.room_cells.items():
        numeric_id = room_id_map.setdefault(rid, len(room_id_map) + 1)
        for cx, cy in cells:
            col, row = cx - ox, cy - oy
            if 0 <= row < H and 0 <= col < W:
                regions[row, col] = numeric_id

    # -- Mask valid: explicit reachability (largest connected component) -----------
    mask_valid = _largest_component_mask(topology)

    # -- Observations: random assignment over passable & valid cells ---------------
    rng = np.random.default_rng(rng_seed)
    observations = np.full((H, W), -1, dtype=np.int32)
    valid_coords = np.argwhere(mask_valid)
    obs_ids = rng.integers(0, n_observations, size=len(valid_coords))
    for (r, c), obs in zip(valid_coords, obs_ids):
        observations[r, c] = obs

    return {
        CHANNEL_TOPOLOGY: topology,
        CHANNEL_REGIONS: regions,
        CHANNEL_MASK_VALID: mask_valid,
        CHANNEL_OBSERVATIONS: observations,
    }


# =================================================================================================
def _pad_channels_to_common_shape(  # ------------------------------------------------------------
    all_channels: list[dict[str, np.ndarray]],
) -> list[dict[str, np.ndarray]]:  # fmt: skip
    """Pad every channel array to the maximum (H, W) found across the batch.

    Padding uses ``False`` for bool channels and ``-1`` for int channels
    (matching the convention that ``-1`` means "no observation / wall").
    The topology channel is padded with ``False`` (wall), so padded cells are
    automatically excluded from walkability.
    """
    max_h = max(c[CHANNEL_TOPOLOGY].shape[0] for c in all_channels)
    max_w = max(c[CHANNEL_TOPOLOGY].shape[1] for c in all_channels)

    padded: list[dict[str, np.ndarray]] = []
    for ch_dict in all_channels:
        new = {}
        for name, arr in ch_dict.items():
            h, w = arr.shape
            if h == max_h and w == max_w:
                new[name] = arr
                continue
            pad_h, pad_w = max_h - h, max_w - w
            fill = False if arr.dtype == bool else -1
            new[name] = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=fill)
        padded.append(new)
    return padded


# =================================================================================================
def _largest_component_mask(  # -------------------------------------------------------------------
    topology: np.ndarray,
) -> np.ndarray:  # fmt: skip
    """Return a bool mask selecting only the largest 4-connected component of *topology*.

    Uses ``scipy.sparse.csgraph.connected_components`` via flood-fill labelling.
    Small disconnected pockets (unreachable tiles) are excluded.
    """
    from scipy.ndimage import label

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])  # 4-connected
    labelled, n_components = label(topology, structure=structure)
    if n_components <= 1:
        return topology.copy()

    # Find component with most cells.
    component_sizes = np.bincount(labelled.ravel())
    component_sizes[0] = 0  # ignore background
    largest = component_sizes.argmax()
    return labelled == largest


# =================================================================================================
def _dungeon_to_dict(  # --------------------------------------------------------------------------
    dungeon: Dungeon,
) -> dict:  # fmt: skip
    """Convert a :class:`Dungeon` to a plain dict suitable for JSON serialisation."""
    return {
        "seed": dungeon.seed,
        "n_rooms": len(dungeon.rooms),
        "n_passages": len(dungeon.passages),
        "bounds": list(dungeon.bounds),
        "rooms": {
            rid: {"x": r.x, "y": r.y, "w": r.width, "h": r.height, "shape": r.shape.name, "number": r.number}
            for rid, r in dungeon.rooms.items()
        },
        "passages": {
            pid: {"start": p.start_room, "end": p.end_room, "waypoints": p.waypoints}
            for pid, p in dungeon.passages.items()
        },
    }


# =================================================================================================
def _save_raw_split(  # ---------------------------------------------------------------------------
    records: list[dict], path: Path,
) -> None:  # fmt: skip
    """Write all raw dungeon records for one split as a gzip-compressed JSONL file.

    One JSON object per line, compressed with gzip. Typical compression ratio
    ~10× on dungeon geometry JSON, reducing thousands of small files to a
    single ~100 KB archive per split.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")


# =================================================================================================
def _build_metadata(  # ---------------------------------------------------------------------------
    split: str, *, channels: list[dict[str, np.ndarray]],
) -> MazeMetadata:  # fmt: skip
    """Extract metadata from channel dicts for ``dataset.json``."""
    return MazeMetadata(
        source="dungeongen",
        split=split,
        n_samples=len(channels),
        shape=list(channels[0][CHANNEL_TOPOLOGY].shape),
        channels=list(channels[0].keys()),
    )


# =================================================================================================
def _build_idx_entry(  # --------------------------------------------------------------------------
    meta: MazeMetadata, channels: list[dict[str, np.ndarray]], *,
    i: int, start_id: int = 0,
) -> MazeIndexEntry:  # fmt: skip
    """Build one :class:`MazeIndexEntry` for the *i*-th dungeon in a split."""
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
    )


# =================================================================================================
if __name__ == "__main__":
    app()
