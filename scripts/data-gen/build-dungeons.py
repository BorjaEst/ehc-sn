"""Generate canonical dungeon datasets and benchmark OOD corpora."""

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

from ehc_sn.data._canonical import (
    binary_structural_landmarks,
    farthest_reachable_cell,
    first_true_cell,
    largest_component_mask,
    sample_observations,
    shortest_path_distances,
    singleton_mask,
)
from ehc_sn.data.contracts import build_layout_benchmark_contract
from ehc_sn.data.datasets import MazeMetadata
from ehc_sn.data.index import MazeIndexEntry, write_index
from ehc_sn.data.schema import (
    CHANNEL_GOALS,
    CHANNEL_LANDMARKS,
    CHANNEL_MASK_VALID,
    CHANNEL_OBSERVATIONS,
    CHANNEL_REGIONS,
    CHANNEL_START,
    CHANNEL_TOPOLOGY,
)

# =================================================================================================
# Configuration
# -------------------------------------------------------------------------------------------------

app = Typer(pretty_exceptions_enable=False)
RAW_PATH = "data/raw/dungeons"
PROCESSED_PATH = "data/processed/dungeons"
PASSABLE_TYPES = frozenset({CellType.ROOM, CellType.PASSAGE, CellType.DOOR})
OOD_BENCHMARK_RECIPES = {
    "medium-classic": {
        "size": "medium",
        "archetype": "classic",
        "seed": 4300,
        "out_dirname": "dungeons-ood-medium-classic",
    },
    "large-classic": {
        "size": "large",
        "archetype": "classic",
        "seed": 4400,
        "out_dirname": "dungeons-ood-large-classic",
    },
    "small-temple": {
        "size": "small",
        "archetype": "temple",
        "seed": 4500,
        "out_dirname": "dungeons-ood-small-temple",
    },
    "small-cavern": {
        "size": "small",
        "archetype": "cavern",
        "seed": 4600,
        "out_dirname": "dungeons-ood-small-cavern",
    },
}


# =================================================================================================
# CLI Commands
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
    seed: int = Option(43, "--seed", help="Base RNG seed (incremented per dungeon)."),
    margin: int = Option(1, "--margin", help="Grid margin (wall border) around the dungeon bounding box."),
    require_benchmark_contract: bool = Option(False, "--require-benchmark-contract", help="Reject generated layouts that cannot satisfy the canonical B23 benchmark contract."),
    max_attempt_multiplier: int = Option(20, "--max-attempt-multiplier", min=1, help="Maximum candidate layouts to try per accepted layout when benchmark filtering is enabled."),
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
    candidate_seed_offset = 0

    for split, count in splits:
        echo(f"Generating {count} {split} dungeons (size={size}, archetype={archetype}) …")
        if count == 0:
            echo(f"  → skipping empty split '{split}'")
            continue

        all_channels: list[dict[str, np.ndarray]] = []
        raw_records: list[dict] = []
        rejected = 0
        attempts = 0
        max_attempts = count * max_attempt_multiplier if require_benchmark_contract else count

        # Generate candidate layouts until the split reaches its target count.
        while len(all_channels) < count:
            if attempts >= max_attempts:
                raise ValueError(
                    f"Failed to generate {count} benchmark-compatible dungeons for split '{split}' after {attempts} attempts; "
                    f"accepted {len(all_channels)}, rejected {rejected}."
                )

            dungeon_seed = seed + candidate_seed_offset
            candidate_seed_offset += 1
            attempts += 1
            params = GenerationParams(size=dg_size, archetype=dg_arch, density=density, seed=dungeon_seed)
            generator = DungeonGenerator(params)
            dungeon = generator.generate(seed=dungeon_seed)

            # Rasterize dungeon → channel arrays.
            channels = _rasterize_dungeon(generator, dungeon, n_observations, margin, rng_seed=dungeon_seed)
            if require_benchmark_contract:
                try:
                    build_layout_benchmark_contract(
                        component=channels[CHANNEL_MASK_VALID],
                        entry_id=str(dungeon_seed),
                        split=split,
                        preferred_start=first_true_cell(channels[CHANNEL_START]),
                    )
                except ValueError:
                    rejected += 1
                    continue

            # Collect raw dungeon record in memory (written as compressed archive later).
            raw_records.append(_dungeon_to_dict(dungeon))
            all_channels.append(channels)

        # Persist the accepted layouts and derived processed tensors for this split.
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
        if require_benchmark_contract:
            echo(f"  → {count} dungeons written (rejected {rejected} infeasible layouts; total so far: {maze_id})")
        else:
            echo(f"  → {count} dungeons written (total so far: {maze_id})")

    echo(f"\nDone. Index written to {index_path}")


# =================================================================================================
@app.command("process-benchmark-ood")
def process_benchmark_ood(  # --------------------------------------------------------------------
    recipe: str = Option(..., "--recipe", help="Benchmark OOD recipe alias."),
    out_root: Path = Option(Path("data/processed"), "--out-root", help="Parent directory for processed benchmark datasets."),
    raw_root: Path = Option(Path(RAW_PATH), "--raw-root", help="Parent directory for raw benchmark dungeon archives."),
    n_observations: int = Option(45, "--n-obs", help="Observation vocabulary size for random assignment."),
    density: float = Option(0.6, "--density", help="Room packing density."),
    margin: int = Option(1, "--margin", help="Grid margin around the dungeon bounding box."),
) -> None:  # fmt: skip
    """Generate one of the canonical B1 OOD corpora with deterministic naming and seed policy."""
    if recipe not in OOD_BENCHMARK_RECIPES:
        known = ", ".join(sorted(OOD_BENCHMARK_RECIPES))
        raise ValueError(f"Unknown benchmark OOD recipe '{recipe}'. Expected one of: {known}.")

    config = OOD_BENCHMARK_RECIPES[recipe]
    process_dungeongen(
        out_dir=out_root / config["out_dirname"],
        raw_dir=raw_root / config["out_dirname"],
        n_train=0,
        n_val=0,
        n_test=100,
        size=str(config["size"]),
        archetype=str(config["archetype"]),
        n_observations=n_observations,
        density=density,
        seed=int(config["seed"]),
        margin=margin,
        require_benchmark_contract=False,
        max_attempt_multiplier=20,
    )


# =================================================================================================
# Rasterization Helpers
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

    # -- Valid cells + semantic singleton channels --------------------------------
    mask_valid = largest_component_mask(topology)
    start_cell = _canonical_entrance_cell(dungeon, topology, mask_valid, ox=ox, oy=oy)
    goal_cell = _canonical_goal_cell(dungeon, topology, mask_valid, start_cell, ox=ox, oy=oy)
    start = singleton_mask((H, W), start_cell)
    goals = singleton_mask((H, W), goal_cell)

    # -- Portable TEM cues ---------------------------------------------------------
    observations = sample_observations(mask_valid, n_observations, seed=rng_seed)
    landmarks = binary_structural_landmarks(mask_valid)

    return {
        CHANNEL_TOPOLOGY: topology,
        CHANNEL_REGIONS: regions,
        CHANNEL_MASK_VALID: mask_valid,
        CHANNEL_START: start,
        CHANNEL_GOALS: goals,
        CHANNEL_OBSERVATIONS: observations,
        CHANNEL_LANDMARKS: landmarks,
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
# Exit Mapping Helpers
# -------------------------------------------------------------------------------------------------


# =================================================================================================
def _canonical_entrance_cell(  # ------------------------------------------------------------------
    dungeon: Dungeon, topology: np.ndarray, mask_valid: np.ndarray, *,
    ox: int, oy: int,
) -> tuple[int, int]:  # fmt: skip
    """Return one canonical entrance cell for the processed dungeon contract."""
    raw_exits = list(getattr(dungeon, "exits", {}).values())
    priorities = (
        lambda ex: bool(getattr(ex, "is_main", False)),
        _is_entrance_exit,
        lambda ex: getattr(ex, "room_id", "") == getattr(dungeon, "spine_start_room", None),
    )

    for predicate in priorities:
        mapped = [_map_exit_to_valid_cell(ex, dungeon, topology, mask_valid, ox=ox, oy=oy) for ex in raw_exits if predicate(ex)]
        mapped = [cell for cell in mapped if cell is not None]
        if mapped:
            return min(mapped)

    fallback = first_true_cell(mask_valid)
    if fallback is None:
        raise ValueError("Cannot derive a canonical entrance cell from an empty valid mask.")
    return fallback


# =================================================================================================
def _canonical_goal_cell(  # ----------------------------------------------------------------------
    dungeon: Dungeon, topology: np.ndarray, mask_valid: np.ndarray, start_cell: tuple[int, int], *,
    ox: int, oy: int,
) -> tuple[int, int]:  # fmt: skip
    """Return one canonical goal cell, preferring non-entrance exits."""
    raw_exits = list(getattr(dungeon, "exits", {}).values())
    goal_candidates = [ex for ex in raw_exits if not getattr(ex, "is_main", False) and not _is_entrance_exit(ex)]
    if not goal_candidates:
        goal_candidates = [ex for ex in raw_exits if _map_exit_identity(ex) != _entrance_identity(raw_exits)]

    distances = shortest_path_distances(mask_valid, start_cell)
    best_goal: tuple[int, int] | None = None
    best_distance = -1
    for exit_obj in goal_candidates:
        cell = _map_exit_to_valid_cell(exit_obj, dungeon, topology, mask_valid, ox=ox, oy=oy)
        if cell is None or cell == start_cell or distances[cell] < 0:
            continue
        distance = int(distances[cell])
        if distance > best_distance or (distance == best_distance and (best_goal is None or cell < best_goal)):
            best_distance = distance
            best_goal = cell

    return farthest_reachable_cell(mask_valid, start_cell) if best_goal is None else best_goal


# =================================================================================================
def _map_exit_to_valid_cell(  # -------------------------------------------------------------------
    exit_obj: object, dungeon: Dungeon, topology: np.ndarray, mask_valid: np.ndarray, *,
    ox: int, oy: int,
) -> tuple[int, int] | None:  # fmt: skip
    """Map one raw dungeongen exit to a passable valid raster cell."""
    world_x = getattr(exit_obj, "x", None)
    world_y = getattr(exit_obj, "y", None)
    if world_x is None or world_y is None:
        return None

    room_center = _room_center(dungeon, getattr(exit_obj, "room_id", ""))
    candidates: set[tuple[int, int]] = set()
    for cand_x, cand_y in (
        (world_x, world_y),
        (world_x - 1, world_y),
        (world_x + 1, world_y),
        (world_x, world_y - 1),
        (world_x, world_y + 1),
    ):
        row = cand_y - oy
        col = cand_x - ox
        if row < 0 or row >= topology.shape[0] or col < 0 or col >= topology.shape[1]:
            continue
        if topology[row, col] and mask_valid[row, col]:
            candidates.add((int(row), int(col)))

    if not candidates:
        return None
    if room_center is None:
        return min(candidates)
    return min(candidates, key=lambda cell: (abs(cell[0] - room_center[1]) + abs(cell[1] - room_center[0]), cell))


# =================================================================================================
def _room_center(  # ------------------------------------------------------------------------------
    dungeon: Dungeon, room_id: str,
) -> tuple[int, int] | None:  # fmt: skip
    """Return the integer grid center of the room attached to an exit."""
    room = getattr(dungeon, "rooms", {}).get(room_id)
    if room is None:
        return None
    x = int(getattr(room, "x", 0))
    y = int(getattr(room, "y", 0))
    width = int(getattr(room, "width", 1))
    height = int(getattr(room, "height", 1))
    return x + width // 2, y + height // 2


# =================================================================================================
def _is_entrance_exit(  # -------------------------------------------------------------------------
    exit_obj: object,
) -> bool:  # fmt: skip
    """Return ``True`` if an exit object should be treated as an entrance."""
    exit_type = getattr(exit_obj, "exit_type", None)
    type_name = getattr(exit_type, "name", str(exit_type))
    return type_name == "ENTRANCE"


# =================================================================================================
def _map_exit_identity(  # ------------------------------------------------------------------------
    exit_obj: object,
) -> str:  # fmt: skip
    """Return a stable identity string for one exit object."""
    return str(getattr(exit_obj, "id", f"{getattr(exit_obj, 'x', '?')}:{getattr(exit_obj, 'y', '?')}"))


# =================================================================================================
def _entrance_identity(  # ------------------------------------------------------------------------
    raw_exits: list[object],
) -> str | None:  # fmt: skip
    """Return the identity of the first raw entrance-like exit, if any."""
    for exit_obj in raw_exits:
        if getattr(exit_obj, "is_main", False) or _is_entrance_exit(exit_obj):
            return _map_exit_identity(exit_obj)
    return None


# =================================================================================================
# Raw Serialization
# -------------------------------------------------------------------------------------------------


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
        "rooms": {rid: _parse_room(r) for rid, r in dungeon.rooms.items()},
        "passages": {pid: _parse_passage(p) for pid, p in dungeon.passages.items()},
        "exits": {eid: _parse_exit(ex) for eid, ex in getattr(dungeon, "exits", {}).items()},
    }


def _parse_room(room: dict) -> tuple[str, object]:
    """Parse a room dict from the raw JSON into a plain object."""
    return {
        "x": room.x,
        "y": room.y,
        "w": room.width,
        "h": room.height,
        "shape": room.shape.name,
        "number": room.number,
    }


def _parse_passage(passage: dict) -> tuple[str, object]:
    """Parse a passage dict from the raw JSON into a plain object."""
    return {
        "start": passage.start_room,
        "end": passage.end_room,
        "waypoints": passage.waypoints,
    }


def _parse_exit(exit_obj: dict) -> tuple[str, object]:
    """Parse an exit dict from the raw JSON into a plain object."""
    return {
        "x": exit_obj.x,
        "y": exit_obj.y,
        "direction": exit_obj.direction,
        "exit_type": getattr(exit_obj.exit_type, "name", str(exit_obj.exit_type)),
        "room_id": exit_obj.room_id,
        "is_main": exit_obj.is_main,
    }


# =================================================================================================
# Processed Metadata
# -------------------------------------------------------------------------------------------------


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
# Entry Point
# -------------------------------------------------------------------------------------------------


# =================================================================================================
if __name__ == "__main__":
    app()
