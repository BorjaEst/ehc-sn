"""Raw dungeongen topology snapshot: canonical immutable tar-sharded format.

This module manages the canonical raw snapshot for dungeongen topologies.

Layout::

    data/raw/dungeongen/
    ├── manifest.json
    ├── train/
    │   ├── shard-00000.tar
    │   └── ...
    ├── val/
    │   └── shard-00000.tar
    └── test/
        └── shard-00000.tar

Each tar shard contains NPZ members named ``sample-<NNNNNN>.npz``.
Each NPZ record contains:

- ``sample_id``: scalar int64 — stable sample index within split.
- ``seed``:      scalar int64 — dungeongen seed used.
- ``topology``:  bool (H, W) — passable cells.
- ``regions``:   int32 (H, W) — room number per cell, -1 for non-room.

The ``manifest.json`` at the raw root is the authoritative identity descriptor.
fetch-raw is a no-op when the manifest identity matches the request.
fetch-raw hard-fails when an existing manifest's identity differs.
If the raw root exists without ``manifest.json`` (legacy loose-file layout),
fetch-raw hard-fails with an actionable error.

Passable cell encoding from dungeongen:

- ``'ROOM'``    — interior of a room (passable)
- ``'PASSAGE'`` — corridor between rooms (passable)
- ``'DOOR'``    — door cell at room entrance (passable)
- all others    — wall / empty (not passable)

Writers may import ``dungeongen``. Readers must not.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any, Final, Iterator

import numpy as np

# =============================================================================
_PASSABLE_CELL_TYPES: Final[frozenset[str]] = frozenset(
    {"ROOM", "PASSAGE", "DOOR"}
)
"""OccupancyGrid cell type strings that are traversable."""

_SHARD_SIZE: Final[int] = 1000
"""Number of samples per tar shard (internal constant; not a CLI param)."""

_SPLIT_SEED_OFFSET: Final[dict[str, int]] = {
    "train": 0,
    "val": 100_000,
    "test": 200_000,
}
"""Per-split seed offsets: sample i in split uses base_seed + offset[split] + i."""

_DEFAULT_ATTEMPT_BUDGET: Final[int] = 100
"""Default attempt budget for ``generate_bounded_topology``."""

_MANIFEST_FILENAME: Final[str] = "manifest.json"
_SOURCE_ID: Final[str] = "dungeongen"
_SNAPSHOT_KIND: Final[str] = "materialized_snapshot"
_RECORD_FORMAT: Final[str] = "tar+npz"
_RECORD_SCHEMA_VERSION: Final[int] = 1
_RAW_MANIFEST_SCHEMA_VERSION: Final[int] = 1


# =============================================================================
def default_raw_root() -> Path:
    """Return the canonical raw root for dungeongen topology snapshots."""
    return Path("data/raw/dungeongen")


# =============================================================================
def generate_topology_and_regions(
    seed: int,
    *,
    generator_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Generate one dungeon topology and room-region map using ``dungeongen``.

    Requires ``dungeongen`` at call time (writer-only dependency).

    Args:
        seed: Integer seed forwarded to ``DungeonGenerator.generate()``.
        generator_config: Optional dict of keyword arguments forwarded to
            ``GenerationParams(**generator_config)``.  When ``None``, the
            generator uses default ``DungeonGenerator()`` (no explicit params).
            Typical keys: ``size`` (DungeonSize), ``room_size_bias`` (float),
            ``density`` (float), ``symmetry`` (SymmetryType), etc.

    Returns:
        ``(topology, regions, natural_h, natural_w)`` where:

        - ``topology`` is a bool ``(H, W)`` array (``True`` = passable).
        - ``regions`` is an int32 ``(H, W)`` array with the room number at each
          passable room cell (``-1`` for corridor/passage cells and walls).
        - ``natural_h``, ``natural_w`` are the raster dimensions derived from
          the generated dungeon's geometric bounds.
    """
    from dungeongen.layout import DungeonGenerator
    from dungeongen.layout.params import GenerationParams

    if generator_config is not None:
        params = GenerationParams(**generator_config)
        gen = DungeonGenerator(params=params)
    else:
        gen = DungeonGenerator()

    dungeon = gen.generate(seed=seed)
    grid = gen.occupancy

    min_x, min_y, max_x, max_y = dungeon.bounds
    grid_h = max_y - min_y + 1
    grid_w = max_x - min_x + 1

    topology = np.zeros((grid_h, grid_w), dtype=bool)
    regions = np.full((grid_h, grid_w), -1, dtype=np.int32)

    for grid_y in range(min_y, max_y + 1):
        for grid_x in range(min_x, max_x + 1):
            cell_type = grid.get_cell(grid_x, grid_y)
            row = grid_y - min_y
            col = grid_x - min_x
            if cell_type in _PASSABLE_CELL_TYPES:
                topology[row, col] = True

    # Assign room numbers to ROOM cells.
    for room in dungeon.rooms.values():
        room_x, room_y = room.x, room.y
        room_w, room_h = room.width, room.height
        room_num = room.number
        for dy in range(room_h):
            for dx in range(room_w):
                gx = room_x + dx
                gy = room_y + dy
                cell_type = grid.get_cell(gx, gy)
                if cell_type == "ROOM":
                    row = gy - min_y
                    col = gx - min_x
                    if 0 <= row < grid_h and 0 <= col < grid_w:
                        regions[row, col] = room_num

    return topology, regions, grid_h, grid_w


def generate_bounded_topology(
    seed: int,
    max_height: int,
    max_width: int,
    *,
    generator_config: dict[str, Any] | None = None,
    attempt_budget: int = _DEFAULT_ATTEMPT_BUDGET,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Call ``generate_topology_and_regions`` with retry until extent fits.

    Each failed attempt increments the attempt seed by 1 while keeping the
    base seed unchanged, producing a deterministic retry sequence.

    Args:
        seed: Base seed for the first attempt.
        max_height: Maximum allowed natural height (exclusive upper bound).
        max_width: Maximum allowed natural width (exclusive upper bound).
        generator_config: Forwarded to ``generate_topology_and_regions``.
        attempt_budget: Maximum number of generation attempts.

    Returns:
        ``(topology, regions, natural_h, natural_w, attempt)`` — the same
        as ``generate_topology_and_regions`` with an extra ``attempt`` int
        indicating which attempt succeeded (0-based).

    Raises:
        RuntimeError: When the budget is exhausted without a fitting layout.
    """
    for attempt in range(attempt_budget):
        topology, regions, h, w = generate_topology_and_regions(
            seed + attempt,
            generator_config=generator_config,
        )
        if h <= max_height and w <= max_width:
            return topology, regions, h, w, attempt
    raise RuntimeError(
        f"generate_bounded_topology: exhausted {attempt_budget} attempts "
        f"for seed {seed}.  Last extent: [{h}, {w}]."
    )


# =============================================================================
def _dungeongen_version() -> str:
    """Return the installed dungeongen package version string."""
    try:
        from importlib.metadata import version

        return version("dungeongen")
    except Exception:
        raise RuntimeError(
            "Cannot determine installed dungeongen version. "
            "Ensure dungeongen is installed in the active environment."
        )


def _ehc_sn_version() -> str:
    """Return the installed ehc_sn package version string."""
    try:
        from importlib.metadata import version

        return version("ehp-sn")
    except Exception:
        # Fallback for editable installs where metadata may not be populated yet.
        try:
            _ver_file = Path(__file__).parent.parent / "VERSION"
            return _ver_file.read_text().strip()
        except Exception:
            return "unknown"


def _manifest_path(raw_root: Path) -> Path:
    return raw_root / _MANIFEST_FILENAME


def _shard_name(shard_idx: int) -> str:
    return f"shard-{shard_idx:05d}.tar"


def _member_name(member_idx: int) -> str:
    return f"sample-{member_idx:06d}.npz"


def _build_shard_inventory(n: int) -> list[str]:
    """Return the ordered list of shard filenames for *n* samples."""
    n_shards = max(1, (n + _SHARD_SIZE - 1) // _SHARD_SIZE)
    return [_shard_name(i) for i in range(n_shards)]


def _build_raw_manifest(
    *,
    source_revision: str,
    producer_revision: str,
    split_counts: dict[str, int],
    base_seed: int,
) -> dict[str, Any]:
    """Build the raw manifest dict."""
    splits: dict[str, Any] = {}
    for split, n in split_counts.items():
        splits[split] = {
            "n_samples": n,
            "shards": _build_shard_inventory(n),
        }
    return {
        "schema_version": _RAW_MANIFEST_SCHEMA_VERSION,
        "snapshot_kind": _SNAPSHOT_KIND,
        "source_id": _SOURCE_ID,
        "source_revision": source_revision,
        "record_format": _RECORD_FORMAT,
        "record_schema_version": _RECORD_SCHEMA_VERSION,
        "splits": splits,
        "generator_params": {"base_seed": base_seed},
        "producer_revision": producer_revision,
    }


def _manifest_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    """Extract identity-bearing fields for mismatch detection."""
    return {
        "source_revision": manifest["source_revision"],
        "record_format": manifest["record_format"],
        "record_schema_version": manifest["record_schema_version"],
        "generator_params": manifest["generator_params"],
        "split_counts": {
            s: v["n_samples"] for s, v in manifest["splits"].items()
        },
    }


def _check_legacy_layout(raw_root: Path) -> None:
    """Hard-fail if raw_root looks like a legacy loose-file layout."""
    for split_dir in raw_root.iterdir():
        if not split_dir.is_dir():
            continue
        if any(split_dir.glob("topology_*.npy")):
            raise RuntimeError(
                f"Legacy loose-file raw layout detected at {raw_root!s}.\n"
                "This format is no longer supported. Delete data/raw/dungeongen "
                "and rerun fetch-raw to create a canonical tar-sharded snapshot."
            )


def _write_shard(
    shard_path: Path,
    samples: list[tuple[int, int, np.ndarray, np.ndarray, int]],
) -> None:
    """Write a tar shard containing one NPZ record per sample.

    Args:
        shard_path: Destination ``.tar`` file path.
        samples: List of ``(sample_id, seed, topology, regions, attempt)``
            tuples, where ``sample_id`` determines the member name within
            the shard and ``attempt`` is the generation attempt index.
    """
    with tarfile.open(shard_path, "w") as tf:
        for member_idx, (
            sample_id,
            seed,
            topology,
            regions,
            attempt,
        ) in enumerate(samples):
            buf = io.BytesIO()
            np.savez_compressed(
                buf,
                sample_id=np.int64(sample_id),
                seed=np.int64(seed),
                topology=topology,
                regions=regions,
                generation_attempt=np.int32(attempt),
            )
            buf.seek(0)
            raw = buf.read()
            info = tarfile.TarInfo(name=_member_name(member_idx))
            info.size = len(raw)
            tf.addfile(info, io.BytesIO(raw))


def _write_snapshot_split(
    raw_root: Path,
    split: str,
    n: int,
    base_seed: int,
    *,
    generator_config: dict[str, Any] | None = None,
    max_extent: tuple[int, int] | None = None,
    attempt_budget: int = _DEFAULT_ATTEMPT_BUDGET,
) -> None:
    """Generate and write all shards for one split.

    When ``max_extent`` is provided, uses ``generate_bounded_topology``
    with retry; otherwise uses the default ``generate_topology_and_regions``.
    """
    split_dir = raw_root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    shards = _build_shard_inventory(n)
    for shard_idx, shard_filename in enumerate(shards):
        start = shard_idx * _SHARD_SIZE
        end = min(start + _SHARD_SIZE, n)
        samples: list[tuple[int, int, np.ndarray, np.ndarray, int]] = []
        for sample_idx in range(start, end):
            seed = base_seed + sample_idx
            if max_extent is not None:
                max_h, max_w = max_extent
                topology, regions, _, _, attempt = generate_bounded_topology(
                    seed,
                    max_h,
                    max_w,
                    generator_config=generator_config,
                    attempt_budget=attempt_budget,
                )
            else:
                topology, regions, _, _ = generate_topology_and_regions(
                    seed, generator_config=generator_config
                )
                attempt = 0
            samples.append((sample_idx, seed, topology, regions, attempt))
        _write_shard(split_dir / shard_filename, samples)


def _read_raw_manifest(raw_root: Path) -> dict[str, Any]:
    """Read and parse the raw manifest JSON.

    Raises:
        FileNotFoundError: When the manifest file does not exist.
    """
    path = _manifest_path(raw_root)
    if not path.exists():
        raise FileNotFoundError(f"Raw manifest not found: {path}")
    with path.open() as f:
        return json.load(f)


# =============================================================================
def ensure_raw_snapshot(
    raw_root: Path,
    base_seed: int,
    split_counts: dict[str, int],
    *,
    generator_config: dict[str, Any] | None = None,
    max_extent: tuple[int, int] | None = None,
    attempt_budget: int = _DEFAULT_ATTEMPT_BUDGET,
) -> None:
    """Create or validate the canonical tar-sharded raw snapshot for all splits.

    Policy:

    - If *raw_root* does not exist, generates all splits and writes
      ``manifest.json`` in a single atomic pass.
    - If ``manifest.json`` exists and the snapshot-wide identity matches,
      is a no-op (idempotent).
    - If ``manifest.json`` exists and the identity differs, raises
      ``RuntimeError`` with an actionable message.
    - If *raw_root* exists without ``manifest.json`` (legacy loose-file
      layout), raises ``RuntimeError`` telling the user to delete and rerun.

    Per-split seeds are derived as ``base_seed + _SPLIT_SEED_OFFSET[split]``.
    The root ``base_seed`` is stored in ``manifest.json`` under
    ``generator_params`` and is the same for all splits.

    When ``max_extent`` is provided (e.g. ``(30, 30)``), the generator uses
    ``generate_bounded_topology`` with deterministic retry up to
    ``attempt_budget`` per layout.

    Args:
        raw_root: Canonical raw root (e.g. ``data/raw/dungeongen``).
        base_seed: Root base seed (train base seed). Per-split seeds use
            ``_SPLIT_SEED_OFFSET`` internally.
        split_counts: Mapping of split name to number of samples, e.g.
            ``{"train": 200, "val": 40, "test": 40}``.
        generator_config: Optional ``GenerationParams`` kwargs forwarded to
            the dungeon generator.
        max_extent: Optional ``(max_height, max_width)`` bound.  When set,
            every exported layout fits within this extent via retry.
        attempt_budget: Max attempts per layout (default 100).

    Raises:
        RuntimeError: On manifest identity mismatch or legacy layout, or
            when any bounded generation exhausts its attempt budget.
    """
    manifest_path = _manifest_path(raw_root)

    if raw_root.exists() and not manifest_path.exists():
        _check_legacy_layout(raw_root)
        raise RuntimeError(
            f"Raw root exists at {raw_root!s} but manifest.json is missing. "
            "Delete data/raw/dungeongen and rerun fetch-raw."
        )

    source_revision = _dungeongen_version()
    producer_revision = _ehc_sn_version()

    generator_params: dict[str, Any] = {
        "base_seed": base_seed,
    }
    if generator_config is not None:
        generator_params["generator_config"] = generator_config
    if max_extent is not None:
        generator_params["max_extent"] = list(max_extent)
        generator_params["attempt_budget"] = attempt_budget

    if manifest_path.exists():
        existing = _read_raw_manifest(raw_root)
        existing_identity = _manifest_identity(existing)
        requested_identity = {
            "source_revision": source_revision,
            "record_format": _RECORD_FORMAT,
            "record_schema_version": _RECORD_SCHEMA_VERSION,
            "generator_params": generator_params,
            "split_counts": split_counts,
        }
        if existing_identity != requested_identity:
            raise RuntimeError(
                f"Raw snapshot at {raw_root!s} exists but its identity does not match "
                "the requested parameters.\n"
                f"  existing : {existing_identity}\n"
                f"  requested: {requested_identity}\n"
                "Delete data/raw/dungeongen and rerun fetch-raw to create a new snapshot."
            )
        return  # No-op — identity matches.

    # First-time creation.
    raw_root.mkdir(parents=True, exist_ok=True)
    for split, n in split_counts.items():
        split_base_seed = base_seed + _SPLIT_SEED_OFFSET.get(split, 0)
        _write_snapshot_split(
            raw_root,
            split,
            n,
            split_base_seed,
            generator_config=generator_config,
            max_extent=max_extent,
            attempt_budget=attempt_budget,
        )

    manifest = _build_raw_manifest(
        source_revision=source_revision,
        producer_revision=producer_revision,
        split_counts=split_counts,
        base_seed=base_seed,
    )
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)


# =============================================================================
def iter_raw_topologies(
    raw_root: Path,
    split: str,
) -> Iterator[tuple[np.ndarray, np.ndarray, int, int]]:
    """Yield ``(topology, regions, seed, generation_attempt)`` for each sample.

    Reads from the canonical tar-sharded snapshot in deterministic shard/member
    order.  Does **not** import ``dungeongen`` — safe to call in environments
    where ``dungeongen`` is not installed.

    Args:
        raw_root: Canonical raw root (e.g. ``data/raw/dungeongen``).
        split: Split name.

    Yields:
        ``(topology, regions, seed, generation_attempt)`` tuples in
        deterministic sample order.  ``generation_attempt`` is the 0-based
        attempt index that produced the accepted layout; 0 for legacy
        snapshots that lack the field.

    Raises:
        FileNotFoundError: When the manifest or raw root does not exist.
        RuntimeError: When the raw root exists without a manifest.
    """
    manifest_path = _manifest_path(raw_root)
    if not manifest_path.exists():
        if raw_root.exists():
            _check_legacy_layout(raw_root)
            raise RuntimeError(
                f"Raw manifest not found at {manifest_path!s}. "
                "Delete data/raw/dungeongen and rerun fetch-raw."
            )
        raise FileNotFoundError(
            f"Raw root not found: {raw_root!s}. Run fetch-raw first."
        )

    manifest = _read_raw_manifest(raw_root)
    if split not in manifest["splits"]:
        raise FileNotFoundError(
            f"Split {split!r} not found in raw manifest. "
            f"Available: {sorted(manifest['splits'])}."
        )

    split_dir = raw_root / split
    for shard_name in manifest["splits"][split]["shards"]:
        shard_path = split_dir / shard_name
        with tarfile.open(shard_path, "r") as tf:
            # Members are stored in insertion order = deterministic sample order.
            for member in tf.getmembers():
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                data = np.load(io.BytesIO(fobj.read()))
                attempt = int(data.get("generation_attempt", np.int32(0)))
                yield (
                    data["topology"],
                    data["regions"],
                    int(data["seed"]),
                    attempt,
                )


# =============================================================================
def count_raw_topologies(raw_root: Path, split: str) -> int:
    """Return the number of samples for *split* from the raw manifest.

    Uses manifest metadata, not filesystem globbing.

    Args:
        raw_root: Canonical raw root (e.g. ``data/raw/dungeongen``).
        split: Split name.

    Returns:
        Sample count, or ``0`` if the manifest does not exist.
    """
    manifest_path = _manifest_path(raw_root)
    if not manifest_path.exists():
        return 0
    manifest = _read_raw_manifest(raw_root)
    return manifest.get("splits", {}).get(split, {}).get("n_samples", 0)


# =============================================================================
__all__ = [
    "default_raw_root",
    "generate_topology_and_regions",
    "generate_bounded_topology",
    "ensure_raw_snapshot",
    "iter_raw_topologies",
    "count_raw_topologies",
]
