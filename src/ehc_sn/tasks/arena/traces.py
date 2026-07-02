"""Arena producer-side trace supplement API.

This module owns:

- :class:`ArenaEvaluationSourceContext` — typed, frozen provider context for Arena cases.
- :class:`ArenaTraceSupplements` — canonical content (not figure objects) ready to attach.
- :func:`build_arena_trace_supplements` — constructs supplements from source context.
- :func:`apply_arena_trace_supplements` — attaches supplement data to a :class:`TraceTree`.

The supplement builder reconstructs spatial geometry and trajectory data from
the arena corpus (self-contained — no parent substrate resolution).
The primary concept is *trace supplements* — not figure join, not callback patches.

Canonical trace keys produced:
    - ``"world_step/location_ids"`` — ``(T, B)`` int32 row-major grid ids.
    - ``"world_step/observation"`` — ``(T, B, V)`` float32 one-hot observations.
    - ``"environments"`` — list of B world dicts (attached as metadata).

Location-id convention (frozen, row-major full-grid):
    ``location_id = row * W + col``

Padded steps have ``trajectory_row == -1`` and ``trajectory_col == -1``, yielding a
negative location id which is safely ignored by downstream consumers (e.g.
``aggregate_rate_map`` skips ``loc_id < 0``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ehc_sn.data.index import read_index
from ehc_sn.data.manifest import read_manifest
from ehc_sn.traces.trace_tree import TraceTree


# =============================================================================
@dataclass(frozen=True)
class ArenaEvaluationSourceContext:
    """Typed, frozen provider-side context for an Arena evaluation case batch.

    This replaces the legacy free-form ``metadata`` dict emitted by Arena providers.
    Carry only the fields needed to reconstruct world/context supplements.

    Attributes:
        task_family: Always ``"arena"``. Used as a discriminant for isinstance checks.
        dataset_path: Absolute path to the processed Arena dataset root.
        split: Dataset split the samples belong to (e.g. ``"val"``).
        sample_ids: Ordered list of Arena sample ids in this batch.
        task_evidence_arrays: Optional dict of name → ndarray for task-context
            figures (e.g. ``task_overview_arena``). Keys are namespaced trace
            paths such as ``"arena/wall_mask"``, ``"arena/observation_ids"``,
            ``"arena/trajectory_locations"``, ``"arena/revisit_mask"``.
            Populated by the Arena provider.  ``None`` when not available.
        case_metadata: Optional dict of compact scalar metadata for the
            evaluation-artifact manifest case row.  Populated by the Arena provider.
            ``None`` when not available.
    """

    task_family: str
    dataset_path: Path
    split: str
    sample_ids: tuple[str, ...]
    task_evidence_arrays: dict[str, np.ndarray] | None = None
    case_metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.task_family != "arena":
            raise ValueError(
                "ArenaEvaluationSourceContext.task_family must be 'arena', got "
                f"{self.task_family!r}"
            )
        if not self.sample_ids:
            raise ValueError(
                "ArenaEvaluationSourceContext.sample_ids must not be empty",
            )


# =============================================================================
@dataclass(frozen=True)
class ArenaTraceSupplements:
    """Canonical world/context supplement content for Arena traces.

    Carries numpy arrays and world dicts ready to attach to a :class:`TraceTree`.
    Does not carry figure objects or rendering state.

    Attributes:
        worlds: List of B world dicts, each with ``"locations"``, ``"n_locations"``,
            and ``"spatial_geometry"``. One entry per episode.
        location_ids: ``(T, B)`` int32 array of row-major cell ids.
            Padded steps have negative values.
        observation_onehot: ``(T, B, V)`` float32 one-hot observation array where
            ``V = observation_vocab_size`` from the Arena manifest.
    """

    worlds: list[dict[str, Any]]
    location_ids: np.ndarray  # (T, B) int32
    observation_onehot: np.ndarray  # (T, B, V) float32


# =============================================================================
def build_arena_trace_supplements(
    source_context: ArenaEvaluationSourceContext,
    trace_length: int,
    *,
    repo_root: Path | None = None,
) -> ArenaTraceSupplements:
    """Build Arena world/context supplements from a typed source context.

    Reconstructs geometry and trajectory data from the arena corpus.
    Spatial arrays are loaded directly from the arena split directory
    (self-contained task corpus — no parent substrate resolution).

    The ``repo_root`` parameter is accepted for backward compatibility
    but is unused (deprecated).

    Args:
        source_context: Typed Arena evaluation source context.
        trace_length: Number of time steps to cover (arrays are truncated to this).
        repo_root: Optional override for the repo root used to resolve the
            ``parent_substrate`` path from the Arena manifest. Auto-detected if ``None``.

    Returns:
        :class:`ArenaTraceSupplements` with world dicts and dense arrays.

    Raises:
        FileNotFoundError: When the arena root or required spatial array is absent.
        ValueError: When sample ids are not found in the index.
    """
    dataset_path = source_context.dataset_path
    sample_ids: list[str] = list(source_context.sample_ids)

    # ── 1. Load Arena manifest ────────────────────────────────────────────────
    arena_manifest = read_manifest(dataset_path)
    observation_vocab_size: int = arena_manifest["observation_vocab_size"]

    # spatial arrays are loaded directly from the arena corpus
    # (self-contained task corpus).  Parent provenance is in parents.spatial_topology.root.
    _ = arena_manifest.get("parents", {}).get("spatial_topology", {})

    # ── 2. Load Arena index → sample order ────────────────────────────────────
    arena_all = read_index(dataset_path / "index.jsonl")
    arena_by_id = {e.id: e for e in arena_all}

    missing = [sid for sid in sample_ids if sid not in arena_by_id]
    if missing:
        raise ValueError(
            "build_arena_trace_supplements: sample ids not found in Arena "
            f"index: {missing}"
        )

    ordered_entries = [arena_by_id[sid] for sid in sample_ids]
    split = ordered_entries[0].split
    split_entries = [e for e in arena_all if e.split == split]
    arena_split_pos = {e.id: i for i, e in enumerate(split_entries)}
    arena_positions = [arena_split_pos[sid] for sid in sample_ids]

    # ── 3. Load spatial arrays from arena split dir (self-contained) ─────────
    arena_split_dir = dataset_path / split
    topology_all = np.load(arena_split_dir / "topology.npy", mmap_mode="r")
    mask_valid_all = np.load(arena_split_dir / "mask_valid.npy", mmap_mode="r")
    _h, _w = topology_all.shape[1], topology_all.shape[2]

    # ── 4. Load Arena trajectory arrays (mmap, sliced per sample) ────────────
    traj_row_all = np.load(
        arena_split_dir / "trajectory_row.npy", mmap_mode="r"
    )
    traj_col_all = np.load(
        arena_split_dir / "trajectory_col.npy", mmap_mode="r"
    )
    traj_obs_all = np.load(
        arena_split_dir / "trajectory_observation_id.npy", mmap_mode="r"
    )
    t_max = traj_row_all.shape[1]

    B = len(sample_ids)
    T = min(trace_length, t_max)
    V = observation_vocab_size

    # ── 5. Build per-episode World dicts and dense arrays ────────────────────
    worlds: list[dict[str, Any]] = []
    location_ids_buf = np.empty((B, T), dtype=np.int32)
    observation_onehot_buf = np.zeros((B, T, V), dtype=np.float32)

    for b, ap in enumerate(arena_positions):
        mask_valid = np.asarray(mask_valid_all[ap], dtype=bool)  # (H, W)
        worlds.append(_build_world(mask_valid, _h, _w))

        row_seq = np.asarray(traj_row_all[ap, :T], dtype=np.int32)
        col_seq = np.asarray(traj_col_all[ap, :T], dtype=np.int32)
        obs_seq = np.asarray(traj_obs_all[ap, :T], dtype=np.int32)

        location_ids_buf[b] = row_seq * np.int32(_w) + col_seq

        valid_mask = (obs_seq >= 0) & (obs_seq < V)
        safe_obs = np.where(valid_mask, obs_seq, 0)
        observation_onehot_buf[b, np.arange(T), safe_obs] = np.where(
            valid_mask, 1.0, 0.0
        )

    # Transpose to (T, B) / (T, B, V) for trace convention.
    location_ids = location_ids_buf.T.copy()  # (T, B) int32
    observation_onehot = observation_onehot_buf.transpose(
        1, 0, 2
    ).copy()  # (T, B, V) float32

    return ArenaTraceSupplements(
        worlds=worlds,
        location_ids=location_ids,
        observation_onehot=observation_onehot,
    )


# =============================================================================
def apply_arena_trace_supplements(  # -----------------------------------------
    trace: TraceTree,
    supplements: ArenaTraceSupplements,
) -> None:
    """Attach Arena world/context supplement content to *trace* in-place.

    Injects the three canonical consumer keys:
        - ``"world_step/observation"`` — ``(T, B, V)`` float32 one-hot observations.
        - ``"world_step/location_ids"`` — ``(T, B)`` int32 row-major grid ids.
        - ``"environments"`` — list of B world dicts (attached as metadata).

    Args:
        trace: The :class:`~ehp_sn.traces.trace_tree.TraceTree` to modify in-place.
        supplements: The supplement content to attach.
    """
    trace.attach_dense(
        "world_step/location_ids", supplements.location_ids, overwrite=True
    )
    trace.attach_dense(
        "world_step/observation", supplements.observation_onehot, overwrite=True
    )
    trace.attach_meta({"environments": supplements.worlds}, overwrite=True)


# =============================================================================
def _build_world(  # ----------------------------------------------------------
    mask_valid: np.ndarray,
    h: int,
    w: int,
) -> dict[str, Any]:
    """Build a world dict for a single parent-substrate sample."""
    locations: list[dict[str, Any]] = []
    for row in range(h):
        for col in range(w):
            locations.append(
                {
                    "o": col,
                    "y": row,
                    "valid": bool(mask_valid[row, col]),
                }
            )
    return {
        "locations": locations,
        "n_locations": h * w,
        "spatial_geometry": "grid2d",
    }


# =============================================================================
__all__ = [
    "ArenaEvaluationSourceContext",
    "ArenaTraceSupplements",
    "build_arena_trace_supplements",
    "apply_arena_trace_supplements",
]
