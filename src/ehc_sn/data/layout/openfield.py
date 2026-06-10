"""Openfield layout generator — legacy-TEM-compatible square, rectangle, and future hex worlds.

Generates :class:`~ehc_sn.data.layout.SpatialLayout` records that match the
topology, sensory assignment, and action semantics of the legacy TEM
environment generator (``legacy_tem/tem_tf2/environments.py``).

Unified entry point: :func:`generate_openfield_layouts`.

Square openfield
    - ``width * width`` states, 4-neighbor + stay adjacency
    - random observation_id per graph state (``sensory_seed`` controls mapping)
    - ``DEFAULT_GRID_ACTION_SPACE`` (5 actions)

Rectangle openfield
    - ``width * height`` states, 4-neighbor + stay adjacency
    - same action space as square

Hex openfield (future)
    - ``width * width`` graph states, 6-neighbor offset adjacency
    - ``square2hex`` pruning via ``valid_state_mask``
    - ``HEX_ACTION_SPACE`` (7 actions)

Named presets (matching legacy TEM default sizes)
    - ``"tem-square"``: 16 square grids [10,10,11,11,8,9,10,11,8,9,10,11,8,8,9,9]
    - ``"tem-rectangle"``: 16 rectangle grids (same widths/heights as square
      in the legacy TF2 defaults)
    - ``"tem-hex"``: 16 hex grids [6,6,7,7,5,5,6,7,5,6,6,7,5,5,6,6]
      (effective grid width = 2*w - 1, applied internally)
    - ``"small"``: 4 small square grids [8,8,9,9] for smoke tests
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

import numpy as np

from ehc_sn.data.layout import (
    DEFAULT_GRID_ACTION_SPACE,
    SpatialLayout,
    validate_spatial_layout,
)

_SQUARE_ACTION_DELTAS: Final[tuple[tuple[int, int], ...]] = (
    (0, 0),  # STAY
    (-1, 0),  # UP
    (0, 1),  # RIGHT
    (1, 0),  # DOWN
    (0, -1),  # LEFT
)


def rectangle_adjacency(
    width: int, height: int, *, stay_still: bool = True
) -> np.ndarray:
    """Build adjacency matrix for a rectangular (or square) world.

    Row-major indexing: state ``i`` corresponds to cell
    ``(row = i // width, col = i % width)``.
    """
    states = width * height
    adj = np.zeros((states, states), dtype=bool)

    for i in range(states):
        if stay_still:
            adj[i, i] = True
        # down
        if i + width < states:
            adj[i, i + width] = True
            adj[i + width, i] = True
        # right
        if i % width != 0:
            adj[i, i - 1] = True
            adj[i - 1, i] = True

    return adj


def _build_layout_record(
    *,
    width: int,
    height: int,
    topology_type: str,
    adj: np.ndarray,
    observation_id: np.ndarray,
    topology_seed: int,
    sensory_seed: int,
    s_size: int,
    valid_state_mask: np.ndarray | None = None,
) -> SpatialLayout:
    """Assemble a single :class:`SpatialLayout` from precomputed arrays."""
    n_states = width * height
    state_to_row_col = np.zeros((n_states, 2), dtype=np.int32)
    for s in range(n_states):
        state_to_row_col[s, 0] = s // width
        state_to_row_col[s, 1] = s % width

    if valid_state_mask is None:
        valid_state_mask = np.ones(n_states, dtype=bool)

    tm = adj.astype(np.float64)
    row_sums = tm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    tm = tm / row_sums

    layout_id = (
        f"openfield-{topology_type}-w{width}"
        + (f"-h{height}" if height != width else "")
        + f"-topo{topology_seed}"
        + (f"-sens{sensory_seed}" if sensory_seed >= 0 else "-topo-only")
    )

    return {
        "layout_id": layout_id,
        "layout_family": "openfield",
        "topology_type": topology_type,
        "graph_state_count": n_states,
        "valid_state_mask": valid_state_mask,
        "state_to_row_col": state_to_row_col,
        "observation_id": observation_id,
        "adjacency": adj,
        "action_space": dict(DEFAULT_GRID_ACTION_SPACE),
        "transition_matrix": tm,
        "topology_seed": topology_seed,
        "sensory_seed": sensory_seed,
        "sensory_vocab_size": s_size,
    }


# ── Sensory enrichment (genuine two-stage pipeline) ──────────────────────


def enrich_layout_with_sensory(
    layout: SpatialLayout,
    *,
    s_size: int,
    sensory_seed: int,
) -> SpatialLayout:
    """Return a new layout with random observation IDs assigned.

    Takes a topology-only layout (``observation_id = -1`` sentinel) and
    returns a deep-ish copy with randomised ``observation_id``, updated
    ``sensory_seed``, ``sensory_vocab_size``, and ``layout_id``.

    Args:
        layout: Topology-only layout to enrich.
        s_size: Sensory vocabulary size (number of distinct observation ids).
        sensory_seed: Seed for the observation_id RNG.

    Returns:
        New :class:`SpatialLayout` with sensory assignment.
    """
    N = layout["graph_state_count"]
    rng = np.random.default_rng(np.uint64(sensory_seed))
    obs_ids = rng.integers(0, s_size, size=N).astype(np.int32)

    new_id = layout["layout_id"].replace("-topo-only", f"-sens{sensory_seed}")
    # If the layout_id doesn't have the suffix (unlikely but defensive), append.
    if new_id == layout["layout_id"]:
        new_id = layout["layout_id"] + f"-sens{sensory_seed}"

    enriched: SpatialLayout = {
        **layout,
        "layout_id": new_id,
        "observation_id": obs_ids,
        "sensory_seed": sensory_seed,
        "sensory_vocab_size": s_size,
    }
    validate_spatial_layout(enriched)
    return enriched


# ── Private generators ───────────────────────────────────────────────────


def _generate_square_layouts(
    widths: list[int],
    *,
    s_size: int = 45,
    n_sensory_instances: int = 1,
    topology_seed: int = 42,
    sensory_assign: bool = True,
) -> list[SpatialLayout]:
    """Generate square openfield layouts (height == width)."""
    return _generate_rectangle_layouts(
        widths,
        None,
        s_size=s_size,
        n_sensory_instances=n_sensory_instances,
        topology_seed=topology_seed,
        sensory_assign=sensory_assign,
    )


def _generate_rectangle_layouts(
    widths: list[int],
    heights: list[int] | None = None,
    *,
    s_size: int = 45,
    n_sensory_instances: int = 1,
    topology_seed: int = 42,
    sensory_assign: bool = True,
) -> list[SpatialLayout]:
    """Generate rectangle (or square) openfield layouts.

    For each (width, height) pair, produces one topology template instantiated
    ``n_sensory_instances`` times with a different ``sensory_seed``.

    When ``sensory_assign=False``, produces one record per topology with
    ``observation_id = -1`` sentinel and ``sensory_seed = -1``.
    """
    n_envs = len(widths)
    if heights is None:
        heights = list(widths)
    if len(heights) != n_envs:
        raise ValueError(
            f"widths and heights must have the same length "
            f"({n_envs} vs {len(heights)})."
        )

    layouts: list[SpatialLayout] = []
    topo_rng = np.random.default_rng(np.uint64(topology_seed))
    _ = topo_rng  # consumed for future topology randomization if needed

    for env_idx, (w, h) in enumerate(zip(widths, heights)):
        n_states = w * h
        adj = rectangle_adjacency(w, h, stay_still=True)

        for inst_idx in range(n_sensory_instances):
            if sensory_assign:
                sensory_seed = (
                    topology_seed + env_idx * n_sensory_instances + inst_idx
                )
                sensory_rng = np.random.default_rng(np.uint64(sensory_seed))
                obs_ids = sensory_rng.integers(0, s_size, size=n_states).astype(
                    np.int32
                )
            else:
                sensory_seed = -1
                obs_ids = np.full(n_states, -1, dtype=np.int32)

            layout = _build_layout_record(
                width=w,
                height=h,
                topology_type="square" if w == h else "rectangle",
                adj=adj,
                observation_id=obs_ids,
                topology_seed=topology_seed,
                sensory_seed=sensory_seed,
                s_size=s_size,
            )
            validate_spatial_layout(layout)
            layouts.append(layout)

    return layouts


# ── Presets ──────────────────────────────────────────────────────────────


OPENFIELD_PRESETS: Final[dict] = {
    "tem-square": {
        "topology_type": "square",
        "widths": [10, 10, 11, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 8, 9, 9],
        "description": (
            "Legacy TEM square grid (TF1 & TF2 default sizes, 16 envs)."
        ),
    },
    "tem-rectangle": {
        "topology_type": "rectangle",
        "widths": [11, 11, 12, 12, 8, 8, 9, 9, 11, 11, 12, 12, 8, 8, 9, 9],
        "heights": [8, 8, 9, 9, 11, 11, 12, 12, 8, 8, 9, 9, 11, 11, 12, 12],
        "description": ("Legacy TEM rectangle (TF2 default sizes, 16 envs)."),
    },
    "tem-hex": {
        "topology_type": "hex",
        "widths": [6, 6, 7, 7, 5, 5, 6, 7, 5, 6, 6, 7, 5, 5, 6, 6],
        "description": (
            "Legacy TEM hexagonal grid (TF1 default, TF2 sizes, 16 envs)."
        ),
    },
    "small": {
        "topology_type": "square",
        "widths": [8, 8, 9, 9],
        "description": "Small square grids for quick smoke tests.",
    },
}
"""Named preset configurations matching legacy TEM default environment sizes.

Each preset defines a ``topology_type``, a list of ``widths``, and optionally
``heights``.  The preset dict is consumed by :func:`generate_openfield_layouts`.
"""


# ── Unified entry point ──────────────────────────────────────────────────


def generate_openfield_layouts(
    topology_type: Literal["square", "rectangle", "hex"],
    widths: list[int],
    heights: list[int] | None = None,
    *,
    s_size: int = 45,
    n_sensory_instances: int = 1,
    topology_seed: int = 42,
    sensory_assign: bool = True,
) -> list[SpatialLayout]:
    """Unified entry point for all openfield topology types.

    Args:
        topology_type: ``\"square\"``, ``\"rectangle\"``, or ``\"hex\"``.
        widths: List of grid widths.
        heights: Optional list of grid heights (required for rectangle).
        s_size: Sensory vocabulary size (legacy TEM default = 45).
            Ignored when ``sensory_assign=False``.
        n_sensory_instances: Sensory maps per topology template.
            Ignored when ``sensory_assign=False`` (produces one topology-only
            record per width).
        topology_seed: Base seed for deterministic topology generation.
        sensory_assign: When ``False``, produce topology-only records with
            ``observation_id = -1`` sentinel and ``sensory_seed = -1``.

    Returns:
        List of :class:`SpatialLayout` records.
    """
    if topology_type == "hex":
        raise NotImplementedError("Hex openfield is not yet implemented.")

    if sensory_assign:
        n_instances = n_sensory_instances
        effective_s_size = s_size
    else:
        n_instances = 1
        effective_s_size = 0

    if topology_type == "rectangle":
        if heights is None:
            heights = list(widths)
        return _generate_rectangle_layouts(
            widths,
            heights,
            s_size=effective_s_size,
            n_sensory_instances=n_instances,
            topology_seed=topology_seed,
            sensory_assign=sensory_assign,
        )
    # square
    return _generate_square_layouts(
        widths,
        s_size=effective_s_size,
        n_sensory_instances=n_instances,
        topology_seed=topology_seed,
        sensory_assign=sensory_assign,
    )


# ── Interim I/O (thin wrappers around generic io.py) ─────────────────────


def load_openfield_layouts(
    version_root: Path,
) -> list[SpatialLayout]:
    """Load openfield layouts — delegates to :func:`load_layout_dataset`.

    Provided for backward compatibility; prefer calling
    :func:`~ehc_sn.data.layout.io.load_layout_dataset` directly.
    """
    from ehc_sn.data.layout.io import load_layout_dataset

    return load_layout_dataset(version_root)


__all__ = [
    "OPENFIELD_PRESETS",
    "generate_openfield_layouts",
    "enrich_layout_with_sensory",
    "load_openfield_layouts",
    "rectangle_adjacency",
]
