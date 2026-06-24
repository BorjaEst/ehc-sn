"""Routebind task corpus materialization.

Orchestrates query selection, oracle invocation, target encoding, and
corpus I/O over spatial topology and semantic DAG parent artifacts.

The hidden semantic DAG is consumed from a dagflow substrate (never
generated inline).  Observation identities are read from the topology
substrate verbatim — routebind does not place, modify, or remap
observations.

Path written: ``data/processed/routebind/<corpus>/v<version>/``
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from ehc_sn.data.lifecycle import (
    extract_version,
    staging_root,
    validate_version_root,
    write_index_at_root,
    write_split,
)
from ehc_sn.data.manifest import read_manifest, write_manifest
from ehc_sn.data.substrate.dagflow import validate_dagflow_layout_sample
from ehc_sn.data.substrate.reader import find_artifact_by_id
from ehc_sn.tasks.routebind.contracts import ROUTEBIND_SCHEMA
from ehc_sn.tasks.routebind.oracle import (
    _build_dag_csr,
    compute_goal_distance_table,
    reconstruct_from_policy,
)
from ehc_sn.tasks.routebind.targets import (
    encode_trajectory_field,
    encode_waypoint_field,
)
from ehc_sn.tasks.routebind.validation import (
    validate_generated_sample,
    validate_stored_sample,
)
from ehc_sn.utils.graph import canonical_dag_digest

# =============================================================================
TASK_FAMILY: Final[str] = "routebind"
"""Task namespace for the routebind task corpus."""

# =============================================================================
# Channel group constants — derived from canonical schema descriptor
# =============================================================================

ROUTEBIND_MODEL_INPUT_CHANNELS: Final[tuple[str, ...]] = (
    ROUTEBIND_SCHEMA.model_input_channels
)
"""Channels passed to the HRM model adapter (model input only).

Targets and evaluation metadata are intentionally excluded — they live
in other channel groups so the adapter never receives them.
"""

ROUTEBIND_TARGET_CHANNELS: Final[tuple[str, ...]] = (
    ROUTEBIND_SCHEMA.target_channels
)
"""Channels used as supervision targets (objective / binding contract)."""

ROUTEBIND_CORPUS_CHANNELS: Final[tuple[str, ...]] = (
    ROUTEBIND_SCHEMA.all_channels
)
"""All channels persisted in the routebind task corpus (union of all groups)."""

ROUTEBIND_TASK_CHANNEL_DTYPES: dict[str, np.dtype] = ROUTEBIND_SCHEMA.dtypes
"""Expected numpy dtypes for routebind task corpus channels."""

_SPLITS: tuple[str, ...] = ("train", "val", "test")


# =============================================================================
# Sampling profile — target distributions for deficit-driven generation
# =============================================================================


@dataclass(frozen=True)
class RoutebindSamplingProfile:
    """Target distribution over physical and semantic route-length buckets.

    The builder uses this profile to drive deficit-aware query selection:
    goals and starts are chosen to fill underfilled (physical, semantic)
    joint buckets, prioritised in ``priority_order``.

    Physical route length means the number of **positions** in the projected
    spatial route (i.e. physical-move cost + 1).  Semantic waypoint count
    means the number of accepted observation events (including the start
    observation).

    Two modes are supported:

    * ``"distribution"`` — the profile shapes which valid candidates are
      selected but never rejects a valid sample.  Samples that fall outside
      all named bins are mapped to a catch-all overflow zone.
    * ``"restricted_domain"`` — samples outside the hard bin boundaries are
      rejected and counted in ``rejected_bucket_mismatch``.

    Attributes:
        mode: ``"distribution"`` or ``"restricted_domain"``.
        physical_length_bins: Inclusive ``(lo, hi)`` bins for physical
            route length, in order.  For ``mode="restricted_domain"`` bins
            must be contiguous and cover
            ``[hard_min_route_length, hard_max_route_length]``.
            For ``mode="distribution"`` bins need not be contiguous.
        physical_length_targets: Target proportion per bin (must sum to 1).
        semantic_length_bins: Inclusive ``(lo, hi)`` bins for semantic
            waypoint count, in order.  For ``mode="restricted_domain"`` bins
            must be contiguous.
        semantic_length_targets: Target proportion per bin (must sum to 1).
        hard_min_route_length: Absolute minimum physical route length.
        hard_max_route_length: Absolute maximum physical route length.
        tolerances: Per-joint-bucket tolerance ``(min_frac, max_frac)``.
            Key ``"*"`` applies to all unspecified buckets.
        priority_order: Lexicographic deficit dimension order, e.g.
            ``["physical_length", "semantic_length"]``.
        attempt_budget: Maximum start attempts per layout (replaces
            ``n_queries_per_layout`` semantics).
        description: Human-readable label.
    """

    semantic_length_targets: list[float]
    mode: str
    physical_length_bins: list[tuple[int, int]]
    physical_length_targets: list[float]
    semantic_length_bins: list[tuple[int, int]]
    # All fields below have defaults
    hard_min_route_length: int = 2
    hard_max_route_length: int = 150
    tolerances: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {"*": (0.0, 1.0)}
    )
    priority_order: list[str] = field(
        default_factory=lambda: ["physical_length", "semantic_length"]
    )
    attempt_budget: int = 10
    description: str = ""

    def __post_init__(self) -> None:
        """Validate structural invariants at construction time."""
        if self.mode not in ("distribution", "restricted_domain"):
            raise ValueError(
                f"mode={self.mode!r} must be 'distribution' or 'restricted_domain'."
            )
        is_restricted = self.mode == "restricted_domain"
        # --- Physical bins ---
        if not self.physical_length_bins:
            raise ValueError("physical_length_bins must not be empty.")
        if len(self.physical_length_bins) != len(self.physical_length_targets):
            raise ValueError(
                f"physical_length_bins ({len(self.physical_length_bins)}) and "
                f"physical_length_targets ({len(self.physical_length_targets)}) "
                "must have the same length."
            )
        total_phys = sum(self.physical_length_targets)
        if abs(total_phys - 1.0) > 0.001:
            raise ValueError(
                f"physical_length_targets sum to {total_phys}, expected ~1.0."
            )
        for i, ((lo1, hi1), tgt) in enumerate(
            zip(self.physical_length_bins, self.physical_length_targets)
        ):
            if lo1 < self.hard_min_route_length:
                raise ValueError(
                    f"physical_length_bins[{i}] lower bound {lo1} < "
                    f"hard_min_route_length ({self.hard_min_route_length})."
                )
            if hi1 > self.hard_max_route_length:
                raise ValueError(
                    f"physical_length_bins[{i}] upper bound {hi1} > "
                    f"hard_max_route_length ({self.hard_max_route_length})."
                )
            if lo1 > hi1:
                raise ValueError(
                    f"physical_length_bins[{i}] empty ({lo1}, {hi1})."
                )
            if tgt < 0.0 or tgt > 1.0:
                raise ValueError(
                    f"physical_length_targets[{i}] = {tgt} outside [0, 1]."
                )
            if is_restricted and i > 0:
                prev_hi = self.physical_length_bins[i - 1][1]
                if lo1 != prev_hi + 1:
                    raise ValueError(
                        f"physical_length_bins[{i - 1}] and [{i}] are not "
                        f"contiguous: previous ends at {prev_hi}, next starts "
                        f"at {lo1}."
                    )

        # --- Semantic bins ---
        if not self.semantic_length_bins:
            raise ValueError("semantic_length_bins must not be empty.")
        if len(self.semantic_length_bins) != len(self.semantic_length_targets):
            raise ValueError(
                f"semantic_length_bins ({len(self.semantic_length_bins)}) and "
                f"semantic_length_targets ({len(self.semantic_length_targets)}) "
                "must have the same length."
            )
        total_sem = sum(self.semantic_length_targets)
        if abs(total_sem - 1.0) > 0.001:
            raise ValueError(
                f"semantic_length_targets sum to {total_sem}, expected ~1.0."
            )
        for i, ((lo1, hi1), tgt) in enumerate(
            zip(self.semantic_length_bins, self.semantic_length_targets)
        ):
            if lo1 < 2:
                raise ValueError(
                    f"semantic_length_bins[{i}] lower bound {lo1} < 2."
                )
            if lo1 > hi1:
                raise ValueError(
                    f"semantic_length_bins[{i}] empty ({lo1}, {hi1})."
                )
            if tgt < 0.0 or tgt > 1.0:
                raise ValueError(
                    f"semantic_length_targets[{i}] = {tgt} outside [0, 1]."
                )

    def physical_bucket_index(self, route_len: int) -> int | None:
        """Return the physical bin index for *route_len*, or None."""
        for i, (lo, hi) in enumerate(self.physical_length_bins):
            if lo <= route_len <= hi:
                return i
        return None

    def semantic_bucket_index(self, waypoint_count: int) -> int | None:
        """Return the semantic bin index for *waypoint_count*, or None."""
        for i, (lo, hi) in enumerate(self.semantic_length_bins):
            if lo <= waypoint_count <= hi:
                return i
        return None

    def target_n_per_physical_bucket(self, total_n: int) -> list[int]:
        """Target count per physical-length bucket for *total_n* samples."""
        return [
            max(0, round(p * total_n)) for p in self.physical_length_targets
        ]

    def joint_targets(self) -> dict[tuple[int, int], float]:
        """Return joint ``(phys_idx, sem_idx) -> target proportion``.

        Assumes independence between physical and semantic dimensions when
        no explicit joint target is given.
        """
        targets: dict[tuple[int, int], float] = {}
        for pi, pt in enumerate(self.physical_length_targets):
            for si, st in enumerate(self.semantic_length_targets):
                targets[(pi, si)] = pt * st
        return targets


# =============================================================================
# Funnel metrics for generation diagnostics
# =============================================================================


@dataclass
class GenerationFunnel:
    """Candidate-flow counters binned by physical-distance bucket.

    Each dict key is a bucket index ``(phys_idx, sem_idx)`` or
    just ``phys_idx`` for pre-reconstruction stages.
    """

    examined: dict[tuple[int, int], int] = field(default_factory=dict)
    unreachable: dict[tuple[int, int], int] = field(default_factory=dict)
    ambiguous: dict[tuple[int, int], int] = field(default_factory=dict)
    eligible: dict[tuple[int, int], int] = field(default_factory=dict)
    reconstructed: dict[tuple[int, int], int] = field(default_factory=dict)
    bucket_full: dict[tuple[int, int], int] = field(default_factory=dict)
    accepted: dict[tuple[int, int], int] = field(default_factory=dict)
    rejected_non_simple: int = 0
    rejected_route_too_long: int = 0
    rejected_trivial_waypoint: int = 0
    rejected_invalid_next_dir: int = 0
    rejected_target_validation: int = 0
    rejected_bucket_mismatch: int = 0

    def _bkey(self, phys_idx: int) -> tuple[int, int]:
        return (phys_idx, -1)

    def _jkey(self, phys_idx: int, sem_idx: int) -> tuple[int, int]:
        return (phys_idx, sem_idx)

    def inc_examined(self, phys_idx: int) -> None:
        k = self._bkey(phys_idx)
        self.examined[k] = self.examined.get(k, 0) + 1

    def inc_unreachable(self, phys_idx: int) -> None:
        k = self._bkey(phys_idx)
        self.unreachable[k] = self.unreachable.get(k, 0) + 1

    def inc_ambiguous(self, phys_idx: int) -> None:
        k = self._bkey(phys_idx)
        self.ambiguous[k] = self.ambiguous.get(k, 0) + 1

    def inc_eligible(self, phys_idx: int) -> None:
        k = self._bkey(phys_idx)
        self.eligible[k] = self.eligible.get(k, 0) + 1

    def inc_reconstructed(self, phys_idx: int, sem_idx: int) -> None:
        k = self._jkey(phys_idx, sem_idx)
        self.reconstructed[k] = self.reconstructed.get(k, 0) + 1

    def inc_bucket_full(self, phys_idx: int, sem_idx: int) -> None:
        k = self._jkey(phys_idx, sem_idx)
        self.bucket_full[k] = self.bucket_full.get(k, 0) + 1

    def inc_accepted(self, phys_idx: int, sem_idx: int) -> None:
        k = self._jkey(phys_idx, sem_idx)
        self.accepted[k] = self.accepted.get(k, 0) + 1

    def serialize(self) -> dict:
        return {
            "examined": {str(k): v for k, v in self.examined.items()},
            "unreachable": {str(k): v for k, v in self.unreachable.items()},
            "ambiguous": {str(k): v for k, v in self.ambiguous.items()},
            "eligible": {str(k): v for k, v in self.eligible.items()},
            "reconstructed": {str(k): v for k, v in self.reconstructed.items()},
            "bucket_full": {str(k): v for k, v in self.bucket_full.items()},
            "accepted": {str(k): v for k, v in self.accepted.items()},
            "rejected_non_simple": self.rejected_non_simple,
            "rejected_route_too_long": self.rejected_route_too_long,
            "rejected_trivial_waypoint": self.rejected_trivial_waypoint,
            "rejected_invalid_next_dir": self.rejected_invalid_next_dir,
            "rejected_target_validation": self.rejected_target_validation,
            "rejected_bucket_mismatch": self.rejected_bucket_mismatch,
        }


# =============================================================================
# Named sampling presets
# =============================================================================


ROUTEBIND_PRESETS: dict[str, RoutebindSamplingProfile] = {
    "smoke": RoutebindSamplingProfile(
        mode="distribution",
        physical_length_bins=[(2, 150)],
        physical_length_targets=[1.0],
        semantic_length_bins=[(2, 20)],
        semantic_length_targets=[1.0],
        hard_min_route_length=2,
        hard_max_route_length=150,
        tolerances={"*": (0.0, 1.0)},
        attempt_budget=10,
        description="Accept-all preset for tests and calibration runs.",
    ),
    "balanced": RoutebindSamplingProfile(
        mode="distribution",
        physical_length_bins=[
            (2, 7),
            (8, 15),
            (16, 30),
            (31, 50),
            (51, 80),
        ],
        physical_length_targets=[0.10, 0.20, 0.35, 0.25, 0.10],
        semantic_length_bins=[
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 6),
            (7, 10),
        ],
        semantic_length_targets=[0.15, 0.25, 0.25, 0.25, 0.10],
        hard_min_route_length=2,
        hard_max_route_length=80,
        tolerances={
            "*": (0.0, 1.0),
        },
        attempt_budget=10,
        description=(
            "Broad physical and semantic distribution for canonical "
            "training corpus."
        ),
    ),
    "long-spatial": RoutebindSamplingProfile(
        mode="distribution",
        physical_length_bins=[
            (2, 25),
            (26, 50),
            (51, 80),
            (81, 120),
        ],
        physical_length_targets=[0.15, 0.35, 0.35, 0.15],
        semantic_length_bins=[
            (2, 3),
            (4, 5),
            (6, 10),
        ],
        semantic_length_targets=[0.45, 0.40, 0.15],
        hard_min_route_length=2,
        hard_max_route_length=120,
        tolerances={
            "*": (0.0, 1.0),
        },
        attempt_budget=50,
        description=(
            "Emphasize physical planning; moderate semantic complexity.  "
            "Recommended with dungeongen topology and a sparse DAG."
        ),
    ),
    "long-semantic": RoutebindSamplingProfile(
        mode="distribution",
        physical_length_bins=[
            (2, 20),
            (21, 50),
            (51, 80),
            (81, 120),
        ],
        physical_length_targets=[0.20, 0.45, 0.25, 0.10],
        semantic_length_bins=[
            (4, 5),
            (6, 8),
            (9, 12),
            (13, 15),
        ],
        semantic_length_targets=[0.20, 0.50, 0.25, 0.05],
        hard_min_route_length=2,
        hard_max_route_length=120,
        tolerances={
            "*": (0.0, 1.0),
        },
        attempt_budget=50,
        description=(
            "Emphasize DAG composition; many waypoints.  "
            "Recommended with chain16 or another low-shortcut DAG."
        ),
    ),
    "joint-hard": RoutebindSamplingProfile(
        mode="distribution",
        physical_length_bins=[
            (2, 19),
            (20, 40),
            (41, 70),
            (71, 100),
            (101, 140),
        ],
        physical_length_targets=[0.05, 0.20, 0.35, 0.30, 0.10],
        semantic_length_bins=[
            (2, 3),
            (4, 5),
            (6, 8),
            (9, 12),
            (13, 15),
        ],
        semantic_length_targets=[0.05, 0.15, 0.40, 0.30, 0.10],
        hard_min_route_length=2,
        hard_max_route_length=140,
        tolerances={
            "*": (0.0, 3.0),
        },
        attempt_budget=200,
        description=(
            "Jointly long spatial and semantic solutions (distribution mode).  "
            "Short/easy samples go into low-target catch-all bins.  "
            "Recommended with dungeongen topology and a sparse DAG."
        ),
    ),
    "joint-hard-only": RoutebindSamplingProfile(
        mode="restricted_domain",
        physical_length_bins=[
            (20, 40),
            (41, 70),
            (71, 100),
            (101, 140),
        ],
        physical_length_targets=[0.20, 0.40, 0.30, 0.10],
        semantic_length_bins=[
            (4, 5),
            (6, 8),
            (9, 12),
            (13, 15),
        ],
        semantic_length_targets=[0.15, 0.45, 0.30, 0.10],
        hard_min_route_length=20,
        hard_max_route_length=140,
        tolerances={
            "*": (0.0, 1.0),
        },
        attempt_budget=200,
        description=(
            "Jointly long only — samples outside hard bounds are rejected.  "
            "Requires large attempt budgets.  Recommended with dungeongen "
            "topology and a sparse DAG."
        ),
    ),
}


def resolve_preset(
    name: str,
    overrides: dict | None = None,
) -> RoutebindSamplingProfile:
    """Look up a named preset, optionally overriding fields.

    Args:
        name: Preset key in ``ROUTEBIND_PRESETS``.
        overrides: Optional dict of profile fields to override
            (e.g. ``{"attempt_budget": 20}``).

    Returns:
        A (possibly modified) ``RoutebindSamplingProfile``.
    """
    base = ROUTEBIND_PRESETS.get(name)
    if base is None:
        raise ValueError(
            f"Unknown routebind preset {name!r}. "
            f"Valid: {sorted(ROUTEBIND_PRESETS)}."
        )
    if not overrides:
        return base
    # Build a new profile with overrides
    kwargs = {
        "mode": base.mode,
        "physical_length_bins": base.physical_length_bins,
        "physical_length_targets": base.physical_length_targets,
        "semantic_length_bins": base.semantic_length_bins,
        "semantic_length_targets": base.semantic_length_targets,
        "hard_min_route_length": base.hard_min_route_length,
        "hard_max_route_length": base.hard_max_route_length,
        "tolerances": base.tolerances,
        "priority_order": base.priority_order,
        "attempt_budget": base.attempt_budget,
        "description": base.description,
    }
    kwargs.update(overrides)
    return RoutebindSamplingProfile(**kwargs)


# =============================================================================
# Private helpers
# =============================================================================


def _compute_connected_components(
    physical_neighbors: np.ndarray,
    n_slots: int,
) -> np.ndarray:
    """Label connected components of the traversable grid via BFS.

    Returns (n_slots,) int32 component labels, -1 for non-traversable cells.
    """
    labels = np.full(n_slots, -1, dtype=np.int32)
    next_label = 0
    for p in range(n_slots):
        if labels[p] >= 0:
            continue
        # Check if p has any neighbor (i.e. is traversable)
        has_neighbor = bool(np.any(physical_neighbors[p] >= 0))
        if not has_neighbor:
            continue
        # BFS
        labels[p] = np.int32(next_label)
        queue = deque([p])
        while queue:
            u = queue.popleft()
            for k in range(4):
                v = int(physical_neighbors[u, k])
                if v >= 0 and labels[v] < 0:
                    labels[v] = np.int32(next_label)
                    queue.append(v)
        next_label += 1
    return labels


def _observations_per_component(
    component_labels: np.ndarray,
    node_at_position: np.ndarray,
    n_obs: int,
) -> list[set[int]]:
    """Which DAG observation nodes appear in each connected component.

    Returns list of sets: index = component label.
    """
    n_components = (
        int(np.max(component_labels)) + 1
        if np.any(component_labels >= 0)
        else 0
    )
    result: list[set[int]] = [set() for _ in range(n_components)]
    for p in range(len(component_labels)):
        c = int(component_labels[p])
        if c < 0:
            continue
        obs = int(node_at_position[p])
        if 0 <= obs < n_obs:
            result[c].add(obs)
    return result


def _build_dense_neighbors_from_layout(
    layout: dict,
    action_space: dict,
    canvas_h: int,
    canvas_w: int,
) -> np.ndarray:
    """Build ``(S, 4)`` dense canvas neighbor index from parent layout.

    Reads the layout's ``next_state`` and ``action_valid`` in compact
    graph index space and maps them to dense row-major canvas positions
    using ``state_to_row_col``.  Non-traversable canvas positions have
    all-1 neighbor entries.

    Direction order: UP=0, RIGHT=1, DOWN=2, LEFT=3.

    Args:
        layout: Compact ``SpatialLayout`` record.
        action_space: ActionSpace descriptor (must be ``grid4_dir``).
        canvas_h: Grid height in cells.
        canvas_w: Grid width in cells.

    Returns:
        ``(S, 4)`` int32 neighbor index array, -1 for no neighbor.
    """
    from ehc_sn.tasks.routebind.contracts import (
        DIRECTION_DELTA,
        Direction,
    )

    num_slots = canvas_h * canvas_w
    neighbors = np.full((num_slots, 4), -1, dtype=np.int32)

    # Map direction deltas to action indices.
    action_names = action_space["action_names"]
    stay_idx = action_space.get("stay_action")
    dir_to_action: dict[int, int] = {}
    for dir_val in [
        Direction.UP,
        Direction.RIGHT,
        Direction.DOWN,
        Direction.LEFT,
    ]:
        delta = DIRECTION_DELTA[Direction(dir_val)]
        for a, name in enumerate(action_names):
            if a == stay_idx:
                continue
            ad = tuple(action_space["action_deltas"][a])
            if ad == delta:
                dir_to_action[dir_val] = a
                break

    next_state = layout["next_state"]
    action_valid = layout["action_valid"]
    row_col = layout["state_to_row_col"]
    n_comp = layout["graph_state_count"]

    # Build dense-position → compact-state lookup.
    dense_to_compact = np.full(num_slots, -1, dtype=np.int32)
    for s in range(n_comp):
        r = int(row_col[s, 0])
        c = int(row_col[s, 1])
        p = r * canvas_w + c
        dense_to_compact[p] = s

    for s in range(n_comp):
        p = int(row_col[s, 0]) * canvas_w + int(row_col[s, 1])
        for dir_val in [
            Direction.UP,
            Direction.RIGHT,
            Direction.DOWN,
            Direction.LEFT,
        ]:
            a = dir_to_action.get(dir_val)
            if a is None:
                continue
            if action_valid[s, a]:
                q = int(next_state[s, a])
                qr = int(row_col[q, 0])
                qc = int(row_col[q, 1])
                qp = qr * canvas_w + qc
                neighbors[p, int(dir_val)] = qp

    return neighbors


def _canonicalize_layout_to_canvas(
    layout: dict,
    storage_h: int,
    storage_w: int,
    n_obs: int,
    *,
    natural_h: int,
    natural_w: int,
    row_offset: int,
    col_offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Canonicalize a compact SpatialLayout into a padded storage canvas.

    The topology substrate stores a compact graph of traversable states.
    Routebind requires a dense fixed-size storage canvas with centered
    embedding of the natural layout.

    - Positions within the embedded natural extent are populated from the
      compact graph: traversable states become ``CELL_OBSERVATION`` or
      ``CELL_FREE``; positions without a compact state become ``CELL_WALL``.
    - Positions outside the embedded natural extent become ``CELL_PAD``
      (storage padding, not part of the topology).

    Args:
        layout: Compact ``SpatialLayout`` record.
        storage_h: Storage canvas height in cells.
        storage_w: Storage canvas width in cells.
        n_obs: Number of DAG observation nodes (vocabulary size).
        natural_h: Natural layout height in cells.
        natural_w: Natural layout width in cells.
        row_offset: Row offset for embedding the natural layout into the
            storage canvas (centered placement).
        col_offset: Col offset for embedding.

    Returns:
        Tuple ``(cell_type, observation_id, node_at_position, spatial_mask)``:
        - cell_type: ``(num_slots,)`` int32.
        - observation_id: ``(num_slots,)`` int32.
        - node_at_position: ``(num_slots,)`` int32.
        - spatial_mask: ``(num_slots,)`` bool — ``True`` inside the embedded
          natural extent.
    """
    from ehc_sn.tasks.routebind.contracts import (
        CELL_OBSERVATION,
        CELL_PAD,
        CELL_WALL,
    )

    num_slots = storage_h * storage_w
    cell_type = np.full(num_slots, CELL_PAD, dtype=np.int32)
    observation_id = np.full(num_slots, -1, dtype=np.int32)
    node_at_position = np.full(num_slots, -1, dtype=np.int32)
    spatial_mask = np.zeros(num_slots, dtype=bool)

    # Compute the bounding box of the natural extent within the storage canvas.
    r_start = row_offset
    r_end = row_offset + natural_h
    c_start = col_offset
    c_end = col_offset + natural_w

    # Mark positions inside the natural extent.
    for r in range(natural_h):
        for c in range(natural_w):
            p = (r_start + r) * storage_w + (c_start + c)
            spatial_mask[p] = True
            # Default: WALL inside natural extent (no compact state at this position).
            cell_type[p] = CELL_WALL

    # Scatter compact graph states into the natural extent.
    row_col = layout["state_to_row_col"]
    compact_obs = layout["observation_id"]
    for i in range(layout["graph_state_count"]):
        r = int(row_col[i, 0])
        c = int(row_col[i, 1])
        p = (r_start + r) * storage_w + (c_start + c)
        oid = int(compact_obs[i])
        observation_id[p] = oid
        # Every traversable state with a valid observation is OBSERVATION.
        # CELL_FREE is reserved for future use.
        cell_type[p] = CELL_OBSERVATION
        if 0 <= oid < n_obs:
            node_at_position[p] = oid

    return cell_type, observation_id, node_at_position, spatial_mask


# =============================================================================
# Goal selection and start selection helpers
# =============================================================================


def _bin_starts_by_distance(
    distance: np.ndarray,
    opt_count: np.ndarray,
    node_at_position: np.ndarray,
    n_slots: int,
    n_obs: int,
    goal_node_idx: int,
    profile: RoutebindSamplingProfile,
    funnel: GenerationFunnel,
) -> dict[int, list[int]]:
    """Bin eligible start positions by physical-distance bucket.

    For every start candidate (traversable, valid observation, not goal,
    reachable, unique optimum), bin its flat index into the profile's
    physical-length buckets based on ``distance``.  Unreachable and
    ambiguous starts are counted in the funnel but excluded from results.

    Returns:
        ``{phys_bucket_idx: [start_position, ...]}`` for eligible starts.
    """
    INF = np.iinfo(np.int32).max
    buckets: dict[int, list[int]] = {
        i: [] for i in range(len(profile.physical_length_bins))
    }

    for p in range(n_slots):
        obs = int(node_at_position[p])
        if obs < 0 or obs >= n_obs:
            continue
        if obs == goal_node_idx:
            continue
        s = p * n_obs + obs
        d = int(distance[s])
        if d >= INF:
            continue
        # Map distance to physical route length = distance + 1
        rlen = d + 1
        bi = profile.physical_bucket_index(rlen)
        if bi is None:
            if profile.mode == "distribution":
                # In distribution mode, assign to the nearest physical bin
                # rather than dropping the candidate.  The catch-all first
                # bin (always present in distribution presets) will absorb
                # routes below its lo; routes above hard_max_route_length
                # go into the last bin.
                for i, (lo, hi) in enumerate(profile.physical_length_bins):
                    if rlen < lo:
                        bi = i
                        break
                if bi is None:
                    bi = len(profile.physical_length_bins) - 1
            else:
                # restricted_domain: reject silently (funnel counts below)
                funnel.rejected_bucket_mismatch += 1
                continue
        funnel.inc_examined(bi)
        if opt_count[s] >= 2:
            funnel.inc_ambiguous(bi)
            continue
        funnel.inc_eligible(bi)
        buckets[bi].append(p)

    return buckets


def _compute_joint_deficits(
    accepted: dict[tuple[int, int], int],
    total_samples: int,
    profile: RoutebindSamplingProfile,
) -> dict[tuple[int, int], float]:
    """Compute deficit (target - realized) per joint bucket.

    Returns a dict mapping ``(phys_idx, sem_idx) -> deficit`` where
    positive values indicate underfilled buckets.
    """
    targets = profile.joint_targets()
    deficits: dict[tuple[int, int], float] = {}
    for k, tgt_frac in targets.items():
        tgt_n = tgt_frac * total_samples if total_samples > 0 else 0
        cur = accepted.get(k, 0)
        deficits[k] = max(0.0, tgt_n - cur)
    return deficits


def _select_goals_deficit_driven(
    adjacency: list[list[int]],
    node_at_position: np.ndarray,
    component_labels: np.ndarray,
    obs_per_component: list[set[int]],
    n_obs: int,
    rng: np.random.Generator,
) -> list[int]:
    """Return all valid goal IDs for this layout, shuffled.

    Unlike the old random-cap selector, this returns **all** valid goals
    so the main loop can iterate them and stop when ``attempt_budget``
    is reached or deficits are filled.

    A goal must satisfy:
    - it has at least one physical occurrence in a traversable component;
    - it has at least one DAG successor (not a sink).
    """
    obs_present: set[int] = set()
    for comp_obs in obs_per_component:
        obs_present.update(comp_obs)

    candidates: list[int] = []
    for o in range(n_obs):
        if o not in obs_present:
            continue
        if len(adjacency[o]) == 0:
            continue
        candidates.append(o)

    rng.shuffle(candidates)
    return candidates


# =============================================================================
# Builder
# =============================================================================


def build_routebind_task_corpus(
    version_root: Path,
    *,
    layouts: list[dict],
    topology_manifest: dict,
    dagflow_root: Path,
    dagflow_graph_id: str,
    corpus: str = "default",
    storage_height: int = 32,
    storage_width: int = 32,
    field_decay_spatial: float = 0.9848,
    field_decay_semantic: float = 0.8,
    max_supported_route_length: int = 150,
    n_queries_per_layout: int = 10,
    seed: int = 42,
    preset: RoutebindSamplingProfile | None = None,
    target_samples_total: int | None = None,
    allow_partial: bool = False,
) -> None:
    """Build the routebind task corpus at *version_root* over spatial
    topology and semantic DAG parent artifacts.

    Consumes:
    - topology layout records (complete visible spatial world including
      per-position observation identities, preserved unchanged);
    - one dagflow graph artifact (hidden directed edges over the same
      public observation vocabulary, shared across all splits).

    Routebind does not place, modify, or remap observations.  It reads
    observation identities from the topology substrate verbatim.

    Each layout yields up to *n_queries_per_layout* distinct start/goal
    query samples.  Split assignment follows the topology layout's own
    ``split`` field.

    Heterogeneous parental natural extents are permitted.  Every corpus
    declares one configured *storage_extent* ``[storage_height, storage_width]``.
    Each parent layout must fit within that storage extent and is embedded
    via centered placement.

    Args:
        version_root: Destination versioned root
            (e.g. ``data/processed/routebind/default/v1``).  Must not exist.
        layouts: Pre-validated topology layout records
            (``list[SpatialLayout]``).  Observation identities are read from
            ``layout["observation_id"]`` and preserved unchanged.
        topology_manifest: Parsed manifest from the topology root.
        dagflow_root: Path to dagflow versioned root containing the
            semantic DAG artifact.
        dagflow_graph_id: Stable artifact ID within *dagflow_root*
            identifying the single DAG used for this corpus.
        corpus: Corpus label (e.g. ``"default"``).
        storage_height: Storage canvas height in cells (default: 32).
        storage_width: Storage canvas width in cells (default: 32).
        field_decay_spatial: Spatial field decay factor gamma_space in
            ``(0, 1)``.
        field_decay_semantic: Semantic field decay factor gamma_sem in
            ``(0, 1)``.
        max_supported_route_length: Maximum route length for terminal
            activation consistency check.
        n_queries_per_layout: Number of start/goal queries to attempt per
            layout (default: 10).  When a ``preset`` is provided,
            this is interpreted as ``attempt_budget`` and the profile's
            own budget takes precedence if set.
        seed: Deterministic base seed for overall reproducibility.
        preset: Target distribution for deficit-driven query
            selection.  When ``None``, defaults to ``balanced``.
        target_samples_total: Hard output size target across all splits.
            When set, the builder raises ``ValueError`` if the accepted
            sample count falls short unless *allow_partial* is ``True``.
        allow_partial: When ``True`` and *target_samples_total* is set,
            a shortfall writes ``build_status: "partial"`` to the manifest
            instead of raising.

    Raises:
        FileExistsError: When *version_root* already exists (immutable root).
        ValueError: When topology family is unsupported, any layout exceeds
            the configured storage extent, ``n_queries_per_layout < 1``,
            or target_samples_total is not met and allow_partial is False.
    """
    version = extract_version(version_root)

    # --- Validate topology family ---
    valid_families = ("openfield", "dungeongen")
    topology_family = topology_manifest.get("family", "")
    if topology_family not in valid_families:
        raise ValueError(
            f"Routebind task corpus requires an 'openfield' or 'dungeongen' "
            f"topology dataset, got family={topology_family!r}."
        )

    if n_queries_per_layout < 1:
        raise ValueError(
            f"n_queries_per_layout={n_queries_per_layout} must be >= 1."
        )

    if storage_height <= 0 or storage_width <= 0:
        raise ValueError(
            f"Storage dimensions must be positive, got "
            f"[{storage_height}, {storage_width}]."
        )

    num_slots = storage_height * storage_width

    # --- Validate topology compatibility per layout ---
    # Routebind v1 requires 4-neighbor rectangular grid movement (grid4_canonical).
    for ly in layouts:
        ly_action = ly.get("action_space", {})
        ly_movement = ly_action.get("movement_kind", "")
        if ly_movement != "grid4":
            raise ValueError(
                f"Layout {ly['layout_id']!r} has action_space.movement_kind="
                f"{ly_movement!r}. Routebind v1 requires movement_kind='grid4' "
                f"(4-direction orthogonal movement: UP, DOWN, LEFT, RIGHT)."
            )

    if not layouts:
        raise ValueError("At least one topology layout is required.")

    # --- Derive canonical parent paths ---
    topology_preset = topology_manifest.get("preset", "default")
    canonical_topology = (
        f"data/interim/{topology_family}/{topology_preset}/"
        f"v{topology_manifest['version']}"
    )

    # --- Compute natural extent statistics and validate fit ---
    natural_heights: list[int] = []
    natural_widths: list[int] = []
    for ly in layouts:
        ly_ext = ly.get("extent")
        if ly_ext is None or len(ly_ext) != 2:
            raise ValueError(
                f"Layout {ly['layout_id']!r} is missing per-layout 'extent'."
            )
        ly_h, ly_w = int(ly_ext[0]), int(ly_ext[1])
        if ly_h <= 0 or ly_w <= 0:
            raise ValueError(
                f"Layout {ly['layout_id']!r} has invalid extent: [{ly_h}, {ly_w}]."
            )
        if ly_h > storage_height or ly_w > storage_width:
            raise ValueError(
                f"Layout {ly['layout_id']!r} has natural extent [{ly_h}, {ly_w}] "
                f"which exceeds configured storage extent "
                f"[{storage_height}, {storage_width}]. "
                "All layouts must fit within the storage extent."
            )
        natural_heights.append(ly_h)
        natural_widths.append(ly_w)

    # --- Resolve corpus-level sample target ---
    if target_samples_total is None:
        target_samples_total = n_queries_per_layout * len(layouts)

    # --- Load semantic DAG from dagflow parent ---
    dagflow_entry, dagflow_sample = find_artifact_by_id(
        dagflow_root, dagflow_graph_id
    )
    validate_dagflow_layout_sample(dagflow_sample)

    dagflow_manifest = read_manifest(dagflow_root)
    dagflow_max_out_degree = dagflow_manifest.get("max_out_degree", 4)

    # Build adjacency in TWO index spaces:
    #   rank_adjacency[rank]  → [successor_rank]   — for digest verification
    #   pub_adjacency[pub_id] → [successor_pub_id]  — for oracle use
    # *_DAG = O_topology = {0..N_obs-1} is the invariant contract.
    n_actual = int(dagflow_sample["node_mask"].sum())
    n_obs = n_actual
    dagflow_obs_ids = dagflow_sample["node_obs_id"][:n_obs].astype(np.int32)
    dagflow_rank_to_obs = dagflow_sample["rank_to_obs_id"][:n_obs].astype(
        np.int32
    )
    dagflow_obs_to_rank = dagflow_sample["obs_id_to_rank"][:n_obs].astype(
        np.int32
    )

    rank_adjacency: list[list[int]] = [[] for _ in range(n_obs)]
    pub_adjacency: list[list[int]] = [[] for _ in range(n_obs)]
    for rank in range(n_obs):
        src_pub = int(dagflow_obs_ids[rank])
        for k in range(dagflow_max_out_degree):
            if dagflow_sample["successor_mask"][rank, k]:
                succ_pub = int(dagflow_sample["successor_indices"][rank, k])
                if 0 <= succ_pub < n_obs:
                    succ_rank = int(dagflow_obs_to_rank[succ_pub])
                    rank_adjacency[rank].append(succ_rank)
                    pub_adjacency[src_pub].append(succ_pub)

    # Sort each successor list for deterministic BFS ordering.
    for u in range(n_obs):
        rank_adjacency[u].sort()
        pub_adjacency[u].sort()

    pred_offsets, pred_nodes = _build_dag_csr(pub_adjacency, n_obs)

    # Verify content digest if stored
    if dagflow_entry.content_digest:
        computed = canonical_dag_digest(rank_adjacency, dagflow_obs_ids)
        if computed != dagflow_entry.content_digest:
            raise ValueError(
                f"DAG content digest mismatch for {dagflow_graph_id!r}: "
                f"computed {computed}, "
                f"stored {dagflow_entry.content_digest}."
            )

    layouts_by_split: dict[str, list] = {"train": [], "val": [], "test": []}
    for ly in layouts:
        spl = ly.get("split", "train")
        layouts_by_split[spl].append(ly)

    natural_extent_homogeneous = (
        len(set(natural_heights)) == 1 and len(set(natural_widths)) == 1
    )
    stage_params: dict[str, Any] = {
        "corpus": corpus,
        "n_observations": n_actual,
        "storage_extent": [storage_height, storage_width],
        "num_spatial_slots": num_slots,
        "field_decay_spatial": field_decay_spatial,
        "field_decay_semantic": field_decay_semantic,
        "max_supported_route_length": max_supported_route_length,
        "n_queries_per_layout": n_queries_per_layout,
        "seed": seed,
        "topology_family": topology_family,
        "topology_preset": topology_preset,
        "topology_version": topology_manifest["version"],
        "dagflow_root": str(dagflow_root),
        "dagflow_graph_id": dagflow_graph_id,
        "preset": ("balanced" if preset is None else "custom"),
        "spatial_storage_policy": "pad_to_configured_max",
        "placement_policy": "center",
        "natural_extent_homogeneous": natural_extent_homogeneous,
        "natural_height_range": [min(natural_heights), max(natural_heights)],
        "natural_width_range": [min(natural_widths), max(natural_widths)],
        "requested_sample_count": target_samples_total,
    }
    # Diagnostic counters (will be filled per-split)
    search_metrics: dict[str, dict[str, int]] = {
        split: {"searches": 0, "starts_examined": 0, "accepted": 0}
        for split in _SPLITS
    }

    rejection_counts: dict[str, dict[str, int]] = {
        split: {} for split in _SPLITS
    }

    with staging_root(version_root) as tmp:
        all_entries: list = []

        for split in _SPLITS:
            split_layouts = layouts_by_split.get(split, [])
            if not split_layouts:
                continue

            split_rng = np.random.default_rng(
                int(seed)
                + (0 if split == "train" else 1 if split == "val" else 2)
            )
            indices = list(range(len(split_layouts)))
            split_rng.shuffle(indices)
            shuffled_layouts = [split_layouts[i] for i in indices]

            samples: list[dict[str, np.ndarray]] = []
            split_rejections: dict[str, int] = {}
            split_funnels: dict[int, GenerationFunnel] = {}

            for layout_idx, layout in enumerate(shuffled_layouts):
                # --- Determine natural extent and centered placement ---
                ly_ext = layout["extent"]
                natural_h = int(ly_ext[0])
                natural_w = int(ly_ext[1])
                row_offset = (storage_height - natural_h) // 2
                col_offset = (storage_width - natural_w) // 2

                # --- Dense canonicalization: compact SpatialLayout → padded storage canvas ---
                n_slots = (
                    num_slots  # storage canvas size, not graph_state_count
                )
                cell_type, observation_id, node_at_position, spatial_mask = (
                    _canonicalize_layout_to_canvas(
                        layout,
                        storage_height,
                        storage_width,
                        n_actual,
                        natural_h=natural_h,
                        natural_w=natural_w,
                        row_offset=row_offset,
                        col_offset=col_offset,
                    )
                )

                from ehc_sn.tasks.routebind.contracts import CELL_OBSERVATION

                # Build dense canvas row/col array for oracle reconstruction.
                # Shape (num_slots, 2) — row-major: slot p = r * storage_w + c.
                dense_row_col = np.stack(
                    [
                        np.repeat(np.arange(storage_height), storage_width),
                        np.tile(np.arange(storage_width), storage_height),
                    ],
                    axis=1,
                ).astype(np.int32)

                # Build traversable mask and physical neighbors from parent layout.
                # Only consider positions inside the natural extent (spatial_mask).
                traversable = (cell_type == CELL_OBSERVATION) & spatial_mask
                traversable_positions = [
                    p for p in range(n_slots) if traversable[p]
                ]

                # Build physical neighbors from the compact layout (unpadded).
                physical_neighbors = _build_dense_neighbors_from_layout(
                    layout, layout["action_space"], natural_h, natural_w
                )
                # Build dense-position → compact-state lookup for the natural grid.
                # _build_dense_neighbors_from_layout returns an array indexed by
                # dense row-major position (p_nat = r * natural_w + c) containing
                # dense row-major destination positions.  We need compact state
                # indices to look up row/col from state_to_row_col.
                nat_n_slots = natural_h * natural_w
                nat_dense_to_compact = np.full(nat_n_slots, -1, dtype=np.int32)
                for s in range(layout["graph_state_count"]):
                    r = int(layout["state_to_row_col"][s, 0])
                    c = int(layout["state_to_row_col"][s, 1])
                    nat_dense_to_compact[r * natural_w + c] = s
                # Remap natural-grid dense neighbors to storage-canvas coordinates.
                remapped_neighbors = np.full((n_slots, 4), -1, dtype=np.int32)
                for s in range(layout["graph_state_count"]):
                    rc = layout["state_to_row_col"][s]
                    p_nat = int(rc[0]) * natural_w + int(rc[1])
                    p_storage = (int(rc[0]) + row_offset) * storage_width + (
                        int(rc[1]) + col_offset
                    )
                    for d in range(4):
                        q_nat = int(physical_neighbors[p_nat, d])
                        if q_nat >= 0:
                            q_compact = int(nat_dense_to_compact[q_nat])
                            if q_compact >= 0:
                                qr = int(
                                    layout["state_to_row_col"][q_compact, 0]
                                )
                                qc = int(
                                    layout["state_to_row_col"][q_compact, 1]
                                )
                                qp = (qr + row_offset) * storage_width + (
                                    qc + col_offset
                                )
                                remapped_neighbors[p_storage, d] = qp
                physical_neighbors = remapped_neighbors

                component_labels = _compute_connected_components(
                    physical_neighbors, n_slots
                )

                obs_per_component = _observations_per_component(
                    component_labels, node_at_position, n_actual
                )

                # --- Goal-driven query generation ---
                # One reverse BFS per goal; reuse the distance/policy table
                # for multiple start queries sharing the same goal.
                query_count = 0
                funnel = GenerationFunnel()

                # --- Deficit-driven query generation using the profile ---
                profile = (
                    preset if preset is not None else resolve_preset("balanced")
                )
                attempt_budget = profile.attempt_budget
                if n_queries_per_layout != 10:
                    # User explicitly set n_queries_per_layout → use as budget
                    attempt_budget = n_queries_per_layout

                # Track accepted joint-bucket counts for deficit computation
                joint_accepted: dict[tuple[int, int], int] = {}

                # --- Goal-driven deficit-aware query generation ---
                query_count = 0

                # All observation-bearing traversable positions
                dag_obs_positions = [
                    p for p in traversable_positions if node_at_position[p] >= 0
                ]
                if not dag_obs_positions:
                    continue

                # Collect valid start positions (per layout RNG)
                layout_rng = np.random.default_rng(
                    int(seed)
                    + layout_idx * 10000
                    + (
                        0
                        if split == "train"
                        else 100000 if split == "val" else 200000
                    )
                )

                # Pre-allocate the kernel workspace arrays (reused per goal)
                n_states = n_slots * n_actual
                _kernel_distance = np.full(
                    n_states, np.iinfo(np.int32).max, dtype=np.int32
                )
                _kernel_policy_kind = np.zeros(n_states, dtype=np.int8)
                _kernel_policy_next = np.full(n_states, -1, dtype=np.int32)
                _kernel_opt_count = np.zeros(n_states, dtype=np.uint8)
                _kernel_deque_buf = np.zeros(2 * n_states, dtype=np.int32)

                # Track accepted joint-bucket counts for deficit computation
                joint_accepted: dict[tuple[int, int], int] = {}

                goals_for_layout = _select_goals_deficit_driven(
                    pub_adjacency,
                    node_at_position,
                    component_labels,
                    obs_per_component,
                    n_actual,
                    layout_rng,
                )
                if not goals_for_layout:
                    split_rejections["no_valid_goal"] = (
                        split_rejections.get("no_valid_goal", 0) + 1
                    )
                    continue

                for goal_idx in goals_for_layout:
                    if query_count >= attempt_budget:
                        break

                    total_accepted_sofar = sum(joint_accepted.values())
                    deficits = _compute_joint_deficits(
                        joint_accepted,
                        total_accepted_sofar + 1,  # avoid division by zero
                        profile,
                    )
                    # If every joint bucket is at or above target, stop
                    max_deficit = max(deficits.values()) if deficits else 0
                    if max_deficit <= 0 and total_accepted_sofar > 0:
                        break

                    goal_positions = np.where(node_at_position == goal_idx)[0]
                    if goal_positions.shape[0] == 0:
                        continue

                    search_metrics[split]["searches"] += 1

                    # Reset kernel workspaces for this goal
                    _kernel_distance[:] = np.iinfo(np.int32).max
                    _kernel_policy_kind[:] = 0
                    _kernel_policy_next[:] = -1
                    _kernel_opt_count[:] = 0

                    workspace = dict(
                        distance=_kernel_distance,
                        policy_kind=_kernel_policy_kind,
                        policy_next=_kernel_policy_next,
                        opt_count=_kernel_opt_count,
                        deque_buf=_kernel_deque_buf,
                    )
                    table = compute_goal_distance_table(
                        physical_neighbors=physical_neighbors,
                        node_at_position=node_at_position,
                        pred_offsets=pred_offsets,
                        pred_nodes=pred_nodes,
                        goal_occurrences=goal_positions,
                        goal_node_idx=goal_idx,
                        n_slots=n_slots,
                        n_obs=n_actual,
                        _workspace=workspace,
                    )

                    if table["deque_overflow"]:
                        split_rejections["goal_table_overflow"] = (
                            split_rejections.get("goal_table_overflow", 0) + 1
                        )
                        continue

                    # Bin eligible starts by physical-distance bucket
                    binned_starts = _bin_starts_by_distance(
                        table["distance"],
                        table["opt_count"],
                        node_at_position,
                        n_slots,
                        n_actual,
                        goal_idx,
                        profile,
                        funnel,
                    )

                    # Iterate buckets in deficit priority order
                    sorted_buckets = sorted(
                        binned_starts.keys(),
                        key=lambda bi: -deficits.get((bi, 0), 0),
                    )

                    for phys_bi in sorted_buckets:
                        start_positions = binned_starts.get(phys_bi, [])
                        if not start_positions:
                            continue
                        layout_rng.shuffle(start_positions)

                        for start_pos in start_positions:
                            if query_count >= attempt_budget:
                                break

                            start_obs = int(node_at_position[start_pos])
                            search_metrics[split]["starts_examined"] += 1

                            # Quick component check
                            start_component = int(component_labels[start_pos])
                            if start_component < 0:
                                continue
                            available_obs = obs_per_component[start_component]
                            if node_at_position[start_pos] not in available_obs:
                                continue

                            result = reconstruct_from_policy(
                                start_pos=start_pos,
                                start_obs=start_obs,
                                goal_node_idx=goal_idx,
                                n_obs=n_actual,
                                distance=table["distance"],
                                policy_kind=table["policy_kind"],
                                policy_next=table["policy_next"],
                                opt_count=table["opt_count"],
                                row_col=dense_row_col,
                                node_at_position=node_at_position,
                            )

                            if result is None:
                                continue

                            # --- Post-reconstruction validation ---
                            physical_route = result.physical_route
                            waypoints = result.waypoints

                            if len(set(physical_route)) != len(physical_route):
                                funnel.rejected_non_simple += 1
                                continue
                            if len(physical_route) > max_supported_route_length:
                                funnel.rejected_route_too_long += 1
                                continue
                            if len(waypoints) < 2:
                                funnel.rejected_trivial_waypoint += 1
                                continue
                            if result.next_dir < 0 or result.next_dir > 3:
                                funnel.rejected_invalid_next_dir += 1
                                continue

                            # --- Bucket check ---
                            route_len = len(physical_route)
                            sem_len = len(waypoints)
                            phy_bi = profile.physical_bucket_index(route_len)
                            sem_bi = profile.semantic_bucket_index(sem_len)
                            if phy_bi is None or sem_bi is None:
                                if profile.mode == "distribution":
                                    # Distribution mode: map to nearest
                                    # bucket (catch-all) instead of dropping.
                                    if phy_bi is None:
                                        for i, (lo, hi) in enumerate(
                                            profile.physical_length_bins
                                        ):
                                            if route_len < lo:
                                                phy_bi = i
                                                break
                                        if phy_bi is None:
                                            phy_bi = (
                                                len(
                                                    profile.physical_length_bins
                                                )
                                                - 1
                                            )
                                    if sem_bi is None:
                                        for i, (lo, hi) in enumerate(
                                            profile.semantic_length_bins
                                        ):
                                            if sem_len < lo:
                                                sem_bi = i
                                                break
                                        if sem_bi is None:
                                            sem_bi = (
                                                len(
                                                    profile.semantic_length_bins
                                                )
                                                - 1
                                            )
                                else:
                                    funnel.rejected_bucket_mismatch += 1
                                    continue

                            funnel.inc_reconstructed(phy_bi, sem_bi)

                            # Check if joint bucket is full (tolerance max)
                            joint_key = (phy_bi, sem_bi)
                            tol = profile.tolerances.get(
                                str(joint_key),
                                profile.tolerances.get("*", (0.0, 1.0)),
                            )
                            current_n = joint_accepted.get(joint_key, 0)
                            if profile.mode == "distribution":
                                # The catch-all joint bin (first bin in both
                                # dimensions) is never capped — it absorbs
                                # overflow when hard supply is short.
                                if phy_bi == 0 and sem_bi == 0:
                                    max_n = 2**31 - 1
                                else:
                                    joint_targets = profile.joint_targets()
                                    tgt_frac = joint_targets.get(joint_key, 0.0)
                                    max_n = max(
                                        1,
                                        round(
                                            tgt_frac
                                            * (query_count + 1)
                                            * tol[1]
                                        ),
                                    )
                            else:
                                max_n = max(
                                    1, round(tol[1] * (query_count + 1))
                                )
                            if current_n >= max_n:
                                funnel.inc_bucket_full(phy_bi, sem_bi)
                                continue

                            # --- Encode targets ---
                            goal_mask = node_at_position == goal_idx
                            target_trajectory = encode_trajectory_field(
                                list(physical_route),
                                n_slots,
                                field_decay_spatial,
                            )
                            target_waypoint = encode_waypoint_field(
                                list(waypoints),
                                n_slots,
                                field_decay_semantic,
                            )

                            next_obs = int(result.next_obs)
                            target_next_obs = np.int32(
                                next_obs if next_obs >= 0 else -1
                            )

                            start_flag = np.zeros(n_slots, dtype=bool)
                            start_flag[start_pos] = True

                            sample_data = {
                                "cell_type": cell_type.astype(np.int32),
                                "observation_id": observation_id.astype(
                                    np.int32
                                ),
                                "start_flag": start_flag,
                                "goal_flag": goal_mask,
                                "spatial_mask": spatial_mask,
                                "natural_height": np.int32(natural_h),
                                "natural_width": np.int32(natural_w),
                                "row_offset": np.int32(row_offset),
                                "col_offset": np.int32(col_offset),
                                "target_trajectory": target_trajectory,
                                "target_waypoint": target_waypoint,
                                "target_next_dir": np.int32(result.next_dir),
                                "target_next_obs": target_next_obs,
                            }

                            target_issues = validate_generated_sample(
                                sample_data,
                                oracle_result=result,
                                n_obs=n_actual,
                                topo_vocab_size=n_actual,
                                S=n_slots,
                                gamma_space=field_decay_spatial,
                                gamma_semantic=field_decay_semantic,
                            )
                            target_errors = [
                                i
                                for i in target_issues
                                if i.severity == "ERROR"
                            ]
                            if target_errors:
                                funnel.rejected_target_validation += 1
                                continue

                            samples.append(sample_data)
                            search_metrics[split]["accepted"] += 1
                            funnel.inc_accepted(phy_bi, sem_bi)
                            joint_accepted[joint_key] = (
                                joint_accepted.get(joint_key, 0) + 1
                            )
                            query_count += 1

                if query_count == 0:
                    split_rejections["no_valid_query"] = (
                        split_rejections.get("no_valid_query", 0) + 1
                    )

                # Accumulate funnel into per-split funnel counter
                split_funnels[layout_idx] = funnel

            rejection_counts[split] = split_rejections

            # Merge per-layout funnels
            merged_funnel = GenerationFunnel()
            for lf in split_funnels.values():
                for d in [
                    "examined",
                    "unreachable",
                    "ambiguous",
                    "eligible",
                    "reconstructed",
                    "bucket_full",
                    "accepted",
                ]:
                    src = getattr(lf, d)
                    dst = getattr(merged_funnel, d)
                    for k, v in src.items():
                        dst[k] = dst.get(k, 0) + v
                merged_funnel.rejected_non_simple += lf.rejected_non_simple
                merged_funnel.rejected_route_too_long += (
                    lf.rejected_route_too_long
                )
                merged_funnel.rejected_trivial_waypoint += (
                    lf.rejected_trivial_waypoint
                )
                merged_funnel.rejected_invalid_next_dir += (
                    lf.rejected_invalid_next_dir
                )
                merged_funnel.rejected_target_validation += (
                    lf.rejected_target_validation
                )
                merged_funnel.rejected_bucket_mismatch += (
                    lf.rejected_bucket_mismatch
                )
            stage_params[f"funnel_{split}"] = merged_funnel.serialize()

            # Record realized distribution
            realized: dict[str, float] = {}
            accepted_dict = merged_funnel.accepted
            total_acc = sum(accepted_dict.values()) or 1
            for (phy_bi, sem_bi), cnt in accepted_dict.items():
                realized[f"({phy_bi},{sem_bi})"] = cnt / total_acc
            stage_params[f"realized_{split}"] = realized

            if not samples:
                continue

            entries = write_split(
                output_root=tmp,
                split=split,
                samples=samples,
                source=f"{TASK_FAMILY}/{corpus}",
                channels=ROUTEBIND_CORPUS_CHANNELS,
                topology_kind="grid2d",
                n_states=num_slots,
                extent=[storage_height, storage_width],
                index_kwargs={
                    "task_metadata": {
                        "task": TASK_FAMILY,
                        "corpus": corpus,
                    },
                },
            )
            all_entries.extend(entries)

        write_index_at_root(all_entries, tmp)

        final_counts: dict[str, int] = {}
        for e in all_entries:
            final_counts[e.split] = final_counts.get(e.split, 0) + 1

        total_accepted = sum(final_counts.values())
        if (
            target_samples_total is not None
            and total_accepted < target_samples_total
        ):
            if not allow_partial:
                # Build a detailed shortfall message
                lines = [
                    f"Routebind corpus target not met.",
                    f"",
                    f"  Requested samples: {target_samples_total}",
                    f"  Accepted samples: {total_accepted}",
                    f"  Attempt budget: {profile.attempt_budget if preset else n_queries_per_layout} per layout",
                    f"  Preset: {preset.description if preset else 'balanced'}",
                    f"",
                ]
                # Per-split counts
                for s in _SPLITS:
                    cnt = final_counts.get(s, 0)
                    lines.append(f"  Split {s}: {cnt}")
                lines.append("")
                # Rejection breakdown
                for s, reasons in rejection_counts.items():
                    for reason, count in reasons.items():
                        lines.append(f"  Rejected {s}/{reason}: {count}")
                lines.append("")
                raise ValueError("\n".join(lines))
            stage_params["build_status"] = "partial"
        else:
            stage_params["build_status"] = "complete"

        for s, reasons in rejection_counts.items():
            for reason, count in reasons.items():
                stage_params[f"rejected_{s}_{reason}"] = count

        for s, metrics in search_metrics.items():
            for key, val in metrics.items():
                stage_params[f"{s}_{key}"] = val

        write_manifest(
            tmp,
            dataset_class="task_corpus",
            family=TASK_FAMILY,
            version=version,
            channels=ROUTEBIND_CORPUS_CHANNELS,
            topology_kind="grid2d",
            n_states=num_slots,
            extent=[storage_height, storage_width],
            n_samples=final_counts,
            source_id=f"synthetic/{topology_family}",
            builder="ehc_sn.tasks.routebind.builder.build_routebind_task_corpus",
            seed=seed,
            stage_params=stage_params,
            task=TASK_FAMILY,
            corpus=corpus,
            task_schema_version=1,
            task_protocol_version=2,
            manifest_schema_version=1,
            parents={
                "spatial_topology": {
                    "family": topology_family,
                    "root": canonical_topology,
                    "version": topology_manifest["version"],
                },
                "semantic_graph": {
                    "family": "dagflow",
                    "root": str(dagflow_root),
                    "version": dagflow_manifest["version"],
                    "artifact_id": dagflow_graph_id,
                    "split": dagflow_entry.split,
                    "content_digest": dagflow_entry.content_digest,
                },
            },
            n_observations=n_actual,
            topology_observation_vocabulary_size=n_actual,
            observation_sentinel_id=n_actual,
            storage_extent=[storage_height, storage_width],
            num_spatial_slots=num_slots,
            canvas_height=storage_height,
            canvas_width=storage_width,
            num_slots=num_slots,
            field_decay_spatial=field_decay_spatial,
            field_decay_semantic=field_decay_semantic,
            max_supported_route_length=max_supported_route_length,
            minimum_terminal_activation=field_decay_spatial
            ** max_supported_route_length,
        )


def validate_routebind_root(root: Path) -> dict:
    """Validate a routebind task corpus root against task-owned semantics.

    Delegates to ``validate_corpus_root`` in ``validation.py`` and raises
    ``ValueError`` on any ERROR-severity issue.

    Args:
        root: Resolved versioned routebind task corpus root.

    Returns:
        Parsed manifest dict.

    Raises:
        ValueError: On any contract violation.
    """
    from ehc_sn.tasks.routebind.validation import validate_corpus_root

    manifest, issues = validate_corpus_root(root)
    errors = [i for i in issues if i.severity == "ERROR"]
    if errors:
        raise ValueError(
            f"Corpus root validation failed with {len(errors)} error(s). "
            f"First: [{errors[0].code}] {errors[0].message}"
        )
    return manifest


# =============================================================================
__all__ = [
    "ROUTEBIND_CORPUS_CHANNELS",
    "ROUTEBIND_MODEL_INPUT_CHANNELS",
    "ROUTEBIND_TARGET_CHANNELS",
    "ROUTEBIND_TASK_CHANNEL_DTYPES",
    "TASK_FAMILY",
    "build_routebind_task_corpus",
    "validate_routebind_root",
]
