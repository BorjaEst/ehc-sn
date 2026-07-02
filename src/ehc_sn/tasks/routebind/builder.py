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

import hashlib
import json
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
    derive_optimal_transition_masks,
    reconstruct_from_policy,
    traverse_optimal_subgraph,
)
from ehc_sn.tasks.routebind.targets import (
    encode_trajectory_field,
    encode_trajectory_support,
    encode_waypoint_field,
    encode_waypoint_support,
)
from ehc_sn.tasks.routebind.validation import (
    validate_generated_sample,
    validate_stored_sample,
    validate_support_channels,
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
# Three-layer preset contract
#
# 1. QuerySelectionProfile  — pre-oracle, normative (controls semantic demand)
# 2. RealizedAdmissionPolicy — post-oracle, conditional (balances accepted samples)
# 3. CompletionPolicy        — artifact validity (strict / allow_degraded)
# =============================================================================


@dataclass(frozen=True)
class QuerySelectionProfile:
    """Pre-oracle query-selection properties.

    Controls which ``(start_position, goal_observation)`` pairs the builder
    samples by precomputed optimal physical route length.

    ``physical_route_position_bins`` = bins over the number of positions in
    the projected physical route (i.e. physical-move cost + 1), computed
    exactly from the product-state oracle query table.
    """

    physical_route_position_bins: list[tuple[int, int]]
    physical_route_position_targets: list[float]
    hard_min_route_positions: int = 2
    hard_max_route_positions: int = 150
    minimum_bin_support: int = 1
    max_instances_per_split: int | None = None
    tolerances: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {"*": (0.0, 1.0)}
    )
    description: str = ""

    def __post_init__(self) -> None:
        """Validate structural invariants."""
        if not self.physical_route_position_bins:
            raise ValueError("physical_route_position_bins must not be empty.")
        n_bins = len(self.physical_route_position_bins)
        if n_bins != len(self.physical_route_position_targets):
            raise ValueError(
                f"physical_route_position_bins ({n_bins}) and "
                f"physical_route_position_targets "
                f"({len(self.physical_route_position_targets)}) "
                "must have the same length."
            )
        total = sum(self.physical_route_position_targets)
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"physical_route_position_targets sum to {total}, "
                "expected ~1.0."
            )
        for i, ((lo, hi), tgt) in enumerate(
            zip(
                self.physical_route_position_bins,
                self.physical_route_position_targets,
            )
        ):
            if lo < self.hard_min_route_positions:
                raise ValueError(
                    f"physical_route_position_bins[{i}] lower bound "
                    f"{lo} < hard_min_route_positions "
                    f"({self.hard_min_route_positions})."
                )
            if hi > self.hard_max_route_positions:
                raise ValueError(
                    f"physical_route_position_bins[{i}] upper bound "
                    f"{hi} > hard_max_route_positions "
                    f"({self.hard_max_route_positions})."
                )
            if lo > hi:
                raise ValueError(
                    f"physical_route_position_bins[{i}] empty " f"({lo}, {hi})."
                )
            if tgt < 0.0 or tgt > 1.0:
                raise ValueError(
                    f"physical_route_position_targets[{i}] = {tgt} "
                    "outside [0, 1]."
                )

    def physical_bucket_index(self, route_positions: int) -> int | None:
        """Return the bin index for *route_positions*, or None."""
        for i, (lo, hi) in enumerate(self.physical_route_position_bins):
            if lo <= route_positions <= hi:
                return i
        return None

    def target_n_per_physical_bucket(self, total_n: int) -> list[int]:
        """Target count per physical-position bin for *total_n* samples."""
        return [
            max(0, round(p * total_n))
            for p in self.physical_route_position_targets
        ]


@dataclass(frozen=True)
class RealizedAdmissionPolicy:
    """Post-oracle admission quotas on realized instance properties.

    Controls waypoint-count distribution only. Physical route length is
    controlled pre-oracle by ``QuerySelectionProfile``.
    """

    waypoint_count_bins: list[tuple[int, int]] | None = None
    waypoint_count_targets: list[float] | None = None

    def semantic_bucket_index(self, waypoint_count: int) -> int | None:
        """Return the bin index for *waypoint_count*, or None."""
        if self.waypoint_count_bins is None:
            return None
        for i, (lo, hi) in enumerate(self.waypoint_count_bins):
            if lo <= waypoint_count <= hi:
                return i
        return None


@dataclass(frozen=True)
class CompletionPolicy:
    """Defines artifact validity when quotas are unmet."""

    mode: str = "strict"
    minimum_eligible_support: float = 0.0
    maximum_bucket_deficit: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in ("strict", "allow_degraded"):
            raise ValueError(
                f"CompletionPolicy.mode={self.mode!r} must be "
                "'strict' or 'allow_degraded'."
            )


@dataclass(frozen=True)
class RoutebindPreset:
    """Complete preset: selection profile + optional admission + completion."""

    selection: QuerySelectionProfile
    admission: RealizedAdmissionPolicy = field(
        default_factory=RealizedAdmissionPolicy
    )
    completion: CompletionPolicy = field(default_factory=CompletionPolicy)
    description: str = ""


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
    rejected_route_too_long: int = 0
    rejected_trivial_waypoint: int = 0
    rejected_invalid_next_dir: int = 0
    rejected_target_validation: int = 0

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
            "rejected_route_too_long": self.rejected_route_too_long,
            "rejected_trivial_waypoint": self.rejected_trivial_waypoint,
            "rejected_invalid_next_dir": self.rejected_invalid_next_dir,
            "rejected_target_validation": self.rejected_target_validation,
        }


# =============================================================================
# Named sampling presets
# =============================================================================


ROUTEBIND_PRESETS: dict[str, RoutebindPreset] = {
    "smoke": RoutebindPreset(
        selection=QuerySelectionProfile(
            physical_route_position_bins=[(2, 150)],
            physical_route_position_targets=[1.0],
            description="Accept-all single-bin preset.",
        ),
        # No admission quotas — accept all unique valid samples.
        admission=RealizedAdmissionPolicy(),
        completion=CompletionPolicy(mode="allow_degraded"),
        description="Accept-all preset for tests and calibration runs.",
    ),
    "balanced": RoutebindPreset(
        selection=QuerySelectionProfile(
            physical_route_position_bins=[
                (2, 7),
                (8, 15),
                (16, 30),
                (31, 50),
                (51, 80),
            ],
            physical_route_position_targets=[0.10, 0.20, 0.35, 0.25, 0.10],
            description="Broad physical distribution for canonical training.",
        ),
        admission=RealizedAdmissionPolicy(
            waypoint_count_bins=[
                (2, 2),
                (3, 3),
                (4, 4),
                (5, 6),
                (7, 10),
            ],
            waypoint_count_targets=[0.15, 0.25, 0.25, 0.25, 0.10],
        ),
        description=(
            "Broad physical and semantic distribution for canonical "
            "training corpus."
        ),
    ),
    "long-spatial": RoutebindPreset(
        selection=QuerySelectionProfile(
            physical_route_position_bins=[
                (2, 25),
                (26, 50),
                (51, 80),
                (81, 120),
            ],
            physical_route_position_targets=[0.15, 0.35, 0.35, 0.15],
            description="Emphasize physical planning; moderate semantic complexity.",
        ),
        admission=RealizedAdmissionPolicy(
            waypoint_count_bins=[
                (2, 3),
                (4, 5),
                (6, 10),
            ],
            waypoint_count_targets=[0.45, 0.40, 0.15],
        ),
        description=(
            "Emphasize physical planning; moderate semantic complexity."
        ),
    ),
    "long-semantic": RoutebindPreset(
        selection=QuerySelectionProfile(
            physical_route_position_bins=[
                (2, 20),
                (21, 50),
                (51, 80),
                (81, 120),
            ],
            physical_route_position_targets=[0.20, 0.45, 0.25, 0.10],
            description="Broad physical distribution; many waypoints expected.",
        ),
        admission=RealizedAdmissionPolicy(
            waypoint_count_bins=[
                (4, 5),
                (6, 8),
                (9, 12),
                (13, 15),
            ],
            waypoint_count_targets=[0.20, 0.50, 0.25, 0.05],
        ),
        description=("Emphasize DAG composition; many waypoints."),
    ),
    "joint-hard": RoutebindPreset(
        selection=QuerySelectionProfile(
            physical_route_position_bins=[
                (20, 40),
                (41, 70),
                (71, 100),
                (101, 140),
            ],
            physical_route_position_targets=[0.20, 0.40, 0.30, 0.10],
            tolerances={"*": (0.0, 3.0)},
            description="Long physical routes; expects many waypoints.",
        ),
        admission=RealizedAdmissionPolicy(
            waypoint_count_bins=[
                (4, 5),
                (6, 8),
                (9, 12),
                (13, 15),
            ],
            waypoint_count_targets=[0.15, 0.45, 0.30, 0.10],
        ),
        completion=CompletionPolicy(mode="allow_degraded"),
        description=(
            "Long physical routes with post-oracle waypoint admission."
        ),
    ),
    "joint-hard-only": RoutebindPreset(
        selection=QuerySelectionProfile(
            physical_route_position_bins=[
                (20, 40),
                (41, 70),
                (71, 100),
                (101, 140),
            ],
            physical_route_position_targets=[0.20, 0.40, 0.30, 0.10],
            description="Long physical routes, strict waypoint admission.",
        ),
        admission=RealizedAdmissionPolicy(
            waypoint_count_bins=[
                (4, 5),
                (6, 8),
                (9, 12),
                (13, 15),
            ],
            waypoint_count_targets=[0.15, 0.45, 0.30, 0.10],
        ),
        completion=CompletionPolicy(mode="strict"),
        description=(
            "Long physical routes with strict post-oracle waypoint "
            "admission.  Samples outside bins are rejected."
        ),
    ),
}


def resolve_preset(
    name: str,
    overrides: dict | None = None,
) -> RoutebindPreset:
    """Look up a named preset, optionally overriding selection fields.

    Args:
        name: Preset key in ``ROUTEBIND_PRESETS``.
        overrides: Optional dict of selection profile fields to override
            (e.g. ``{"attempt_budget": 20}``).

    Returns:
        A (possibly modified) ``RoutebindPreset``.
    """
    base = ROUTEBIND_PRESETS.get(name)
    if base is None:
        raise ValueError(
            f"Unknown routebind preset {name!r}. "
            f"Valid: {sorted(ROUTEBIND_PRESETS)}."
        )
    if not overrides:
        return base
    # Build a new selection profile with overrides, preserving admission/completion.
    sel_kwargs = {
        "physical_route_position_bins": base.selection.physical_route_position_bins,
        "physical_route_position_targets": base.selection.physical_route_position_targets,
        "hard_min_route_positions": base.selection.hard_min_route_positions,
        "hard_max_route_positions": base.selection.hard_max_route_positions,
        "minimum_bin_support": base.selection.minimum_bin_support,
        "max_instances_per_split": base.selection.max_instances_per_split,
        "tolerances": base.selection.tolerances,
        "description": base.selection.description,
    }
    sel_kwargs.update(overrides)
    return RoutebindPreset(
        selection=QuerySelectionProfile(**sel_kwargs),
        admission=base.admission,
        completion=base.completion,
        description=base.description,
    )


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
        CELL_FREE,
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
        if 0 <= oid < n_obs:
            cell_type[p] = CELL_OBSERVATION
            node_at_position[p] = oid
        else:
            # observation_id == -1 sentinel → traversable but no semantic content
            cell_type[p] = CELL_FREE

    return cell_type, observation_id, node_at_position, spatial_mask


# =============================================================================
# Goal selection and start selection helpers
# =============================================================================


INF = np.iinfo(np.int32).max


def _build_query_table(
    physical_neighbors: np.ndarray,
    node_at_position: np.ndarray,
    pred_offsets: np.ndarray,
    pred_nodes: np.ndarray,
    dense_row_col: np.ndarray,
    n_slots: int,
    n_obs: int,
    goal_obs: list[int],
    max_supported_route_length: int,
) -> dict:
    """Precompute the oracle distance table for one layout.

    Runs one reverse 0-1 BFS per goal observation and extracts
    ``move_count`` and ``reachable`` for all ``(position, goal)`` pairs
    that have a finite-cost product-state solution.

    The returned table is start-position independent and requires no
    path reconstruction — ambiguity is not a rejection criterion.

    Returns a dict with keys:

        move_count: (n_slots, n_obs) int32 — minimum physical moves from
            each (position, observation_at_position) to a valid accepted goal.
        reachable: (n_slots, n_obs) bool — True iff a finite-cost path
            exists.
    """
    n_states = n_slots * n_obs
    INF_VAL = np.iinfo(np.int32).max
    move_count = np.full((n_slots, n_obs), INF_VAL, dtype=np.int32)
    reachable = np.zeros((n_slots, n_obs), dtype=bool)

    # Pre-allocated kernel workspace (reused per goal)
    _dist = np.full(n_states, INF_VAL, dtype=np.int32)
    _pkind = np.zeros(n_states, dtype=np.int8)
    _pnext = np.full(n_states, -1, dtype=np.int32)
    _optct = np.zeros(n_states, dtype=np.uint8)
    _deque = np.zeros(2 * n_states, dtype=np.int32)
    workspace = dict(
        distance=_dist,
        policy_kind=_pkind,
        policy_next=_pnext,
        opt_count=_optct,
        deque_buf=_deque,
    )

    for goal_idx in goal_obs:
        goal_positions = np.where(node_at_position == goal_idx)[0]
        if goal_positions.shape[0] == 0:
            continue

        # Reset workspace
        _dist[:] = INF_VAL
        _pkind[:] = 0
        _pnext[:] = -1
        _optct[:] = 0

        table = compute_goal_distance_table(
            physical_neighbors=physical_neighbors,
            node_at_position=node_at_position,
            pred_offsets=pred_offsets,
            pred_nodes=pred_nodes,
            goal_occurrences=goal_positions,
            goal_node_idx=goal_idx,
            n_slots=n_slots,
            n_obs=n_obs,
            _workspace=workspace,
        )

        if table["deque_overflow"]:
            continue

        d_arr = table["distance"]

        for p in range(n_slots):
            obs = int(node_at_position[p])
            if obs < 0 or obs >= n_obs:
                continue
            if obs == goal_idx:
                continue
            s = p * n_obs + obs
            d = int(d_arr[s])
            if d >= INF_VAL:
                continue
            if d > max_supported_route_length:
                continue
            move_count[p, goal_idx] = d
            reachable[p, goal_idx] = True

    return {
        "move_count": move_count,
        "reachable": reachable,
    }


@dataclass(frozen=True)
class CatalogEntry:
    """One eligible query in the global catalog."""

    layout_idx: int
    start_pos: int
    goal_obs: int
    waypoint_count: int
    physical_bin: int
    waypoint_bin: int


def _build_candidate_pools(
    query_table: dict,
    n_slots: int,
    n_obs: int,
    node_at_position: np.ndarray,
    selection: QuerySelectionProfile,
    admission: RealizedAdmissionPolicy | None,
    layout_idx: int,
    funnel: GenerationFunnel,
) -> dict[tuple[int, int], list[CatalogEntry]]:
    """Group eligible (start_pos, goal_obs) candidates by joint bin.

    Reads the precomputed query table and returns
    ``{(phys_bucket_idx, waypoint_bucket_idx): [CatalogEntry, ...]}``.

    Unreachable and out-of-bin pairs are excluded.  Ambiguous pairs
    (multiple optimal solutions) are NOT rejected — ambiguity is
    informational only.

    When ``admission`` has no waypoint bins, all candidates with
    physical route positions >= 2 are accepted into bin 0.
    """
    INF_VAL = np.iinfo(np.int32).max

    pools: dict[tuple[int, int], list[CatalogEntry]] = {}
    move_count = query_table["move_count"]
    reachable = query_table["reachable"]

    for p in range(n_slots):
        obs_start = int(node_at_position[p])
        if obs_start < 0 or obs_start >= n_obs:
            continue
        for g in range(n_obs):
            if g == obs_start:
                continue
            if not reachable[p, g]:
                continue
            route_positions = int(move_count[p, g]) + 1
            phy_bi = selection.physical_bucket_index(route_positions)
            if phy_bi is None:
                continue
            funnel.inc_examined(phy_bi)
            funnel.inc_eligible(phy_bi)
            if (
                admission is not None
                and admission.waypoint_count_bins is not None
            ):
                # Waypoint count unknown pre-reconstruction; treat as
                # post-oracle admission: allocate to all sem bins,
                # check during Phase D.
                sem_bi = 0
            else:
                sem_bi = 0
            key = (phy_bi, sem_bi)
            if key not in pools:
                pools[key] = []
            pools[key].append(
                CatalogEntry(
                    layout_idx=layout_idx,
                    start_pos=p,
                    goal_obs=int(g),
                    waypoint_count=0,  # unknown until Phase D
                    physical_bin=phy_bi,
                    waypoint_bin=sem_bi,
                )
            )

    return pools


def _select_goal_obs_for_layout(
    node_at_position: np.ndarray,
    n_slots: int,
    n_obs: int,
) -> list[int]:
    """Return all observation IDs that physically occur in this layout."""
    present: set[int] = set()
    for p in range(n_slots):
        obs = int(node_at_position[p])
        if 0 <= obs < n_obs:
            present.add(obs)
    return sorted(present)


# =============================================================================
# Build-status determination
# =============================================================================


def _compute_bin_support_from_pools(
    pools: dict[int, list],
    bins: list[tuple[int, int]],
) -> dict[str, int]:
    """Count eligible candidates in each physical-position bin.

    Returns dict mapping ``"(lo,hi)"`` to candidate count.
    """
    support: dict[str, int] = {}
    for i, (lo, hi) in enumerate(bins):
        cnt = len(pools.get(i, []))
        support[f"({lo},{hi})"] = cnt
    return support


def _determine_build_status(
    selection: QuerySelectionProfile,
    completion: CompletionPolicy,
    total_accepted: int,
    target_samples_total: int | None,
    admission_deficits: dict[str, dict],
) -> str:
    """Return ``"complete"``, ``"degraded"``, or ``"failed"``.

    Rules:
    1. ``completion.mode == "strict"`` and ``target_samples_total`` not met → ``"failed"``.
    2. Any admission deficit > 0 → ``"degraded"``.
    3. Otherwise → ``"complete"``.
    """
    if (
        target_samples_total is not None
        and total_accepted < target_samples_total
    ):
        if completion.mode == "strict":
            return "failed"
        return "degraded"

    has_any_deficit = False
    for split_name, split_deficits in admission_deficits.items():
        for bin_key, info in split_deficits.items():
            target_val = info.get("target", 0.0)
            achieved_val = info.get("achieved", 0.0)
            if target_val > 0 and achieved_val == 0:
                if completion.mode == "strict":
                    return "failed"
                has_any_deficit = True
            elif achieved_val < target_val:
                has_any_deficit = True

    if has_any_deficit:
        return "degraded"
    return "complete"


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
    storage_height: int = 30,
    storage_width: int = 30,
    field_decay_spatial: float = 0.9848,
    field_decay_semantic: float = 0.8,
    max_supported_route_length: int = 150,
    n_queries_per_layout: int = 10,
    seed: int = 42,
    preset: RoutebindPreset | None = None,
    target_samples_total: int | None = None,
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
        storage_height: Storage canvas height in cells (default: 30).
        storage_width: Storage canvas width in cells (default: 30).
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

    # Build public-ID-indexed successor mask/indices for
    # ``derive_optimal_transition_masks`` (Phase D).
    max_out_degree = max((len(succs) for succs in pub_adjacency), default=0)
    pub_succ_mask = np.zeros((n_obs, max_out_degree), dtype=bool)
    pub_succ_indices = np.full((n_obs, max_out_degree), -1, dtype=np.int32)
    for src_pub in range(n_obs):
        for k, dst_pub in enumerate(pub_adjacency[src_pub]):
            pub_succ_mask[src_pub, k] = True
            pub_succ_indices[src_pub, k] = np.int32(dst_pub)

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
        "target_semantics": "optimal_subgraph_support",
        "target_schema_version": 1,
        "depth_sentinel": -1,
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

            # =============================================================
            # Phase A: Build query table per layout → populate global catalog
            # =============================================================
            global_catalog: dict[tuple[int, int], list[CatalogEntry]] = {}
            per_layout_state: dict[int, dict] = (
                {}
            )  # layout_idx → tensors for reconstruction
            funnel = GenerationFunnel()

            preset_ = (
                preset if preset is not None else resolve_preset("balanced")
            )
            selection_profile = preset_.selection
            admission_policy = preset_.admission
            completion_policy = preset_.completion

            max_instances = selection_profile.max_instances_per_split
            if max_instances is None:
                max_instances = len(shuffled_layouts)

            for layout_idx, layout in enumerate(shuffled_layouts):
                if layout_idx >= max_instances:
                    break

                # --- Determine natural extent and centered placement ---
                ly_ext = layout["extent"]
                natural_h = int(ly_ext[0])
                natural_w = int(ly_ext[1])
                row_offset = (storage_height - natural_h) // 2
                col_offset = (storage_width - natural_w) // 2

                # --- Dense canonicalization ---
                n_slots = num_slots
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

                dense_row_col = np.stack(
                    [
                        np.repeat(np.arange(storage_height), storage_width),
                        np.tile(np.arange(storage_width), storage_height),
                    ],
                    axis=1,
                ).astype(np.int32)

                traversable = (cell_type == CELL_OBSERVATION) & spatial_mask
                traversable_positions = [
                    p for p in range(n_slots) if traversable[p]
                ]

                physical_neighbors = _build_dense_neighbors_from_layout(
                    layout, layout["action_space"], natural_h, natural_w
                )
                nat_n_slots = natural_h * natural_w
                nat_dense_to_compact = np.full(nat_n_slots, -1, dtype=np.int32)
                for s in range(layout["graph_state_count"]):
                    r = int(layout["state_to_row_col"][s, 0])
                    c = int(layout["state_to_row_col"][s, 1])
                    nat_dense_to_compact[r * natural_w + c] = s
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

                # Build the full query table (eager reconstruction included).
                goal_obs = _select_goal_obs_for_layout(
                    node_at_position, n_slots, n_actual
                )
                if not goal_obs:
                    continue

                query_table = _build_query_table(
                    physical_neighbors=physical_neighbors,
                    node_at_position=node_at_position,
                    pred_offsets=pred_offsets,
                    pred_nodes=pred_nodes,
                    dense_row_col=dense_row_col,
                    n_slots=n_slots,
                    n_obs=n_actual,
                    goal_obs=goal_obs,
                    max_supported_route_length=max_supported_route_length,
                )

                # Build exact candidate pools (pre-filtered for simple,
                # waypoint-bin, etc.).
                pools = _build_candidate_pools(
                    query_table=query_table,
                    n_slots=n_slots,
                    n_obs=n_actual,
                    node_at_position=node_at_position,
                    selection=selection_profile,
                    admission=admission_policy,
                    layout_idx=layout_idx,
                    funnel=funnel,
                )

                # Merge per-layout pools into global catalog.
                for key, entries in pools.items():
                    if key not in global_catalog:
                        global_catalog[key] = []
                    global_catalog[key].extend(entries)

                # Store per-layout tensors for target encoding (referenced by
                # layout_idx in CatalogEntry).
                per_layout_state[layout_idx] = {
                    "cell_type": cell_type.astype(np.int32),
                    "observation_id": observation_id.astype(np.int32),
                    "spatial_mask": spatial_mask,
                    "node_at_position": node_at_position,
                    "dense_row_col": dense_row_col,
                    "physical_neighbors": physical_neighbors,
                    "pred_offsets": pred_offsets,
                    "pred_nodes": pred_nodes,
                    "natural_height": np.int32(natural_h),
                    "natural_width": np.int32(natural_w),
                    "row_offset": np.int32(row_offset),
                    "col_offset": np.int32(col_offset),
                }

            # =============================================================
            # Phase B: Pre-materialization bin-support gate
            # =============================================================
            total_instances_processed = min(
                max_instances, len(shuffled_layouts)
            )

            # Build a list of mandatory physical bins (non-zero target proportion).
            # Waypoint-count bins are a post-hoc admission filter applied in
            # Phase D; the pool only knows physical bins.
            mandatory_bins: list[tuple[int, str]] = []
            for phy_bi, phy_tgt in enumerate(
                selection_profile.physical_route_position_targets
            ):
                if phy_tgt > 0.0:
                    lo_p, hi_p = selection_profile.physical_route_position_bins[
                        phy_bi
                    ]
                    mandatory_bins.append((phy_bi, f"({lo_p},{hi_p})"))

            min_support = selection_profile.minimum_bin_support
            insufficient_bins: list[str] = []
            for phy_bi, label in mandatory_bins:
                # Sum all semantic-bin entries for this physical bin
                actual = sum(
                    len(entries)
                    for (pb, _), entries in global_catalog.items()
                    if pb == phy_bi
                )
                if actual < min_support:
                    insufficient_bins.append(
                        f"  {label}: {actual} candidates "
                        f"(minimum required: {min_support})"
                    )

            if insufficient_bins and completion_policy.mode == "strict":
                raise ValueError(
                    f"Preset '{preset_}' has insufficient candidate support "
                    f"across {total_instances_processed} layout(s).\n"
                    + "\n".join(insufficient_bins)
                    + "\nProvide more topology layouts or choose a different preset."
                )

            # =============================================================
            # Phase C: Corpus-wide stratified allocation from global catalog
            # =============================================================
            total_target = (
                target_samples_total
                if target_samples_total is not None
                else sum(len(v) for v in global_catalog.values())
            )

            # Compute per-bin target counts.
            bin_targets: dict[tuple[int, int], int] = {}
            all_joint_keys = list(global_catalog.keys())
            for phy_bi, sem_bi in all_joint_keys:
                p_tgt = selection_profile.physical_route_position_targets[
                    phy_bi
                ]
                if (
                    admission_policy.waypoint_count_bins is not None
                    and admission_policy.waypoint_count_targets is not None
                    and sem_bi < len(admission_policy.waypoint_count_targets)
                ):
                    s_tgt = admission_policy.waypoint_count_targets[sem_bi]
                else:
                    s_tgt = 1.0
                joint_frac = p_tgt * s_tgt
                tgt = max(0, round(joint_frac * total_target))
                bin_targets[(phy_bi, sem_bi)] = tgt

            # Allocate without replacement.
            selected_entries: list[CatalogEntry] = []
            split_rng = np.random.default_rng(
                int(seed)
                + (0 if split == "train" else 1 if split == "val" else 2)
            )

            for key in all_joint_keys:
                entries = list(global_catalog[key])
                split_rng.shuffle(entries)
                n_take = min(bin_targets.get(key, 0), len(entries))
                selected_entries.extend(entries[:n_take])

            # If targets undershoot available pool, top up from the largest
            # surplus bin (distribution-mode fallback).
            if len(selected_entries) < total_target:
                # Collect surplus candidates from all bins (sorted by pool size).
                surplus: list[CatalogEntry] = []
                for key in all_joint_keys:
                    entries = list(global_catalog[key])
                    taken = bin_targets.get(key, 0)
                    surplus.extend(entries[taken:])
                split_rng.shuffle(surplus)
                remaining = total_target - len(selected_entries)
                selected_entries.extend(surplus[:remaining])

            split_rng.shuffle(selected_entries)

            # =============================================================
            # Phase D: Traverse optimal subgraph and encode targets
            # =============================================================
            samples: list[dict[str, np.ndarray]] = []
            query_count = 0

            # Pre-allocate kernel workspace for per-goal BFS.
            n_states = num_slots * n_actual
            _kernel_distance = np.full(
                n_states, np.iinfo(np.int32).max, dtype=np.int32
            )
            _kernel_policy_kind = np.zeros(n_states, dtype=np.int8)
            _kernel_policy_next = np.full(n_states, -1, dtype=np.int32)
            _kernel_opt_count = np.zeros(n_states, dtype=np.uint8)
            _kernel_deque_buf = np.zeros(2 * n_states, dtype=np.int32)

            # Pre-allocate optimal-subgraph traversal workspace (reused
            # across queries for the same goal).
            _traversal_workspace: dict | None = None

            for entry in selected_entries:
                state = per_layout_state[entry.layout_idx]
                start_pos = entry.start_pos
                goal_idx = entry.goal_obs
                start_obs = int(state["node_at_position"][start_pos])

                search_metrics[split]["searches"] += 1
                search_metrics[split]["starts_examined"] += 1

                # Re-run per-goal BFS for this candidate.
                goal_positions = np.where(
                    state["node_at_position"] == goal_idx
                )[0]
                if goal_positions.shape[0] == 0:
                    continue

                _kernel_distance[:] = np.iinfo(np.int32).max
                _kernel_policy_kind[:] = 0
                _kernel_policy_next[:] = -1
                _kernel_opt_count[:] = 0

                ws = dict(
                    distance=_kernel_distance,
                    policy_kind=_kernel_policy_kind,
                    policy_next=_kernel_policy_next,
                    opt_count=_kernel_opt_count,
                    deque_buf=_kernel_deque_buf,
                )
                gdt = compute_goal_distance_table(
                    physical_neighbors=state["physical_neighbors"],
                    node_at_position=state["node_at_position"],
                    pred_offsets=state["pred_offsets"],
                    pred_nodes=state["pred_nodes"],
                    goal_occurrences=goal_positions,
                    goal_node_idx=goal_idx,
                    n_slots=num_slots,
                    n_obs=n_actual,
                    _workspace=ws,
                )

                if gdt["deque_overflow"]:
                    continue

                # Derive optimal transition masks from distance table.
                phys_opt, accept_opt = derive_optimal_transition_masks(
                    distance=gdt["distance"],
                    physical_neighbors=state["physical_neighbors"],
                    node_at_position=state["node_at_position"],
                    succ_mask=pub_succ_mask,
                    succ_indices=pub_succ_indices,
                    n_slots=num_slots,
                    n_obs=n_actual,
                )

                # Traverse the optimal subgraph — no path reconstruction.
                try:
                    sup = traverse_optimal_subgraph(
                        start_pos=start_pos,
                        start_obs=start_obs,
                        distance=gdt["distance"],
                        physical_optimal_mask=phys_opt,
                        accept_optimal_mask=accept_opt,
                        physical_neighbors=state["physical_neighbors"],
                        node_at_position=state["node_at_position"],
                        n_slots=num_slots,
                        n_obs=n_actual,
                        _workspace=_traversal_workspace,
                    )
                except ValueError:
                    continue

                # Compute waypoint count from waypoint_support
                waypoint_count = int(sup.waypoint_support.sum())

                # Waypoint-count admission filter
                if (
                    preset_.admission is not None
                    and preset_.admission.waypoint_count_bins is not None
                ):
                    sem_bi = preset_.admission.semantic_bucket_index(
                        waypoint_count
                    )
                    if sem_bi is None:
                        funnel.inc_examined(entry.physical_bin)
                        continue  # out of waypoint bin — skip
                else:
                    sem_bi = 0

                if waypoint_count < 2:
                    funnel.rejected_trivial_waypoint += 1
                    continue

                funnel.inc_reconstructed(entry.physical_bin, sem_bi)

                # --- Encode targets ---
                goal_mask = state["node_at_position"] == goal_idx
                target_trajectory = encode_trajectory_support(
                    sup.trajectory_support,
                    sup.trajectory_forward_depth,
                    field_decay_spatial,
                )
                target_waypoint = encode_waypoint_support(
                    sup.waypoint_support,
                    sup.waypoint_semantic_depth,
                    field_decay_semantic,
                )

                # Multi-label auxiliary masks
                target_optimal_directions = sup.target_optimal_directions.copy()
                target_optimal_next_observations = (
                    sup.target_optimal_next_observations.copy()
                )

                start_flag = np.zeros(num_slots, dtype=bool)
                start_flag[start_pos] = True

                sample_data = {
                    "cell_type": state["cell_type"],
                    "observation_id": state["observation_id"],
                    "start_flag": start_flag,
                    "goal_flag": goal_mask,
                    "spatial_mask": state["spatial_mask"],
                    "natural_height": state["natural_height"],
                    "natural_width": state["natural_width"],
                    "row_offset": state["row_offset"],
                    "col_offset": state["col_offset"],
                    "target_trajectory": target_trajectory,
                    "target_waypoint": target_waypoint,
                    "trajectory_support": sup.trajectory_support,
                    "trajectory_forward_depth": sup.trajectory_forward_depth,
                    "trajectory_remaining_cost": sup.trajectory_remaining_cost,
                    "waypoint_support": sup.waypoint_support,
                    "waypoint_semantic_depth": sup.waypoint_semantic_depth,
                    "target_optimal_directions": target_optimal_directions,
                    "target_optimal_next_observations": target_optimal_next_observations,
                    "total_physical_cost": np.int16(sup.total_physical_cost),
                }

                # Build-time support/algebra validation — catches encoding
                # defects before writing the corpus.
                support_issues = validate_support_channels(
                    sample_data,
                    S=num_slots,
                    gamma_space=field_decay_spatial,
                    gamma_semantic=field_decay_semantic,
                )
                support_errors = [
                    i for i in support_issues if i.severity == "ERROR"
                ]
                if support_errors:
                    funnel.rejected_target_validation += 1
                    continue

                # Simplified target validation — no single-path assumptions.
                target_issues = validate_generated_sample(
                    sample_data,
                    oracle_result=None,
                    n_obs=n_actual,
                    topo_vocab_size=n_actual,
                    S=num_slots,
                    gamma_space=field_decay_spatial,
                    gamma_semantic=field_decay_semantic,
                )
                target_errors = [
                    i for i in target_issues if i.severity == "ERROR"
                ]
                if target_errors:
                    funnel.rejected_target_validation += 1
                    continue

                samples.append(sample_data)
                search_metrics[split]["accepted"] += 1
                funnel.inc_accepted(entry.physical_bin, sem_bi)
                query_count += 1

            if query_count == 0:
                split_rejections["no_valid_query"] = (
                    split_rejections.get("no_valid_query", 0) + 1
                )

            stage_params[f"funnel_{split}"] = funnel.serialize()

            # Record realized distribution
            realized: dict[str, float] = {}
            accepted_dict = funnel.accepted
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

        # ── Compute admission deficits per split ──────────────────────────
        admission_deficits: dict[str, dict] = {}
        for split in _SPLITS:
            split_realized = stage_params.get(f"realized_{split}", {})
            deficits: dict[str, dict] = {}
            if preset is not None and preset.admission is not None:
                wpt = preset.admission.waypoint_count_targets
                if wpt is not None:
                    for si, tgt_frac in enumerate(wpt):
                        joint_key_str = f"sem_{si}"
                        achieved = 0.0
                        for rk, rv in split_realized.items():
                            # rk is "(phys_bi,sem_bi)"
                            parts = rk.strip("()").split(",")
                            if len(parts) == 2 and int(parts[1]) == si:
                                achieved += rv
                        deficit = max(0.0, tgt_frac - achieved)
                        deficits[joint_key_str] = {
                            "target": tgt_frac,
                            "achieved": achieved,
                            "deficit": deficit,
                        }
            admission_deficits[split] = deficits

        # ── Determine build status ────────────────────────────────────────
        build_status = _determine_build_status(
            selection=(
                preset.selection
                if preset is not None
                else QuerySelectionProfile(
                    physical_route_position_bins=[(2, 150)],
                    physical_route_position_targets=[1.0],
                )
            ),
            completion=(
                preset.completion if preset is not None else CompletionPolicy()
            ),
            total_accepted=total_accepted,
            target_samples_total=target_samples_total,
            admission_deficits=admission_deficits,
        )

        # ── Build capability report ───────────────────────────────────────
        capability_report: dict[str, Any] = {
            "generation_funnel": {
                split: stage_params.get(f"funnel_{split}", {})
                for split in _SPLITS
            },
            "realized_distribution": {
                split: stage_params.get(f"realized_{split}", {})
                for split in _SPLITS
            },
            "admission_deficits": admission_deficits,
            "build_status": build_status,
            "rejection_counts": {
                split: {
                    r: stage_params.get(f"rejected_{split}_{r}", 0)
                    for r in (
                        "ambiguous",
                        "route_too_long",
                        "trivial_waypoint",
                        "invalid_next_dir",
                        "target_validation",
                    )
                }
                for split in _SPLITS
            },
        }

        # Digest excludes itself to avoid circular dependency
        digest_source = json.dumps(
            {
                k: v
                for k, v in capability_report.items()
                if k != "capability_calibration_digest"
            },
            sort_keys=True,
        )
        capability_report["capability_calibration_digest"] = hashlib.sha256(
            digest_source.encode("utf-8")
        ).hexdigest()

        # Write capability report to corpus root (inside staging tmp)
        report_path = tmp / "capability_report.json"
        with open(report_path, "w") as f:
            json.dump(capability_report, f, indent=2, sort_keys=True)

        # ── Populate stage_params and write manifest ──────────────────────
        stage_params["build_status"] = build_status
        stage_params["capability_calibration_digest"] = capability_report[
            "capability_calibration_digest"
        ]
        stage_params["corpus_schema_version"] = 1
        stage_params["target_semantics"] = "optimal_subgraph_support"
        stage_params["target_schema_version"] = 1
        stage_params["depth_sentinel"] = -1

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
            builder="ehp_sn.tasks.routebind.builder.build_routebind_task_corpus",
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
