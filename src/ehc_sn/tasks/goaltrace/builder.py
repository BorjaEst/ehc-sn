"""Goaltrace task corpus materialization.

Owns goaltrace-specific channel schema, oracle path selection, target field
encoding, validation, and the builder that produces the goaltrace task corpus
over a parent dagflow shared substrate.

Task corpus channels extend the dagflow structural channels with weight-sample
channels and oracle-derived target fields.

Path written: ``data/processed/goaltrace/<corpus>/v<version>/``
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

from ehc_sn.data.lifecycle import (
    extract_version,
    staging_root,
    validate_version_root,
    write_index_at_root,
    write_split,
)
from ehc_sn.data.manifest import read_manifest, write_manifest
from ehc_sn.data.substrate.reader import (
    iter_substrate_entries_and_samples,
)
from ehc_sn.utils.graph import shortest_path as bfs_shortest_path

# =============================================================================
TASK_FAMILY: Final[str] = "goaltrace"
"""Task namespace for the goaltrace task corpus."""

# =============================================================================
# Channel group constants — explicit information-boundary declarations
# =============================================================================

GOALTRACE_MODEL_INPUT_CHANNELS: Final[tuple[str, ...]] = (
    "observation_id",
    "weight",
    "current_flag",
    "goal_flag",
    "node_mask",
)
"""Channels passed to the HRM model adapter (model input only).

Topology and supervision targets are intentionally excluded — they live
in other channel groups so the adapter never receives them.
"""

GOALTRACE_TARGET_CHANNELS: Final[tuple[str, ...]] = ("target_field",)
"""Channels used as supervision targets (objective / binding contract)."""

GOALTRACE_EVALUATION_META_CHANNELS: Final[tuple[str, ...]] = (
    "successor_indices",
    "successor_mask",
)
"""Channels captured as evaluation metadata for figures and diagnostics.

These are persisted in the processed corpus and travel with the sample
through collation, but are projected out of the model input surface by
the adapter.  Intentionally omitted from ``GOALTRACE_MODEL_INPUT_CHANNELS``.
"""

GOALTRACE_CORPUS_CHANNELS: Final[tuple[str, ...]] = (
    *GOALTRACE_MODEL_INPUT_CHANNELS,
    *GOALTRACE_TARGET_CHANNELS,
    *GOALTRACE_EVALUATION_META_CHANNELS,
)
"""All channels persisted in the goaltrace task corpus (union of all groups)."""

# Legacy alias — do not use in new code.
GOALTRACE_TASK_CHANNELS: Final[list[str]] = list(GOALTRACE_CORPUS_CHANNELS)

_REQUIRED_PARENT_CHANNELS: tuple[str, ...] = (
    "node_obs_id",
    "successor_indices",
    "successor_mask",
    "node_mask",
    "obs_id_to_rank",
    "rank_to_obs_id",
)
"""Channels the goaltrace task builder requires in the parent dagflow substrate."""

GOALTRACE_TASK_CHANNEL_DTYPES: dict[str, np.dtype] = {
    "observation_id": np.dtype(np.int32),
    "weight": np.dtype(np.float32),
    "current_flag": np.dtype(bool),
    "goal_flag": np.dtype(bool),
    "node_mask": np.dtype(bool),
    "target_field": np.dtype(np.float32),
    "successor_indices": np.dtype(np.int32),
    "successor_mask": np.dtype(bool),
}
"""Expected numpy dtypes for goaltrace task corpus channels."""

_ORACLE_SEMANTICS_VALID: Final[tuple[str, ...]] = (
    "reliability",
    "linear_cost",
    "preference",
)

_SPLITS: tuple[str, ...] = ("train", "val", "test")

_EPS: float = 1e-8
"""Small epsilon for log-transform stability in cost computation."""

_COST_TIE_EPS: float = 1e-7
"""Maximum absolute cost difference for two paths to be considered tied."""


# =============================================================================
# Oracle: weight-aware optimal path selection
# =============================================================================


def _edge_cost(
    w: float,
    semantics: str,
    step_penalty: float = 0.0,
) -> float:
    """Convert a relational weight to an edge cost under the given semantics.

    Args:
        w: Relational weight in [0, 1].
        semantics: One of ``"reliability"``, ``"linear_cost"``, ``"preference"``.
        step_penalty: Per-step penalty lambda (only used for ``"preference"``).

    Returns:
        Edge cost as a non-negative float.

    Raises:
        ValueError: If *semantics* is unknown.
    """
    match semantics:
        case "reliability":
            return -np.log(max(w, _EPS))
        case "linear_cost":
            return 1.0 - w
        case "preference":
            return -w + step_penalty
        case _:
            raise ValueError(
                f"Unknown oracle semantics: {semantics!r}. "
                f"Valid options: {_ORACLE_SEMANTICS_VALID}."
            )


def _select_optimal_path(
    adjacency: list[list[int]],
    weights: np.ndarray,
    current_idx: int,
    goal_idx: int,
    semantics: str,
    step_penalty: float = 0.0,
) -> list[int]:
    """Select the optimal path from *current_idx* to *goal_idx* under the
    declared oracle semantics.

    Uses Dijkstra over the DAG with continuous edge costs.  When multiple
    paths have costs within ``_COST_TIE_EPS``, the path with fewer edges
    (shorter hop distance) is preferred.

    Args:
        adjacency: Adjacency list for actual (non-padded) nodes.
        weights: Per-edge weight array indexed by ``(u, v)`` pairs.
            Shape ``(n_actual, n_actual)``, with ``np.inf`` for non-edges.
        current_idx: Start node index (0-based original).
        goal_idx: Goal node index (0-based original).

    Returns:
        List of node indices (original indices) representing the optimal
        path from *current_idx* to *goal_idx*, inclusive.

    Raises:
        RuntimeError: If no path exists from current to goal.
    """
    n = len(adjacency)
    # Compute cost matrix once
    costs = np.full((n, n), np.inf, dtype=np.float64)
    for u in range(n):
        for v in adjacency[u]:
            costs[u, v] = _edge_cost(weights[u, v], semantics, step_penalty)

    # Dijkstra
    dist = np.full(n, np.inf, dtype=np.float64)
    prev = np.full(n, -1, dtype=np.int32)
    hops = np.full(n, np.inf, dtype=np.float64)
    visited = np.zeros(n, dtype=bool)

    dist[current_idx] = 0.0
    hops[current_idx] = 0

    for _ in range(n):
        # Find unvisited node with smallest distance
        u = int(np.argmin(np.where(visited, np.inf, dist)))
        if u == goal_idx or np.isinf(dist[u]):
            break
        visited[u] = True

        for v in adjacency[u]:
            alt = dist[u] + costs[u, v]
            alt_hops = hops[u] + 1
            if alt < dist[v] - _COST_TIE_EPS:
                dist[v] = alt
                prev[v] = u
                hops[v] = alt_hops
            elif abs(alt - dist[v]) <= _COST_TIE_EPS and alt_hops < hops[v]:
                # Tie: prefer fewer hops
                prev[v] = u
                hops[v] = alt_hops

    if np.isinf(dist[goal_idx]):
        raise RuntimeError(
            f"No path from current ({current_idx}) to goal ({goal_idx}) "
            f"after weight assignment."
        )

    # Reconstruct path
    path: list[int] = []
    v = goal_idx
    while v != -1:
        path.append(v)
        v = int(prev[v])
    path.reverse()
    return path


def _encode_target_field(
    path: list[int],
    n_nodes: int,
    decay: float,
    current_idx: int,
) -> np.ndarray:
    """Encode the target firing field along the optimal path.

    The field assigns ``1.0`` to the current location and ``decay ** distance``
    to each subsequent node on the path.  Nodes not on the path receive ``0.0``.

    Args:
        path: Optimal path node indices (original indices), inclusive of
            current and goal.
        n_nodes: Total number of actual nodes (N).
        decay: Decay factor gamma in ``(0, 1)``.
        current_idx: Current location node index.

    Returns:
        Target field array of shape ``(n_nodes,)`` float32 with values in ``[0, 1]``.
    """
    field = np.zeros(n_nodes, dtype=np.float32)

    # Find start of path (distance = 0 at current_idx)
    try:
        start_pos = path.index(current_idx)
    except ValueError:
        raise RuntimeError(
            f"Current location {current_idx} not found in optimal path {path}."
        )

    for dist_offset, node in enumerate(path[start_pos:]):
        field[node] = float(decay**dist_offset)

    return field


def _sample_weight_vector(
    n_nodes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a continuous weight matrix for the DAG.

    Weights are sampled uniformly in ``(0, 1]`` for existing edges and
    ``np.inf`` for non-edges.  No zero weights to avoid log(0) issues
    with reliability semantics.

    Args:
        n_nodes: Number of actual nodes.
        rng: Seeded random generator.

    Returns:
        Weight matrix of shape ``(n_nodes, n_nodes)`` float32.
    """
    weights = rng.uniform(1e-6, 1.0, size=(n_nodes, n_nodes)).astype(np.float32)
    return weights


def _apply_adjacency_mask(
    weights: np.ndarray,
    adjacency: list[list[int]],
) -> np.ndarray:
    """Mask weights to only keep existing edges, setting non-edges to ``np.inf``.

    Args:
        weights: Full weight matrix of shape ``(n_nodes, n_nodes)``.
        adjacency: Adjacency list for actual nodes.

    Returns:
        Masked weight matrix with ``np.inf`` for non-edges.
    """
    n = len(adjacency)
    masked = np.full((n, n), np.inf, dtype=np.float64)
    for u in range(n):
        for v in adjacency[u]:
            masked[u, v] = float(weights[u, v])
    return masked


def _assign_goaltrace_sample(
    substrate_sample: dict[str, np.ndarray],
    n_nodes: int,
    adjacency: list[list[int]],
    rng: np.random.Generator,
    semantics: str,
    field_decay: float,
    step_penalty: float,
    num_observations: int,
) -> dict[str, np.ndarray]:
    """Generate one goaltrace sample from a DAG substrate sample.

    Samples a (current, goal) pair, relational weights, computes the oracle
    path, and encodes the target field.

    The dagflow substrate stores nodes in **rank order** (row = rank).
    ``obs_id_to_rank`` provides the bijection from public observation ID
    to rank, used for weight extraction targeting the current node's
    outgoing edges.

    Args:
        substrate_sample: Dagflow substrate sample with structural channels.
            Nodes are in rank order.
        n_nodes: Number of actual nodes.
        adjacency: Adjacency list for actual nodes (rank space).
        rng: Seeded random generator.
        semantics: Oracle semantics string.
        field_decay: Field decay factor gamma.
        step_penalty: Per-step penalty for ``"preference"`` semantics.
        num_observations: Padded observation count N.

    Returns:
        Sample dict with goaltrace task channels.
    """
    actual_indices = list(range(n_nodes))  # row = rank

    # Sample (current, goal) pair — must have a valid path between them
    while True:
        current_rank = int(rng.integers(0, n_nodes))
        goal_rank = int(rng.integers(0, n_nodes))
        if current_rank == goal_rank:
            continue
        sp = bfs_shortest_path(adjacency, current_rank, goal_rank)
        if sp:
            break

    # Sample weights
    full_weights = _sample_weight_vector(n_nodes, rng)
    masked = _apply_adjacency_mask(full_weights, adjacency)

    # Select optimal path (in rank space)
    path_orig = _select_optimal_path(
        adjacency, masked, current_rank, goal_rank, semantics, step_penalty
    )

    # Encode target field (in rank space)
    target_field_orig = _encode_target_field(
        path_orig, n_nodes, field_decay, current_rank
    )

    # Build padded output arrays in rank order (row = rank)
    current_flag = np.zeros(num_observations, dtype=bool)
    goal_flag = np.zeros(num_observations, dtype=bool)
    node_mask_out = np.zeros(num_observations, dtype=bool)
    weight = np.zeros(num_observations, dtype=np.float32)
    padded_target = np.zeros(num_observations, dtype=np.float32)
    o2r = substrate_sample["obs_id_to_rank"]

    for r in range(n_nodes):
        current_flag[r] = r == current_rank
        goal_flag[r] = r == goal_rank
        node_mask_out[r] = True
        # Only expose edge weights to the model; non-edges → 0.0.
        if r in adjacency[current_rank]:
            weight[r] = full_weights[current_rank, r]
        else:
            weight[r] = 0.0
        padded_target[r] = float(target_field_orig[r])

    # Pad observation_id: real nodes get ids from rank_to_obs_id, padding gets n_nodes.
    PAD_OBS_ID = n_nodes
    padded_obs_ids = np.full(num_observations, PAD_OBS_ID, dtype=np.int32)
    r2o = substrate_sample["rank_to_obs_id"]
    for r in range(n_nodes):
        padded_obs_ids[r] = int(r2o[r])

    # Forward topology from parent substrate, padded to num_observations
    K = substrate_sample["successor_indices"].shape[-1]
    raw_succ_idx = substrate_sample["successor_indices"]  # (n_actual, K)
    raw_succ_mask = substrate_sample["successor_mask"]  # (n_actual, K)
    if raw_succ_idx.ndim == 2:
        padded_succ_idx = np.zeros((num_observations, K), dtype=np.int32)
        padded_succ_mask = np.zeros((num_observations, K), dtype=bool)
        n_actual = raw_succ_idx.shape[0]
        padded_succ_idx[:n_actual, :] = raw_succ_idx
        padded_succ_mask[:n_actual, :] = raw_succ_mask
    else:
        padded_succ_idx = raw_succ_idx
        padded_succ_mask = raw_succ_mask

    return {
        "observation_id": padded_obs_ids,
        "weight": weight,
        "current_flag": current_flag,
        "goal_flag": goal_flag,
        "node_mask": node_mask_out,
        "target_field": padded_target,
        "successor_indices": padded_succ_idx,
        "successor_mask": padded_succ_mask,
    }


# =============================================================================
# Validation
# =============================================================================


def validate_goaltrace_sample(data: dict[str, np.ndarray]) -> None:
    """Validate one goaltrace task corpus sample against the task channel schema.

    Validates all channels present in *data* against known dtypes and shape
    constraints.  Extra channels (not in ``GOALTRACE_TASK_CHANNEL_DTYPES``)
    that are present in *data* are validated for shape only.

    Channels absent from *data* are not required — this allows the validator
    to accept both v1 (legacy) and v2 (with topology) samples.

    Args:
        data: Dict of channel arrays for one sample.

    Raises:
        ValueError: On any contract violation.
    """

    for name, arr in data.items():
        if (
            name in GOALTRACE_TASK_CHANNEL_DTYPES
            and arr.dtype != GOALTRACE_TASK_CHANNEL_DTYPES[name]
        ):
            raise ValueError(
                f"Channel '{name}' has dtype {arr.dtype}, expected "
                f"{GOALTRACE_TASK_CHANNEL_DTYPES[name]}."
            )

    n_nodes = data["observation_id"].shape[0]

    if data["weight"].shape[0] != n_nodes:
        raise ValueError(f"weight does not match N ({n_nodes}).")
    if data["current_flag"].shape[0] != n_nodes:
        raise ValueError(f"current_flag does not match N ({n_nodes}).")
    if data["goal_flag"].shape[0] != n_nodes:
        raise ValueError(f"goal_flag does not match N ({n_nodes}).")
    if data["node_mask"].shape[0] != n_nodes:
        raise ValueError(f"node_mask does not match N ({n_nodes}).")
    if "target_field" in data:
        if data["target_field"].shape[0] != n_nodes:
            raise ValueError(f"target_field does not match N ({n_nodes}).")
    if "successor_indices" in data:
        succ = data["successor_indices"]
        if succ.ndim != 2 or succ.shape[0] != n_nodes:
            raise ValueError(
                f"successor_indices must be (N, K) with N={n_nodes}, "
                f"got shape {succ.shape}."
            )
    if "successor_mask" in data:
        _succ_shape = (
            data["successor_indices"].shape
            if "successor_indices" in data
            else None
        )
        if (
            _succ_shape is not None
            and data["successor_mask"].shape != _succ_shape
        ):
            raise ValueError(
                f"successor_mask shape {data['successor_mask'].shape} must match "
                f"successor_indices shape {_succ_shape}."
            )

    # Validate field values in [0, 1]
    field = data["target_field"]
    if np.any(field < 0.0) or np.any(field > 1.0):
        raise ValueError(
            f"target_field values outside [0, 1]: "
            f"min={field.min()}, max={field.max()}."
        )

    # Validate exactly one current and one goal flag
    n_current = int(data["current_flag"].sum())
    n_goal = int(data["goal_flag"].sum())
    if n_current != 1:
        raise ValueError(f"Expected exactly one current_flag, got {n_current}.")
    if n_goal != 1:
        raise ValueError(f"Expected exactly one goal_flag, got {n_goal}.")

    # Validate current and goal are different
    current_idx = int(data["current_flag"].argmax())
    goal_idx = int(data["goal_flag"].argmax())
    if current_idx == goal_idx:
        raise ValueError("current_flag and goal_flag are identical.")

    # Validate current location is anchored at 1.0
    if abs(float(data["target_field"][current_idx]) - 1.0) > 1e-6:
        raise ValueError(
            f"Current location ({current_idx}) target_field "
            f"= {float(data['target_field'][current_idx])}, expected 1.0."
        )

    # Validate weights in [0, 1] for masked nodes
    masked_weights = data["weight"][data["node_mask"]]
    if np.any(masked_weights < 0.0) or np.any(masked_weights > 1.0):
        raise ValueError(
            f"weight values outside [0, 1] for masked nodes: "
            f"min={masked_weights.min()}, max={masked_weights.max()}."
        )

    # Validate observation_id: sentinel for padding, valid for real nodes
    obs = data["observation_id"]
    n_actual = int(data["node_mask"].sum())
    PAD_OBS_ID = n_actual
    real_obs = obs[data["node_mask"]]
    pad_obs = obs[~data["node_mask"]]
    if np.any(real_obs < 0) or np.any(real_obs >= n_actual):
        raise ValueError(
            f"Real-node observation IDs must be in [0, {n_actual}). "
            f"Got min={real_obs.min()}, max={real_obs.max()}."
        )
    if len(pad_obs) > 0 and np.any(pad_obs != PAD_OBS_ID):
        raise ValueError(
            f"Padding observation IDs must be {PAD_OBS_ID} (n_actual). "
            f"Got values: {np.unique(pad_obs).tolist()}."
        )


# =============================================================================
def validate_goaltrace_root(root: Path) -> dict:
    """Validate a goaltrace task corpus root against task-owned semantics.

    Validates against the channels declared in the manifest (backward
    compatible with corpora that predate ``successor_indices`` /
    ``successor_mask``).

    Args:
        root: Resolved versioned goaltrace task corpus root.

    Returns:
        Parsed manifest dict.

    Raises:
        ValueError: On any contract violation.
        FileNotFoundError: When a required file is absent.
    """
    manifest = validate_version_root(root)
    if manifest.get("dataset_class") != "task_corpus":
        raise ValueError("Root is not a task_corpus.")
    if manifest.get("task") != TASK_FAMILY:
        raise ValueError(
            f"Root task is {manifest.get('task')!r}, expected {TASK_FAMILY!r}."
        )

    n_nodes = manifest.get("n_observations")
    if n_nodes is None:
        raise ValueError("Manifest missing n_observations.")

    # Use the manifest's channel list so old corpora (v1 without topology)
    # still validate correctly.  New corpora include all corpus channels.
    declared_channels: list[str] = manifest.get(
        "channels", list(GOALTRACE_CORPUS_CHANNELS)
    )

    for split, n in manifest["n_samples"].items():
        split_dir = root / split
        arrays: dict[str, np.ndarray] = {}
        for ch in declared_channels:
            ch_file = split_dir / f"{ch}.npy"
            if not ch_file.exists():
                raise FileNotFoundError(
                    f"Missing task channel '{ch}' in {split_dir}."
                )
            arrays[ch] = np.load(ch_file, mmap_mode="r")
            if arrays[ch].shape[0] != n:
                raise ValueError(
                    f"Task channel '{ch}' in split '{split}' "
                    f"has {arrays[ch].shape[0]} samples, "
                    f"manifest declares {n}."
                )
            # The first per-sample dimension must be n_nodes, regardless
            # of whether the channel is 1D (N,) or 2D (N, K).
            if arrays[ch].shape[1] != n_nodes:
                raise ValueError(
                    f"Channel '{ch}' first sample dim is "
                    f"{arrays[ch].shape[1]}, expected {n_nodes}."
                )
                raise ValueError(
                    f"Channel '{ch}' has N={arrays[ch].shape[1]}, "
                    f"expected {n_nodes}."
                )

        for i in range(n):
            sample = {ch: arrays[ch][i] for ch in declared_channels}
            validate_goaltrace_sample(sample)

    return manifest


# =============================================================================
# Static-weight helpers
# =============================================================================


def _build_static_corpus_samples(
    adjacency: list[list[int]],
    base_weights: np.ndarray,
    n_nodes: int,
    n_observations: int,
    semantics: str,
    field_decay: float,
    step_penalty: float,
    min_margin: float,
    pad_obs_id: int,
    obs_ids: np.ndarray,
    rng: np.random.Generator,
) -> list[dict[str, np.ndarray]]:
    """Enumerate all valid (current, goal) pairs, run the oracle, enforce
    uniqueness margin, and return a list of sample dicts.

    Args:
        adjacency: DAG adjacency in original (hidden) index space.
        base_weights: ``(N, N)`` float64 relational topology matrix.
        n_nodes: Number of actual nodes.
        n_observations: Padded slot count.
        semantics: Oracle semantics string.
        field_decay: Decay factor gamma.
        step_penalty: Penalty lambda for ``"preference"`` semantics.
        min_margin: Minimum optimality margin (second-best minus best cost).
            0 means no filtering.
        pad_obs_id: Observation ID sentinel for padding slots.
        obs_ids: (N,) int32 observation IDs in original (hidden) index order.
        rng: Seeded RNG for sample ordering and split shuffling.

    Returns:
        List of sample dicts, one per valid (current, goal) pair that passes
        the margin check.  Each dict has the 8 standard goaltrace channels.
    """
    samples: list[dict[str, np.ndarray]] = []

    # Precompute successor mask for padding
    K = max(len(a) for a in adjacency) if adjacency else 0

    rejected_margin = 0

    for current in range(n_nodes):
        for goal in range(current + 1, n_nodes):
            if not bfs_shortest_path(adjacency, current, goal):
                continue  # safety — should not happen with Hamiltonian backbone

            # Run oracle
            cost_matrix = _apply_adjacency_mask(base_weights, adjacency)
            path = _select_optimal_path(
                adjacency, cost_matrix, current, goal, semantics, step_penalty
            )
            target = _encode_target_field(path, n_nodes, field_decay, current)

            # Optionally check optimality margin
            if min_margin > 0:
                # Approximate second-best path cost by finding the second-best
                # distinct route.  We compute the cost of the optimal path,
                # then find the next path that differs by at least one edge.
                opt_cost = sum(
                    _edge_cost(
                        float(base_weights[path[i], path[i + 1]]),
                        semantics,
                        step_penalty,
                    )
                    for i in range(len(path) - 1)
                )
                # Simple heuristic: find the cost of the best alternative first
                # step.  This is approximate but catches obvious ties.
                alt_costs: list[float] = []
                for first_succ in adjacency[current]:
                    if first_succ == path[1]:
                        continue
                    if not bfs_shortest_path(adjacency, first_succ, goal):
                        continue
                    alt_cost = _edge_cost(
                        float(base_weights[current, first_succ]),
                        semantics,
                        step_penalty,
                    )
                    alt_path = _select_optimal_path(
                        adjacency,
                        cost_matrix,
                        first_succ,
                        goal,
                        semantics,
                        step_penalty,
                    )
                    alt_cost += sum(
                        _edge_cost(
                            float(base_weights[alt_path[i], alt_path[i + 1]]),
                            semantics,
                            step_penalty,
                        )
                        for i in range(len(alt_path) - 1)
                    )
                    alt_costs.append(alt_cost)
                if alt_costs:
                    best_alt = min(alt_costs)
                    if opt_cost + min_margin > best_alt:
                        rejected_margin += 1
                        continue

            # Convert from original to padded public space
            weight_row = np.zeros(n_observations, dtype=np.float32)
            public_target = np.zeros(n_observations, dtype=np.float32)
            cur_flag = np.zeros(n_observations, dtype=bool)
            goal_flag = np.zeros(n_observations, dtype=bool)
            node_mask_out = np.zeros(n_observations, dtype=bool)

            # Original-index space: we store observation IDs in their
            # original (hidden rank) order, since the dagflow substrate
            # provides the permutation.  We use original indices directly.
            for j in range(n_nodes):
                node_mask_out[j] = True
                if base_weights[current, j] > 0:
                    weight_row[j] = float(base_weights[current, j])
                cur_flag[j] = j == current
                goal_flag[j] = j == goal
                public_target[j] = float(target[j])
            # Self-identity signal: model should know current position has
            # full support, matching the target field's f[current]=1.0.
            weight_row[current] = 1.0

            # Pad observation IDs
            padded_obs = np.full(n_observations, pad_obs_id, dtype=np.int32)
            padded_obs[:n_nodes] = obs_ids[:n_nodes]

            # Successor padding (K from adjacency)
            succ_idx_pad = np.zeros((n_observations, K), dtype=np.int32)
            succ_mask_pad = np.zeros((n_observations, K), dtype=bool)
            for j in range(n_nodes):
                for k_idx, s in enumerate(adjacency[j]):
                    if k_idx < K:
                        succ_idx_pad[j, k_idx] = s
                        succ_mask_pad[j, k_idx] = True

            samples.append(
                {
                    "observation_id": padded_obs,
                    "weight": weight_row,
                    "current_flag": cur_flag,
                    "goal_flag": goal_flag,
                    "node_mask": node_mask_out,
                    "target_field": public_target,
                    "successor_indices": succ_idx_pad,
                    "successor_mask": succ_mask_pad,
                }
            )

    if rejected_margin > 0:
        print(
            f"  Builder: rejected {rejected_margin} pairs below "
            f"optimality margin ({min_margin})."
        )

    return samples


def _split_pairs_by_identity(
    samples: list[dict[str, np.ndarray]],
    n_train: int,
    n_val: int,
    n_test: int,
    rng: np.random.Generator,
) -> dict[str, list[dict[str, np.ndarray]]]:
    """Assign samples to splits by (current, goal) identity.

    Shuffles the sample list, then assigns the first *n_train* to training,
    next *n_val* to validation, remaining to test (capped at *n_test*).
    Raises if not enough samples to satisfy the requested split sizes.

    Returns dict mapping split name to list of sample dicts.
    """
    total = n_train + n_val + n_test
    if len(samples) < total:
        raise ValueError(
            f"Not enough samples ({len(samples)}) for requested split "
            f"({total}).  Increase the margin or reduce the split sizes."
        )

    indices = list(range(len(samples)))
    rng.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val : n_train + n_val + n_test]

    return {
        "train": [samples[i] for i in train_idx],
        "val": [samples[i] for i in val_idx],
        "test": [samples[i] for i in test_idx],
    }


# =============================================================================
# Builder
# =============================================================================


def build_goaltrace_task_corpus(
    version_root: Path,
    *,
    layout_root: Path,
    corpus: str = "default",
    n_observations: int = 45,
    num_graphs: int = 1,
    dagflow_graph_id: str | None = None,
    oracle_semantics: str = "reliability",
    field_decay: float = 0.8,
    preference_step_penalty: float = 0.0,
    n_train: int = 500,
    n_val: int = 250,
    n_test: int = 240,
    seed: int = 42,
    static_weights: bool = True,
    min_optimality_margin: float = 0.0,
    distance_tau: float = 8.0,
    distance_max: float | None = None,
    grid_width: int = 20,
    grid_height: int = 30,
    geometry_seed: int = 42,
) -> None:
    """Build the goaltrace task corpus at *version_root* over a dagflow layout.

    Consumes dagflow layout samples for graph topology and generates
    weight-per-edge, oracle path selection, and target field encoding.
    When ``num_graphs == 1`` (default), all samples share a single DAG.

    Two modes:

    - **Random-weight mode** (``static_weights=False``, default): each sample
      gets an independent random weight matrix.  This is the v2-compatible path.
    - **Static-weight mode** (``static_weights=True``): a single base relational
      topology matrix $W$ is derived from the DAG geometry and a hidden spatial
      substrate.  All valid (current, goal) pairs are enumerated exactly once,
      producing at most 990 deterministic samples with no contradictions.

    Args:
        version_root: Destination versioned root
            (e.g. ``data/processed/goaltrace/default/v1``).  Must not exist.
        layout_root: Path to the dagflow layout dataset root.  Must contain
            a ``manifest.json`` with all required structural channels.
        corpus: Corpus label (e.g. ``"default"``).
        n_observations: Maximum padded node count N.
        num_graphs: Number of DAGs to sample from the layout pool.  Must be
            at most the number of available layouts per split.
        dagflow_graph_id: Stable artifact ID within *layout_root* identifying
            a single graph.  When provided, overrides ``num_graphs`` and
            selects exactly one graph.  Pass ``entry.id`` from a dagflow index.
            When not provided, falls back to ``num_graphs`` (deprecated).
        oracle_semantics: Weight semantics for path selection.
            One of ``"reliability"``, ``"linear_cost"``, ``"preference"``.
        field_decay: Decay factor gamma in ``(0, 1)``.
        preference_step_penalty: Per-step penalty lambda for ``"preference"``
            semantics.
        n_train: Number of training samples (for random-weight mode) or
            number of (current, goal) pairs assigned to training (for static mode).
        n_val: Number of validation samples/pairs.
        n_test: Number of test samples/pairs.
        seed: Deterministic base seed for reproducibility.
        static_weights: When True, use a single static relational weight matrix
            derived from geometry plus DAG adjacency.  When False (default),
            use per-sample random weights (v2-compatible).
        min_optimality_margin: Minimum required cost difference between the
            optimal path and the second-best path.  Pairs below this margin
            are rejected.  Only used when ``static_weights=True``.
        distance_tau: Temperature / distance scale for the spatial kernel.
            Only used when ``static_weights=True``.
        distance_max: Maximum effective distance for the kernel truncation.
            When ``None`` (default), computed as ``grid_width + grid_height``
            to avoid truncation of any DAG edge.
            Only used when ``static_weights=True``.
        grid_width: Width of the hidden spatial grid in cells.
            Only used when ``static_weights=True``.
        grid_height: Height of the hidden spatial grid in cells.
            Only used when ``static_weights=True``.
        geometry_seed: Seed for anchor placement on the grid.
            Only used when ``static_weights=True``.

    Raises:
        FileExistsError: When *version_root* already exists (immutable root).
        FileNotFoundError: When *layout_root* has no manifest.
        ValueError: When the layout is not a dagflow layout_dataset,
            the layout lacks required structural channels, or
            *oracle_semantics* is invalid.
        RuntimeError: When ``static_weights=True`` and no valid pairs survive
            the optimality margin check.
    """
    version = extract_version(version_root)
    layout_manifest = read_manifest(layout_root)

    if layout_manifest.get("family") != "dagflow":
        raise ValueError(
            f"Goaltrace task corpus requires a 'dagflow' layout "
            f"dataset, got family={layout_manifest.get('family')!r}."
        )

    if oracle_semantics not in _ORACLE_SEMANTICS_VALID:
        raise ValueError(
            f"Unknown oracle_semantics: {oracle_semantics!r}. "
            f"Valid options: {_ORACLE_SEMANTICS_VALID}."
        )

    # Validate layout dataset has required structural channels
    layout_n_max = layout_manifest.get("n_max", 0)
    if n_observations < layout_n_max:
        raise ValueError(
            f"n_observations ({n_observations}) is smaller than substrate "
            f"n_max ({layout_n_max}).  Real graph nodes would be truncated. "
            f"Increase --n-observations or regenerate the dagflow substrate "
            f"with a smaller --n-max."
        )
    if n_observations > layout_n_max:
        warnings.warn(
            f"n_observations ({n_observations}) exceeds substrate n_max "
            f"({layout_n_max}).  Extra {n_observations - layout_n_max} padding "
            f"slots will be added to each sample.",
            UserWarning,
            stacklevel=2,
        )
    layout_channels = set(layout_manifest.get("channels", []))
    missing = [
        ch for ch in _REQUIRED_PARENT_CHANNELS if ch not in layout_channels
    ]
    if missing:
        raise ValueError(
            f"Layout dataset lacks required structural channels: "
            f"{', '.join(missing)}. "
            f"Rebuild with:\n"
            f"    python scripts/data-gen/build-dagflow.py build-all"
        )

    split_counts = {"train": n_train, "val": n_val, "test": n_test}
    layout_n = layout_manifest.get("n_samples", {})
    for split, n in split_counts.items():
        avail = layout_n.get(split, 0)
        if num_graphs > avail:
            raise ValueError(
                f"Requested {num_graphs} graphs for {split!r} "
                f"but layout dataset only has {avail} layouts."
            )

    # Derive layout dataset canonical path for manifest
    canonical_layout = (
        f"data/interim/dagflow/{layout_manifest.get('preset', 'balanced-default')}/"
        f"v{layout_manifest['version']}"
    )

    # Default D_max to grid diameter if not specified
    if distance_max is None:
        distance_max = float(grid_width + grid_height)

    stage_params = {
        "corpus": corpus,
        "n_observations": n_observations,
        "num_graphs": num_graphs,
        "oracle_semantics": oracle_semantics,
        "field_decay": field_decay,
        "preference_step_penalty": preference_step_penalty,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "seed": seed,
        "layout_version": layout_manifest["version"],
        "static_weights": static_weights,
        "min_optimality_margin": min_optimality_margin,
    }
    if static_weights:
        stage_params["distance_tau"] = distance_tau
        stage_params["distance_max"] = distance_max
        stage_params["grid_width"] = grid_width
        stage_params["grid_height"] = grid_height
        stage_params["geometry_seed"] = geometry_seed

    with staging_root(version_root) as tmp:
        all_entries: list = []
        base_rng = np.random.default_rng(seed)

        for split in _SPLITS:
            n_samples = split_counts[split]
            split_rng = np.random.default_rng(base_rng.integers(0, 2**31))
            effective_k = min(num_graphs, n_samples)

            if effective_k == 0:
                # Write empty split
                entries = write_split(
                    output_root=tmp,
                    split=split,
                    samples=[],
                    source=f"{TASK_FAMILY}/{corpus}",
                    channels=GOALTRACE_TASK_CHANNELS,
                    topology_kind="dag",
                    n_states=n_observations,
                    extent=[n_observations],
                    index_kwargs={
                        "task_metadata": {
                            "task": TASK_FAMILY,
                            "corpus": corpus,
                        },
                    },
                    sample_validator=validate_goaltrace_sample,
                )
                all_entries.extend(entries)
                continue

            # Load layout graphs for this split
            entry_sample_pairs = list(
                iter_substrate_entries_and_samples(
                    layout_root,
                    split,
                    list(_REQUIRED_PARENT_CHANNELS),
                )
            )

            if effective_k > len(entry_sample_pairs):
                raise ValueError(
                    f"Requested {effective_k} graphs for {split!r} "
                    f"but only {len(entry_sample_pairs)} layouts available."
                )

            # Build adjacency for each layout graph
            k_max = layout_manifest.get("max_out_degree", 4)
            layout_graphs: list[dict[str, Any]] = []
            for entry_idx in range(effective_k):
                _entry, substrate_sample = entry_sample_pairs[entry_idx]
                n_actual = int(substrate_sample["node_mask"].sum())
                adjacency = [[] for _ in range(n_actual)]
                obs_ids = substrate_sample["node_obs_id"][:n_actual].astype(
                    np.int32
                )
                # Build reverse map: public obs ID → storage row.
                pub_to_row = {int(obs_ids[i]): i for i in range(n_actual)}
                for i in range(n_actual):
                    for k in range(k_max):
                        if substrate_sample["successor_mask"][i, k]:
                            succ_pub = int(
                                substrate_sample["successor_indices"][i, k]
                            )
                            if succ_pub in pub_to_row:
                                adjacency[i].append(pub_to_row[succ_pub])
                layout_graphs.append(
                    {
                        "sample": substrate_sample,
                        "n_actual": n_actual,
                        "adjacency": adjacency,
                        "obs_ids": obs_ids,
                    }
                )

            # --- Static-weight branch ---
            if static_weights:
                # Build all samples once, then split by identity.
                if split == _SPLITS[0]:  # first split
                    from ehc_sn.data.relational import (
                        combine_relational_topology,
                        compute_all_pairs_grid_distances,
                        distance_kernel,
                        generate_anchor_grid,
                    )

                    self_rng = np.random.default_rng(seed)

                    # Use the first graph's topology
                    graph = layout_graphs[0]
                    n_actual = graph["n_actual"]
                    adj = graph["adjacency"]  # permuted-space adjacency
                    obs_ids_g = graph["obs_ids"]

                    # Rebuild a rank-space adjacency with Hamiltonian backbone
                    # so that graph searches work in unpermuted index space.
                    rank_adj: list[list[int]] = [
                        list(range(i + 1, n_actual)) for i in range(n_actual)
                    ]
                    rank_adj[n_actual - 1] = []

                    # Build geometric substrate
                    anchors, grid_adj = generate_anchor_grid(
                        n_actual,
                        grid_width,
                        grid_height,
                        geometry_seed,
                    )
                    D = compute_all_pairs_grid_distances(
                        anchors,
                        grid_adj,
                        grid_width,
                    )
                    G = distance_kernel(D, distance_tau, distance_max)
                    # Combine geometry with the permuted-space adjacency
                    W = combine_relational_topology(G, adj)

                    pad_id = n_actual  # sentinel for padding
                    all_samples = _build_static_corpus_samples(
                        adjacency=rank_adj,
                        base_weights=W,
                        n_nodes=n_actual,
                        n_observations=n_observations,
                        semantics=oracle_semantics,
                        field_decay=field_decay,
                        step_penalty=preference_step_penalty,
                        min_margin=min_optimality_margin,
                        pad_obs_id=pad_id,
                        obs_ids=obs_ids_g,
                        rng=self_rng,
                    )

                    if not all_samples:
                        raise RuntimeError(
                            "No valid (current, goal) pairs survived the "
                            "optimality margin check.  Try lowering "
                            "min_optimality_margin."
                        )

                    # Split by identity
                    split_samples = _split_pairs_by_identity(
                        all_samples,
                        n_train,
                        n_val,
                        n_test,
                        rng=np.random.default_rng(seed + 1),
                    )

                # Use the pre-computed split for this split
                samples = split_samples[split]
                print(
                    f"  {split}: {len(samples)} samples "
                    f"(static weights, {len(adj)} nodes)"
                )

            else:
                # --- Legacy random-weight branch ---
                # Distribute samples across graphs
                samples_per_graph = [
                    n_samples // effective_k
                    + (1 if i < n_samples % effective_k else 0)
                    for i in range(effective_k)
                ]

                samples: list[dict[str, np.ndarray]] = []
                for graph_idx, graph in enumerate(layout_graphs):
                    graph_rng = np.random.default_rng(
                        split_rng.integers(0, 2**31) + graph_idx
                    )
                    for _ in range(samples_per_graph[graph_idx]):
                        sample = _assign_goaltrace_sample(
                            substrate_sample=graph["sample"],
                            n_nodes=graph["n_actual"],
                            adjacency=graph["adjacency"],
                            rng=graph_rng,
                            semantics=oracle_semantics,
                            field_decay=field_decay,
                            step_penalty=preference_step_penalty,
                            num_observations=n_observations,
                        )
                        samples.append(sample)

            entries = write_split(
                output_root=tmp,
                split=split,
                samples=samples,
                source=f"{TASK_FAMILY}/{corpus}",
                channels=GOALTRACE_TASK_CHANNELS,
                topology_kind="dag",
                n_states=n_observations,
                extent=[n_observations],
                index_kwargs={
                    "task_metadata": {
                        "task": TASK_FAMILY,
                        "corpus": corpus,
                    },
                },
                sample_validator=validate_goaltrace_sample,
            )
            all_entries.extend(entries)

        write_index_at_root(all_entries, tmp)

        # Build manifest with role-addressed parents dict.
        manifest_kwargs: dict[str, Any] = dict(
            dataset_class="task_corpus",
            family=TASK_FAMILY,
            version=version,
            channels=GOALTRACE_TASK_CHANNELS,
            topology_kind="dag",
            n_states=n_observations,
            extent=[n_observations],
            n_samples=split_counts,
            source_id="synthetic/dagflow",
            builder="ehc_sn.tasks.goaltrace.builder.build_goaltrace_task_corpus",
            seed=seed,
            stage_params=stage_params,
            task=TASK_FAMILY,
            corpus=corpus,
            task_schema_version=1,
            task_protocol_version=1,
            n_observations=n_observations,
            num_graphs=num_graphs,
            oracle_semantics=oracle_semantics,
            field_decay=field_decay,
            manifest_schema_version=1,
            parents={
                "semantic_graph": {
                    "family": "dagflow",
                    "root": str(layout_root),
                    "version": layout_manifest["version"],
                },
            },
        )
        if dagflow_graph_id is not None:
            # Look up the graph entry for its content_digest
            from ehc_sn.data.substrate.reader import find_artifact_by_id

            graph_entry, _graph_sample = find_artifact_by_id(
                layout_root, dagflow_graph_id
            )
            manifest_kwargs["parents"]["semantic_graph"].update(
                {
                    "artifact_id": dagflow_graph_id,
                    "split": graph_entry.split,
                    "content_digest": graph_entry.content_digest,
                }
            )

        write_manifest(tmp, **manifest_kwargs)


__all__ = [
    "GOALTRACE_TASK_CHANNELS",
    "GOALTRACE_TASK_CHANNEL_DTYPES",
    "TASK_FAMILY",
    "build_goaltrace_task_corpus",
    "validate_goaltrace_root",
    "validate_goaltrace_sample",
]
