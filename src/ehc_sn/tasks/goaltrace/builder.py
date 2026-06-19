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
    "node_start_flag",
    "node_goal_flag",
    "successor_indices",
    "successor_mask",
    "node_mask",
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

    The dagflow substrate stores nodes in permuted (final) order with
    ``node_start_flag`` and ``node_goal_flag`` marking the start and goal
    positions.  This function samples a (current, goal) pair **independent of
    the substrate's start/goal** — every goaltrace sample redefines these.

    Args:
        substrate_sample: Dagflow substrate sample with structural channels.
            Nodes are already in permuted (final) position order.
        n_nodes: Number of actual nodes.
        adjacency: Adjacency list for actual nodes (original index space).
        rng: Seeded random generator.
        semantics: Oracle semantics string.
        field_decay: Field decay factor gamma.
        step_penalty: Per-step penalty for ``"preference"`` semantics.
        num_observations: Padded observation count N.

    Returns:
        Sample dict with goaltrace task channels.
    """
    # Build the permuted-index mapping: original_idx -> permuted_idx
    # The dagflow substrate stores nodes in permuted order, so the
    # i-th actual (masked) node IS the permuted position.
    # We need to find all actual nodes and their observation IDs.
    node_mask_np = substrate_sample["node_mask"]
    obs_ids = substrate_sample["node_obs_id"]
    actual_indices = np.where(node_mask_np)[
        0
    ]  # permuted positions of actual nodes

    # Convert adjacency (original indices) to permuted indices
    perm_to_orig = {pi: oi for oi, pi in enumerate(actual_indices)}
    orig_to_perm = {oi: pi for oi, pi in enumerate(actual_indices)}

    # Sample (current, goal) pair — must have a valid path between them
    while True:
        current_orig = int(rng.integers(0, n_nodes))
        goal_orig = int(rng.integers(0, n_nodes))
        if current_orig == goal_orig:
            continue
        sp = bfs_shortest_path(adjacency, current_orig, goal_orig)
        if sp:
            break

    # Sample weights
    full_weights = _sample_weight_vector(n_nodes, rng)
    masked = _apply_adjacency_mask(full_weights, adjacency)

    # Select optimal path (in original index space)
    path_orig = _select_optimal_path(
        adjacency, masked, current_orig, goal_orig, semantics, step_penalty
    )

    # Encode target field (in original index space)
    target_field_orig = _encode_target_field(
        path_orig, n_nodes, field_decay, current_orig
    )

    # Build padded output arrays in permuted (final) position order
    current_flag = np.zeros(num_observations, dtype=bool)
    goal_flag = np.zeros(num_observations, dtype=bool)
    node_mask_out = np.zeros(num_observations, dtype=bool)
    weight = np.zeros(num_observations, dtype=np.float32)
    padded_target = np.zeros(num_observations, dtype=np.float32)

    for orig_i in range(n_nodes):
        pi = orig_to_perm[orig_i]
        current_flag[pi] = orig_i == current_orig
        goal_flag[pi] = orig_i == goal_orig
        node_mask_out[pi] = True
        weight[pi] = full_weights[current_orig, orig_i]
        padded_target[pi] = float(target_field_orig[orig_i])

    # Pad observation_id to num_observations (matching other channels)
    obs_ids_len = len(obs_ids)
    if obs_ids_len < num_observations:
        padded_obs_ids = np.zeros(num_observations, dtype=np.int32)
        padded_obs_ids[:obs_ids_len] = obs_ids.astype(np.int32)
    else:
        padded_obs_ids = obs_ids[:num_observations].astype(np.int32)

    # Forward topology from parent substrate, padded to num_observations
    K = substrate_sample["successor_indices"].shape[-1]
    raw_succ_idx = substrate_sample["successor_indices"]  # (n_actual, K)
    raw_succ_mask = substrate_sample["successor_mask"]  # (n_actual, K)
    if raw_succ_idx.ndim == 2:
        # Single-graph substrate: shape (n_actual, K), pad rows to N
        padded_succ_idx = np.zeros((num_observations, K), dtype=np.int32)
        padded_succ_mask = np.zeros((num_observations, K), dtype=bool)
        n_actual = raw_succ_idx.shape[0]
        padded_succ_idx[:n_actual, :] = raw_succ_idx
        padded_succ_mask[:n_actual, :] = raw_succ_mask
    else:
        # Multi-sample: keep as-is (already batched)
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
# Builder
# =============================================================================


def build_goaltrace_task_corpus(
    version_root: Path,
    *,
    layout_root: Path,
    corpus: str = "default",
    n_observations: int = 45,
    num_graphs: int = 1,
    oracle_semantics: str = "reliability",
    field_decay: float = 0.8,
    preference_step_penalty: float = 0.0,
    n_train: int = 4000,
    n_val: int = 500,
    n_test: int = 500,
    seed: int = 42,
) -> None:
    """Build the goaltrace task corpus at *version_root* over a dagflow layout.

    Consumes dagflow layout samples for graph topology and generates
    weight-per-edge, oracle path selection, and target field encoding.
    When ``num_graphs == 1`` (default), all samples share a single DAG.

    The version integer is derived from the ``v<N>`` leaf of *version_root*.

    Args:
        version_root: Destination versioned root
            (e.g. ``data/processed/goaltrace/default/v1``).  Must not exist.
        layout_root: Path to the dagflow layout dataset root.  Must contain
            a ``manifest.json`` with all required structural channels.
        corpus: Corpus label (e.g. ``"default"``).
        n_observations: Maximum padded node count N.
        num_graphs: Number of DAGs to sample from the layout pool.  Must be
            at most the number of available layouts per split.
        oracle_semantics: Weight semantics for path selection.
            One of ``"reliability"``, ``"linear_cost"``, ``"preference"``.
        field_decay: Decay factor gamma in ``(0, 1)``.
        preference_step_penalty: Per-step penalty lambda for ``"preference"``
            semantics.
        n_train: Number of training samples.
        n_val: Number of validation samples.
        n_test: Number of test samples.
        seed: Deterministic base seed for reproducibility.

    Raises:
        FileExistsError: When *version_root* already exists (immutable root).
        FileNotFoundError: When *layout_root* has no manifest.
        ValueError: When the layout is not a dagflow layout_dataset,
            the layout lacks required structural channels, or
            *oracle_semantics* is invalid.
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
        f"data/interim/dagflow/{layout_manifest.get('preset', 'default')}/"
        f"v{layout_manifest['version']}"
    )

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
    }

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
                for i in range(n_actual):
                    for k in range(k_max):
                        if substrate_sample["successor_mask"][i, k]:
                            succ = int(
                                substrate_sample["successor_indices"][i, k]
                            )
                            if succ < n_actual:
                                adjacency[i].append(succ)
                layout_graphs.append(
                    {
                        "sample": substrate_sample,
                        "n_actual": n_actual,
                        "adjacency": adjacency,
                    }
                )

            # Distribute samples across graphs (round-robin or even split)
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

        write_manifest(
            tmp,
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
            parent_substrate=canonical_layout,
            parent_family="dagflow",
            parent_version=layout_manifest["version"],
            n_observations=n_observations,
            num_graphs=num_graphs,
            oracle_semantics=oracle_semantics,
            field_decay=field_decay,
        )


__all__ = [
    "GOALTRACE_TASK_CHANNELS",
    "GOALTRACE_TASK_CHANNEL_DTYPES",
    "TASK_FAMILY",
    "build_goaltrace_task_corpus",
    "validate_goaltrace_root",
    "validate_goaltrace_sample",
]
