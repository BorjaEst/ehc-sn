"""Layout-dataset builder for the dagflow source family.

Produces a versioned, immutable interim layout dataset containing DAG
topology, node observations, successor structure, and start/goal flags.
These are reusable structural facts consumed by downstream task CLIs
(build-seqmaze.py).

Layout channels (structural, expressed in public observation IDs):

- ``node_rank``: Topological rank per node row. Padded entries are
  ``n_actual`` (sentinel). Shape ``(N,)`` int32.
- ``node_obs_id``: Public observation ID per node row. Padded entries
  are ``n_actual`` (sentinel outside the public domain). Shape ``(N,)`` int32.
- ``rank_to_obs_id``: Public observation ID for each rank. Strict
  bijection. Shape ``(n_actual,)`` int32.
- ``obs_id_to_rank``: Rank for each public observation ID. Inverse of
  ``rank_to_obs_id``. Shape ``(n_actual,)`` int32.
- ``successor_indices``: Public observation IDs of successors per slot.
  Padded entries are ``n_actual``. Shape ``(N, K)`` int32.
- ``successor_mask``: Valid successor slot mask. Shape ``(N, K)`` bool.
- ``node_mask``: True for actual (non-padded) nodes. Shape ``(N,)`` bool.

Dagflow defines only the public observation vocabulary and the
semantic transition relation.  Task protocol channels (target path,
path length, edge labels, weights, targets) belong in the respective
task corpus, not here.

Layout dataset path: ``data/interim/dagflow/{preset}/v{version}/``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ehc_sn.data.lifecycle import (
    extract_version,
    staging_root,
    validate_version_root,
    write_index_at_root,
    write_split,
)
from ehc_sn.data.manifest import write_manifest
from ehc_sn.utils.graph import (
    canonical_dag_digest,
    generate_hamiltonian_dag,
    remap_obs_ids,
)
from ehc_sn.utils.graph import shortest_path as bfs_shortest_path

# ---------------------------------------------------------------------------
# Rank-span distribution profiles
# ---------------------------------------------------------------------------

_SPAN_PROFILES: dict[str, dict[str, float]] = {
    "local": {
        "short_prob": 0.70,
        "medium_prob": 0.25,
        "long_prob": 0.05,
    },
    "balanced": {
        "short_prob": 0.45,
        "medium_prob": 0.40,
        "long_prob": 0.15,
    },
    "heavy": {
        "short_prob": 0.25,
        "medium_prob": 0.40,
        "long_prob": 0.35,
    },
    "uniform": {
        "short_prob": 0.33,
        "medium_prob": 0.34,
        "long_prob": 0.33,
    },
}
"""Named rank-span distribution profiles (short, medium, long probabilities).

Used by presets to control where optional edges land in rank space.
Probabilities need not sum to 1.0; each eligible edge independently
competes with the profile's per-band probability.
"""

DAGFLOW_PRESETS: dict[str, dict] = {
    "branching": {
        "span_profile": "local",
        "target_edges": 139,
        "n_max": 45,
        "max_out_degree": 4,
        "public_id_policy": "permuted",
        "description": (
            "Branching DAG profile: 45 nodes, 139 edges, high shortcut "
            "density.  Produces graphs with many alternative paths and "
            "high merge/branch counts.  Suitable for multi-path reasoning "
            "stress tests."
        ),
    },
    "small": {
        "span_profile": "balanced",
        "target_edges": 11,
        "n_max": 8,
        "max_out_degree": 3,
        "public_id_policy": "permuted",
        "description": "Small 8-node graphs for smoke tests.",
    },
    "routing": {
        "span_profile": "local",
        "target_edges": 80,
        "n_max": 45,
        "max_out_degree": 4,
        "public_id_policy": "permuted",
        "description": (
            "Canonical routing graph profile.  45 observations, "
            "low extra-edge count with local shortcuts; "
            "supports waypoint_count 5-10 in typical Routebind queries;  "
            "this is a structural graph property, not a routebind preset  "
            "guarantee.  "
            "Suitable for the documented Routebind mixed-difficulty regime."
        ),
    },
    "chain16": {
        "span_profile": "local",
        "target_edges": 18,
        "n_max": 16,
        "max_out_degree": 3,
        "public_id_policy": "permuted",
        "description": (
            "Near-chain semantic graph for long-composition stress tests.  "
            "Minimal shortcuts; supports accepted-observation sequences "
            "up to 15 steps.  Use for curriculum progression, recurrent-depth "
            "evaluation, and oracle performance characterization.  "
            "NOT the canonical Routebind v1 profile."
        ),
    },
}
"""Named generation presets bundling span profile, target edge count, and dimensions.

Each preset defines:
- ``span_profile``: Named rank-span distribution (see ``_SPAN_PROFILES``).
- ``target_edges``: Desired total edge count (backbone + optional).
- ``n_max``: Maximum candidate nodes (N).
- ``max_out_degree``: Maximum out-degree per node (K).
- ``public_id_policy``: ``"permuted"`` (always) — observation IDs are a
  random bijection of ranks, so models cannot infer direction from
  numerical ID order.

A preset is feasible only when ``target_edges ≤ max_feasible`` under the
degree cap (computed by ``_compute_max_edges_under_degree_cap``).  Every
predefined preset satisfies this constraint.

Backward-compatibility note: old preset names (``balanced-default``,
``sparse-default``, ``dense-default``, ``small-balanced``, ``balanced``,
``dense``) are not forward-ported.  Existing layout datasets built with
those names remain valid but the ``build`` command no longer accepts them.
"""


def _resolve_preset(
    preset: str,
    n_max: int | None,
    max_out_degree: int | None,
    extra_edge_density: float | None,
    span_profile: str | None = None,
) -> dict:
    """Resolve a named preset into full generation parameters.

    Returns a dict with keys ``n_max``, ``max_out_degree``,
    ``extra_edge_density``, ``span_profile``, ``target_edges``,
    and the resolved ``short_prob``, ``medium_prob``, ``long_prob``.

    Individual dimension overrides (*n_max*, *max_out_degree*, etc.)
    override the preset defaults when explicitly passed (not None).
    The Hamiltonian backbone is mandatory and non-configurable.
    Public observation IDs are always permuted.

    When *extra_edge_density* is provided as an override, it takes
    precedence over the preset's stored ``target_edges``.  When both
    are present in the preset dict, ``target_edges`` is the primary
    specification and ``extra_edge_density`` is a fallback for CLI
    compatibility.
    """
    # Backward-compatible alias: "sparse" → "branching"
    _PRESET_ALIASES: dict[str, str] = {"sparse": "branching"}
    resolved_preset = _PRESET_ALIASES.get(preset, preset)
    if resolved_preset != preset:
        import warnings

        warnings.warn(
            f"dagflow preset {preset!r} is deprecated; use "
            f"{resolved_preset!r} instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    if resolved_preset not in DAGFLOW_PRESETS:
        raise ValueError(
            f"Unknown dagflow preset {resolved_preset!r}. "
            f"Valid: {sorted(DAGFLOW_PRESETS)}."
        )
    preset = resolved_preset

    p = DAGFLOW_PRESETS[preset]

    resolved_profile = (
        span_profile if span_profile is not None else p["span_profile"]
    )
    if resolved_profile not in _SPAN_PROFILES:
        raise ValueError(
            f"Preset {preset!r} references unknown span profile "
            f"{resolved_profile!r}. Valid: {sorted(_SPAN_PROFILES)}."
        )

    span_params = _SPAN_PROFILES[resolved_profile]

    resolved = {
        "preset": preset,
        "span_profile": resolved_profile,
        "n_max": p["n_max"] if n_max is None else n_max,
        "max_out_degree": (
            p["max_out_degree"] if max_out_degree is None else max_out_degree
        ),
    }

    n = resolved["n_max"]
    max_extra = n * (n - 1) // 2 - (n - 1)

    # Determine target_edges: CLI density override > preset target_edges > preset extra_edge_density
    if extra_edge_density is not None:
        # CLI override: compute from density, validate against degree cap
        extra = round(extra_edge_density * max_extra)
        resolved["target_edges"] = (n - 1) + extra
        resolved["extra_edge_density"] = extra_edge_density
    elif "target_edges" in p:
        resolved["target_edges"] = p["target_edges"]
        # Back-compute density for manifest / display purposes
        resolved["extra_edge_density"] = (
            (p["target_edges"] - (n - 1)) / max_extra if max_extra > 0 else 0.0
        )
    elif "extra_edge_density" in p:
        # Fallback for legacy preset dicts
        extra = round(p["extra_edge_density"] * max_extra)
        resolved["target_edges"] = (n - 1) + extra
        resolved["extra_edge_density"] = p["extra_edge_density"]
    else:
        raise ValueError(
            f"Preset {preset!r} has neither 'target_edges' nor "
            f"'extra_edge_density'."
        )

    resolved["short_prob"] = span_params["short_prob"]
    resolved["medium_prob"] = span_params["medium_prob"]
    resolved["long_prob"] = span_params["long_prob"]

    # Feasibility guard: reject edge counts that exceed the out-degree cap.
    k = resolved["max_out_degree"]
    max_feasible = _compute_max_edges_under_degree_cap(n, k)
    if resolved["target_edges"] > max_feasible:
        feasible_extra = max_feasible - (n - 1)
        feasible_density = feasible_extra / max_extra if max_extra > 0 else 0.0
        raise ValueError(
            f"Preset {preset!r} requests target_edges={resolved['target_edges']}"
            f" (extra_edge_density={resolved['extra_edge_density']}), "
            f"but max_out_degree={k} caps feasible edges at {max_feasible}"
            f" for n_max={n}. "
            f"Reduce extra_edge_density to ≤{feasible_density:.3f}"
            f" or increase max_out_degree."
        )

    return resolved


# ---------------------------------------------------------------------------
# Graph structural diagnostics
# ---------------------------------------------------------------------------


def compute_graph_diagnostics(
    adjacency: list[list[int]],
    n_nodes: int,
    obs_ids: np.ndarray | None = None,
) -> dict:
    """Compute structural diagnostics for a DAG graph.

    Operates on the rank-space adjacency list.  Diagnostics include edge
    counts, degree histograms, span distribution, branch/merge node counts,
    shortest-path-length distribution over all rank-ordered pairs, semantic
    diameter, direct-pair fraction, multi-path-pair fraction, and rank-to-
    public-ID correlation diagnostics.

    Args:
        adjacency: Rank-space adjacency list where ``adj[i]`` is a list of
            successor ranks.
        n_nodes: Number of nodes in the graph.
        obs_ids: Optional array of public observation IDs in rank order.
            When provided, rank-to-public-ID correlation diagnostics are
            included.

    Returns:
        Dict with the following keys:
        - ``total_edges``: Total directed edges.
        - ``backbone_edges``: Edges in the mandatory Hamiltonian backbone.
        - ``extra_edges``: Edges beyond the backbone.
        - ``realized_density``: Fraction of possible extra edges realised.
        - ``out_degree_histogram``: ``list[int]`` of length ``n_nodes``,
          ``out_degree_histogram[d]`` = number of nodes with out-degree d.
        - ``in_degree_histogram``: Same shape for in-degree.
        - ``rank_span_histogram``: ``list[int]`` of length ``n_nodes``,
          ``rank_span_histogram[s]`` = number of edges with rank separation s.
        - ``branch_nodes``: Nodes with out-degree > 1.
        - ``merge_nodes``: Nodes with in-degree > 1.
        - ``shortest_path_histogram``: ``list[int]`` of length ``n_nodes``,
          ``shortest_path_histogram[l]`` = number of ordered pairs (a, b)
          with ``a < b`` whose shortest path length in edges is ``l``.
          The 0-th element counts pairs with ``a == b`` (not stored).
        - ``semantic_diameter``: Longest shortest path length (in edges)
          over all rank-ordered pairs.
        - ``direct_pair_fraction``: Fraction of rank-ordered pairs
          connected by a direct edge.
        - ``multi_path_pair_fraction``: Fraction of rank-ordered pairs
          that have at least two distinct shortest paths.
        - ``rank_obs_id_correlation``: Dict with Spearman correlation
          and rank-order statistics, or ``None`` if ``obs_ids`` not provided.
    """
    total_edges = sum(len(a) for a in adjacency)
    backbone_edges = n_nodes - 1
    extra_edges = total_edges - backbone_edges

    max_extra_possible = n_nodes * (n_nodes - 1) // 2 - backbone_edges
    realized_density = (
        extra_edges / max_extra_possible if max_extra_possible > 0 else 0.0
    )

    # Degree histograms
    out_deg = [len(a) for a in adjacency]
    out_deg_hist = [0] * (n_nodes + 1)
    for d in out_deg:
        out_deg_hist[d] += 1

    in_deg = [0] * n_nodes
    for u in range(n_nodes):
        for v in adjacency[u]:
            in_deg[v] += 1
    in_deg_hist = [0] * (n_nodes + 1)
    for d in in_deg:
        in_deg_hist[d] += 1

    # Rank-span histogram
    span_hist = [0] * n_nodes
    for u in range(n_nodes):
        for v in adjacency[u]:
            span = v - u
            if 0 < span < n_nodes:
                span_hist[span] += 1

    # Branch and merge counts
    branch_nodes = sum(1 for d in out_deg if d > 1)
    merge_nodes = sum(1 for d in in_deg if d > 1)

    # Shortest-path histogram over all rank-ordered pairs (a < b)
    from ehc_sn.utils.graph import count_shortest_paths
    from ehc_sn.utils.graph import shortest_path as bfs_shortest_path

    sp_hist = [0] * n_nodes  # sp_hist[l] = count pairs with shortest path l
    direct_count = 0
    multi_path_count = 0
    total_pairs = n_nodes * (n_nodes - 1) // 2
    semantic_diameter = 0

    for a in range(n_nodes):
        for b in range(a + 1, n_nodes):
            sp = bfs_shortest_path(adjacency, a, b)
            if not sp:
                continue
            length = len(sp) - 1  # number of edges
            sp_hist[length] += 1
            if length > semantic_diameter:
                semantic_diameter = length
            if length == 1:
                direct_count += 1
            if count_shortest_paths(adjacency, a, b) >= 2:
                multi_path_count += 1

    direct_pair_fraction = (
        direct_count / total_pairs if total_pairs > 0 else 0.0
    )
    multi_path_pair_fraction = (
        multi_path_count / total_pairs if total_pairs > 0 else 0.0
    )

    # Rank-to-public-ID correlation diagnostics
    rank_obs_corr = None
    if obs_ids is not None:
        import scipy.stats as stats

        ranks = np.arange(n_nodes, dtype=np.float64)
        obs = np.asarray(obs_ids[:n_nodes], dtype=np.float64)
        spearman_r, spearman_p = stats.spearmanr(ranks, obs)
        rank_obs_corr = {
            "spearman_r": float(spearman_r),
            "spearman_p_value": float(spearman_p),
            "obs_ids_in_rank_order": obs_ids[:n_nodes].tolist(),
        }

    return {
        "total_edges": total_edges,
        "backbone_edges": backbone_edges,
        "extra_edges": extra_edges,
        "realized_density": realized_density,
        "out_degree_histogram": out_deg_hist,
        "in_degree_histogram": in_deg_hist,
        "rank_span_histogram": span_hist,
        "branch_nodes": branch_nodes,
        "merge_nodes": merge_nodes,
        "shortest_path_histogram": sp_hist,
        "semantic_diameter": semantic_diameter,
        "direct_pair_fraction": direct_pair_fraction,
        "multi_path_pair_fraction": multi_path_pair_fraction,
        "rank_obs_id_correlation": rank_obs_corr,
    }


# ---------------------------------------------------------------------------
# Preset acceptance profiles
# ---------------------------------------------------------------------------

# Registry of preset-specific acceptance check functions.
# Each function receives (diagnostics dict) and returns list of error strings
# (empty = accepted).
_PRESET_ACCEPTANCE_CHECKS: dict[str, list[callable]] = {}


def _register_acceptance_checks(preset_name: str) -> callable:
    """Decorator that registers a function as an acceptance checker for a preset."""

    def decorator(fn: callable) -> callable:
        _PRESET_ACCEPTANCE_CHECKS.setdefault(preset_name, []).append(fn)
        return fn

    return decorator


def validate_preset_acceptance_profile(
    diagnostics: dict,
    preset_name: str,
) -> list[str]:
    """Validate graph diagnostics against the preset's acceptance profile.

    Args:
        diagnostics: Dict from ``compute_graph_diagnostics``.
        preset_name: Preset name to check against.

    Returns:
        List of error strings.  Empty list means the graph satisfies the
        preset's acceptance profile.
    """
    errors: list[str] = []
    checks = _PRESET_ACCEPTANCE_CHECKS.get(preset_name, [])
    for check_fn in checks:
        try:
            check_errors = check_fn(diagnostics)
            if check_errors:
                errors.extend(check_errors)
        except Exception as exc:
            errors.append(f"Acceptance check {check_fn.__name__} raised: {exc}")
    return errors


# --- Built-in acceptance checks for each preset ---


@_register_acceptance_checks("routing")
def _check_routing(diag: dict) -> list[str]:
    errors: list[str] = []
    if diag["branch_nodes"] < 2:
        errors.append(
            f"routing requires branch_nodes >= 2, got {diag['branch_nodes']}."
        )
    if diag["merge_nodes"] < 2:
        errors.append(
            f"routing requires merge_nodes >= 2, got {diag['merge_nodes']}."
        )
    if diag["semantic_diameter"] < 8:
        errors.append(
            f"routing requires semantic_diameter >= 8, "
            f"got {diag['semantic_diameter']}."
        )
    if diag["direct_pair_fraction"] > 0.35:
        errors.append(
            f"routing requires direct_pair_fraction <= 0.35, "
            f"got {diag['direct_pair_fraction']:.4f}."
        )
    if diag["extra_edges"] > 50:
        errors.append(
            f"routing requires extra_edges <= 50, "
            f"got {diag['extra_edges']}."
        )
    return errors


@_register_acceptance_checks("chain16")
def _check_chain16(diag: dict) -> list[str]:
    errors: list[str] = []
    if diag["semantic_diameter"] < 12:
        errors.append(
            f"chain16 requires semantic_diameter >= 12, "
            f"got {diag['semantic_diameter']}."
        )
    if diag["extra_edges"] > 5:
        errors.append(
            f"chain16 requires extra_edges <= 5, got {diag['extra_edges']}."
        )
    if diag["total_edges"] > 20:
        errors.append(
            f"chain16 requires total_edges <= 20, "
            f"got {diag['total_edges']}."
        )
    return errors


# ---------------------------------------------------------------------------
# Span-band helpers (scale with N)
# ---------------------------------------------------------------------------


def _scale_spans(
    n_nodes: int,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Return (short_span, medium_span, long_span) scaled to *n_nodes*.

    Boundaries are fractions of N, clamped to valid ranges.
    """
    # Ensure at least one possible edge beyond the backbone in each band.
    short_hi = max(2, n_nodes // 8)
    medium_lo = short_hi + 1
    medium_hi = max(medium_lo, n_nodes // 3)
    long_lo = medium_hi + 1
    long_hi = max(long_lo, n_nodes - 2)

    return (
        (2, short_hi),
        (medium_lo, medium_hi),
        (long_lo, long_hi),
    )


# ---------------------------------------------------------------------------
# Degree-cap feasibility
# ---------------------------------------------------------------------------


def _compute_max_edges_under_degree_cap(
    n_nodes: int,
    max_out_degree: int,
) -> int:
    """Return the maximum total edges possible under ``max_out_degree``.

    Each node ``i`` can have at most ``max_out_degree`` outgoing edges, all
    of which must go to nodes ``j > i`` (forward-only DAG invariant).  The
    mandatory Hamiltonian backbone occupies one out-degree slot per non-
    terminal node; the remaining slots are available for extra edges.

    Args:
        n_nodes: Number of nodes (N).
        max_out_degree: Maximum out-degree per node (K).

    Returns:
        Maximum total edge count achievable under the degree cap.
    """
    total = 0
    for i in range(n_nodes - 1):
        # Available forward candidates: nodes i+1 .. n_nodes-1
        max_forward = n_nodes - 1 - i
        # At most K edges, but capped by available forward candidates
        total += min(max_out_degree, max_forward)
    return total


# =============================================================================
LAYOUT_FAMILY: str = "dagflow"
"""Layout-dataset family name for the dagflow source."""

LAYOUT_CHANNELS: list[str] = [
    "node_rank",
    "node_obs_id",
    "rank_to_obs_id",
    "obs_id_to_rank",
    "successor_indices",
    "successor_mask",
    "node_mask",
]
"""Canonical layout channels for dagflow worlds.

Only the public observation vocabulary and the semantic transition
relation.  No task-query or internal-machinery channels.
"""

_SOURCE_ID: str = "synthetic/dagflow"

_SPLITS: tuple[str, ...] = ("train", "val", "test")


# =============================================================================
def _generate_graph_sample(
    n_max: int,
    k_max: int,
    rng: np.random.Generator,
    *,
    target_edges: int | None = None,
    fixed_n_actual: bool = False,
    min_extra_edges_per_node: int = 0,
    short_prob: float = 0.45,
    medium_prob: float = 0.40,
    long_prob: float = 0.15,
) -> tuple[dict[str, np.ndarray], str]:
    """Generate one random DAG sample with structural channels only.

    Nodes are stored in **rank order** (row ``r`` always contains rank
    ``r``).  Public observation IDs are a random bijection of the rank
    ordering, so the model cannot infer direction from numerical ID order.

    The mandatory Hamiltonian backbone ``0→1→…→n_actual-1`` guarantees
    full reachability: ``rank(i) < rank(j) ⇒ i⇝j``.

    Layout channels include explicit rank↔observation lookup tables so
    downstream consumers never need to reconstruct the mapping.

    Args:
        n_max: Maximum candidate nodes (N). Actual nodes < n_max are padded.
        k_max: Maximum out-degree per node (K).
        rng: Seeded random generator.
        target_edges: Desired total edge count.  ``None`` for probabilistic
            sampling without a fixed target.
        fixed_n_actual: When True, use all ``n_max`` nodes (no randomization).
        min_extra_edges_per_node: Minimum extra edges beyond the mandatory
            backbone edge.
        short_prob: Per-candidate probability for short-span edges.
        medium_prob: Per-candidate probability for medium-span edges.
        long_prob: Per-candidate probability for long-span edges.

    Returns:
        Tuple of (sample dict with layout channels, content_digest string).
    """
    if fixed_n_actual:
        n_actual = n_max
    else:
        n_actual_low = max(4, min(n_max - 1, 10))
        n_actual = int(rng.integers(n_actual_low, n_max + 1))

    base_seed = int(rng.integers(0, 2**31))
    short_span, medium_span, long_span = _scale_spans(n_actual)
    adjacency = generate_hamiltonian_dag(
        n_actual,
        k_max,
        base_seed,
        target_edges=target_edges,
        min_extra_edges_per_node=min_extra_edges_per_node,
        short_edge_prob=short_prob,
        medium_edge_prob=medium_prob,
        long_edge_prob=long_prob,
        short_span=short_span,
        medium_span=medium_span,
        long_span=long_span,
    )

    remap_seed = int(rng.integers(0, 2**31))
    obs_ids = remap_obs_ids(n_actual, remap_seed)

    sentinel = (
        n_actual  # padding sentinel outside public domain {0..n_actual-1}
    )

    # Rank-indexed storage: row r = rank r
    node_rank = np.full(n_max, sentinel, dtype=np.int32)
    node_obs_id = np.full(n_max, sentinel, dtype=np.int32)
    successor_indices = np.full((n_max, k_max), sentinel, dtype=np.int32)
    successor_mask = np.zeros((n_max, k_max), dtype=bool)
    node_mask = np.zeros(n_max, dtype=bool)

    rank_to_obs_id = np.array(obs_ids, dtype=np.int32)  # (n_actual,)
    obs_id_to_rank = np.full(n_actual, sentinel, dtype=np.int32)
    for r in range(n_actual):
        obs_id_to_rank[obs_ids[r]] = r

    for r in range(n_actual):
        node_rank[r] = r
        node_obs_id[r] = obs_ids[r]
        node_mask[r] = True

        for k, succ_rank in enumerate(adjacency[r]):
            # Store the public observation ID of the successor (rank→obs).
            succ_pub = obs_ids[succ_rank]
            successor_indices[r, k] = succ_pub
            successor_mask[r, k] = True

    # Compute content digest from canonical DAG representation (public IDs only)
    obs_ids_arr = np.array(obs_ids, dtype=np.int32)
    content_digest = canonical_dag_digest(adjacency, obs_ids_arr)

    return {
        "node_rank": node_rank,
        "node_obs_id": node_obs_id,
        "rank_to_obs_id": rank_to_obs_id,
        "obs_id_to_rank": obs_id_to_rank,
        "successor_indices": successor_indices,
        "successor_mask": successor_mask,
        "node_mask": node_mask,
    }, content_digest


# =============================================================================
def validate_dagflow_layout_sample(data: dict[str, np.ndarray]) -> None:
    """Validate one dagflow layout sample against the rank-first contract.

    Enforces:

    1. All channels present.
    2. Shape invariants.
    3. Rank permutation bijection (every rank 0..n_actual-1 appears exactly once).
    4. Inverse consistency (obs_id_to_rank[rank_to_obs_id[r]] == r).
    5. Forward-only edges (rank(u) < rank(v) for every edge).
    6. Mandatory backbone: edge r → r+1 exists for every r < n_actual-1.
    7. No stranded non-terminal nodes (each has out-degree ≥ 1; each except
       rank 0 has in-degree ≥ 1).
    8. Public-edge correspondence (successor_indices are obs IDs of rank-space
       successors).
    9. Global rank reachability (BFS from rank 0 reaches all ranks).

    Args:
        data: Dict of channel arrays for one sample.

    Raises:
        ValueError: On any structural or semantic violation.
    """
    missing = set(LAYOUT_CHANNELS) - data.keys()
    if missing:
        raise ValueError(
            f"dagflow layout sample missing channels: {sorted(missing)}"
        )

    n_max = data["node_rank"].shape[0]
    k_max = data["successor_indices"].shape[1]

    # --- Shape invariants ---
    for name in ("node_obs_id", "node_rank"):
        if data[name].shape[0] != n_max:
            raise ValueError(f"{name} does not match N ({n_max}).")
    if data["successor_indices"].shape[0] != n_max:
        raise ValueError("successor_indices first dim does not match N.")
    if data["successor_mask"].shape != (n_max, k_max):
        raise ValueError("successor_mask shape mismatch.")
    if data["node_mask"].shape[0] != n_max:
        raise ValueError("node_mask does not match N.")
    if data["rank_to_obs_id"].ndim != 1:
        raise ValueError("rank_to_obs_id must be 1-D.")
    if data["obs_id_to_rank"].ndim != 1:
        raise ValueError("obs_id_to_rank must be 1-D.")

    n_actual = int(data["node_mask"].sum())
    if n_actual < 2:
        raise ValueError(f"At least 2 actual nodes required, got {n_actual}.")

    # --- 3. Rank permutation bijection ---
    # node_rank[r] == r for r < n_actual, sentinel for padding
    sentinel = n_actual
    for r in range(n_max):
        if data["node_mask"][r]:
            if int(data["node_rank"][r]) != r:
                raise ValueError(
                    f"node_rank[{r}] = {int(data['node_rank'][r])} for masked "
                    f"node, expected {r} (rank-indexed storage)."
                )
        else:
            if int(data["node_rank"][r]) != sentinel:
                raise ValueError(
                    f"node_rank[{r}] = {int(data['node_rank'][r])} for padded "
                    f"node, expected sentinel {sentinel}."
                )

    # Verify rank_to_obs_id is a bijection containing each ID 0..n_actual-1
    r2o = data["rank_to_obs_id"]
    if r2o.shape[0] != n_actual:
        raise ValueError(
            f"rank_to_obs_id has length {r2o.shape[0]}, expected {n_actual}."
        )
    if set(int(x) for x in r2o) != set(range(n_actual)):
        raise ValueError(
            "rank_to_obs_id does not contain every ID in [0, n_actual)."
        )

    # --- 4. Inverse consistency ---
    o2r = data["obs_id_to_rank"]
    if o2r.shape[0] != n_actual:
        raise ValueError(
            f"obs_id_to_rank has length {o2r.shape[0]}, expected {n_actual}."
        )
    for r in range(n_actual):
        pub_id = int(r2o[r])
        if int(o2r[pub_id]) != r:
            raise ValueError(
                f"Inverse consistency violation: rank_to_obs_id[{r}] = {pub_id}, "
                f"but obs_id_to_rank[{pub_id}] = {int(o2r[pub_id])}, expected {r}."
            )

    # Verify node_obs_id entries
    for r in range(n_max):
        oid = int(data["node_obs_id"][r])
        if data["node_mask"][r]:
            if oid < 0 or oid >= n_actual:
                raise ValueError(
                    f"node_obs_id[{r}] = {oid} for a masked node, "
                    f"expected in [0, {n_actual})."
                )
            if int(data["node_rank"][r]) != r:
                raise ValueError(
                    f"Rank-indexed storage invariant: node_rank[{r}] "
                    f"= {int(data['node_rank'][r])}, expected {r}."
                )
        else:
            if oid != sentinel:
                raise ValueError(
                    f"node_obs_id[{r}] = {oid} for a padded node, "
                    f"expected sentinel {sentinel}."
                )

    # Build rank-space adjacency from successor_indices (public IDs).
    # With rank-indexed storage, row r = rank r.
    adjacency: list[list[int]] = [[] for _ in range(n_actual)]
    for r in range(n_actual):
        for k in range(k_max):
            if data["successor_mask"][r, k]:
                succ_pub = int(data["successor_indices"][r, k])
                # Map public ID back to rank
                succ_rank = int(o2r[succ_pub])
                if 0 <= succ_rank < n_actual:
                    adjacency[r].append(succ_rank)

    # --- 5. Forward-only edges ---
    for r in range(n_actual):
        for sr in adjacency[r]:
            if r >= sr:
                raise ValueError(
                    f"Forward-only violation: edge rank {r} → rank {sr} "
                    f"(expected {r} < {sr})."
                )

    # --- 6. Mandatory backbone ---
    for r in range(n_actual - 1):
        if (r + 1) not in adjacency[r]:
            raise ValueError(
                f"Mandatory backbone edge rank {r} → rank {r+1} is missing."
            )

    # --- 7a. No stranded non-terminal nodes (out-degree ≥ 1) ---
    for r in range(n_actual - 1):
        if len(adjacency[r]) < 1:
            raise ValueError(
                f"Non-terminal rank {r} has out-degree 0 (stranded node)."
            )
    # The terminal node (n_actual-1) may have out-degree 0 — that is expected.

    # --- 7b. Every non-source node has in-degree ≥ 1 ---
    in_deg = [0] * n_actual
    for r in range(n_actual):
        for sr in adjacency[r]:
            in_deg[sr] += 1
    for r in range(1, n_actual):
        if in_deg[r] < 1:
            raise ValueError(
                f"Rank {r} has in-degree 0 (unreachable from source)."
            )

    # --- 8. Public-edge correspondence ---
    # Every internal edge r→sr must correspond to
    # successor_indices[r, k] == obs_id_of(sr) for some k.
    for r in range(n_actual):
        for sr in adjacency[r]:
            expected_pub = int(r2o[sr])
            found = False
            for k in range(k_max):
                if (
                    data["successor_mask"][r, k]
                    and int(data["successor_indices"][r, k]) == expected_pub
                ):
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"Public-edge correspondence failure: edge rank {r}→{sr} "
                    f"(obs {int(r2o[r])}→{expected_pub}) not found in "
                    f"successor_indices at row {r}."
                )

    # --- 9. Global rank reachability ---
    from ehc_sn.utils.graph import shortest_path as bfs_shortest_path

    for target in range(1, n_actual):
        sp = bfs_shortest_path(adjacency, 0, target)
        if not sp:
            raise ValueError(
                f"Global reachability failure: no path from rank 0 to rank "
                f"{target}."
            )


# =============================================================================
def build_dagflow_layouts(
    version_root: Path,
    *,
    preset: str = "routing",
    n_max: int | None = None,
    max_out_degree: int | None = None,
    extra_edge_density: float | None = None,
    span_profile: str | None = None,
    n_train: int = 4000,
    n_val: int = 500,
    n_test: int = 500,
    seed: int = 42,
) -> None:
    """Build the dagflow layout dataset at *version_root*.

    Generates random DAG samples with a mandatory Hamiltonian backbone
    and controlled shortcut edges.  Nodes are stored in rank order (row
    ``r`` = rank ``r``).  Public observation IDs are a random bijection,
    preventing the model from inferring direction from numerical ID order.

    Downstream task builders receive explicit ``obs_id_to_rank`` and
    ``rank_to_obs_id`` lookup channels so they never need to reconstruct
    the rank↔observation mapping.

    The version integer is derived from the ``v<N>`` leaf of *version_root*.

    Density and dimensions are selected by *preset* (one of
    ``DAGFLOW_PRESETS``).  Individual parameters override the preset
    defaults when explicitly passed.

    The Hamiltonian backbone is mandatory and non-configurable.  All graphs
    have permuted public observation IDs (``public_id_policy: "permuted"``).

    Args:
        version_root: Destination versioned root
            (e.g. ``data/interim/dagflow/routing/v1``).  Must not
            already exist.
        preset: Named generation preset (one of ``DAGFLOW_PRESETS`` keys,
            default ``"routing"``).
        n_max: Maximum candidate nodes per batch (N).  Overrides preset.
        max_out_degree: Maximum out-degree per node (K).  Overrides preset.
        extra_edge_density: Fraction of possible extra forward edges to add
            beyond the backbone (in ``[0, 1]``).  Overrides the preset's
            density when set.
        span_profile: Named rank-span distribution profile.  One of
            ``_SPAN_PROFILES`` keys.  Overrides the preset's profile
            when set.
        n_train: Number of training samples.
        n_val: Number of validation samples.
        n_test: Number of test samples.
        seed: Deterministic base seed for reproducibility.

    Raises:
        FileExistsError: When *version_root* already exists (immutable root).
        ValueError: When the version leaf name is not ``v<integer>``, or
            when the preset name is unknown.
        RuntimeError: When preset acceptance validation fails after
            ``max_generation_attempts`` retries.
    """
    # Resolve preset with optional overrides
    resolved = _resolve_preset(
        preset,
        n_max,
        max_out_degree,
        extra_edge_density,
        span_profile,
    )
    preset_name = resolved["preset"]
    n_max = resolved["n_max"]
    max_out_degree = resolved["max_out_degree"]
    short_prob = resolved["short_prob"]
    medium_prob = resolved["medium_prob"]
    long_prob = resolved["long_prob"]
    target_edges = resolved["target_edges"]
    max_generation_attempts = 50

    version = extract_version(version_root)

    stage_params = {
        "preset": preset_name,
        "n_max": n_max,
        "max_out_degree": max_out_degree,
        "target_edges": target_edges,
        "extra_edge_density": resolved["extra_edge_density"],
        "span_profile": resolved["span_profile"],
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "seed": seed,
    }
    split_counts = {"train": n_train, "val": n_val, "test": n_test}

    seed_seq = np.random.SeedSequence(seed)
    split_rngs = dict(
        zip(
            _SPLITS,
            [np.random.default_rng(s) for s in seed_seq.spawn(len(_SPLITS))],
        )
    )

    with staging_root(version_root) as tmp:
        all_entries = []
        first_diagnostics: dict | None = None

        for split in _SPLITS:
            n = split_counts[split]
            rng = split_rngs[split]

            samples: list[dict[str, np.ndarray]] = []
            per_sample_ids: list[str] = []
            per_sample_extra: list[dict] = []

            for idx in range(n):
                # Retry loop: generate a graph and validate against preset acceptance profile.
                sample: dict[str, np.ndarray] | None = None
                sample_digest: str | None = None
                attempt_seed: int = int(rng.integers(0, 2**31))

                for attempt in range(max_generation_attempts):
                    attempt_rng = np.random.default_rng(attempt_seed)
                    candidate, digest = _generate_graph_sample(
                        n_max=n_max,
                        k_max=max_out_degree,
                        rng=attempt_rng,
                        target_edges=target_edges,
                        fixed_n_actual=True,
                        min_extra_edges_per_node=0,
                        short_prob=short_prob,
                        medium_prob=medium_prob,
                        long_prob=long_prob,
                    )
                    # Build rank-space adjacency from the candidate sample for diagnostics.
                    n_actual = int(candidate["node_mask"].sum())
                    adj: list[list[int]] = [[] for _ in range(n_actual)]
                    o2r = candidate["obs_id_to_rank"][:n_actual]
                    for r in range(n_actual):
                        for k in range(max_out_degree):
                            if candidate["successor_mask"][r, k]:
                                succ_pub = int(
                                    candidate["successor_indices"][r, k]
                                )
                                if 0 <= succ_pub < n_actual:
                                    adj[r].append(int(o2r[succ_pub]))
                    for u in range(n_actual):
                        adj[u].sort()

                    diag = compute_graph_diagnostics(
                        adj, n_actual, candidate["node_obs_id"]
                    )
                    acceptance_errors = validate_preset_acceptance_profile(
                        diag, preset_name
                    )
                    if not acceptance_errors:
                        sample = candidate
                        sample_digest = digest
                        if first_diagnostics is None:
                            first_diagnostics = diag
                        break
                    # Prepare next attempt with a new seed
                    attempt_seed = int(rng.integers(0, 2**31))

                if sample is None:
                    raise RuntimeError(
                        f"Failed to generate a valid graph for preset "
                        f"{preset_name!r} after {max_generation_attempts} "
                        f"attempts (split={split}, idx={idx})."
                    )

                samples.append(sample)
                artifact_id = (
                    f"dagflow-{preset_name}-v{version}-{split}-{idx:06d}"
                )
                per_sample_ids.append(artifact_id)
                per_sample_extra.append({"content_digest": sample_digest})

            entries = write_split(
                output_root=tmp,
                split=split,
                samples=samples,
                source=LAYOUT_FAMILY,
                channels=LAYOUT_CHANNELS,
                topology_kind="dag",
                n_states=n_max,
                extent=[n_max],
                index_kwargs={},
                per_sample_ids=per_sample_ids,
                per_sample_extra=per_sample_extra,
                sample_validator=validate_dagflow_layout_sample,
            )
            all_entries.extend(entries)

        # Record first-sample diagnostics in stage_params and manifest
        if first_diagnostics is not None:
            # Serialize lists as-is; JSON-friendly.
            stage_params["graph_diagnostics"] = {
                "total_edges": first_diagnostics["total_edges"],
                "backbone_edges": first_diagnostics["backbone_edges"],
                "extra_edges": first_diagnostics["extra_edges"],
                "realized_density": round(
                    first_diagnostics["realized_density"], 6
                ),
                "branch_nodes": first_diagnostics["branch_nodes"],
                "merge_nodes": first_diagnostics["merge_nodes"],
                "semantic_diameter": first_diagnostics["semantic_diameter"],
                "direct_pair_fraction": round(
                    first_diagnostics["direct_pair_fraction"], 4
                ),
                "multi_path_pair_fraction": round(
                    first_diagnostics["multi_path_pair_fraction"], 4
                ),
            }

        write_index_at_root(all_entries, tmp)

        write_manifest(
            tmp,
            dataset_class="layout_dataset",
            family=LAYOUT_FAMILY,
            version=version,
            channels=LAYOUT_CHANNELS,
            topology_kind="dag",
            n_states=n_max,
            extent=[n_max],
            n_samples=split_counts,
            source_id=_SOURCE_ID,
            builder="ehc_sn.data.substrate.dagflow.build_dagflow_layouts",
            seed=seed,
            stage_params=stage_params,
            preset=preset_name,
            n_max=n_max,
            max_out_degree=max_out_degree,
            target_edges=target_edges,
            extra_edge_density=resolved["extra_edge_density"],
            span_profile=resolved["span_profile"],
        )

    n_total = n_train + n_val + n_test
    print(
        f"dagflow layout dataset written to {version_root}  "
        f"({n_total} samples.)"
    )


def validate_dagflow_layout_root(root: Path) -> dict:
    """Validate a dagflow layout dataset root.

    Args:
        root: Resolved versioned dagflow layout dataset root.

    Returns:
        Parsed manifest dict.

    Raises:
        ValueError: On any contract violation.
        FileNotFoundError: When a required file is absent.
    """
    manifest = validate_version_root(root)
    if manifest.get("family") != LAYOUT_FAMILY:
        raise ValueError(
            f"Root family is {manifest.get('family')!r}, "
            f"expected {LAYOUT_FAMILY!r}."
        )
    if manifest.get("dataset_class") != "layout_dataset":
        raise ValueError(
            f"Expected dataset_class 'layout_dataset', "
            f"got {manifest.get('dataset_class')!r}."
        )

    missing_ch = set(LAYOUT_CHANNELS) - set(manifest.get("channels", []))
    if missing_ch:
        raise ValueError(
            f"Manifest missing required dagflow layout channels: "
            f"{sorted(missing_ch)}"
        )

    for split, n in manifest["n_samples"].items():
        split_dir = root / split
        arrays: dict[str, np.ndarray] = {}
        for ch in LAYOUT_CHANNELS:
            ch_file = split_dir / f"{ch}.npy"
            if not ch_file.exists():
                raise FileNotFoundError(
                    f"Missing channel '{ch}' in {split_dir}."
                )
            arrays[ch] = np.load(ch_file, mmap_mode="r")
            if arrays[ch].shape[0] != n:
                raise ValueError(
                    f"Channel '{ch}' in split '{split}' "
                    f"has {arrays[ch].shape[0]} samples, "
                    f"manifest declares {n}."
                )

        for i in range(n):
            sample = {ch: arrays[ch][i] for ch in LAYOUT_CHANNELS}
            validate_dagflow_layout_sample(sample)

    return manifest


__all__ = [
    "DAGFLOW_PRESETS",
    "LAYOUT_FAMILY",
    "LAYOUT_CHANNELS",
    "_SPAN_PROFILES",
    "build_dagflow_layouts",
    "compute_graph_diagnostics",
    "validate_dagflow_layout_root",
    "validate_dagflow_layout_sample",
    "validate_preset_acceptance_profile",
]
