"""Goaltrace corpus diagnostics — aggregate statistics and baselines.

Computes corpus-level statistics from persisted arrays only (no oracle).
Pattern mirrors ``routebind/diagnostics.py``, with content promoted from
``temp/checkdata.py``.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from ehc_sn.data.manifest import read_manifest
from ehc_sn.tasks.goaltrace.builder import GOALTRACE_CORPUS_CHANNELS
from ehc_sn.tasks.goaltrace.corpus import load_split_arrays

# =============================================================================
# Internal helpers
# =============================================================================


def _adjacency_from_stored(
    succ_idx: np.ndarray, succ_mask: np.ndarray, n: int
) -> list[list[int]]:
    """Rebuild adjacency list from stored successor channels."""
    adj: list[list[int]] = [[] for _ in range(n)]
    for u in range(n):
        for k in range(succ_mask.shape[-1]):
            if succ_mask[u, k]:
                v = int(succ_idx[u, k])
                if v < n:
                    adj[u].append(v)
    return adj


def _bfs_shortest_path(
    adj: list[list[int]], start: int, goal: int
) -> list[int] | None:
    """BFS shortest path in a DAG."""
    q: deque[int] = deque([start])
    visited = {start: None}
    while q:
        u = q.popleft()
        if u == goal:
            path: list[int] = []
            while u is not None:
                path.append(u)
                u = visited[u]
            path.reverse()
            return path
        for v in adj[u]:
            if v not in visited:
                visited[v] = u
                q.append(v)
    return None


# =============================================================================
# Corpus-level checks (promoted from checkdata.py)
# =============================================================================


def _check_decay_consistency(
    target_field: np.ndarray,
    node_mask: np.ndarray,
    field_decay: float = 0.8,
    tolerance: float = 0.02,
) -> dict:
    """Verify target field values on optimal path follow gamma^d.

    Returns dict with pass_count, total_count, failure_indices.
    """
    ns = target_field.shape[0]
    passes = 0
    failures: list[int] = []
    for i in range(ns):
        n_i = int(node_mask[i].sum())
        tf = target_field[i, :n_i]
        on_path = np.where(tf > 0)[0]
        if len(on_path) == 0:
            continue
        order = np.argsort(-tf[on_path])
        path_nodes = on_path[order].tolist()
        ok = True
        for d, node in enumerate(path_nodes):
            expected = float(field_decay**d)
            actual = float(tf[node])
            if abs(actual - expected) > tolerance:
                ok = False
                break
        if ok:
            passes += 1
        else:
            failures.append(i)
    return {"passes": passes, "total": ns, "failure_indices": failures[:10]}


def _check_input_uniqueness(
    arrays: dict[str, np.ndarray], check_n: int = 4000
) -> dict:
    """Hash distinct (current, goal, weight) tuples and detect contradictory targets.

    Returns dict with unique_keys, duplicate_keys, contradictions.
    """
    target = arrays.get("target_field", None)
    if target is None:
        return {"unique_keys": 0, "duplicate_keys": 0, "contradictions": 0}

    mask = arrays["node_mask"]
    cur_f = arrays["current_flag"]
    goal_f = arrays["goal_flag"]
    target = arrays["target_field"]
    w = arrays["weight"]
    ns = min(check_n, target.shape[0])

    seen: dict[tuple, np.ndarray] = {}
    duplicates = 0
    contradictions = 0

    for i in range(ns):
        n_i = int(mask[i].sum())
        cur = int(cur_f[i, :n_i].argmax())
        g = int(goal_f[i, :n_i].argmax())
        w_key = tuple(np.round(w[i, :n_i].astype(np.float64), 6))
        m_key = tuple(mask[i, :n_i].tolist())
        key = (cur, g, w_key, m_key)

        if key in seen:
            duplicates += 1
            if not np.allclose(seen[key], target[i, :n_i], atol=1e-5):
                contradictions += 1
        else:
            seen[key] = target[i, :n_i].copy()

    return {
        "unique_keys": len(seen),
        "duplicate_keys": duplicates,
        "contradictions": contradictions,
    }


def _compute_weight_overlap(
    arrays: dict[str, np.ndarray],
    adjacency: list[list[int]] | None,
    check_n: int = 500,
) -> dict:
    """Compute edge vs non-edge weight distributions and overlap coefficient.

    Returns dict with edge_mean, edge_std, noedge_mean, noedge_std, overlap,
    dense_dag flags.
    """
    succ_idx = arrays.get("successor_indices", None)
    succ_mask = arrays.get("successor_mask", None)
    node_mask = arrays["node_mask"]
    cur_f = arrays["current_flag"]
    weights = arrays["weight"]
    ns = min(check_n, weights.shape[0])

    w_edge_vals: list[float] = []
    w_noedge_vals: list[float] = []

    for i in range(ns):
        n_i = int(node_mask[i].sum())
        if succ_idx is not None:
            adj_i = _adjacency_from_stored(
                succ_idx[i, :n_i], succ_mask[i, :n_i], n_i
            )
        else:
            adj_i = adjacency or [[] for _ in range(n_i)]
        cur = int(cur_f[i, :n_i].argmax())
        for j in range(n_i):
            if j == cur:
                continue
            wv = float(weights[i, j])
            if j in adj_i[cur]:
                w_edge_vals.append(wv)
            else:
                w_noedge_vals.append(wv)

    w_edge = np.array(w_edge_vals) if w_edge_vals else np.array([0.0])
    w_noedge = np.array(w_noedge_vals) if w_noedge_vals else np.array([0.0])

    bins = np.linspace(0, 1, 51)
    he, _ = np.histogram(w_edge, bins=bins, density=True)
    hn, _ = np.histogram(w_noedge, bins=bins, density=True)
    overlap = float(
        np.clip(np.sum(np.minimum(he, hn)) * (bins[1] - bins[0]), 0, 1)
    )

    # Detect dense DAG: all forward pairs are edges
    dense_dag = False
    if adjacency and len(adjacency) > 1:
        n = len(adjacency)
        dense_dag = all(len(adjacency[u]) == n - 1 - u for u in range(n))

    return {
        "edge_mean": float(np.mean(w_edge)),
        "edge_std": float(np.std(w_edge)),
        "noedge_mean": float(np.mean(w_noedge)),
        "noedge_std": float(np.std(w_noedge)),
        "overlap": overlap,
        "dense_dag": dense_dag,
    }


def _check_reachable_pairs(
    adjacency: list[list[int]] | None, n_real: int
) -> dict:
    """Count ordered (u,v) pairs with u<v where a directed path exists."""
    if adjacency is None:
        return {"reachable": 0, "total": 0}
    total = n_real * (n_real - 1) // 2
    count = 0
    for u in range(n_real):
        for v in range(u + 1, n_real):
            if _bfs_shortest_path(adjacency, u, v):
                count += 1
    return {"reachable": count, "total": total}


# =============================================================================
# Baseline computation
# =============================================================================


def _compute_baselines(arrays: dict[str, np.ndarray]) -> dict:
    """Compute baseline MSE values for a split.

    Returns dict with zero, per_sample_mean, current_only, random_uniform.
    """
    target = arrays["target_field"]
    mask = arrays["node_mask"]
    vc = mask.sum(axis=1)
    mse0 = (target**2 * mask).sum(axis=1) / vc
    pm = (target * mask).sum(axis=1) / vc
    mse_m = (((pm[:, None] * mask) - target) ** 2 * mask).sum(axis=1) / vc
    pc = arrays["current_flag"].astype(np.float32)
    mse_c = ((pc - target) ** 2 * mask).sum(axis=1) / vc
    mse_r = ((1 / 3 - target + target**2) * mask).sum(axis=1) / vc
    return {
        "zero": float(mse0.mean()),
        "per_sample_mean": float(mse_m.mean()),
        "current_only": float(mse_c.mean()),
        "random_uniform": float(mse_r.mean()),
    }


# =============================================================================
# Public API
# =============================================================================


def compute_corpus_statistics(
    corpus_path: Path,
    manifest: dict | None = None,
    splits: list[str] | None = None,
    max_samples: int = -1,
) -> dict:
    """Compute aggregate corpus statistics for goaltrace.

    Args:
        corpus_path: Path to the versioned corpus root.
        manifest: Parsed manifest (loaded if None).
        splits: Splits to analyze (default: all declared in manifest).
        max_samples: Max samples per split (-1 for all).

    Returns:
        Dict with keys:
            per_split, edge_density, reachable_pairs, weight_overlap,
            decay_consistency, input_uniqueness, baselines, dense_dag.
    """
    if manifest is None:
        manifest = read_manifest(corpus_path)
    if splits is None:
        splits = list(manifest.get("n_samples", {}).keys())

    n_obs = manifest.get("n_observations", 0)
    field_decay = manifest.get("field_decay", 0.8)

    result: dict[str, Any] = {
        "per_split": {},
        "edge_density": {},
        "reachable_pairs": {},
        "weight_overlap": {},
        "decay_consistency": {},
        "input_uniqueness": {},
        "baselines": {},
        "dense_dag": {},
    }

    for split in splits:
        arrays = load_split_arrays(corpus_path, split)
        if arrays is None:
            continue
        ns = next(iter(arrays.values())).shape[0]
        result["per_split"][split] = ns

        # Constrain samples
        ns_check = min(ns, max_samples) if max_samples > 0 else ns
        clipped: dict[str, np.ndarray] = {}
        for ch, arr in arrays.items():
            if arr.shape[0] >= ns_check:
                clipped[ch] = arr[:ns_check]
            else:
                clipped[ch] = arr

        node_mask = clipped["node_mask"]
        n_real = int(node_mask[0].sum())

        # Adjacency from first sample
        succ_idx = clipped.get("successor_indices")
        succ_mask = clipped.get("successor_mask")
        adjacency: list[list[int]] | None = None
        if succ_idx is not None and succ_mask is not None:
            adjacency = _adjacency_from_stored(
                succ_idx[0], succ_mask[0], n_real
            )

        # Edge density
        if adjacency:
            total_edges = sum(len(a) for a in adjacency)
            max_edges = n_real * (n_real - 1) / 2
            result["edge_density"][split] = (
                total_edges / max_edges if max_edges > 0 else 0.0
            )
            result["reachable_pairs"][split] = _check_reachable_pairs(
                adjacency, n_real
            )
        else:
            result["edge_density"][split] = 0.0
            result["reachable_pairs"][split] = {"reachable": 0, "total": 0}

        # Decay consistency
        result["decay_consistency"][split] = _check_decay_consistency(
            clipped["target_field"], node_mask, field_decay
        )

        # Input uniqueness
        result["input_uniqueness"][split] = _check_input_uniqueness(
            clipped, check_n=ns_check
        )

        # Weight overlap
        result["weight_overlap"][split] = _compute_weight_overlap(
            clipped, adjacency, check_n=ns_check
        )

        # Dense DAG flag
        result["dense_dag"][split] = result["weight_overlap"][split][
            "dense_dag"
        ]

        # Baselines
        result["baselines"][split] = _compute_baselines(clipped)

    return result


def select_samples(
    corpus_path: Path,
    splits: list[str],
    *,
    policy: str = "random",
    n: int = 1,
    seed: int | None = None,
) -> list[tuple[str, int]]:
    """Select sample references from a goaltrace corpus using the given policy.

    Policies:
        ``"random"`` — uniform random selection across splits.
        ``"stratified"`` — proportional allocation across path-length buckets
            (short: ≤4 hops, medium: 5-10 hops, long: ≥11 hops) using
            largest-remainder rounding.
        ``"longest_path"`` — samples with the most hops on the optimal path.
        ``"shortest_path"`` — samples with the fewest hops on the optimal path.

    Args:
        corpus_path: Versioned corpus root.
        splits: Splits to search (order determines priority).
        policy: Selection policy.
        n: Maximum number of samples to return.
        seed: RNG seed for deterministic selection.

    Returns:
        List of ``(split, index)`` tuples.
    """
    valid_policies = {"random", "stratified", "longest_path", "shortest_path"}
    if policy not in valid_policies:
        raise ValueError(
            f"Unknown selection policy '{policy}'. "
            f"Valid: {', '.join(sorted(valid_policies))}."
        )

    rng = np.random.default_rng(seed)

    candidates: list[tuple[str, int, dict]] = []

    for split in splits:
        arrays = load_split_arrays(corpus_path, split)
        if arrays is None:
            continue
        n_total = next(iter(arrays.values())).shape[0]
        for idx in range(n_total):
            tf = np.asarray(arrays["target_field"][idx])
            nm = np.asarray(arrays["node_mask"][idx])
            n_valid = int(nm.sum())
            path_nodes = np.where(tf[:n_valid] > 0)[0]
            path_len = max(len(path_nodes) - 1, 0)  # hop count
            candidates.append((split, idx, {"path_len": path_len}))

    if not candidates:
        return []

    if policy == "random":
        indices = rng.choice(
            len(candidates), size=min(n, len(candidates)), replace=False
        )
        return [(candidates[int(i)][0], candidates[int(i)][1]) for i in indices]

    if policy == "stratified":
        buckets: dict[int, list[tuple[str, int]]] = {}
        for split, idx, meta in candidates:
            pl = meta["path_len"]
            bucket = 0 if pl <= 4 else (1 if pl <= 10 else 2)
            buckets.setdefault(bucket, []).append((split, idx))

        total = sum(len(v) for v in buckets.values())
        n_actual = min(n, total)
        bk_sorted = sorted(buckets.keys())

        raw = {bk: len(buckets[bk]) * n_actual / total for bk in bk_sorted}
        alloc = {bk: int(raw[bk]) for bk in bk_sorted}
        remainder = n_actual - sum(alloc.values())
        for bk in sorted(raw, key=lambda b: raw[b] - int(raw[b]), reverse=True):
            if remainder <= 0:
                break
            alloc[bk] += 1
            remainder -= 1

        selected: list[tuple[str, int]] = []
        for bk in bk_sorted:
            pool = buckets[bk]
            rng.shuffle(pool)
            selected.extend(pool[: alloc[bk]])
        return selected

    reverse = policy in ("longest_path",)
    candidates.sort(key=lambda x: x[2].get("path_len", 0), reverse=reverse)
    return [(c[0], c[1]) for c in candidates[:n]]


__all__ = [
    "compute_corpus_statistics",
    "select_samples",
]
