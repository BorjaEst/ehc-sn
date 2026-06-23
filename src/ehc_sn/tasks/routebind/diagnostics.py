"""Routebind corpus diagnostics — aggregate statistics and baselines.

Computes corpus-level distributions (route lengths, semantic lengths,
rejection counts, coverage) and baseline scores (zero-field MSE,
start-only MSE).

This module imports from the persisted arrays only — no oracle logic.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from ehc_sn.data.manifest import read_manifest
from ehc_sn.tasks.routebind.contracts import CELL_WALL, ROUTEBIND_SCHEMA
from ehc_sn.tasks.routebind.corpus import load_split_arrays
from ehc_sn.tasks.routebind.decoding import (
    extract_route_from_trajectory_field,
    extract_waypoints_from_field,
)

# Channel names
_TRAJECTORY = ROUTEBIND_SCHEMA.target_trajectory
_WAYPOINT = ROUTEBIND_SCHEMA.target_waypoint
_CELL_TYPE = ROUTEBIND_SCHEMA.cell_type
_START_FLAG = ROUTEBIND_SCHEMA.start_flag
_GOAL_FLAG = ROUTEBIND_SCHEMA.goal_flag
_OBSERVATION_ID = ROUTEBIND_SCHEMA.observation_id


# =============================================================================
# Internal
# =============================================================================


def _iter_samples(
    arrays: dict[str, np.ndarray],
    split: str,
    max_samples: int = -1,
):
    n = next(iter(arrays.values())).shape[0]
    if max_samples > 0:
        n = min(n, max_samples)
    for i in range(n):
        yield i, {
            ch: arrays[ch][i]
            for ch in ROUTEBIND_SCHEMA.all_channels
            if ch in arrays
        }, split


def _format_grid(manifest: dict) -> str:
    """Derive a ``"HxW"`` grid string from the manifest.

    Checks ``extent`` first (canonical v2+), then ``canvas_height``/
    ``canvas_width`` (builder extra fields), then legacy
    ``grid_height``/``grid_width``, falling back to ``"?x?"``.
    """
    ext = manifest.get("extent")
    if ext and len(ext) == 2:
        return f"{ext[0]}x{ext[1]}"
    h = manifest.get("canvas_height") or manifest.get("grid_height")
    w = manifest.get("canvas_width") or manifest.get("grid_width")
    if h and w:
        return f"{h}x{w}"
    return "?x?"


def _spatial_bfs(
    cell_type: np.ndarray,
    start: int,
    goals: np.ndarray,
    grid_width: int | None = None,
) -> list[int] | None:
    """Simple BFS over free cells to find any route to a goal position.

    Args:
        cell_type: ``(S,)`` int32 cell type codes.
        start: Start position index.
        goals: Array of goal position indices.
        grid_width: Grid width in columns.  When ``None``, assumes a square
            grid and infers width from ``sqrt(S)``.

    Returns:
        List of positions from start to a goal, or ``None`` if unreachable.
    """
    S = len(cell_type)
    if grid_width is not None:
        width = grid_width
        height = S // width
        if height * width != S:
            return None
    else:
        width = int(math.sqrt(S))
        if width * width != S:
            return None
        height = width
    gset = {int(p) for p in goals}
    visited = {start}
    q: list[tuple[int, list[int]]] = [(start, [start])]
    while q:
        pos, path = q.pop(0)
        if pos in gset:
            return path
        r, c = divmod(pos, width)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                npos = nr * width + nc
                if npos not in visited and int(cell_type[npos]) != CELL_WALL:
                    visited.add(npos)
                    q.append((npos, path + [npos]))
    return None


def _dist(arr: list[float | int]) -> dict:
    if not arr:
        return {"count": 0}
    a = np.array(arr)
    return {
        "min": float(a.min()),
        "max": float(a.max()),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "count": len(arr),
    }


# =============================================================================
# Public API
# =============================================================================


def compute_corpus_statistics(
    corpus_path: Path,
    manifest: dict | None = None,
    splits: list[str] | None = None,
    max_samples: int = -1,
    grid_width: int | None = None,
) -> dict:
    """Compute aggregate corpus statistics.

    Args:
        corpus_path: Path to the versioned corpus root.
        manifest: Parsed manifest (loaded if None).
        splits: Splits to analyze (default: all declared in manifest).
        max_samples: Max samples per split (-1 for all).

    Returns:
        Dict with keys:
            per_split, route_length, semantic_length, wall_density,
            goal_occurrences, duplicate_observations, detour_ratio,
            zero_field_mse_trajectory, zero_field_mse_waypoint,
            start_only_mse_trajectory, rejection_summary.
    """
    if manifest is None:
        manifest = read_manifest(corpus_path)

    if splits is None:
        splits = list(manifest.get("n_samples", {}).keys())

    route_lens: list[int] = []
    semantic_lens: list[int] = []
    wall_dens: list[float] = []
    goal_occs: list[int] = []
    dup_counts: list[int] = []
    split_counts: dict[str, int] = {}
    rejection_summary: dict[str, dict[str, int]] = {}

    n_spatial_exact = 0
    n_closest_goal = 0
    n_checked = 0
    detour_ratios: list[float] = []

    for split in splits:
        arrays = load_split_arrays(corpus_path, split)
        if arrays is None:
            split_counts[split] = 0
            continue
        n = next(iter(arrays.values())).shape[0]
        split_counts[split] = n

        for _, sample, _ in _iter_samples(arrays, split, max_samples):
            ct = np.asarray(sample[_CELL_TYPE])
            tf = np.asarray(sample[_TRAJECTORY])
            wf = np.asarray(sample[_WAYPOINT])
            sf = np.asarray(sample[_START_FLAG])
            gf = np.asarray(sample[_GOAL_FLAG])
            oid = np.asarray(sample[_OBSERVATION_ID])
            S = len(ct)

            if int(sf.sum()) == 1:
                sp = int(np.argmax(sf))
                r = extract_route_from_trajectory_field(
                    tf, sp, grid_width=grid_width
                )
                route_lens.append(len(r))
                wps = extract_waypoints_from_field(wf, r)
                semantic_lens.append(len(wps))

            wall_dens.append(float((ct == CELL_WALL).mean()))
            goal_occs.append(int(gf.sum()))

            valid_obs = oid[(ct != CELL_WALL) & (oid >= 0)]
            if len(valid_obs) > 0:
                dup_counts.append(len(valid_obs) - len(set(valid_obs.tolist())))

            # Degeneracy analysis
            if int(sf.sum()) == 1 and int(gf.sum()) >= 1:
                n_checked += 1
                sp = int(np.argmax(sf))
                gp = np.where(gf)[0]
                route = extract_route_from_trajectory_field(
                    tf, sp, grid_width=grid_width
                )

                if route:
                    w = (
                        grid_width
                        if grid_width is not None
                        else int(math.sqrt(S))
                    )
                    sr, sc = divmod(sp, w)
                    min_d = float("inf")
                    closest = -1
                    for g in gp:
                        gr, gc = g // w, g % w
                        d = abs(sr - gr) + abs(sc - gc)
                        if d < min_d:
                            min_d = d
                            closest = g

                    manhattan = abs(sr - route[-1] // w) + abs(
                        sc - route[-1] % w
                    )
                    detour_ratios.append(
                        len(route) / manhattan if manhattan > 0 else 1.0
                    )

                    spatial = _spatial_bfs(ct, sp, gp, grid_width=grid_width)
                    if spatial:
                        n_spatial_exact += 1 if spatial == route else 0
                    n_closest_goal += 1 if route[-1] == closest else 0

    # Rejection stats from manifest stage_params
    for key, val in manifest.get("stage_params", {}).items():
        if key.startswith("rejected_"):
            parts = key.split("_", 2)
            if len(parts) >= 3:
                s, r = parts[1], "_".join(parts[2:])
                rejection_summary.setdefault(s, {})[r] = int(val)

    # Funnel and distribution stats from stage_params
    funnel_data: dict[str, dict] = {}
    realized_dist: dict[str, dict] = {}
    for key, val in manifest.get("stage_params", {}).items():
        if key.startswith("funnel_"):
            split = key[len("funnel_") :]
            funnel_data[split] = val
        if key.startswith("realized_"):
            split = key[len("realized_") :]
            realized_dist[split] = val

    stats: dict[str, Any] = {
        "task": manifest.get("task", "routebind"),
        "corpus": manifest.get("corpus", "default"),
        "version": manifest.get("version", 1),
        "n_observations": manifest.get("n_observations", 0),
        "grid": (_format_grid(manifest)),
    }

    # Baselines on a subset
    zero_traj, zero_wp, start_only = [], [], []
    for split in splits[:1]:
        arrays = load_split_arrays(corpus_path, split)
        if arrays is None:
            continue
        for _, s, _ in _iter_samples(
            arrays, split, min(100, max(10, max_samples // 10))
        ):
            tf = np.asarray(s[_TRAJECTORY])
            wf = np.asarray(s[_WAYPOINT])
            zero_traj.append(float((tf**2).mean()))
            zero_wp.append(float((wf**2).mean()))
            sf = np.asarray(s[_START_FLAG])
            if int(sf.sum()) == 1:
                pred = np.zeros(len(tf), dtype=np.float32)
                pred[int(np.argmax(sf))] = 1.0
                start_only.append(float(((pred - tf) ** 2).mean()))

    stats["per_split"] = split_counts
    stats["rejection_summary"] = rejection_summary
    stats["generation_funnel"] = funnel_data
    stats["realized_distribution"] = realized_dist
    stats["route_length"] = _dist(route_lens)
    stats["semantic_length"] = _dist(semantic_lens)
    stats["wall_density"] = _dist(wall_dens)
    stats["goal_occurrences"] = _dist(goal_occs)
    stats["duplicate_observations"] = _dist(dup_counts)
    stats["detour_ratio"] = _dist(detour_ratios)
    stats["zero_field_mse_trajectory"] = _dist(zero_traj)
    stats["zero_field_mse_waypoint"] = _dist(zero_wp)
    stats["start_only_mse_trajectory"] = _dist(start_only)
    if n_checked > 0:
        stats["spatial_only_exact_match_rate"] = n_spatial_exact / n_checked
        stats["closest_goal_rate"] = n_closest_goal / n_checked

    return stats


def select_samples(
    corpus_path: Path,
    splits: list[str],
    *,
    policy: str = "random",
    n: int = 1,
    seed: int | None = None,
    grid_width: int | None = None,
) -> list[tuple[str, int]]:
    """Select sample references from a corpus using the given policy.

    Policies:
        ``"random"`` — uniform random selection.
        ``"stratified"`` — proportional allocation across joint (physical,
            semantic) buckets using largest-remainder rounding.
            Returns exactly ``min(n, total_candidates)`` samples.
        ``"longest_route"`` — samples with the longest extracted physical route.
        ``"shortest_route"`` — samples with the shortest extracted physical route.
        ``"most_waypoints"`` — samples with the most semantic waypoints.
        ``"most_goal_occurrences"`` — samples with the most goal occurrences.
        ``"largest_spatial_detour"`` — samples with the largest detour ratio.

    Args:
        corpus_path: Versioned corpus root.
        splits: Splits to search (order determines priority).
        policy: Selection policy.
        n: Maximum number of samples to return.
        seed: RNG seed for deterministic selection ("random" and "stratified").
        grid_width: Grid width in columns for route extraction.  When ``None``
            (default), the route extractor falls back to ``sqrt(S)``
            (square-grid assumption).

    Returns:
        List of ``(split, index)`` tuples.
    """
    rng = np.random.default_rng(seed)

    candidates: list[tuple[str, int, dict]] = []

    for split in splits:
        arrays = load_split_arrays(corpus_path, split)
        if arrays is None:
            continue
        n_total = next(iter(arrays.values())).shape[0]
        for idx in range(n_total):
            sample = {
                ch: arrays[ch][idx]
                for ch in ROUTEBIND_SCHEMA.all_channels
                if ch in arrays
            }
            ct = np.asarray(sample[_CELL_TYPE])
            sf = np.asarray(sample[_START_FLAG])
            gf = np.asarray(sample[_GOAL_FLAG])
            tf = np.asarray(sample[_TRAJECTORY])
            wf = np.asarray(sample[_WAYPOINT])

            meta: dict = {}
            if int(sf.sum()) == 1:
                sp = int(np.argmax(sf))
                route = extract_route_from_trajectory_field(
                    tf, sp, grid_width=grid_width
                )
                meta["route_len"] = len(route) if route else 0
                wps = extract_waypoints_from_field(wf, route) if route else []
                meta["waypoint_count"] = len(wps)
                meta["goal_occurrences"] = int(gf.sum())
                if route and meta["route_len"] >= 2:
                    S = len(ct)
                    width = int(math.sqrt(S))
                    sr, sc = divmod(sp, width)
                    manhattan = abs(sr - route[-1] // width) + abs(
                        sc - route[-1] % width
                    )
                    meta["detour"] = (
                        meta["route_len"] / manhattan if manhattan > 0 else 1.0
                    )
                else:
                    meta["detour"] = 1.0
            candidates.append((split, idx, meta))

    if not candidates:
        return []

    if policy == "random":
        indices = rng.choice(
            len(candidates), size=min(n, len(candidates)), replace=False
        )
        return [(candidates[int(i)][0], candidates[int(i)][1]) for i in indices]

    if policy == "stratified":
        # Group by buckets
        buckets: dict[tuple[int, int], list[tuple[str, int]]] = {}
        for split, idx, meta in candidates:
            rl = meta.get("route_len", 0)
            wc = meta.get("waypoint_count", 0)
            # Simple coarse bins: physical in [1,10],[11,30],[31,150]; semantic in [1,2],[3,5],[6,20]
            pb = 0 if rl <= 10 else (1 if rl <= 30 else 2)
            sb = 0 if wc <= 2 else (1 if wc <= 5 else 2)
            buckets.setdefault((pb, sb), []).append((split, idx))

        total = sum(len(v) for v in buckets.values())
        n_actual = min(n, total)
        bk_sorted = sorted(buckets.keys())

        # Proportional allocation with largest-remainder rounding (Hamilton)
        raw = {bk: len(buckets[bk]) * n_actual / total for bk in bk_sorted}
        alloc = {bk: int(raw[bk]) for bk in bk_sorted}
        remainder = n_actual - sum(alloc.values())
        # Distribute remainders — one each to buckets with largest fractional parts
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

    # Sorting-based policies
    reverse = policy in (
        "longest_route",
        "most_waypoints",
        "most_goal_occurrences",
        "largest_spatial_detour",
    )
    sort_key_map = {
        "longest_route": "route_len",
        "shortest_route": "route_len",
        "most_waypoints": "waypoint_count",
        "most_goal_occurrences": "goal_occurrences",
        "largest_spatial_detour": "detour",
    }
    key_name = sort_key_map.get(policy, "route_len")
    candidates.sort(key=lambda x: x[2].get(key_name, 0), reverse=reverse)
    return [(c[0], c[1]) for c in candidates[:n]]


__all__ = [
    "compute_corpus_statistics",
    "select_samples",
]
