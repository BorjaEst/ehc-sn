"""Routebind corpus diagnostics — aggregate statistics and baselines.

Computes corpus-level distributions (route lengths, semantic lengths,
rejection counts, coverage) and baseline scores (zero-field MSE,
start-only MSE).

This module imports from the persisted arrays only — no oracle logic.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ehc_sn.data.manifest import read_manifest
from ehc_sn.tasks.routebind.contracts import CELL_WALL, ROUTEBIND_SCHEMA
from ehc_sn.tasks.routebind.decoding import (
    extract_route_from_trajectory_field,
    extract_waypoints_from_field,
)
from ehc_sn.tasks.routebind.validation import ValidationIssue, _load_split_data

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


def _spatial_bfs(
    cell_type: np.ndarray,
    start: int,
    goals: np.ndarray,
) -> list[int] | None:
    """Simple BFS over free cells to find any route to a goal position."""
    S = len(cell_type)
    width = int(math.sqrt(S))
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
            if 0 <= nr < width and 0 <= nc < width:
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
        arrays = _load_split_data(corpus_path, split)
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
                r = extract_route_from_trajectory_field(tf, sp)
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
                route = extract_route_from_trajectory_field(tf, sp)

                if route:
                    width = int(math.sqrt(S))
                    sr, sc = divmod(sp, width)
                    min_d = float("inf")
                    closest = -1
                    for g in gp:
                        gr, gc = g // width, g % width
                        d = abs(sr - gr) + abs(sc - gc)
                        if d < min_d:
                            min_d = d
                            closest = g

                    manhattan = abs(sr - route[-1] // width) + abs(
                        sc - route[-1] % width
                    )
                    detour_ratios.append(
                        len(route) / manhattan if manhattan > 0 else 1.0
                    )

                    spatial = _spatial_bfs(ct, sp, gp)
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
        "grid": (
            f"{manifest.get('grid_height', '?')}x"
            f"{manifest.get('grid_width', '?')}"
        ),
    }

    # Baselines on a subset
    zero_traj, zero_wp, start_only = [], [], []
    for split in splits[:1]:
        arrays = _load_split_data(corpus_path, split)
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


def write_summary(
    stats: dict,
    issues: list[ValidationIssue],
    path: Path,
) -> None:
    """Write a human-readable validation summary to *path*."""
    I = "  "
    L: list[str] = []
    L.append("=" * 72)
    L.append("Routebind corpus validation")
    L.append("=" * 72)
    L.append("")

    for k in ("task", "corpus", "version", "n_observations", "grid"):
        if k in stats:
            L.append(f"{I}{k}: {stats[k]}")
    L.append("")

    L.append("Samples:")
    for s, c in stats.get("per_split", {}).items():
        L.append(f"{I}{s}: {c}")
    L.append("")

    errs = [i for i in issues if i.severity == "ERROR"]
    warns = [i for i in issues if i.severity == "WARNING"]
    L.append(f"Errors: {len(errs)}")
    L.append(f"Warnings: {len(warns)}")
    L.append("")

    if errs:
        L.append("Errors (first 10):")
        for e in errs[:10]:
            L.append(
                f"{I}[{e.code}] {e.message} "
                f"(split={e.split}, idx={e.sample_index})"
            )
        if len(errs) > 10:
            L.append(f"{I}... and {len(errs) - 10} more")
        L.append("")

    for key, label in [
        ("route_length", "Route length"),
        ("semantic_length", "Semantic length"),
        ("wall_density", "Wall density"),
        ("goal_occurrences", "Goal occurrences"),
        ("duplicate_observations", "Duplicate obs"),
        ("detour_ratio", "Detour ratio"),
    ]:
        d = stats.get(key, {})
        if d.get("count", 0) > 0:
            L.append(
                f"{I}{label}: min={d['min']:.1f} median={d['median']:.1f} "
                f"max={d['max']:.1f} mean={d['mean']:.1f}"
            )

    zt = stats.get("zero_field_mse_trajectory", {})
    if zt.get("count", 0) > 0:
        L.append("")
        L.append("Baselines:")
        L.append(f"{I}zero-field traj MSE: median={zt['median']:.6f}")
        st = stats.get("start_only_mse_trajectory", {})
        L.append(f"{I}start-only traj MSE: median={st.get('median', 0):.6f}")
        if "spatial_only_exact_match_rate" in stats:
            L.append(
                f"{I}spatial-only exact: {stats['spatial_only_exact_match_rate']:.1%}"
            )
            L.append(f"{I}closest-goal: {stats['closest_goal_rate']:.1%}")

    rej = stats.get("rejection_summary", {})
    if rej:
        L.append("")
        L.append("Rejections:")
        for s, reasons in sorted(rej.items()):
            L.append(f"{I}{s}:")
            for r, c in sorted(reasons.items()):
                L.append(f"{I}{I}{r}: {c}")

    # Generation funnel (per-split)
    funnel_data = stats.get("generation_funnel", {})
    if funnel_data:
        L.append("")
        L.append("Generation Funnel:")
        for split, fdict in sorted(funnel_data.items()):
            L.append(f"{I}{split}:")
            for stage in [
                "examined",
                "unreachable",
                "ambiguous",
                "eligible",
                "reconstructed",
                "bucket_full",
                "accepted",
            ]:
                entries = fdict.get(stage, {})
                if entries:
                    total = sum(int(v) for v in entries.values())
                    L.append(f"{I}{I}{stage}: total={total}")
                    for k, v in sorted(entries.items()):
                        L.append(f"{I}{I}{I}{k}: {v}")
            for scalar_key in [
                "rejected_non_simple",
                "rejected_route_too_long",
                "rejected_trivial_waypoint",
                "rejected_invalid_next_dir",
                "rejected_target_validation",
            ]:
                val = fdict.get(scalar_key, 0)
                if val:
                    L.append(f"{I}{I}{scalar_key}: {val}")

    # Realized vs target distribution
    realized_dist = stats.get("realized_distribution", {})
    if realized_dist:
        L.append("")
        L.append("Target vs Realized Distribution (joint bucket proportions):")
        for split, rdict in sorted(realized_dist.items()):
            L.append(f"{I}{split}:")
            for k, v in sorted(rdict.items()):
                L.append(f"{I}{I}{k}: {v:.4f}")

    L.append("")
    L.append("=" * 72)
    path.write_text("\n".join(L) + "\n")


__all__ = [
    "compute_corpus_statistics",
    "write_summary",
]
