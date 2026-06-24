"""MazeHard corpus diagnostics — aggregate statistics.

Computes corpus-level statistics from persisted arrays only.
Pattern mirrors ``goaltrace/diagnostics.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ehc_sn.data.manifest import read_manifest
from ehc_sn.tasks.mazehard.builder import MAZEHARD_TASK_CHANNELS
from ehc_sn.tasks.mazehard.corpus import load_split_arrays


# =============================================================================
def compute_corpus_statistics(
    root: Path,
    manifest: dict[str, Any],
    splits: list[str],
    max_samples: int = -1,
) -> dict[str, Any]:
    """Compute corpus-level statistics from persisted mazehard arrays.

    Args:
        root: Versioned corpus root.
        manifest: Parsed corpus manifest.
        splits: List of split names.
        max_samples: Max samples to check per split (default: all).

    Returns:
        Dict with per-split stats (wall density, solution length, grid extent).
    """
    stats: dict[str, Any] = {
        "per_split": {},
        "wall_density": {},
        "solution_length": {},
        "grid_extent": {},
    }

    for split in splits:
        arrays = load_split_arrays(root, split)
        if arrays is None:
            continue

        n = next(iter(arrays.values())).shape[0]
        stats["per_split"][split] = n
        n_check = min(n, max_samples) if max_samples > 0 else n

        wall_densities: list[float] = []
        sol_lengths: list[int] = []
        extents: set[tuple[int, int]] = set()

        for idx in range(n_check):
            topo = np.asarray(arrays["topology"][idx])
            sol = np.asarray(arrays["solution"][idx])
            H, W = topo.shape
            extents.add((H, W))
            wall_densities.append(float(topo.sum()) / float(H * W))
            sol_lengths.append(int((sol > 0).sum()))

        if wall_densities:
            stats["wall_density"][split] = {
                "min": float(np.min(wall_densities)),
                "mean": float(np.mean(wall_densities)),
                "median": float(np.median(wall_densities)),
                "max": float(np.max(wall_densities)),
            }
        if sol_lengths:
            stats["solution_length"][split] = {
                "min": int(np.min(sol_lengths)),
                "mean": float(np.mean(sol_lengths)),
                "median": float(np.median(sol_lengths)),
                "max": int(np.max(sol_lengths)),
            }
        if extents:
            stats["grid_extent"][split] = {
                "unique_extents": sorted(str(e) for e in extents),
                "homogeneous": len(extents) == 1,
            }

    return stats


def select_samples(
    root: Path,
    splits: list[str],
    *,
    policy: str = "random",
    n: int = 1,
    seed: int | None = None,
) -> list[tuple[str, int]]:
    """Select sample references from a mazehard corpus using the given policy.

    Policies:
        ``"random"`` — uniform random selection across splits.
        ``"stratified"`` — proportional allocation across solution-length
            buckets (short: ≤25, medium: 26-50, long: ≥51) using
            largest-remainder rounding.
        ``"longest"`` — samples with the longest solution paths.
        ``"shortest"`` — samples with the shortest solution paths.

    Args:
        root: Versioned corpus root.
        splits: Splits to search (order determines priority).
        policy: Selection policy.
        n: Maximum number of samples to return.
        seed: RNG seed for deterministic selection.

    Returns:
        List of ``(split, index)`` tuples.
    """
    valid_policies = {"random", "stratified", "longest", "shortest"}
    if policy not in valid_policies:
        raise ValueError(
            f"Unknown selection policy '{policy}'. "
            f"Valid: {', '.join(sorted(valid_policies))}."
        )

    rng = np.random.default_rng(seed)

    candidates: list[tuple[str, int, dict[str, Any]]] = []

    for split in splits:
        arrays = load_split_arrays(root, split)
        if arrays is None:
            continue
        n_total = next(iter(arrays.values())).shape[0]
        for idx in range(n_total):
            sol = np.asarray(arrays["solution"][idx])
            sol_len = int((sol > 0).sum())
            candidates.append((split, idx, {"solution_length": sol_len}))

    if not candidates:
        return []

    if policy == "random":
        indices = rng.choice(
            len(candidates), size=min(n, len(candidates)), replace=False
        )
        return [(candidates[int(i)][0], candidates[int(i)][1]) for i in indices]

    if policy == "longest":
        candidates.sort(key=lambda x: x[2]["solution_length"], reverse=True)
        return [(c[0], c[1]) for c in candidates[:n]]

    if policy == "shortest":
        candidates.sort(key=lambda x: x[2]["solution_length"])
        return [(c[0], c[1]) for c in candidates[:n]]

    # Stratified: bucket by solution length
    buckets: dict[str, list[tuple[str, int]]] = {
        "short": [],
        "medium": [],
        "long": [],
    }
    for split, idx, meta in candidates:
        sl = meta["solution_length"]
        if sl <= 25:
            buckets["short"].append((split, idx))
        elif sl <= 50:
            buckets["medium"].append((split, idx))
        else:
            buckets["long"].append((split, idx))

    # Filter empty buckets
    non_empty = {k: v for k, v in buckets.items() if v}
    if not non_empty:
        return []

    # Largest-remainder proportional allocation
    total_available = sum(len(v) for v in non_empty.values())
    bucket_alloc: dict[str, float] = {}
    for name, items in non_empty.items():
        bucket_alloc[name] = len(items) / total_available

    # Assign integer seats via largest-remainder
    seats: dict[str, int] = {}
    remainders: dict[str, float] = {}
    assigned = 0
    for name, frac in bucket_alloc.items():
        raw = frac * n
        seats[name] = int(raw)
        remainders[name] = raw - int(raw)
        assigned += seats[name]

    # Distribute remaining seats by largest remainder
    for name in sorted(remainders, key=remainders.get, reverse=True):  # type: ignore[arg-type]
        if assigned >= n:
            break
        seats[name] = seats.get(name, 0) + 1  # type: ignore[operator]
        assigned += 1

    result: list[tuple[str, int]] = []
    for name in ("short", "medium", "long"):
        items = non_empty.get(name, [])
        alloc = seats.get(name, 0)
        if alloc > 0 and items:
            chosen = rng.choice(
                len(items), size=min(alloc, len(items)), replace=False
            )
            result.extend(items[int(i)] for i in chosen)

    # Shuffle to avoid sorted-by-bucket ordering
    rng.shuffle(result)
    return result


__all__ = [
    "compute_corpus_statistics",
    "select_samples",
]
