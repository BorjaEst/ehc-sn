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


__all__ = [
    "compute_corpus_statistics",
]
