"""Arena corpus diagnostics — aggregate statistics and baselines.

Computes corpus-level statistics from persisted arrays only.
Pattern mirrors ``routebind/diagnostics.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ehc_sn.data.manifest import read_manifest
from ehc_sn.tasks.arena.corpus import load_split_arrays

# =============================================================================
# Internal
# =============================================================================


def _dist(arr: list[float | int]) -> dict:
    """Compute distribution stats for a list of values."""
    if not arr:
        return {"count": 0, "min": 0.0, "median": 0.0, "max": 0.0, "mean": 0.0}
    a = np.array(arr, dtype=np.float64)
    return {
        "count": len(a),
        "min": float(a.min()),
        "median": float(np.median(a)),
        "max": float(a.max()),
        "mean": float(a.mean()),
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
    """Compute aggregate corpus statistics for arena.

    Args:
        corpus_path: Path to the versioned corpus root.
        manifest: Parsed manifest (loaded if None).
        splits: Splits to analyze (default: all declared in manifest).
        max_samples: Max samples per split (-1 for all).

    Returns:
        Dict with keys:
            per_split, episode_length, revisit_rate, revisit_count,
            wall_density, trajectory_validity_rate.
    """
    if manifest is None:
        manifest = read_manifest(corpus_path)
    if splits is None:
        splits = list(manifest.get("n_samples", {}).keys())

    result: dict[str, Any] = {
        "per_split": {},
        "episode_length": {},
        "revisit_rate": {},
        "revisit_count": {},
        "wall_density": {},
        "trajectory_validity_rate": {},
    }

    for split in splits:
        arrays = load_split_arrays(corpus_path, split)
        if arrays is None:
            continue
        ns = next(iter(arrays.values())).shape[0]
        result["per_split"][split] = ns
        n_check = min(ns, max_samples) if max_samples > 0 else ns

        ep_lens: list[int] = []
        revisit_rates: list[float] = []
        revisit_counts: list[int] = []
        wall_densities: list[float] = []
        validity_ok = 0

        for idx in range(n_check):
            tl = int(np.asarray(arrays["trajectory_length"][idx]).flat[0])
            ep_lens.append(tl)

            rows = arrays["trajectory_row"][idx][:tl]
            cols = arrays["trajectory_col"][idx][:tl]
            coords = list(zip(rows.tolist(), cols.tolist()))
            unique_cells = len(set(coords))
            revisit_count = tl - unique_cells
            revisit_counts.append(revisit_count)
            revisit_rates.append(revisit_count / max(tl, 1))

            topo = arrays["topology"][idx]
            wall_densities.append(float((~topo).sum() / topo.size))

            # Validity: all steps within bounds, valid_step matches length
            valid_step = arrays["trajectory_valid_step"][idx]
            expected_valid = np.arange(len(valid_step)) < tl
            if np.array_equal(valid_step, expected_valid):
                validity_ok += 1

        result["episode_length"][split] = _dist(ep_lens)
        result["revisit_rate"][split] = _dist(revisit_rates)
        result["revisit_count"][split] = _dist(revisit_counts)
        result["wall_density"][split] = _dist(wall_densities)
        result["trajectory_validity_rate"][split] = {
            "valid": validity_ok,
            "total": n_check,
            "rate": validity_ok / max(n_check, 1),
        }

    return result


__all__ = [
    "compute_corpus_statistics",
]
