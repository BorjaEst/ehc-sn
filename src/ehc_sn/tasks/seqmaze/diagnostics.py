"""SeqMaze corpus diagnostics — aggregate statistics.

Computes corpus-level statistics from persisted arrays only.
Pattern mirrors ``goaltrace/diagnostics.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ehc_sn.data.manifest import read_manifest
from ehc_sn.tasks.seqmaze.builder import SEQMAZE_TASK_CHANNELS
from ehc_sn.tasks.seqmaze.corpus import load_split_arrays


# =============================================================================
def compute_corpus_statistics(
    root: Path,
    manifest: dict[str, Any],
    splits: list[str],
    max_samples: int = -1,
) -> dict[str, Any]:
    """Compute corpus-level statistics from persisted seqmaze arrays.

    Args:
        root: Versioned corpus root.
        manifest: Parsed corpus manifest.
        splits: List of split names.
        max_samples: Max samples to check per split (default: all).

    Returns:
        Dict with per-split stats (path length, graph diameter, out-degree).
    """
    stats: dict[str, Any] = {
        "per_split": {},
        "path_length": {},
        "n_actual": {},
    }
    n_max = manifest.get("n_max", 45)
    t_max = manifest.get("t_max", 16)
    stats["n_max"] = n_max
    stats["t_max"] = t_max
    stats["path_vocab_size"] = manifest.get("path_vocab_size", n_max + 2)

    for split in splits:
        arrays = load_split_arrays(root, split)
        if arrays is None:
            continue

        n = next(iter(arrays.values())).shape[0]
        stats["per_split"][split] = n
        n_check = min(n, max_samples) if max_samples > 0 else n

        path_lengths: list[int] = []
        n_actuals: list[int] = []

        for idx in range(n_check):
            pl = int(np.asarray(arrays["path_length"][idx]).flat[0])
            path_lengths.append(pl)
            nm = np.asarray(arrays["node_mask"][idx], dtype=bool)
            n_actuals.append(int(nm.sum()))

        if path_lengths:
            stats["path_length"][split] = {
                "min": int(np.min(path_lengths)),
                "mean": float(np.mean(path_lengths)),
                "median": float(np.median(path_lengths)),
                "max": int(np.max(path_lengths)),
            }
        if n_actuals:
            stats["n_actual"][split] = {
                "min": int(np.min(n_actuals)),
                "mean": float(np.mean(n_actuals)),
                "median": float(np.median(n_actuals)),
                "max": int(np.max(n_actuals)),
            }

    return stats


__all__ = [
    "compute_corpus_statistics",
]
