"""SeqMaze artifact invariant checking.

Three validation levels mirroring ``goaltrace/validation.py``:

1. Stored-sample validation (from persisted arrays only).
2. Corpus-level checks.

Pattern adapted from ``goaltrace/validation.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ehc_sn.data.lifecycle import validate_version_root
from ehc_sn.tasks.seqmaze.builder import (
    SEQMAZE_TASK_CHANNELS,
    TASK_FAMILY,
    validate_seqmaze_sample,
)
from ehc_sn.tasks.seqmaze.corpus import load_split_arrays

# =============================================================================
# Validation issue type
# =============================================================================


class SeqMazeValidationIssue:
    """One issue found during seqmaze validation.

    Mirrors ``goaltrace.validation.GoaltraceValidationIssue`` interface so
    reporting code can treat both uniformly.
    """

    def __init__(
        self,
        severity: str,
        code: str,
        split: str = "",
        sample_index: int = -1,
        message: str = "",
        observed: Any = None,
        expected: Any = None,
    ) -> None:
        self.severity = severity  # "ERROR", "WARNING", "INFO"
        self.code = code
        self.split = split
        self.sample_index = sample_index
        self.message = message
        self.observed = observed
        self.expected = expected


# =============================================================================
# Internal helpers
# =============================================================================


def _issue(
    severity: str,
    code: str,
    split: str = "",
    idx: int = -1,
    msg: str = "",
    obs: Any = None,
    exp: Any = None,
) -> SeqMazeValidationIssue:
    return SeqMazeValidationIssue(
        severity=severity,
        code=code,
        split=split,
        sample_index=idx,
        message=msg,
        observed=obs,
        expected=exp,
    )


# =============================================================================
# Public validators
# =============================================================================


def validate_stored_sample(
    sample: dict[str, np.ndarray],
    *,
    n_max: int = 45,
    t_max: int = 16,
    split: str = "",
    idx: int = -1,
) -> list[SeqMazeValidationIssue]:
    """Validate a stored seqmaze sample, returning structured issues.

    Delegates structural invariants to ``validate_seqmaze_sample()``
    (raises ``ValueError``), then runs additional checks.

    Args:
        sample: Channel dict for one sample.
        n_max: Maximum candidate nodes N.
        t_max: Maximum path length T.
        split: Split label (for issue reporting).
        idx: Sample index (for issue reporting).

    Returns:
        List of issues (empty = clean).
    """
    issues: list[SeqMazeValidationIssue] = []

    # Structural invariants via existing builder validator.
    try:
        validate_seqmaze_sample(sample)
    except ValueError as e:
        issues.append(_issue("ERROR", "structural", split, idx, str(e)))
        return issues

    # --- Path length bounds ---
    path_length = int(sample["path_length"])
    if path_length < 1 or path_length > t_max:
        issues.append(
            _issue(
                "ERROR",
                "path_length_bounds",
                split,
                idx,
                f"path_length={path_length} out of [1, t_max={t_max}].",
                obs=path_length,
                exp=f"[1, {t_max}]",
            )
        )

    # --- Node count ---
    node_mask = np.asarray(sample.get("node_mask", np.array([])), dtype=bool)
    n_actual = int(node_mask.sum())
    if n_actual < 2 or n_actual > n_max:
        issues.append(
            _issue(
                "WARNING",
                "node_count",
                split,
                idx,
                f"n_actual={n_actual} (n_max={n_max}).",
                obs=n_actual,
                exp=f"[2, {n_max}]",
            )
        )

    return issues


def validate_all_samples(
    root: Path,
    splits: list[str],
    max_samples: int = -1,
) -> tuple[list[SeqMazeValidationIssue], dict[str, int]]:
    """Validate all samples across the given splits.

    Args:
        root: Versioned corpus root.
        splits: Split names to validate.
        max_samples: Max samples to check per split (default: all).

    Returns:
        Tuple of (issues list, sample_counts dict).
    """
    from ehc_sn.data.manifest import read_manifest

    manifest = read_manifest(root)
    n_max = manifest.get("n_max", 45)
    t_max = manifest.get("t_max", 16)

    issues: list[SeqMazeValidationIssue] = []
    sample_counts: dict[str, int] = {}

    for split in splits:
        arrays = load_split_arrays(root, split)
        if arrays is None:
            sample_counts[split] = 0
            continue

        n = next(iter(arrays.values())).shape[0]
        sample_counts[split] = n
        n_check = min(n, max_samples) if max_samples > 0 else n

        for idx in range(n_check):
            sample = {
                ch: arrays[ch][idx]
                for ch in SEQMAZE_TASK_CHANNELS
                if ch in arrays
            }
            sample_issues = validate_stored_sample(
                sample, n_max=n_max, t_max=t_max, split=split, idx=idx
            )
            issues.extend(sample_issues)

    return issues, sample_counts


__all__ = [
    "SeqMazeValidationIssue",
    "validate_stored_sample",
    "validate_all_samples",
]
