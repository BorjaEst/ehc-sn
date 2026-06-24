"""MazeHard artifact invariant checking.

Two validation levels mirroring ``goaltrace/validation.py``:

1. Stored-sample validation (from persisted arrays only).
2. Corpus-level checks.

Pattern adapted from ``goaltrace/validation.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ehc_sn.data.lifecycle import validate_version_root
from ehc_sn.tasks.mazehard.builder import (
    MAZEHARD_TASK_CHANNELS,
    TASK_FAMILY,
    validate_mazehard_sample,
)
from ehc_sn.tasks.mazehard.corpus import load_split_arrays

# =============================================================================
# Validation issue type
# =============================================================================


class MazeHardValidationIssue:
    """One issue found during mazehard validation.

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
) -> MazeHardValidationIssue:
    return MazeHardValidationIssue(
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
    split: str = "",
    idx: int = -1,
) -> list[MazeHardValidationIssue]:
    """Validate a stored mazehard sample, returning structured issues.

    Delegates structural invariants to ``validate_mazehard_sample()``
    (raises ``ValueError``), then runs additional checks.

    Args:
        sample: Channel dict for one sample.
        split: Split label (for issue reporting).
        idx: Sample index (for issue reporting).

    Returns:
        List of issues (empty = clean).
    """
    issues: list[MazeHardValidationIssue] = []

    # Structural invariants via existing builder validator.
    try:
        validate_mazehard_sample(sample)
    except ValueError as e:
        issues.append(_issue("ERROR", "structural", split, idx, str(e)))
        return issues

    # --- Start/goal count ---
    start = np.asarray(sample.get("start", np.array([])), dtype=bool)
    goals = np.asarray(sample.get("goals", np.array([])), dtype=bool)

    n_start = int(start.sum())
    if n_start != 1:
        issues.append(
            _issue(
                "ERROR",
                "start_count",
                split,
                idx,
                f"Expected exactly 1 start cell, got {n_start}.",
                obs=n_start,
                exp=1,
            )
        )

    n_goals = int(goals.sum())
    if n_goals < 1:
        issues.append(
            _issue(
                "ERROR",
                "goal_count",
                split,
                idx,
                f"Expected at least 1 goal cell, got {n_goals}.",
                obs=n_goals,
                exp=">= 1",
            )
        )

    return issues


def validate_all_samples(
    root: Path,
    splits: list[str],
    max_samples: int = -1,
) -> tuple[list[MazeHardValidationIssue], dict[str, int]]:
    """Validate all samples across the given splits.

    Args:
        root: Versioned corpus root.
        splits: Split names to validate.
        max_samples: Max samples to check per split (default: all).

    Returns:
        Tuple of (issues list, sample_counts dict).
    """
    issues: list[MazeHardValidationIssue] = []
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
                for ch in MAZEHARD_TASK_CHANNELS
                if ch in arrays
            }
            sample_issues = validate_stored_sample(sample, split=split, idx=idx)
            issues.extend(sample_issues)

    return issues, sample_counts


__all__ = [
    "MazeHardValidationIssue",
    "validate_stored_sample",
    "validate_all_samples",
]
