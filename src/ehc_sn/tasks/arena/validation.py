"""Arena artifact invariant checking.

Pattern mirrors ``routebind/validation.py`` with arena-specific checks:
- Trajectory step validity (padded positions use sentinels).
- Revisit mask consistency.
- Episode boundary correctness.
- Observation ID bounds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ehc_sn.tasks.arena.builder import (
    ARENA_TASK_CHANNELS,
    TASK_FAMILY,
    validate_arena_task_sample,
)
from ehc_sn.tasks.arena.corpus import load_split_arrays

# =============================================================================
# Validation issue type
# =============================================================================


class ArenaValidationIssue:
    """One issue found during arena validation."""

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
) -> ArenaValidationIssue:
    return ArenaValidationIssue(
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
) -> list[ArenaValidationIssue]:
    """Validate a stored arena sample, returning structured issues.

    Delegates structural invariants to ``validate_arena_task_sample()``,
    then runs additional trajectory correctness checks.

    Args:
        sample: Channel dict for one sample.
        split: Split label.
        idx: Sample index.

    Returns:
        List of issues (empty = clean).
    """
    issues: list[ArenaValidationIssue] = []

    # Structural invariants via existing builder validator.
    try:
        validate_arena_task_sample(sample)
    except ValueError as e:
        issues.append(_issue("ERROR", "structural", split, idx, str(e)))
        return issues

    traj_len = int(np.asarray(sample["trajectory_length"]).flat[0])
    valid_step = np.asarray(sample["trajectory_valid_step"])
    T = valid_step.shape[-1]

    # --- Trajectory length consistency ---
    expected_valid = np.arange(T) < traj_len
    if not np.array_equal(valid_step, expected_valid):
        issues.append(
            _issue(
                "ERROR",
                "valid_step_mismatch",
                split,
                idx,
                "trajectory_valid_step does not match (t < trajectory_length)",
            )
        )

    # --- Coordinates within bounds ---
    rows = np.asarray(sample["trajectory_row"])
    cols = np.asarray(sample["trajectory_col"])
    topology = np.asarray(sample["topology"])
    H, W = topology.shape
    for t in range(traj_len):
        r, c = int(rows[t]), int(cols[t])
        if r < 0 or r >= H or c < 0 or c >= W:
            issues.append(
                _issue(
                    "ERROR",
                    "coord_out_of_bounds",
                    split,
                    idx,
                    f"Step {t}: (r={r}, c={c}) out of bounds ({H}x{W})",
                )
            )

    # --- Observation ID at valid steps ---
    obs_ids = np.asarray(sample["trajectory_observation_id"])
    for t in range(traj_len):
        oid = int(obs_ids[t])
        if oid < 0:
            issues.append(
                _issue(
                    "WARNING",
                    "negative_obs_id",
                    split,
                    idx,
                    f"Step {t}: observation_id={oid}",
                )
            )

    # --- Revisit mask consistency ---
    if "trajectory_is_revisit" in sample:
        revisit = np.asarray(sample["trajectory_is_revisit"])
        cell_set: set[tuple[int, int]] = set()
        for t in range(traj_len):
            r, c = int(rows[t]), int(cols[t])
            cell = (r, c)
            expected_revisit = cell in cell_set
            actual_revisit = bool(revisit[t])
            if expected_revisit and not actual_revisit:
                issues.append(
                    _issue(
                        "WARNING",
                        "revisit_mask_missed",
                        split,
                        idx,
                        f"Step {t}: revisit expected at ({r},{c}) "
                        f"but mask is False",
                    )
                )
            elif not expected_revisit and actual_revisit:
                issues.append(
                    _issue(
                        "WARNING",
                        "revisit_mask_false_positive",
                        split,
                        idx,
                        f"Step {t}: revisit unexpected at ({r},{c}) "
                        f"but mask is True",
                    )
                )
            cell_set.add(cell)

    return issues


# =============================================================================
# Corpus-level validation
# =============================================================================


def validate_arena_root(root: Path) -> dict:
    """Validate an arena corpus root, returning the manifest.

    Args:
        root: Versioned corpus root.

    Returns:
        Parsed manifest dict.

    Raises:
        ValueError: On contract violation.
        FileNotFoundError: When required files are absent.
    """
    from ehc_sn.tasks.arena.builder import validate_arena_task_root

    return validate_arena_task_root(root)


def validate_all_samples(
    root: Path,
    splits: list[str] | None = None,
    max_samples: int = -1,
) -> tuple[list[ArenaValidationIssue], dict[str, int]]:
    """Run stored-sample validation over all/selected splits.

    Args:
        root: Versioned corpus root.
        splits: Splits to check (default: all from manifest).
        max_samples: Max per split (-1 = all).

    Returns:
        Tuple of (issues, sample_counts).
    """
    from ehc_sn.data.manifest import read_manifest

    manifest = read_manifest(root)
    all_splits = list(manifest.get("n_samples", {}).keys())
    if splits is None:
        splits = all_splits

    issues: list[ArenaValidationIssue] = []
    sample_counts: dict[str, int] = {}

    for split in splits:
        arrays = load_split_arrays(root, split)
        if arrays is None:
            sample_counts[split] = 0
            continue
        n = next(iter(arrays.values())).shape[0]
        n_check = min(n, max_samples) if max_samples > 0 else n
        sample_counts[split] = n_check

        # Check that all required channels exist
        missing = [ch for ch in ARENA_TASK_CHANNELS if ch not in arrays]
        if missing:
            issues.append(
                _issue(
                    "ERROR",
                    "missing_channels",
                    split,
                    -1,
                    f"Split {split}: missing channels: {missing}",
                )
            )
            continue

        for idx in range(n_check):
            sample = {ch: arrays[ch][idx] for ch in ARENA_TASK_CHANNELS}
            sample_issues = validate_stored_sample(sample, split=split, idx=idx)
            issues.extend(sample_issues)

    return issues, sample_counts


__all__ = [
    "ArenaValidationIssue",
    "validate_stored_sample",
    "validate_arena_root",
    "validate_all_samples",
]
