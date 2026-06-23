"""Goaltrace artifact invariant checking.

Three validation levels mirroring ``routebind/validation.py``:

1. Generated-sample validation (oracle metadata available).
2. Stored-sample validation (from persisted arrays only).
3. Corpus-level checks.

Pattern adapted from ``routebind/validation.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ehc_sn.data.lifecycle import validate_version_root
from ehc_sn.tasks.goaltrace.builder import (
    GOALTRACE_CORPUS_CHANNELS,
    TASK_FAMILY,
    validate_goaltrace_sample,
)
from ehc_sn.tasks.goaltrace.corpus import load_split_arrays

# =============================================================================
# Validation issue type
# =============================================================================


class GoaltraceValidationIssue:
    """One issue found during goaltrace validation.

    Mirrors ``routebind.validation.ValidationIssue`` interface so reporting
    code can treat both uniformly.
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
) -> GoaltraceValidationIssue:
    return GoaltraceValidationIssue(
        severity=severity,
        code=code,
        split=split,
        sample_index=idx,
        message=msg,
        observed=obs,
        expected=exp,
    )


# =============================================================================
# Public validators — stored-sample checks
# =============================================================================


def validate_stored_sample(
    sample: dict[str, np.ndarray],
    n_observations: int,
    field_decay: float,
    *,
    split: str = "",
    idx: int = -1,
) -> list[GoaltraceValidationIssue]:
    """Validate a stored goaltrace sample, returning structured issues.

    Delegates structural invariants to ``validate_goaltrace_sample()``
    (raises ``ValueError``), then runs additional field-level checks.

    Args:
        sample: Channel dict for one sample.
        n_observations: Padded observation count N.
        field_decay: Field decay factor gamma.
        split: Split label (for issue reporting).
        idx: Sample index (for issue reporting).

    Returns:
        List of issues (empty = clean).
    """
    issues: list[GoaltraceValidationIssue] = []

    # Structural invariants via existing builder validator.
    try:
        validate_goaltrace_sample(sample)
    except ValueError as e:
        issues.append(_issue("ERROR", "structural", split, idx, str(e)))
        return issues

    # --- Field-decay consistency ---
    target = np.asarray(sample["target_field"])
    node_mask = np.asarray(sample["node_mask"])
    n_actual = int(node_mask.sum())
    tf = target[:n_actual]

    on_path = np.where(tf > 0)[0]
    if len(on_path) > 0:
        order = np.argsort(-tf[on_path])
        path_nodes = on_path[order].tolist()
        ok = True
        for d, node in enumerate(path_nodes):
            expected = float(field_decay**d)
            actual = float(tf[node])
            if abs(actual - expected) > 0.02:
                ok = False
                break
        if not ok:
            issues.append(
                _issue(
                    "WARNING",
                    "decay_inconsistency",
                    split,
                    idx,
                    "Target field values deviate from gamma^d pattern",
                )
            )

    # --- Current location anchored at 1.0 ---
    current_idx = int(np.asarray(sample["current_flag"]).argmax())
    current_val = float(tf[current_idx])
    if abs(current_val - 1.0) > 1e-6:
        issues.append(
            _issue(
                "ERROR",
                "current_not_one",
                split,
                idx,
                f"Current location target = {current_val}, expected 1.0",
                current_val,
                1.0,
            )
        )

    # --- Weight bounds for real nodes ---
    weights = np.asarray(sample["weight"])
    masked_w = weights[node_mask]
    if np.any(masked_w < 0.0) or np.any(masked_w > 1.0):
        issues.append(
            _issue(
                "ERROR",
                "weight_out_of_bounds",
                split,
                idx,
                f"Weights outside [0, 1]: min={masked_w.min()}, "
                f"max={masked_w.max()}",
            )
        )

    # --- Non-edge weights should be zero for model input ---
    # In the stored corpus, non-edge positions for the current row get 0.
    # Check that they are not NaN or negative.
    padded_w = weights[~node_mask] if (~node_mask).sum() > 0 else np.array([])
    if len(padded_w) > 0 and np.any(padded_w < 0):
        issues.append(
            _issue(
                "WARNING",
                "padding_weight_negative",
                split,
                idx,
                f"Padding weights contain negative values: "
                f"min={padded_w.min()}",
            )
        )

    return issues


# =============================================================================
# Public validators — corpus-level
# =============================================================================


def validate_goaltrace_root(root: Path) -> dict:
    """Validate a goaltrace corpus root, returning the manifest.

    Re-exports the builder's validator for CLI convenience.

    Raises:
        ValueError: On contract violation.
        FileNotFoundError: When required files are absent.
    """
    from ehc_sn.tasks.goaltrace.builder import validate_goaltrace_root as _vgr

    return _vgr(root)


# =============================================================================
# Batch-level: run stored-sample checks over many samples
# =============================================================================


def validate_all_samples(
    root: Path,
    splits: list[str] | None = None,
    max_samples: int = -1,
) -> tuple[list[GoaltraceValidationIssue], dict[str, int]]:
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
    n_obs = manifest.get("n_observations", 0)
    field_decay = manifest.get("field_decay", 0.8)
    all_splits = list(manifest.get("n_samples", {}).keys())
    if splits is None:
        splits = all_splits

    issues: list[GoaltraceValidationIssue] = []
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
        missing = [ch for ch in GOALTRACE_CORPUS_CHANNELS if ch not in arrays]
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
            sample = {ch: arrays[ch][idx] for ch in GOALTRACE_CORPUS_CHANNELS}
            sample_issues = validate_stored_sample(
                sample, n_obs, field_decay, split=split, idx=idx
            )
            issues.extend(sample_issues)

    return issues, sample_counts


__all__ = [
    "GoaltraceValidationIssue",
    "validate_stored_sample",
    "validate_goaltrace_root",
    "validate_all_samples",
]
