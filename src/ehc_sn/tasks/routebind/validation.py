"""Routebind artifact invariant checking.

Separates three validation levels:
1. Generated-sample validation (oracle metadata available).
2. Stored-sample validation (from persisted arrays only).
3. Corpus-root validation (cross-file and cross-sample checks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ehc_sn.data.lifecycle import validate_version_root
from ehc_sn.tasks.routebind.contracts import (
    CELL_FREE,
    CELL_OBSERVATION,
    CELL_PAD,
    CELL_WALL,
    ROUTEBIND_SCHEMA,
)
from ehc_sn.tasks.routebind.corpus import load_split_arrays
from ehc_sn.tasks.routebind.oracle import OracleResult
from ehc_sn.tasks.routebind.targets import validate_decay_consistency

# =============================================================================
# Validation issue type
# =============================================================================


@dataclass
class ValidationIssue:
    """One issue found during validation."""

    severity: str  # "ERROR", "WARNING", "INFO"
    code: str
    split: str = ""
    sample_index: int = -1
    message: str = ""
    observed: Any = None
    expected: Any = None


# =============================================================================
# Internal helpers
# =============================================================================


# =============================================================================
# Public validators
# =============================================================================


def _issue(
    severity: str,
    code: str,
    split: str = "",
    idx: int = -1,
    msg: str = "",
    obs: Any = None,
    exp: Any = None,
) -> ValidationIssue:
    return ValidationIssue(
        severity=severity,
        code=code,
        split=split,
        sample_index=idx,
        message=msg,
        observed=obs,
        expected=exp,
    )


def _check_shape(
    data: dict[str, np.ndarray],
    issues: list[ValidationIssue],
    split: str,
    idx: int,
    name: str,
    S: int,
) -> None:
    """Check that *name* has shape ``(S,)`` (spatial), scalar (aux),
    or a known special shape (mask channels)."""
    arr = np.asarray(data[name])
    scalar_channels = {
        "natural_height",
        "natural_width",
        "row_offset",
        "col_offset",
        "total_physical_cost",
    }
    mask_channels = {
        "target_optimal_directions",  # (4,) bool
        "target_optimal_next_observations",  # (n_obs,) bool
    }
    if name in scalar_channels:
        if arr.ndim != 0:
            issues.append(
                _issue(
                    "ERROR",
                    "invalid_shape",
                    split,
                    idx,
                    f"'{name}' must be scalar, got shape {arr.shape}",
                    str(arr.shape),
                    "scalar",
                )
            )
    elif name in mask_channels:
        # Defer detailed shape check to calling context (n_obs not available here).
        if arr.ndim != 1:
            issues.append(
                _issue(
                    "ERROR",
                    "invalid_shape",
                    split,
                    idx,
                    f"'{name}' must be 1-D, got shape {arr.shape}",
                    str(arr.shape),
                    "1-D",
                )
            )
    elif arr.shape != (S,):
        issues.append(
            _issue(
                "ERROR",
                "invalid_shape",
                split,
                idx,
                f"'{name}' shape {arr.shape}, expected ({S},)",
                str(arr.shape),
                f"({S},)",
            )
        )


# =============================================================================
# Public validators
# =============================================================================


def validate_generated_sample(
    data: dict[str, np.ndarray],
    *,
    oracle_result: OracleResult | None,
    n_obs: int,
    topo_vocab_size: int,
    S: int,
    gamma_space: float,
    gamma_semantic: float,
) -> list[ValidationIssue]:
    """Validate a generated sample where oracle metadata is available.

    In addition to structural checks (shapes, dtypes, one start, etc.),
    verifies that the target fields match the oracle result when provided.

    When *oracle_result* is ``None``, oracle-specific consistency checks
    (decay consistency, off-route zeros) are skipped.

    Args:
        data: Sample channel dict.
        oracle_result: ``OracleResult`` or ``None``.  When ``None``,
            oracle consistency checks are skipped.
        n_obs: DAG observation count.
        topo_vocab_size: Topology observation vocabulary size.
        S: Number of spatial positions.
        gamma_space: Spatial decay factor.
        gamma_semantic: Semantic decay factor.

    Returns:
        List of validation issues (empty = clean).
    """
    issues: list[ValidationIssue] = []

    # Structural checks
    for ch in ROUTEBIND_SCHEMA.all_channels:
        if ch not in data:
            issues.append(
                _issue("ERROR", "missing_channel", msg=f"'{ch}' missing")
            )
            continue
        _check_shape(data, issues, "", -1, ch, S)

    if any(i.code == "missing_channel" for i in issues):
        return issues

    ct = np.asarray(data["cell_type"])
    oid = np.asarray(data["observation_id"])
    sf = np.asarray(data["start_flag"])
    gf = np.asarray(data["goal_flag"])
    tf = np.asarray(data["target_trajectory"])
    wf = np.asarray(data["target_waypoint"])

    # Dtype checks
    expected_dtypes = ROUTEBIND_SCHEMA.dtypes
    for ch, edt in expected_dtypes.items():
        arr = np.asarray(data[ch])
        adt = arr.dtype
        if edt.kind == "i" and adt.kind == "i":
            continue
        if edt.kind == "b" and adt.kind == "b":
            continue
        if edt.kind == "f" and adt.kind == "f":
            continue
        if adt != edt:
            issues.append(
                _issue(
                    "ERROR",
                    "invalid_dtype",
                    msg=f"'{ch}' dtype {adt}, expected {edt}",
                    obs=str(adt),
                    exp=str(edt),
                )
            )

    # Value ranges
    invalid_ct = set(ct.tolist()) - {
        CELL_WALL,
        CELL_FREE,
        CELL_OBSERVATION,
        CELL_PAD,
    }
    if invalid_ct:
        issues.append(
            _issue(
                "ERROR",
                "invalid_cell_type",
                msg=f"Invalid cell types: {sorted(invalid_ct)}",
                obs=sorted(invalid_ct),
                exp="{0,1,2,3}",
            )
        )

    # Spatial mask consistency
    if "spatial_mask" in data:
        sm = np.asarray(data["spatial_mask"])
        for p in range(S):
            if sm[p] and int(ct[p]) == CELL_PAD:
                issues.append(
                    _issue(
                        "ERROR",
                        "spatial_mask_inconsistency",
                        msg=f"Pos {p}: spatial_mask True but cell_type is CELL_PAD",
                    )
                )
            if not sm[p] and int(ct[p]) != CELL_PAD:
                issues.append(
                    _issue(
                        "ERROR",
                        "spatial_mask_inconsistency",
                        msg=f"Pos {p}: spatial_mask False but cell_type is {int(ct[p])} (not CELL_PAD)",
                    )
                )
        if "target_trajectory" in data:
            padding_nonzero = np.where(
                (~sm) & (np.asarray(data["target_trajectory"]) > 1e-7)
            )[0]
            if len(padding_nonzero) > 0:
                issues.append(
                    _issue(
                        "ERROR",
                        "target_in_padding",
                        msg=f"{len(padding_nonzero)} padding positions have non-zero trajectory target",
                    )
                )
        if "target_waypoint" in data:
            padding_nonzero = np.where(
                (~sm) & (np.asarray(data["target_waypoint"]) > 1e-7)
            )[0]
            if len(padding_nonzero) > 0:
                issues.append(
                    _issue(
                        "ERROR",
                        "target_in_padding",
                        msg=f"{len(padding_nonzero)} padding positions have non-zero waypoint target",
                    )
                )

    for name, field in [("target_trajectory", tf), ("target_waypoint", wf)]:
        if np.any(np.isnan(field)):
            issues.append(
                _issue("ERROR", "nan_value", msg=f"'{name}' contains NaN")
            )
        if np.any(np.isinf(field)):
            issues.append(
                _issue("ERROR", "inf_value", msg=f"'{name}' contains Inf")
            )
        mn, mx = float(field.min()), float(field.max())
        if mn < 0 or mx > 1:
            issues.append(
                _issue(
                    "ERROR",
                    "field_out_of_range",
                    msg=f"'{name}' outside [0,1]: min={mn:.4f}, max={mx:.4f}",
                    obs=f"[{mn:.4f}, {mx:.4f}]",
                    exp="[0, 1]",
                )
            )

    # Observation ID semantics
    for p in range(S):
        c = int(ct[p])
        o = int(oid[p])
        if c == CELL_OBSERVATION and o < 0:
            issues.append(
                _issue(
                    "ERROR",
                    "invalid_observation_id",
                    msg=f"OBS at pos {p} has sentinel ID {o}",
                    obs=o,
                    exp=">=0",
                )
            )
        if c in (CELL_WALL, CELL_FREE) and o >= 0:
            issues.append(
                _issue(
                    "ERROR",
                    "invalid_observation_id",
                    msg=f"Non-OBS at pos {p} (type={c}) has ID {o}",
                    obs=o,
                    exp="<0",
                )
            )
        if c == CELL_OBSERVATION and o >= topo_vocab_size:
            issues.append(
                _issue(
                    "ERROR",
                    "invalid_observation_id",
                    msg=f"Pos {p}: ID {o} >= topo_vocab_size={topo_vocab_size}",
                    obs=o,
                    exp=f"<{topo_vocab_size}",
                )
            )

    # Start invariant
    ns = int(sf.sum())
    if ns == 0:
        issues.append(_issue("ERROR", "missing_start"))
    elif ns > 1:
        issues.append(
            _issue(
                "ERROR",
                "multiple_starts",
                msg=f"{ns} start_flags",
                obs=ns,
                exp=1,
            )
        )
    else:
        sp = int(np.argmax(sf))
        if int(ct[sp]) == CELL_WALL:
            issues.append(
                _issue("ERROR", "start_on_wall", msg=f"Start at {sp} is WALL")
            )
        if int(oid[sp]) < 0:
            issues.append(
                _issue(
                    "ERROR",
                    "start_without_observation",
                    msg=f"Start at {sp} has no obs ID",
                )
            )

    # Goal invariant
    ng = int(gf.sum())
    if ng == 0:
        issues.append(_issue("ERROR", "missing_goal_occurrence"))
    else:
        gp = np.where(gf)[0]
        goal_obs = set(int(oid[p]) for p in gp)
        if len(goal_obs) != 1:
            issues.append(
                _issue(
                    "ERROR",
                    "goal_flag_identity_mismatch",
                    msg=f"Goal cells have multiple obs IDs: {sorted(goal_obs)}",
                    obs=sorted(goal_obs),
                    exp="single ID",
                )
            )

    # Oracle-specific consistency checks removed.
    # Routebind v1 now uses the optimal-subgraph contract:
    #   support/depth channels are validated by validate_support_channels.
    #   Bellman-optimal product-state recomputation is done by
    #   check_oracle_optimal_subgraph.
    pass

    # Start != goal observation
    if ns == 1 and ng >= 1:
        sp = int(np.argmax(sf))
        so = int(oid[sp])
        go = int(oid[int(np.argmax(gf))])
        if so == go and so >= 0:
            issues.append(
                _issue(
                    "ERROR",
                    "start_equals_goal",
                    msg=f"Start and goal obs identical ({so})",
                    obs=so,
                    exp="different",
                )
            )

    return issues


def validate_stored_sample(
    data: dict[str, np.ndarray],
    *,
    n_obs: int,
    topo_vocab_size: int,
    S: int,
    gamma_space: float,
    gamma_semantic: float,
    grid_width: int | None = None,
) -> list[ValidationIssue]:
    """Validate a stored sample from persisted arrays (no oracle metadata).

    Performs structural checks only: shapes, dtypes, one start, goal
    occurrences, field ranges, and decay consistency against the stored
    target fields (without recomputing the oracle).

    Args:
        data: Sample channel dict from persisted arrays.
        n_obs: DAG observation count.
        topo_vocab_size: Topology observation vocabulary size.
        S: Number of spatial positions.
        gamma_space: Spatial decay factor.
        gamma_semantic: Semantic decay factor.
        grid_width: Grid width in cells.  When provided, enables non-square
            grid route extraction.  Falls back to square-grid inference when
            ``None``.

    Returns:
        List of validation issues.
    """
    issues: list[ValidationIssue] = []

    # Structural checks (shared with validate_generated_sample)
    for ch in ROUTEBIND_SCHEMA.all_channels:
        if ch not in data:
            issues.append(
                _issue("ERROR", "missing_channel", msg=f"'{ch}' missing")
            )
            continue
        _check_shape(data, issues, "", -1, ch, S)

    if any(i.code == "missing_channel" for i in issues):
        return issues

    ct = np.asarray(data["cell_type"])
    oid = np.asarray(data["observation_id"])
    sf = np.asarray(data["start_flag"])
    gf = np.asarray(data["goal_flag"])
    tf = np.asarray(data["target_trajectory"])
    wf = np.asarray(data["target_waypoint"])

    # Dtype checks
    expected_dtypes = ROUTEBIND_SCHEMA.dtypes
    for ch, edt in expected_dtypes.items():
        arr = np.asarray(data[ch])
        adt = arr.dtype
        if edt.kind == "i" and adt.kind == "i":
            continue
        if edt.kind == "b" and adt.kind == "b":
            continue
        if edt.kind == "f" and adt.kind == "f":
            continue
        if adt != edt:
            issues.append(
                _issue(
                    "ERROR",
                    "invalid_dtype",
                    msg=f"'{ch}' dtype {adt}, expected {edt}",
                    obs=str(adt),
                    exp=str(edt),
                )
            )

    # Value ranges
    invalid_ct = set(ct.tolist()) - {
        CELL_WALL,
        CELL_FREE,
        CELL_OBSERVATION,
        CELL_PAD,
    }
    if invalid_ct:
        issues.append(
            _issue(
                "ERROR",
                "invalid_cell_type",
                msg=f"Invalid cell types: {sorted(invalid_ct)}",
                obs=sorted(invalid_ct),
                exp="{0,1,2,3}",
            )
        )

    # Spatial mask consistency
    if "spatial_mask" in data:
        sm = np.asarray(data["spatial_mask"])
        for p in range(S):
            if sm[p] and int(ct[p]) == CELL_PAD:
                issues.append(
                    _issue(
                        "ERROR",
                        "spatial_mask_inconsistency",
                        msg=f"Pos {p}: spatial_mask True but cell_type is CELL_PAD",
                    )
                )
            if not sm[p] and int(ct[p]) != CELL_PAD:
                issues.append(
                    _issue(
                        "ERROR",
                        "spatial_mask_inconsistency",
                        msg=f"Pos {p}: spatial_mask False but cell_type is {int(ct[p])} (not CELL_PAD)",
                    )
                )
        if "target_trajectory" in data:
            padding_nonzero = np.where(
                (~sm) & (np.asarray(data["target_trajectory"]) > 1e-7)
            )[0]
            if len(padding_nonzero) > 0:
                issues.append(
                    _issue(
                        "ERROR",
                        "target_in_padding",
                        msg=f"{len(padding_nonzero)} padding positions have non-zero trajectory target",
                    )
                )
        if "target_waypoint" in data:
            padding_nonzero = np.where(
                (~sm) & (np.asarray(data["target_waypoint"]) > 1e-7)
            )[0]
            if len(padding_nonzero) > 0:
                issues.append(
                    _issue(
                        "ERROR",
                        "target_in_padding",
                        msg=f"{len(padding_nonzero)} padding positions have non-zero waypoint target",
                    )
                )

    for name, field in [("target_trajectory", tf), ("target_waypoint", wf)]:
        if np.any(np.isnan(field)):
            issues.append(
                _issue("ERROR", "nan_value", msg=f"'{name}' contains NaN")
            )
        if np.any(np.isinf(field)):
            issues.append(
                _issue("ERROR", "inf_value", msg=f"'{name}' contains Inf")
            )
        mn, mx = float(field.min()), float(field.max())
        if mn < 0 or mx > 1:
            issues.append(
                _issue(
                    "ERROR",
                    "field_out_of_range",
                    msg=f"'{name}' outside [0,1]: min={mn:.4f}, max={mx:.4f}",
                    obs=f"[{mn:.4f}, {mx:.4f}]",
                    exp="[0, 1]",
                )
            )

    # Start invariant
    ns = int(sf.sum())
    if ns == 0:
        issues.append(_issue("ERROR", "missing_start"))
    elif ns > 1:
        issues.append(
            _issue(
                "ERROR",
                "multiple_starts",
                msg=f"{ns} start_flags",
                obs=ns,
                exp=1,
            )
        )
    else:
        sp = int(np.argmax(sf))
        if int(ct[sp]) == CELL_WALL:
            issues.append(
                _issue("ERROR", "start_on_wall", msg=f"Start at {sp} is WALL")
            )
        if int(oid[sp]) < 0:
            issues.append(
                _issue(
                    "ERROR",
                    "start_without_observation",
                    msg=f"Start at {sp} has no obs ID",
                )
            )
        # Decay consistency (approximate — route extracted from field)
        # Skip for v2 corpora whose fields encode optimal-support (not single-path decay).
        if "trajectory_support" not in data:
            from ehc_sn.tasks.routebind.decoding import (
                extract_route_from_trajectory_field,
            )

            route = extract_route_from_trajectory_field(
                tf, sp, ct, grid_width=grid_width
            )
            if route:
                violations = validate_decay_consistency(tf, route, gamma_space)
                for v in violations[:3]:
                    issues.append(
                        _issue("WARNING", "field_decay_mismatch", msg=v)
                    )

    # Goal invariant
    ng = int(gf.sum())
    if ng == 0:
        issues.append(_issue("ERROR", "missing_goal_occurrence"))
    else:
        gp = np.where(gf)[0]
        goal_obs = set(int(oid[p]) for p in gp)
        if len(goal_obs) != 1:
            issues.append(
                _issue(
                    "ERROR",
                    "goal_flag_identity_mismatch",
                    msg=f"Goal cells have multiple obs IDs: {sorted(goal_obs)}",
                    obs=sorted(goal_obs),
                    exp="single ID",
                )
            )

    return issues


# =============================================================================
# Corpus root validation
# =============================================================================


_VALID_CORPUS_CHANNELS = frozenset(ROUTEBIND_SCHEMA.all_channels)


def validate_corpus_root(
    root: Path,
    max_root_samples: int = -1,
) -> tuple[dict, list[ValidationIssue]]:
    """Validate a routebind task corpus root against task-owned semantics.

    Returns (manifest, issues).  The caller decides severity thresholds.

    Args:
        root: Resolved versioned routebind corpus root.
        max_root_samples: Maximum per-split samples validated at the root
            level (structural scan).  ``-1`` (default) validates all.

    Returns:
        Tuple of (manifest dict, issues list).
    """
    issues: list[ValidationIssue] = []

    try:
        manifest = validate_version_root(root)
    except (FileNotFoundError, ValueError) as e:
        issues.append(
            _issue("ERROR", "invalid_root", msg=f"Root validation failed: {e}")
        )
        return ({}, issues)

    if manifest.get("dataset_class") != "task_corpus":
        issues.append(
            _issue(
                "ERROR",
                "invalid_dataset_class",
                msg=f"Expected task_corpus, got {manifest.get('dataset_class')!r}",
            )
        )

    task_name = manifest.get("task", "")
    if task_name != "routebind":
        issues.append(
            _issue(
                "ERROR",
                "invalid_task",
                msg=f"Expected routebind, got {task_name!r}",
            )
        )

    num_slots = manifest.get("num_spatial_slots") or manifest.get("n_states")
    if num_slots is None:
        issues.append(_issue("ERROR", "missing_num_spatial_slots"))

    declared_channels: list[str] = manifest.get("channels", [])
    for ch in _VALID_CORPUS_CHANNELS:
        if ch not in declared_channels:
            issues.append(
                _issue(
                    "WARNING",
                    "missing_channel",
                    msg=f"Channel '{ch}' not declared in manifest",
                )
            )

    # Validate per-split data
    for split, n in manifest.get("n_samples", {}).items():
        if n == 0:
            continue
        split_dir = root / split
        for ch in declared_channels:
            ch_file = split_dir / f"{ch}.npy"
            if not ch_file.exists():
                issues.append(
                    _issue(
                        "ERROR",
                        "missing_channel_file",
                        split=split,
                        msg=f"Missing '{ch}' in {split_dir}",
                    )
                )
                continue
            arr = np.load(ch_file, mmap_mode="r")
            if arr.shape[0] != n:
                issues.append(
                    _issue(
                        "ERROR",
                        "sample_count_mismatch",
                        split=split,
                        msg=f"Channel '{ch}' has {arr.shape[0]} samples, manifest declares {n}",
                        obs=arr.shape[0],
                        exp=n,
                    )
                )

        # Validate a sample subset
        arrays: dict[str, np.ndarray] = {}
        for ch in declared_channels:
            ch_file = split_dir / f"{ch}.npy"
            if ch_file.exists():
                arrays[ch] = np.load(ch_file, mmap_mode="r")

        if arrays:
            n_root_samples = (
                min(n, max_root_samples) if max_root_samples > 0 else n
            )
            for i in range(n_root_samples):
                sample = {
                    ch: arrays[ch][i]
                    for ch in declared_channels
                    if ch in arrays
                }
                # Use stored-sample validation (no oracle metadata available)
                sample_issues = validate_stored_sample(
                    sample,
                    n_obs=manifest.get("n_observations", 0),
                    topo_vocab_size=manifest.get("source", {}).get(
                        "observation_vocab_size", 0
                    )
                    or manifest.get("n_observations", 0),
                    S=num_slots or 900,
                    gamma_space=manifest.get("field_decay_spatial", 0.9848),
                    gamma_semantic=manifest.get("field_decay_semantic", 0.8),
                    grid_width=manifest.get("canvas_width", None),
                )
                for si in sample_issues:
                    si.split = split
                    si.sample_index = i
                issues.extend(sample_issues)

    return (manifest, issues)


# =============================================================================
# Detailed stored-sample checks (structural child checks)
# =============================================================================


# =============================================================================
# Layer 1 — Structural support/depth/field algebra validation (v2)
# =============================================================================


def _si(
    severity: str, code: str, split: str, idx: int, msg: str, obs=None, exp=None
) -> ValidationIssue:
    """Shorthand for creating a ValidationIssue."""
    return ValidationIssue(
        severity=severity,
        code=code,
        split=split,
        sample_index=idx,
        message=msg,
        observed=obs,
        expected=exp,
    )


def validate_support_channels(
    data: dict[str, np.ndarray],
    *,
    S: int,
    gamma_space: float,
    gamma_semantic: float,
    split: str = "",
    idx: int = -1,
) -> list[ValidationIssue]:
    """Layer 1 structural validation for optimal-subgraph v2 corpora.

    Validates bidirectional support/depth/field algebra, start invariants,
    goal presence, padding zeros, and mask shape/dtype/non-emptiness.

    Does NOT extract a route, does NOT check single-path decay,
    does NOT check waypoint order, does NOT check DAG transitions.

    Args:
        data: Sample channel dict.
        S: Number of spatial positions.
        gamma_space: Spatial decay factor.
        gamma_semantic: Semantic decay factor.
        split: Optional split label.
        idx: Optional sample index.

    Returns:
        List of validation issues (empty = clean).
    """
    issues: list[ValidationIssue] = []

    ct = np.asarray(data["cell_type"])
    oid = np.asarray(data["observation_id"])
    sf = np.asarray(data["start_flag"])
    gf = np.asarray(data["goal_flag"])
    sm = np.asarray(data.get("spatial_mask", np.ones(S, dtype=bool)))

    # Guard: must have support channels
    has_v2 = all(
        ch in data
        for ch in (
            "trajectory_support",
            "trajectory_forward_depth",
            "trajectory_remaining_cost",
            "waypoint_support",
            "waypoint_semantic_depth",
            "target_optimal_directions",
            "target_optimal_next_observations",
        )
    )
    if not has_v2:
        return []

    ts = np.asarray(data["trajectory_support"])
    td = np.asarray(data["trajectory_forward_depth"])
    trc = np.asarray(data["trajectory_remaining_cost"])
    ws = np.asarray(data["waypoint_support"])
    wd = np.asarray(data["waypoint_semantic_depth"])
    tf = np.asarray(data["target_trajectory"])
    wf = np.asarray(data["target_waypoint"])
    gvf = tf  # alias for backward compat during migration
    dir_mask = np.asarray(data["target_optimal_directions"])
    obs_mask = np.asarray(data["target_optimal_next_observations"])
    n_obs = obs_mask.shape[0]

    tol = 1e-6

    # ── Auxiliary masks: shape, dtype, non-empty ─────────────────────
    if dir_mask.shape != (4,):
        issues.append(
            _si(
                "ERROR",
                "direction_mask_out_of_range",
                split,
                idx,
                f"target_optimal_directions shape {dir_mask.shape}, expected (4,)",
            )
        )
    if dir_mask.dtype != bool:
        issues.append(
            _si(
                "ERROR",
                "direction_mask_out_of_range",
                split,
                idx,
                "target_optimal_directions dtype not bool",
            )
        )
    if dir_mask.size == 4 and not dir_mask.any():
        issues.append(
            _si(
                "ERROR",
                "direction_mask_out_of_range",
                split,
                idx,
                "target_optimal_directions is empty (no optimal directions)",
            )
        )

    if obs_mask.shape != (n_obs,):
        issues.append(
            _si(
                "ERROR",
                "observation_mask_out_of_range",
                split,
                idx,
                f"target_optimal_next_observations shape {obs_mask.shape}, expected ({n_obs},)",
            )
        )
    if obs_mask.dtype != bool:
        issues.append(
            _si(
                "ERROR",
                "observation_mask_out_of_range",
                split,
                idx,
                "target_optimal_next_observations dtype not bool",
            )
        )
    if not obs_mask.any():
        issues.append(
            _si(
                "ERROR",
                "observation_mask_out_of_range",
                split,
                idx,
                "target_optimal_next_observations is empty",
            )
        )

    # ── Per-position checks ──────────────────────────────────────────
    for p in range(S):
        c = int(ct[p])

        # Trajectory
        if not ts[p]:
            if td[p] != -1:
                issues.append(
                    _si(
                        "ERROR",
                        "trajectory_depth_invalid",
                        split,
                        idx,
                        f"Pos {p}: trajectory_support False but depth={int(td[p])} (expected -1)",
                    )
                )
            if abs(tf[p]) > tol:
                issues.append(
                    _si(
                        "ERROR",
                        "trajectory_support_field_mismatch",
                        split,
                        idx,
                        f"Pos {p}: trajectory_support False but field={float(tf[p]):.6f} (expected 0)",
                    )
                )
        else:
            if td[p] < 0:
                issues.append(
                    _si(
                        "ERROR",
                        "trajectory_depth_invalid",
                        split,
                        idx,
                        f"Pos {p}: trajectory_support True but depth={int(td[p])} (expected >=0)",
                    )
                )
            if tf[p] <= 0.0:
                issues.append(
                    _si(
                        "ERROR",
                        "trajectory_support_field_mismatch",
                        split,
                        idx,
                        f"Pos {p}: trajectory_support True but field={float(tf[p]):.6f} (expected >0)",
                    )
                )
            if td[p] >= 0:
                expected = float(gamma_space ** int(td[p]))
                if abs(tf[p] - expected) > tol:
                    issues.append(
                        _si(
                            "ERROR",
                            "trajectory_support_field_mismatch",
                            split,
                            idx,
                            f"Pos {p}: field={float(tf[p]):.6f} expected γ^{int(td[p])}={expected:.6f}",
                        )
                    )
            if c in (CELL_WALL, CELL_PAD):
                issues.append(
                    _si(
                        "ERROR",
                        "trajectory_support_on_nontraversable",
                        split,
                        idx,
                        f"Pos {p}: trajectory_support on cell_type={c}",
                    )
                )

        # Remaining cost sentinel
        if not ts[p]:
            if trc[p] != -1:
                issues.append(
                    _si(
                        "ERROR",
                        "remaining_cost_invalid",
                        split,
                        idx,
                        f"Pos {p}: trajectory_support False but "
                        f"remaining_cost={int(trc[p])} (expected -1)",
                    )
                )
        else:
            if trc[p] < 0:
                issues.append(
                    _si(
                        "ERROR",
                        "remaining_cost_invalid",
                        split,
                        idx,
                        f"Pos {p}: trajectory_support True but "
                        f"remaining_cost={int(trc[p])} (expected >=0)",
                    )
                )

        # Waypoint
        if not ws[p]:
            if wd[p] != -1:
                issues.append(
                    _si(
                        "ERROR",
                        "waypoint_depth_invalid",
                        split,
                        idx,
                        f"Pos {p}: waypoint_support False but depth={int(wd[p])} (expected -1)",
                    )
                )
            if abs(wf[p]) > tol:
                issues.append(
                    _si(
                        "ERROR",
                        "waypoint_support_field_mismatch",
                        split,
                        idx,
                        f"Pos {p}: waypoint_support False but field={float(wf[p]):.6f} (expected 0)",
                    )
                )
        else:
            if wd[p] < 0:
                issues.append(
                    _si(
                        "ERROR",
                        "waypoint_depth_invalid",
                        split,
                        idx,
                        f"Pos {p}: waypoint_support True but depth={int(wd[p])} (expected >=0)",
                    )
                )
            if wf[p] <= 0.0:
                issues.append(
                    _si(
                        "ERROR",
                        "waypoint_support_field_mismatch",
                        split,
                        idx,
                        f"Pos {p}: waypoint_support True but field={float(wf[p]):.6f} (expected >0)",
                    )
                )
            if wd[p] >= 0:
                expected = float(gamma_semantic ** int(wd[p]))
                if abs(wf[p] - expected) > tol:
                    issues.append(
                        _si(
                            "ERROR",
                            "waypoint_support_field_mismatch",
                            split,
                            idx,
                            f"Pos {p}: field={float(wf[p]):.6f} expected γ^{int(wd[p])}={expected:.6f}",
                        )
                    )
            if c != CELL_OBSERVATION:
                issues.append(
                    _si(
                        "ERROR",
                        "waypoint_support_on_nonobservation",
                        split,
                        idx,
                        f"Pos {p}: waypoint_support on cell_type={c}",
                    )
                )

    # ── Start invariants ─────────────────────────────────────────────
    if int(sf.sum()) == 1:
        sp = int(np.argmax(sf))
        if not ts[sp]:
            issues.append(
                _si(
                    "ERROR",
                    "trajectory_start_invalid",
                    split,
                    idx,
                    f"Start {sp}: trajectory_support False",
                )
            )
        if int(td[sp]) != 0:
            issues.append(
                _si(
                    "ERROR",
                    "trajectory_start_invalid",
                    split,
                    idx,
                    f"Start {sp}: trajectory_forward_depth={int(td[sp])} (expected 0)",
                )
            )
        if abs(float(tf[sp]) - 1.0) > tol:
            issues.append(
                _si(
                    "ERROR",
                    "trajectory_start_invalid",
                    split,
                    idx,
                    f"Start {sp}: target_trajectory={float(tf[sp]):.6f} (expected 1.0)",
                )
            )
        if not ws[sp]:
            issues.append(
                _si(
                    "ERROR",
                    "waypoint_start_invalid",
                    split,
                    idx,
                    f"Start {sp}: waypoint_support False",
                )
            )
        if int(wd[sp]) != 0:
            issues.append(
                _si(
                    "ERROR",
                    "waypoint_start_invalid",
                    split,
                    idx,
                    f"Start {sp}: waypoint_semantic_depth={int(wd[sp])} (expected 0)",
                )
            )
        if abs(float(wf[sp]) - 1.0) > tol:
            issues.append(
                _si(
                    "ERROR",
                    "waypoint_start_invalid",
                    split,
                    idx,
                    f"Start {sp}: target_waypoint={float(wf[sp]):.6f} (expected 1.0)",
                )
            )

    # ── Goal: at least one goal_flag position in trajectory_support ──
    if int(gf.sum()) >= 1:
        goal_supported = np.any(gf & ts)
        if not goal_supported:
            issues.append(
                _si(
                    "ERROR",
                    "trajectory_support_field_mismatch",
                    split,
                    idx,
                    "No goal_flag position has trajectory_support",
                )
            )

    # ── Padding zeros ────────────────────────────────────────────────
    for p in range(S):
        if sm[p]:
            continue
        if abs(tf[p]) > tol:
            issues.append(
                _si(
                    "ERROR",
                    "trajectory_support_field_mismatch",
                    split,
                    idx,
                    f"Pad {p}: target_trajectory non-zero",
                )
            )
        if ts[p]:
            issues.append(
                _si(
                    "ERROR",
                    "trajectory_support_field_mismatch",
                    split,
                    idx,
                    f"Pad {p}: trajectory_support True",
                )
            )
        if abs(wf[p]) > tol:
            issues.append(
                _si(
                    "ERROR",
                    "waypoint_support_field_mismatch",
                    split,
                    idx,
                    f"Pad {p}: target_waypoint non-zero",
                )
            )
        if ws[p]:
            issues.append(
                _si(
                    "ERROR",
                    "waypoint_support_field_mismatch",
                    split,
                    idx,
                    f"Pad {p}: waypoint_support True",
                )
            )

    return issues


# =============================================================================
# Layer 2 — Oracle optimal-subgraph recomputation validation (v2)
# =============================================================================


@dataclass
class OracleValidationContext:
    """Typed context for recomputing the optimal product-state subgraph.

    Fields correspond to the inputs of ``compute_goal_distance_table``
    and ``traverse_optimal_subgraph``.
    """

    physical_neighbors: np.ndarray  # (S, 4) int32
    node_at_position: np.ndarray  # (S,) int32
    pred_offsets: np.ndarray  # (n_obs+1,) int32
    pred_nodes: np.ndarray  # (total_pred_edges,) int32
    pub_succ_mask: np.ndarray  # (n_obs, K) bool
    pub_succ_indices: np.ndarray  # (n_obs, K) int32
    n_slots: int
    n_obs: int


def check_oracle_optimal_subgraph(
    data: dict[str, np.ndarray],
    ctx: OracleValidationContext,
    *,
    gamma_space: float,
    gamma_semantic: float,
    split: str = "",
    idx: int = -1,
    _workspace: dict | None = None,
) -> list[ValidationIssue]:
    """Layer 2 semantic validation for optimal-subgraph v2 corpora.

    Recomputes the optimal product-state subgraph for the given sample
    via ``compute_goal_distance_table`` + ``derive_optimal_transition_masks``
    + ``traverse_optimal_subgraph``, and compares the expected support/depth/
    mask channels against the stored values.

    Args:
        data: Sample channel dict.
        ctx: ``OracleValidationContext`` with topology and DAG data.
        gamma_space: Spatial decay factor.
        gamma_semantic: Semantic decay factor.
        split: Optional split label.
        idx: Optional sample index.
        _workspace: Optional pre-allocated workspace dict.

    Returns:
        List of validation issues (empty = matches oracle).
    """
    from ehc_sn.tasks.routebind.oracle import (
        INF,
        compute_goal_distance_table,
        derive_optimal_transition_masks,
        traverse_optimal_subgraph,
    )

    E = lambda c, m, o=None, e=None: _si("ERROR", c, split, idx, m, o, e)

    sf = np.asarray(data["start_flag"])
    gf = np.asarray(data["goal_flag"])
    oid = np.asarray(data["observation_id"])

    if int(sf.sum()) != 1 or int(gf.sum()) < 1:
        return []

    start_pos = int(np.argmax(sf))
    start_obs = int(oid[start_pos])
    if start_obs < 0 or start_obs >= ctx.n_obs:
        return []

    goal_positions = np.where(gf)[0]
    goal_obs = int(oid[goal_positions[0]])

    # Recompute distance table for this goal
    n_states = ctx.n_slots * ctx.n_obs
    if _workspace is None:
        dist = np.full(n_states, np.iinfo(np.int32).max, dtype=np.int32)
        pkind = np.zeros(n_states, dtype=np.int8)
        pnext = np.full(n_states, -1, dtype=np.int32)
        oct = np.zeros(n_states, dtype=np.uint8)
        dqb = np.zeros(2 * n_states, dtype=np.int32)
        ws = dict(
            distance=dist,
            policy_kind=pkind,
            policy_next=pnext,
            opt_count=oct,
            deque_buf=dqb,
        )
    else:
        ws = _workspace

    table = compute_goal_distance_table(
        physical_neighbors=ctx.physical_neighbors,
        node_at_position=ctx.node_at_position,
        pred_offsets=ctx.pred_offsets,
        pred_nodes=ctx.pred_nodes,
        goal_occurrences=goal_positions,
        goal_node_idx=goal_obs,
        n_slots=ctx.n_slots,
        n_obs=ctx.n_obs,
        _workspace=ws,
    )

    if table["deque_overflow"]:
        return [
            E(
                "total_optimal_cost_mismatch",
                "Deque overflow during oracle recomputation",
            )
        ]

    # Derive optimal transition masks
    phys_mask, accept_mask = derive_optimal_transition_masks(
        distance=table["distance"],
        physical_neighbors=ctx.physical_neighbors,
        node_at_position=ctx.node_at_position,
        succ_mask=ctx.pub_succ_mask,
        succ_indices=ctx.pub_succ_indices,
        n_slots=ctx.n_slots,
        n_obs=ctx.n_obs,
    )

    # Traverse optimal subgraph
    try:
        expected = traverse_optimal_subgraph(
            start_pos=start_pos,
            start_obs=start_obs,
            distance=table["distance"],
            physical_optimal_mask=phys_mask,
            accept_optimal_mask=accept_mask,
            physical_neighbors=ctx.physical_neighbors,
            node_at_position=ctx.node_at_position,
            n_slots=ctx.n_slots,
            n_obs=ctx.n_obs,
        )
    except ValueError as e:
        return [
            E("total_optimal_cost_mismatch", f"Subgraph traversal failed: {e}")
        ]

    # Compare stored channels against expected
    issues: list[ValidationIssue] = []
    tol = 1e-6

    # Trajectory support and depth
    stored_ts = np.asarray(data["trajectory_support"])
    stored_td = np.asarray(data["trajectory_forward_depth"])
    if not np.array_equal(stored_ts, expected.trajectory_support):
        n_diff = int(np.sum(stored_ts != expected.trajectory_support))
        issues.append(
            E(
                "trajectory_optimal_support_mismatch",
                f"{n_diff} positions differ from oracle trajectory_support",
            )
        )
    if not np.array_equal(stored_td, expected.trajectory_forward_depth):
        n_diff = int(np.sum(stored_td != expected.trajectory_forward_depth))
        issues.append(
            E(
                "trajectory_optimal_support_mismatch",
                f"{n_diff} positions differ from oracle trajectory_forward_depth",
            )
        )
    # Trajectory remaining cost
    stored_trc = np.asarray(data.get("trajectory_remaining_cost", stored_td))
    if not np.array_equal(stored_trc, expected.trajectory_remaining_cost):
        n_diff = int(np.sum(stored_trc != expected.trajectory_remaining_cost))
        issues.append(
            E(
                "trajectory_optimal_support_mismatch",
                f"{n_diff} positions differ from oracle trajectory_remaining_cost",
            )
        )

    # Waypoint support and depth
    stored_ws = np.asarray(data["waypoint_support"])
    stored_wd = np.asarray(data["waypoint_semantic_depth"])
    if not np.array_equal(stored_ws, expected.waypoint_support):
        n_diff = int(np.sum(stored_ws != expected.waypoint_support))
        issues.append(
            E(
                "waypoint_optimal_support_mismatch",
                f"{n_diff} positions differ from oracle waypoint_support",
            )
        )
    if not np.array_equal(stored_wd, expected.waypoint_semantic_depth):
        n_diff = int(np.sum(stored_wd != expected.waypoint_semantic_depth))
        issues.append(
            E(
                "waypoint_optimal_support_mismatch",
                f"{n_diff} positions differ from oracle waypoint_semantic_depth",
            )
        )

    # Auxiliary masks
    stored_dir = np.asarray(data["target_optimal_directions"])
    stored_obs = np.asarray(data["target_optimal_next_observations"])
    if not np.array_equal(stored_dir, expected.target_optimal_directions):
        issues.append(
            E(
                "optimal_direction_mask_mismatch",
                "target_optimal_directions differs from oracle",
            )
        )
    if not np.array_equal(
        stored_obs, expected.target_optimal_next_observations
    ):
        issues.append(
            E(
                "optimal_observation_mask_mismatch",
                "target_optimal_next_observations differs from oracle",
            )
        )

    # Total cost
    stored_cost = expected.total_physical_cost
    # (no stored scalar for cost in corpus channels — compare via depth)
    if int(stored_td.max()) != stored_cost:
        issues.append(
            E(
                "total_optimal_cost_mismatch",
                f"max trajectory_forward_depth={int(stored_td.max())} "
                f"!= oracle total_cost={stored_cost}",
            )
        )

    return issues


__all__ = [
    "OracleValidationContext",
    "ValidationIssue",
    "check_oracle_optimal_subgraph",
    "validate_corpus_root",
    "validate_generated_sample",
    "validate_stored_sample",
    "validate_support_channels",
]
