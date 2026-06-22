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
    """Check that *name* has shape ``(S,)`` (spatial) or scalar (aux)."""
    arr = np.asarray(data[name])
    scalar_channels = {
        "target_next_dir",
        "target_next_obs",
        "natural_height",
        "natural_width",
        "row_offset",
        "col_offset",
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
    oracle_result: OracleResult,
    n_obs: int,
    topo_vocab_size: int,
    S: int,
    gamma_space: float,
    gamma_semantic: float,
) -> list[ValidationIssue]:
    """Validate a generated sample where oracle metadata is available.

    In addition to structural checks (shapes, dtypes, one start, etc.),
    verifies that the target fields match the oracle result.

    Args:
        data: Sample channel dict.
        oracle_result: OracleResult from which targets were derived.
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

    # Oracle consistency
    if ns == 1:
        sp = int(np.argmax(sf))
        # Start activation
        if abs(float(tf[sp]) - 1.0) > 1e-6:
            issues.append(
                _issue(
                    "ERROR",
                    "field_decay_mismatch",
                    msg=f"Start activation {float(tf[sp]):.6f} != 1.0",
                    obs=float(tf[sp]),
                    exp=1.0,
                )
            )

        # Decay consistency
        route = list(oracle_result.physical_route)
        violations = validate_decay_consistency(tf, route, gamma_space)
        for v in violations[:5]:
            issues.append(_issue("ERROR", "field_decay_mismatch", msg=v))

        # Off-route zeros (only check if field matches oracle)
        if not violations:
            rs = set(route)
            off = np.where((tf > 1e-7) & (~np.isin(np.arange(S), list(rs))))[0]
            if len(off) > 0:
                issues.append(
                    _issue(
                        "ERROR",
                        "field_decay_mismatch",
                        msg=f"{len(off)} off-route non-zero activations",
                        obs=len(off),
                        exp=0,
                    )
                )

        # Walls positive
        wall_pos = np.where((ct == CELL_WALL) & (tf > 1e-7))[0]
        if len(wall_pos) > 0:
            issues.append(
                _issue(
                    "ERROR",
                    "field_decay_mismatch",
                    msg=f"{len(wall_pos)} WALL cells positive",
                    obs=len(wall_pos),
                    exp=0,
                )
            )

        # Waypoint decay consistency
        wps = oracle_result.waypoints
        wp_decay_violations = validate_decay_consistency(
            wf, [wp[1] for wp in wps], gamma_semantic
        )
        for v in wp_decay_violations[:5]:
            issues.append(_issue("ERROR", "waypoint_decay_mismatch", msg=v))

        # Crossed-but-unaccepted
        wps_set = {wp[1] for wp in wps}
        for pos in route:
            if int(ct[pos]) == CELL_OBSERVATION and pos not in wps_set:
                if wf[pos] > 1e-7:
                    issues.append(
                        _issue(
                            "ERROR",
                            "unaccepted_observation_marked",
                            msg=f"Crossed OBS at {pos} has wp activation {float(wf[pos]):.6f}",
                            obs=float(wf[pos]),
                            exp=0.0,
                        )
                    )

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
        from ehc_sn.tasks.routebind.decoding import (
            extract_route_from_trajectory_field,
        )

        route = extract_route_from_trajectory_field(
            tf, sp, ct, grid_width=grid_width
        )
        if route:
            violations = validate_decay_consistency(tf, route, gamma_space)
            for v in violations[:3]:
                issues.append(_issue("WARNING", "field_decay_mismatch", msg=v))

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


def check_route_field(
    data: dict[str, np.ndarray],
    *,
    S: int,
    gamma_space: float,
    split: str = "",
    idx: int = -1,
    grid_width: int | None = None,
) -> list[ValidationIssue]:
    """Validate the trajectory field against the decay contract.

    Extracts the physical route via the canonical greedy decoder and checks:
    adjacency, simplicity (no revisits), decay consistency, off-route zeros,
    wall-cell zeros, and start activation.

    Args:
        data: Sample channel dict.
        S: Number of spatial positions.
        gamma_space: Spatial decay factor.
        split: Optional split label for issue context.
        idx: Optional sample index for issue context.

    Returns:
        List of validation issues.
    """
    import math
    from collections import Counter

    from ehc_sn.tasks.routebind.decoding import (
        extract_route_from_trajectory_field,
    )

    E = lambda c, m, o=None, e=None: _issue("ERROR", c, split, idx, m, o, e)

    ct = np.asarray(data["cell_type"])
    sf = np.asarray(data["start_flag"])
    tf = np.asarray(data["target_trajectory"])

    if int(sf.sum()) != 1:
        return []
    sp = int(np.argmax(sf))
    route = extract_route_from_trajectory_field(tf, sp, grid_width=grid_width)
    if not route:
        return [
            _issue(
                "ERROR",
                "no_route_extracted",
                split,
                idx,
                "Could not extract route",
            )
        ]

    width = int(math.sqrt(S))
    if grid_width is not None:
        width = grid_width

    if route[0] != sp:
        return [
            E(
                "invalid_physical_step",
                f"First route pos {route[0]} != start {sp}",
            )
        ]

    issues: list[ValidationIssue] = []

    for k in range(len(route) - 1):
        a, b = route[k], route[k + 1]
        ra, ca = divmod(a, width)
        rb, cb = divmod(b, width)
        if abs(ra - rb) + abs(ca - cb) != 1:
            issues.append(
                E("invalid_physical_step", f"Step {k}: {a}->{b} not adjacent")
            )
        if int(ct[b]) == CELL_WALL:
            issues.append(
                E("invalid_physical_step", f"Step {k}: pos {b} is WALL")
            )

    if len(set(route)) != len(route):
        repeats = [p for p, c in Counter(route).items() if c > 1]
        issues.append(
            E(
                "non_simple_route",
                f"Route revisits: {repeats}",
                repeats,
                "no repeats",
            )
        )

    tol = 1e-5
    for k, pos in enumerate(route):
        expected = gamma_space**k
        actual = float(tf[pos])
        if abs(actual - expected) > tol:
            issues.append(
                E(
                    "field_decay_mismatch",
                    f"Pos {pos} step {k}: field={actual:.6f} "
                    f"expected={expected:.6f}",
                    actual,
                    expected,
                )
            )

    if abs(float(tf[sp]) - 1.0) > 1e-6:
        issues.append(
            E(
                "field_decay_mismatch",
                f"Start activation {float(tf[sp]):.6f} != 1.0",
                float(tf[sp]),
                1.0,
            )
        )

    rs = set(route)
    off = np.where((tf > 1e-7) & (~np.isin(np.arange(S), list(rs))))[0]
    if len(off) > 0:
        issues.append(
            E(
                "field_decay_mismatch",
                f"{len(off)} off-route non-zero activations",
                len(off),
                0,
            )
        )

    wall_pos = np.where((ct == CELL_WALL) & (tf > 1e-7))[0]
    if len(wall_pos) > 0:
        issues.append(
            E(
                "field_decay_mismatch",
                f"{len(wall_pos)} WALL cells positive",
                len(wall_pos),
                0,
            )
        )

    return issues


def check_waypoint_field(
    data: dict[str, np.ndarray],
    *,
    S: int,
    gamma_semantic: float,
    split: str = "",
    idx: int = -1,
    grid_width: int | None = None,
) -> list[ValidationIssue]:
    """Validate the waypoint field against the oracle contract.

    Checks: waypoints lie on the extracted route, are at observation cells,
    appear in route order, obey decay, and crossed-but-unaccepted
    observation cells have zero waypoint activation.

    Args:
        data: Sample channel dict.
        S: Number of spatial positions.
        gamma_semantic: Semantic decay factor.
        split: Optional split label.
        idx: Optional sample index.

    Returns:
        List of validation issues.
    """
    import math

    from ehc_sn.tasks.routebind.decoding import (
        extract_route_from_trajectory_field,
        extract_waypoints_from_field,
    )

    E = lambda c, m, o=None, e=None: _issue("ERROR", c, split, idx, m, o, e)

    ct = np.asarray(data["cell_type"])
    sf = np.asarray(data["start_flag"])
    tf = np.asarray(data["target_trajectory"])
    wf = np.asarray(data["target_waypoint"])

    if int(sf.sum()) != 1:
        return []
    sp = int(np.argmax(sf))
    route = extract_route_from_trajectory_field(tf, sp, grid_width=grid_width)
    if not route:
        return []

    waypoints = extract_waypoints_from_field(wf, route)
    if not waypoints:
        return []  # valid — some samples have no intermediate waypoints

    issues: list[ValidationIssue] = []

    if waypoints[0][0] != sp:
        issues.append(
            E(
                "waypoint_not_on_route",
                f"First WP {waypoints[0][0]} != start {sp}",
            )
        )

    rs = set(route)
    for wp_pos, _ in waypoints:
        if wp_pos not in rs:
            issues.append(
                E("waypoint_not_on_route", f"WP at {wp_pos} not on route")
            )
        if int(ct[wp_pos]) != CELL_OBSERVATION:
            issues.append(
                E(
                    "waypoint_not_on_route",
                    f"WP at {wp_pos} is not OBS " f"(type={int(ct[wp_pos])})",
                )
            )

    wpp = [w[0] for w in waypoints]
    pos_idx = {p: i for i, p in enumerate(route)}
    for k in range(len(wpp) - 1):
        if pos_idx[wpp[k]] >= pos_idx[wpp[k + 1]]:
            issues.append(
                E(
                    "waypoint_not_on_route",
                    f"WPs out of order: {wpp[k]} -> {wpp[k+1]}",
                )
            )

    for m, (wp_pos, _) in enumerate(waypoints):
        expected = gamma_semantic**m
        actual = float(wf[wp_pos])
        if abs(actual - expected) > 1e-5:
            issues.append(
                E(
                    "waypoint_decay_mismatch",
                    f"WP {m} pos {wp_pos}: field={actual:.6f} "
                    f"expected={expected:.6f}",
                    actual,
                    expected,
                )
            )

    wps = {w[0] for w in waypoints}
    for pos in route:
        if int(ct[pos]) == CELL_OBSERVATION and pos not in wps:
            if wf[pos] > 1e-7:
                issues.append(
                    E(
                        "unaccepted_observation_marked",
                        f"Crossed OBS at {pos} has wp activation "
                        f"{float(wf[pos]):.6f}",
                        float(wf[pos]),
                        0.0,
                    )
                )

    return issues


def check_auxiliary_targets(
    data: dict[str, np.ndarray],
    *,
    S: int,
    split: str = "",
    idx: int = -1,
    grid_width: int | None = None,
) -> list[ValidationIssue]:
    """Validate auxiliary targets against the oracle route.

    Checks: ``target_next_dir`` matches the first physical step,
    ``target_next_obs`` matches the observation at the second waypoint.

    Args:
        data: Sample channel dict.
        S: Number of spatial positions.
        split: Optional split label.
        idx: Optional sample index.

    Returns:
        List of validation issues.
    """
    import math

    from ehc_sn.tasks.routebind.contracts import DELTA_TO_DIRECTION, Direction
    from ehc_sn.tasks.routebind.decoding import (
        extract_route_from_trajectory_field,
        extract_waypoints_from_field,
    )

    DIRECTION_NAMES = {d.value: d.name for d in Direction}

    E = lambda c, m, o=None, e=None: _issue("ERROR", c, split, idx, m, o, e)

    sf = np.asarray(data["start_flag"])
    tf = np.asarray(data["target_trajectory"])
    wf = np.asarray(data["target_waypoint"])

    nd_raw = data.get("target_next_dir", -1)
    no_raw = data.get("target_next_obs", -1)
    nd = int(nd_raw.item()) if hasattr(nd_raw, "item") else int(nd_raw)
    no = int(no_raw.item()) if hasattr(no_raw, "item") else int(no_raw)

    if int(sf.sum()) != 1:
        return []
    sp = int(np.argmax(sf))
    route = extract_route_from_trajectory_field(tf, sp, grid_width=grid_width)
    if not route or len(route) < 2:
        return []

    width = int(math.sqrt(S))
    if grid_width is not None:
        width = grid_width
    fs = route[1]
    dr = fs // width - sp // width
    dc = fs % width - sp % width
    exp_dir_val: int = -1
    direction = DELTA_TO_DIRECTION.get((dr, dc))
    if direction is not None:
        exp_dir_val = int(direction)

    issues: list[ValidationIssue] = []

    if nd < 0 or nd > 3:
        issues.append(
            E(
                "next_direction_mismatch",
                f"target_next_dir={nd} out of [0,3]",
                nd,
                "[0,3]",
            )
        )
    elif nd != exp_dir_val:
        issues.append(
            E(
                "next_direction_mismatch",
                f"target_next_dir={nd} "
                f"({DIRECTION_NAMES.get(nd, '?')}) "
                f"but first step is "
                f"{DIRECTION_NAMES.get(exp_dir_val, '?')}",
                nd,
                exp_dir_val,
            )
        )

    waypoints = extract_waypoints_from_field(wf, route)
    if len(waypoints) >= 2:
        swp = waypoints[1][0]
        exp_obs = int(np.asarray(data["observation_id"])[swp])
        if no != exp_obs:
            issues.append(
                E(
                    "next_observation_mismatch",
                    f"target_next_obs={no} but second WP at "
                    f"{swp} has obs {exp_obs}",
                    no,
                    exp_obs,
                )
            )

    return issues


def check_dag_transitions(
    data: dict[str, np.ndarray],
    *,
    adjacency: list[list[int]],
    split: str = "",
    idx: int = -1,
    grid_width: int | None = None,
) -> list[ValidationIssue]:
    """Validate that consecutive accepted waypoint observations follow DAG edges.

    For every consecutive pair ``(o_i, o_{i+1})`` of observation IDs in the
    decoded waypoint sequence, checks that ``o_{i+1}`` is a valid successor
    of ``o_i`` in the DAG adjacency list.

    Args:
        data: Sample channel dict.
        adjacency: ``adjacency[o]`` lists valid successor public observation
            IDs for node ``o``.
        split: Optional split label for issue context.
        idx: Optional sample index for issue context.

    Returns:
        List of validation issues (empty = all transitions valid).
    """
    from ehc_sn.tasks.routebind.decoding import (
        extract_route_from_trajectory_field,
        extract_waypoint_sequence,
    )

    E = lambda c, m, o=None, e=None: _issue("ERROR", c, split, idx, m, o, e)

    ct = np.asarray(data["cell_type"])
    oid = np.asarray(data["observation_id"])
    sf = np.asarray(data["start_flag"])
    tf = np.asarray(data["target_trajectory"])
    wf = np.asarray(data["target_waypoint"])

    if int(sf.sum()) != 1:
        return []
    sp = int(np.argmax(sf))
    route = extract_route_from_trajectory_field(tf, sp, grid_width=grid_width)
    if not route:
        return []

    waypoints = extract_waypoint_sequence(wf, route, oid)
    if len(waypoints) < 2:
        return []

    issues: list[ValidationIssue] = []
    for i in range(len(waypoints) - 1):
        src_obs = waypoints[i][1]
        dst_obs = waypoints[i + 1][1]
        if src_obs < 0 or dst_obs < 0:
            continue
        if dst_obs not in adjacency[src_obs]:
            issues.append(
                E(
                    "missing_dag_edge",
                    f"Waypoint {i} (obs {src_obs}) -> "
                    f"waypoint {i + 1} (obs {dst_obs}): "
                    f"edge not in DAG adjacency",
                    f"{src_obs} -> {dst_obs}",
                    f"{dst_obs} in adjacency[{src_obs}]",
                )
            )

    return issues


__all__ = [
    "ValidationIssue",
    "check_auxiliary_targets",
    "check_dag_transitions",
    "check_route_field",
    "check_waypoint_field",
    "validate_corpus_root",
    "validate_generated_sample",
    "validate_stored_sample",
]
