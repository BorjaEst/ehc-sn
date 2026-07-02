"""Report-facing helpers for HRM latent-dynamics diagnostics.

Bridges artifact trace data → compute_hrm_dynamics_metrics → report-ready rows.

Usage::

    from ehc_sn.diagnostics.reporting import compute_hrm_dynamics_report_rows

    rows = compute_hrm_dynamics_report_rows("path/to/eval_artifact")
    for r in rows:
        print(r["metric"], r["value"], r["aggregation"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ehc_sn.diagnostics.hrm_dynamics import (
    compute_hrm_dynamics_metrics_from_trace,
)
from ehc_sn.evaluation.artifacts import load_artifact_run_cases
from ehc_sn.traces.keys import PFC_TRACE_KEY_Z_H, PFC_TRACE_KEY_Z_L

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_hrm_dynamics_report_rows(
    artifact_dir: str | Path,
) -> list[dict[str, Any]]:
    """Load an evaluation artifact and compute aggregated HRM dynamics metrics.

    Loads all case traces from *artifact_dir*, computes
    ``hrm_dynamics_metrics`` per case, and aggregates results across cases.

    Args:
        artifact_dir: Path to a persisted evaluation artifact directory
            containing ``manifest.json`` and case traces.

    Returns:
        List of dict rows, each with keys:
            ``metric`` — metric name string
            ``value`` — aggregated scalar value
            ``unit`` — measurement unit (``"a.u."``)
            ``aggregation`` — how the value was computed
            ``n_cases`` — number of cases contributing to the aggregation

    Raises:
        FileNotFoundError: If the artifact directory does not exist or
            contains no trace-bearing cases.
        ValueError: If no case traces contain both ``pfc/z_H`` and
            ``pfc/z_L``, or if the trace data has unexpected shape.
    """
    artifact_dir = Path(artifact_dir)
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Artifact directory not found: {artifact_dir}")

    loaded = load_artifact_run_cases(artifact_dir)
    if not loaded:
        raise FileNotFoundError(
            f"No trace-bearing cases found in artifact: {artifact_dir}"
        )

    # Compute per-case metrics.
    per_case: list[dict[str, float]] = []
    for case in loaded:
        if case.trace is None:
            continue
        metrics = compute_hrm_dynamics_metrics_from_trace(case.trace)
        per_case.append(metrics)

    if not per_case:
        raise ValueError(
            f"No case traces contained {PFC_TRACE_KEY_Z_H} and {PFC_TRACE_KEY_Z_L} "
            f"in artifact: {artifact_dir}"
        )

    return _aggregate_rows(per_case)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_METRIC_NAMES: tuple[str, ...] = (
    "h_state_norm_mean",
    "l_state_norm_mean",
    "h_state_delta_mean",
    "l_state_delta_mean",
    "h_l_delta_ratio",
)
"""Ordered metric keys produced by compute_hrm_dynamics_metrics."""


def _aggregate_rows(
    per_case: list[dict[str, float]],
) -> list[dict[str, Any]]:
    """Aggregate per-case metrics into report rows with mean/std."""
    n = len(per_case)
    rows: list[dict[str, Any]] = []

    for metric in _METRIC_NAMES:
        values = np.array([c[metric] for c in per_case])
        finite = np.isfinite(values)
        n_finite = int(finite.sum())

        if n_finite == 0:
            # All values were infinite or NaN; skip.
            continue

        mean_val = float(values[finite].mean())
        std_val = float(values[finite].std()) if n_finite > 1 else 0.0

        rows.append(
            {
                "metric": metric,
                "value": mean_val,
                "unit": "a.u.",
                "aggregation": f"mean ± std over {n_finite} cases",
                "n_cases": n_finite,
            }
        )

        # Include standard deviation as a separate row for transparency.
        if n_finite > 1:
            rows.append(
                {
                    "metric": f"{metric}_std",
                    "value": std_val,
                    "unit": "a.u.",
                    "aggregation": f"std over {n_finite} cases",
                    "n_cases": n_finite,
                }
            )

    return rows
