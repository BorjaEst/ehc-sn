"""Routebind report serialization — formats and writes validation/diagnostics results.

This module owns presentation semantics only.  It does not load corpora,
run validation, compute statistics, or render figures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ehc_sn.tasks.routebind.validation import ValidationIssue


# =============================================================================
def serialize_validation_result(
    issues: list[ValidationIssue],
    sample_counts: dict[str, int],
) -> dict:
    """Serialize a validation result to a JSON-serializable dict.

    Args:
        issues: List of validation issues.
        sample_counts: Per-split sample counts.

    Returns:
        Dict with keys ``n_errors``, ``n_warnings``, ``n_info``, ``issues``,
        ``sample_counts``.
    """
    errors = [i for i in issues if i.severity == "ERROR"]
    warnings = [i for i in issues if i.severity == "WARNING"]
    info = [i for i in issues if i.severity == "INFO"]

    def _safe_value(v: object) -> object:
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()[:20]
        if isinstance(v, (list, tuple)) and len(v) > 20:
            return list(v)[:20]
        return v

    return {
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "n_info": len(info),
        "issues": [
            {
                "severity": i.severity,
                "code": i.code,
                "message": i.message,
                **({"split": i.split} if i.split else {}),
                **(
                    {"sample_index": i.sample_index}
                    if i.sample_index >= 0
                    else {}
                ),
                **(
                    {"observed": _safe_value(i.observed)}
                    if i.observed is not None
                    else {}
                ),
                **(
                    {"expected": _safe_value(i.expected)}
                    if i.expected is not None
                    else {}
                ),
            }
            for i in issues
        ],
        "sample_counts": sample_counts,
    }


def serialize_diagnostics(stats: dict) -> dict:
    """Serialize diagnostics to a JSON-serializable dict.

    Args:
        stats: Diagnostics dict from ``compute_corpus_statistics``.

    Returns:
        A copy of *stats* with numpy types converted.
    """
    return json.loads(json.dumps(stats, default=str))


# =============================================================================
def format_validation_summary(
    stats: dict,
    issues: list[ValidationIssue],
) -> str:
    """Format a human-readable validation summary string.

    Args:
        stats: Diagnostics dict (for corpus identity and statistics).
        issues: List of validation issues.

    Returns:
        Formatted multi-line string.
    """
    I = "  "
    L: list[str] = []
    L.append("=" * 72)
    L.append("Routebind corpus validation")
    L.append("=" * 72)
    L.append("")

    for k in ("task", "corpus", "version", "n_observations", "grid"):
        if k in stats:
            L.append(f"{I}{k}: {stats[k]}")
    L.append("")

    L.append("Samples:")
    for s, c in stats.get("per_split", {}).items():
        L.append(f"{I}{s}: {c}")
    L.append("")

    errs = [i for i in issues if i.severity == "ERROR"]
    warns = [i for i in issues if i.severity == "WARNING"]
    L.append(f"Errors: {len(errs)}")
    L.append(f"Warnings: {len(warns)}")
    L.append("")

    if errs:
        L.append("Errors (first 10):")
        for e in errs[:10]:
            L.append(
                f"{I}[{e.code}] {e.message} "
                f"(split={e.split}, idx={e.sample_index})"
            )
        if len(errs) > 10:
            L.append(f"{I}... and {len(errs) - 10} more")
        L.append("")

    for key, label in [
        ("route_length", "Route length"),
        ("semantic_length", "Semantic length"),
        ("wall_density", "Wall density"),
        ("goal_occurrences", "Goal occurrences"),
        ("duplicate_observations", "Duplicate obs"),
        ("detour_ratio", "Detour ratio"),
    ]:
        d = stats.get(key, {})
        if d.get("count", 0) > 0:
            L.append(
                f"{I}{label}: min={d['min']:.1f} median={d['median']:.1f} "
                f"max={d['max']:.1f} mean={d['mean']:.1f}"
            )

    zt = stats.get("zero_field_mse_trajectory", {})
    if zt.get("count", 0) > 0:
        L.append("")
        L.append("Baselines:")
        L.append(f"{I}zero-field traj MSE: median={zt['median']:.6f}")
        st_stats = stats.get("start_only_mse_trajectory", {})
        L.append(
            f"{I}start-only traj MSE: median={st_stats.get('median', 0):.6f}"
        )
        if "spatial_only_exact_match_rate" in stats:
            L.append(
                f"{I}spatial-only exact: {stats['spatial_only_exact_match_rate']:.1%}"
            )
            L.append(f"{I}closest-goal: {stats['closest_goal_rate']:.1%}")

    rej = stats.get("rejection_summary", {})
    if rej:
        L.append("")
        L.append("Rejections:")
        for s, reasons in sorted(rej.items()):
            L.append(f"{I}{s}:")
            for r, c in sorted(reasons.items()):
                L.append(f"{I}{I}{r}: {c}")

    funnel_data = stats.get("generation_funnel", {})
    if funnel_data:
        L.append("")
        L.append("Generation Funnel:")
        for split, fdict in sorted(funnel_data.items()):
            L.append(f"{I}{split}:")
            for stage in [
                "examined",
                "unreachable",
                "ambiguous",
                "eligible",
                "reconstructed",
                "bucket_full",
                "accepted",
            ]:
                entries = fdict.get(stage, {})
                if entries:
                    total = sum(int(v) for v in entries.values())
                    L.append(f"{I}{I}{stage}: total={total}")
                    for k, v in sorted(entries.items()):
                        L.append(f"{I}{I}{I}{k}: {v}")
            for scalar_key in [
                "rejected_non_simple",
                "rejected_route_too_long",
                "rejected_trivial_waypoint",
                "rejected_invalid_next_dir",
                "rejected_target_validation",
            ]:
                val = fdict.get(scalar_key, 0)
                if val:
                    L.append(f"{I}{I}{scalar_key}: {val}")

    realized_dist = stats.get("realized_distribution", {})
    if realized_dist:
        L.append("")
        L.append("Target vs Realized Distribution (joint bucket proportions):")
        for split, rdict in sorted(realized_dist.items()):
            L.append(f"{I}{split}:")
            for k, v in sorted(rdict.items()):
                L.append(f"{I}{I}{k}: {v:.4f}")

    L.append("")
    L.append("=" * 72)
    return "\n".join(L) + "\n"


def write_validation_bundle(
    issues: list[ValidationIssue],
    stats: dict,
    sample_counts: dict[str, int],
    *,
    output_dir: Path,
    json_name: str = "validation.json",
    diagnostics_name: str = "diagnostics.json",
    summary_name: str = "validation-summary.txt",
) -> dict[str, Path]:
    """Write validation JSON, diagnostics JSON, and text summary to *output_dir*.

    Args:
        issues: List of validation issues.
        stats: Diagnostics dict.
        sample_counts: Per-split sample counts.
        output_dir: Output directory (created if not exists).
        json_name: Filename for validation JSON.
        diagnostics_name: Filename for diagnostics JSON.
        summary_name: Filename for text summary.

    Returns:
        Dict mapping role to written ``Path``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_path = output_dir / json_name
    validation_path.write_text(
        json.dumps(serialize_validation_result(issues, sample_counts), indent=2)
    )

    diagnostics_path = output_dir / diagnostics_name
    diagnostics_path.write_text(
        json.dumps(serialize_diagnostics(stats), indent=2, default=str)
    )

    summary_path = output_dir / summary_name
    summary_text = format_validation_summary(stats, issues)
    summary_path.write_text(summary_text)

    return {
        "validation": validation_path,
        "diagnostics": diagnostics_path,
        "summary": summary_path,
    }


__all__ = [
    "format_validation_summary",
    "serialize_diagnostics",
    "serialize_validation_result",
    "write_validation_bundle",
]
