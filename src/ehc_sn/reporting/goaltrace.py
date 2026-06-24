"""Goaltrace report serialization — formats and writes validation/diagnostics results.

Pattern mirrors ``reporting/routebind.py``.  This module owns presentation
semantics only.  It does not load corpora, run validation, compute statistics,
or render figures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ehc_sn.tasks.goaltrace.validation import GoaltraceValidationIssue


# =============================================================================
def serialize_validation_result(
    issues: list[GoaltraceValidationIssue],
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
    issues: list[GoaltraceValidationIssue],
) -> str:
    """Format a human-readable validation summary string.

    Args:
        stats: Diagnostics dict.
        issues: List of validation issues.

    Returns:
        Formatted multi-line string.
    """
    I = "  "
    L: list[str] = []
    L.append("=" * 72)
    L.append("Goaltrace corpus validation")
    L.append("=" * 72)
    L.append("")

    L.append(f"{I}Corpus: {stats.get('corpus', '?')}")
    L.append(f"{I}Version: {stats.get('version', '?')}")
    L.append(f"{I}Observations: {stats.get('n_observations', '?')}")
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

    # Per-split diagnostics
    for split in stats.get("per_split", {}):
        L.append(f"--- {split} ---")
        dc = stats.get("decay_consistency", {}).get(split, {})
        if dc:
            L.append(
                f"{I}Decay consistent: {dc.get('passes', 0)}/"
                f"{dc.get('total', 0)}"
            )
        iu = stats.get("input_uniqueness", {}).get(split, {})
        if iu:
            L.append(
                f"{I}Unique keys: {iu.get('unique_keys', 0)}, "
                f"Contradictions: {iu.get('contradictions', 0)}"
            )
        wo = stats.get("weight_overlap", {}).get(split, {})
        if wo:
            L.append(
                f"{I}Weight edge mu={wo.get('edge_mean', 0):.3f}, "
                f"noedge mu={wo.get('noedge_mean', 0):.3f}, "
                f"overlap={wo.get('overlap', 0):.3f}"
            )
        rp = stats.get("reachable_pairs", {}).get(split, {})
        if rp:
            L.append(
                f"{I}Reachable pairs: {rp.get('reachable', 0)}/"
                f"{rp.get('total', 0)}"
            )
        ed = stats.get("edge_density", {}).get(split, 0.0)
        if ed:
            L.append(f"{I}Edge density: {ed * 100:.1f}%")
        bl = stats.get("baselines", {}).get(split, {})
        if bl:
            L.append(
                f"{I}Baseline current_only: {bl.get('current_only', 0):.4f}"
            )
        L.append("")

    L.append("=" * 72)
    return "\n".join(L) + "\n"


def write_validation_bundle(
    issues: list[GoaltraceValidationIssue],
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
