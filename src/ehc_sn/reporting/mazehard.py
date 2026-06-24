"""MazeHard report serialization — formats and writes validation/diagnostics results.

Pattern mirrors ``reporting/arena.py``.  This module owns presentation
semantics only.  It does not load corpora, run validation, compute statistics,
or render figures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ehc_sn.tasks.mazehard.validation import MazeHardValidationIssue


# =============================================================================
def serialize_validation_result(
    issues: list[MazeHardValidationIssue],
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
    warnings_list = [i for i in issues if i.severity == "WARNING"]
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
        "n_warnings": len(warnings_list),
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
    issues: list[MazeHardValidationIssue],
) -> str:
    """Format a human-readable validation summary string.

    Args:
        stats: Diagnostics dict (for corpus identity and statistics).
        issues: Validation issues list.

    Returns:
        Multiline string.
    """
    I = "  "
    L: list[str] = []
    L.append("=" * 72)
    L.append("MazeHard corpus validation")
    L.append("=" * 72)
    L.append("")

    L.append(f"{I}Corpus: {stats.get('corpus', '?')}")
    L.append(f"{I}Version: {stats.get('version', '?')}")
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

    for split in stats.get("per_split", {}):
        L.append(f"--- {split} ---")
        wd = stats.get("wall_density", {}).get(split, {})
        if wd.get("min") is not None:
            L.append(
                f"{I}Wall density: min={wd['min']:.3f} "
                f"mean={wd['mean']:.3f} max={wd['max']:.3f}"
            )
        sl = stats.get("solution_length", {}).get(split, {})
        if sl.get("min") is not None:
            L.append(
                f"{I}Solution path length: min={sl['min']} "
                f"mean={sl['mean']:.1f} max={sl['max']}"
            )
        L.append("")

    L.append("=" * 72)
    return "\n".join(L) + "\n"


def write_validation_bundle(
    issues: list[MazeHardValidationIssue],
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
        issues: Validation issues list.
        stats: Diagnostics dict.
        sample_counts: Per-split sample counts.
        output_dir: Output directory.
        json_name: Validation JSON filename.
        diagnostics_name: Diagnostics JSON filename.
        summary_name: Summary text filename.

    Returns:
        Dict with keys ``validation``, ``diagnostics``, ``summary`` mapping to
        written file paths.
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
