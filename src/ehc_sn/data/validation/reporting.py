"""Shared validation reporting helpers for data-gen CLIs.

Provides :func:`write_validation_bundle`, :func:`serialize_validation_result`,
and :func:`format_validation_summary` — the three functions called by task-corpus
validate commands that write structured report files, JSON output, and
text summaries.

These helpers are issue-class-agnostic: they use duck-typed access to
``.severity``, ``.code``, ``.split``, ``.sample_index``, and ``.message``
attributes, matching all five task-specific ``*ValidationIssue`` classes
(``ArenaValidationIssue``, ``GoaltraceValidationIssue``,
``MazeHardValidationIssue``, ``SeqMazeValidationIssue``,
``routebind.validation.ValidationIssue``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def write_validation_bundle(
    issues: Iterable[Any],
    stats: dict[str, Any],
    sample_counts: dict[str, int],
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Write validation report, diagnostics, and summary to *output_dir*.

    Args:
        issues: Iterable of issue objects with ``.severity``, ``.code``,
            ``.split``, ``.sample_index``, ``.message`` attributes.
        stats: Corpus statistics dict (task-specific keys).
        sample_counts: Split name → sample count.
        output_dir: Output directory (created if missing).

    Returns:
        Dict with keys ``validation``, ``diagnostics``, ``summary``
        pointing to the written file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    issues_list = list(issues)
    errors = [i for i in issues_list if i.severity == "ERROR"]
    warnings = [i for i in issues_list if i.severity == "WARNING"]

    # Validation report
    report_path = output_dir / "validation_report.txt"
    with report_path.open("w") as f:
        f.write(f"Corpus validation: {stats.get('corpus', '?')}\n")
        f.write(f"  Version: {stats.get('version', '?')}\n")
        f.write(f"  Validated splits: {list(sample_counts.keys())}\n")
        f.write(f"  Errors: {len(errors)}\n")
        f.write(f"  Warnings: {len(warnings)}\n")
        for i in issues_list:
            f.write(f"  [{i.split}/{i.sample_index}] {i.code}: {i.message}\n")

    # Diagnostics
    diag_path = output_dir / "diagnostics.txt"
    with diag_path.open("w") as f:
        f.write("Corpus diagnostics\n")
        f.write("==================\n")
        for split_name, count in sample_counts.items():
            f.write(f"\n{split_name}: {count} samples\n")
            for key, value in stats.items():
                if key in ("corpus", "version", "per_split"):
                    continue
                per_split = value if isinstance(value, dict) else {}
                split_val = per_split.get(split_name)
                if split_val is not None:
                    f.write(f"  {key}: {split_val}\n")

    # Summary
    summary_path = output_dir / "summary.txt"
    with summary_path.open("w") as f:
        f.write(f"Corpus: {stats.get('corpus', '?')}\n")
        f.write(f"Version: {stats.get('version', '?')}\n")
        f.write(f"Total samples: {sum(sample_counts.values())}\n")
        f.write(f"Errors: {len(errors)}\n")
        f.write(f"Warnings: {len(warnings)}\n")

    return {
        "validation": report_path,
        "diagnostics": diag_path,
        "summary": summary_path,
    }


def serialize_validation_result(
    issues: Iterable[Any],
    sample_counts: dict[str, int],
) -> str:
    """Serialize validation issues and sample counts to a JSON string.

    Args:
        issues: Iterable of issue objects with ``.severity``, ``.code``,
            ``.split``, ``.sample_index``, ``.message`` attributes.
        sample_counts: Split name → sample count.

    Returns:
        JSON string with ``n_errors``, ``n_warnings``, ``n_total``,
        ``samples_validated``, and ``issues`` keys.
    """
    issues_list = list(issues)
    errors = [i for i in issues_list if i.severity == "ERROR"]
    warnings = [i for i in issues_list if i.severity == "WARNING"]

    result = {
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "n_total": len(issues_list),
        "samples_validated": sample_counts,
        "issues": [
            {
                "severity": i.severity,
                "code": i.code,
                "split": i.split,
                "sample_index": i.sample_index,
                "message": i.message,
            }
            for i in issues_list
        ],
    }
    return json.dumps(result, indent=2)


def format_validation_summary(
    stats: dict[str, Any],
    issues: Iterable[Any],
) -> str:
    """Format a human-readable validation summary string.

    Args:
        stats: Corpus statistics dict.
        issues: Iterable of issue objects with ``.severity`` attribute.

    Returns:
        Plain-text summary suitable for writing to a summary file.
    """
    issues_list = list(issues)
    errors = [i for i in issues_list if i.severity == "ERROR"]
    warnings = [i for i in issues_list if i.severity == "WARNING"]

    lines = [
        f"Corpus: {stats.get('corpus', '?')}",
        f"Version: {stats.get('version', '?')}",
        f"Errors: {len(errors)}",
        f"Warnings: {len(warnings)}",
        f"Total issues: {len(issues_list)}",
    ]

    if errors:
        lines.append("")
        lines.append("Errors:")
        for e in errors[:10]:
            lines.append(
                f"  [{e.split}/{e.sample_index}] {e.code}: {e.message}"
            )

    return "\n".join(lines) + "\n"
