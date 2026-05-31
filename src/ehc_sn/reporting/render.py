"""Post-hoc Markdown/HTML rendering for completed ``ReportRun`` directories.

This module consumes an existing :class:`ReportRun` (loaded via
:func:`load_report_run`) and produces rendered output files under
``<report_root>/rendered/``.  It is a pure consumer of the report
contract — it does **not** import ``ehc_sn.eval``, Lightning, PyTorch,
or any model/task/adapter module.

Usage::

    from ehc_sn.reporting import load_report_run, render_report_run

    report = load_report_run("reports/tem_v2/version_12_best")
    manifest = render_report_run(report, formats=("markdown",))

    # Or pass a path directly:
    manifest = render_report_run("reports/tem_v2/version_12_best")
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import markdown as _markdown

from ehc_sn.reporting.reader import ReportRun, load_report_run
from ehc_sn.reporting.schema import (
    FigureIndex,
    MetricRecord,
    RenderManifest,
    RenderedOutput,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RENDERED_DIR = "rendered"
_RENDER_MANIFEST_FILENAME = "render_manifest.json"
_HTML_FILENAME = "report.html"

SUPPORTED_FORMATS: frozenset[str] = frozenset({"markdown", "html"})
"""Set of output formats accepted by :func:`render_report_run`."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rendered_dir(report: ReportRun) -> Path:
    """Return the rendered output directory for a report run."""
    return report.root / _RENDERED_DIR


def _ensure_rendered_dir(report: ReportRun, *, overwrite: bool) -> Path:
    """Create or reset the rendered directory.

    Raises ``FileExistsError`` if the directory exists with outputs
    (including partial/orphaned state) and *overwrite* is ``False``.
    """
    target = _rendered_dir(report)
    if not target.exists():
        target.mkdir(parents=True)
        return target

    success_path = target / "_SUCCESS"
    has_success = success_path.exists()
    has_any_output = any(f != "_SUCCESS" for f in (p.name for p in target.iterdir()))

    if has_success:
        if not overwrite:
            raise FileExistsError(
                f"Rendered outputs already exist at {target}. "
                f"Use overwrite=True to replace."
            )
        _clear_rendered_dir(target)
    elif has_any_output:
        if not overwrite:
            raise FileExistsError(
                f"Rendered directory at {target} has partial/orphaned "
                f"outputs from a previous failed render. "
                f"Use overwrite=True to replace."
            )
        _clear_rendered_dir(target)
    else:
        # Empty directory (no outputs, no _SUCCESS) — safe to use.
        pass

    return target


def _clear_rendered_dir(target: Path) -> None:
    """Remove all contents of *target* and recreate it empty."""
    import shutil

    shutil.rmtree(target)
    target.mkdir(parents=True)


def _format_provenance_table(report: ReportRun) -> str:
    """Return a Markdown provenance section."""
    p = report.provenance
    lines = ["## Provenance", ""]
    if p.code:
        if p.code.git_commit:
            lines.append(f"- **Git commit:** `{p.code.git_commit}`")
        if p.code.git_dirty is not None:
            lines.append(f"- **Git dirty:** {p.code.git_dirty}")
    else:
        lines.append("- No code provenance recorded.")
    lines.append("")
    return "\n".join(lines)


def _format_artifacts_table(report: ReportRun) -> str:
    """Return a Markdown evaluated-artifacts table."""
    lines = [
        "## Evaluated Artifacts",
        "",
        "| Regime | Task | Kind | Path |",
        "|--------|------|------|------|",
    ]
    for ref in report.manifest.eval_artifacts:
        lines.append(
            f"| {ref.regime_id} | {ref.task} | {ref.regime_kind} "
            f"| `{ref.path}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _format_metrics_section(report: ReportRun) -> str:
    """Return a Markdown metrics section."""
    lines = ["## Metrics", ""]
    records = report.metrics
    if not records:
        lines.append("_No metrics recorded._")
        lines.append("")
        return "\n".join(lines)

    lines.append(
        "| Metric | Value | Unit | Higher is better | Task | Regime |"
    )
    lines.append("|--------|------:|:----:|:----------------:|:----:|:------:|")
    for r in records:
        val = _format_metric_value(r.value)
        unit = r.unit or "—"
        hib = "✓" if r.higher_is_better else "—" if r.higher_is_better is None else "✗"
        task = r.task or "—"
        regime = r.regime_id or "—"
        lines.append(f"| {r.metric} | {val} | {unit} | {hib} | {task} | {regime} |")
    lines.append("")
    return "\n".join(lines)


def _format_metric_value(value: float | int | bool | str | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _format_figures_section(report: ReportRun) -> str:
    """Return a Markdown figures section."""
    lines = ["## Figures", ""]
    index = report.figures
    if not index.entries:
        lines.append("_No figures recorded._")
        lines.append("")
        return "\n".join(lines)

    lines.append("| ID | Task | Regime | Format | Title |")
    lines.append("|----|------|--------|--------|-------|")
    for entry in index.entries:
        title = entry.title or "—"
        lines.append(
            f"| {entry.figure_id} | {entry.task} | {entry.regime_id} "
            f"| {entry.format} | {title} |"
        )
    lines.append("")
    return "\n".join(lines)


def _build_markdown(report: ReportRun) -> str:
    """Build the full Markdown report content."""
    lines: list[str] = []

    # Header
    lines.append("# Report")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Model family | {report.manifest.model_family} |")
    ckpt = report.manifest.checkpoint
    lines.append(f"| Checkpoint | `{ckpt.path}` |")
    lines.append(f"| Created at | {report.manifest.created_at.isoformat()} |")
    lines.append("")

    # Provenance
    lines.append(_format_provenance_table(report))

    # Evaluated artifacts
    lines.append(_format_artifacts_table(report))

    # Metrics
    lines.append(_format_metrics_section(report))

    # Figures
    lines.append(_format_figures_section(report))

    return "\n".join(lines)


def _write_html(report: ReportRun, target_dir: Path) -> Path:
    """Write ``report.html`` to *target_dir* and return its path.

    Converts the same Markdown content model used for ``report.md``
    to HTML via the ``markdown`` library.
    """
    markdown_text = _build_markdown(report)
    html_body = _markdown.markdown(markdown_text, extensions=["tables"])
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Report</title>
</head>
<body>
{html_body}
</body>
</html>
"""
    path = target_dir / _HTML_FILENAME
    path.write_text(html_doc, encoding="utf-8")
    return path


def _write_markdown(report: ReportRun, target_dir: Path) -> Path:
    """Write ``report.md`` to *target_dir* and return its path."""
    content = _build_markdown(report)
    path = target_dir / "report.md"
    path.write_text(content, encoding="utf-8")
    return path


def _write_render_manifest(
    outputs: list[RenderedOutput],
    target_dir: Path,
) -> Path:
    """Write ``render_manifest.json`` to *target_dir* and return its path."""
    manifest = RenderManifest(
        outputs=outputs,
        created_at=datetime.now(timezone.utc),
    )
    path = target_dir / _RENDER_MANIFEST_FILENAME
    path.write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_report_run(
    report: ReportRun | Path,
    *,
    formats: Sequence[Literal["markdown", "html"]] = ("markdown",),
    overwrite: bool = False,
) -> RenderManifest:
    """Render a completed report run into ``<root>/rendered/``.

    Parameters
    ----------
    report:
        An already-loaded :class:`ReportRun` object, or a ``Path`` to a
        valid report-run directory (auto-loaded via
        :func:`load_report_run`).
    formats:
        Output formats to produce.  Supported values:
        ``"markdown"``, ``"html"``.
    overwrite:
        If ``True``, replace any existing ``rendered/`` directory.
        If ``False``, raise ``FileExistsError`` when ``rendered/``
        already has outputs (including partial/orphaned state).

    Returns
    -------
    RenderManifest
        Manifest recording what was rendered and where.

    Raises
    ------
    FileNotFoundError
        If *report* is a ``Path`` that does not point to a valid
        report-run directory (delegated to :func:`load_report_run`).
    FileExistsError
        If ``rendered/`` already has outputs and *overwrite* is
        ``False``.
    ValueError
        If *formats* contains any format not in
        ``SUPPORTED_FORMATS``.
    """
    # Resolve report.
    if isinstance(report, Path):
        report = load_report_run(report)
    elif not isinstance(report, ReportRun):
        raise TypeError(
            f"Expected ReportRun or Path, got {type(report).__name__}."
        )

    # Validate formats.
    unsupported = set(formats) - SUPPORTED_FORMATS
    if unsupported:
        raise ValueError(
            f"Unsupported render format(s): {sorted(unsupported)}. "
            f"Supported formats: {sorted(SUPPORTED_FORMATS)}."
        )

    # Prepare rendered directory.
    target_dir = _ensure_rendered_dir(report, overwrite=overwrite)

    # Produce outputs.
    now = datetime.now(timezone.utc)
    outputs: list[RenderedOutput] = []

    if "markdown" in formats:
        md_path = _write_markdown(report, target_dir)
        outputs.append(
            RenderedOutput(
                format="markdown",
                relative_path=md_path.relative_to(report.root),
                created_at=now,
            )
        )

    if "html" in formats:
        html_path = _write_html(report, target_dir)
        outputs.append(
            RenderedOutput(
                format="html",
                relative_path=html_path.relative_to(report.root),
                created_at=now,
            )
        )

    # Write manifest.
    _write_render_manifest(outputs, target_dir)

    # Write _SUCCESS last.
    (target_dir / "_SUCCESS").touch()

    return RenderManifest(
        outputs=outputs,
        created_at=now,
    )


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def load_render_manifest(report: ReportRun | Path) -> RenderManifest:
    """Load and validate the render manifest for a completed ReportRun.

    Parameters
    ----------
    report:
        A loaded :class:`ReportRun`, or a ``Path`` to the report-run
        root directory.

    Returns
    -------
    RenderManifest
        Validated render manifest.

    Raises
    ------
    FileNotFoundError
        If the report root does not exist, is not a directory,
        ``rendered/_SUCCESS`` is missing, ``render_manifest.json``
        is missing, or any declared output file is missing from disk.
    ValueError
        If a ``RenderedOutput.relative_path`` is absolute, escapes the
        report-run root, or points to a non-file.
    """
    # Resolve root.
    if isinstance(report, ReportRun):
        root = report.root
    else:
        root = report.resolve()

    if not root.exists():
        raise FileNotFoundError(f"Report-run directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(
            f"Report-run path is not a directory: {root}"
        )

    rendered_dir = root / _RENDERED_DIR
    success_path = rendered_dir / "_SUCCESS"
    manifest_path = rendered_dir / _RENDER_MANIFEST_FILENAME

    if not success_path.exists():
        raise FileNotFoundError(
            f"No rendered outputs at {rendered_dir}: "
            f"_SUCCESS sentinel is missing."
        )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Render manifest not found at {manifest_path}."
        )

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = RenderManifest.model_validate(raw)

    # Validate every declared output.
    root_resolved = root.resolve()
    for out in manifest.outputs:
        rel = out.relative_path
        if rel.is_absolute():
            raise ValueError(
                f"Corrupt render manifest: relative_path must be "
                f"relative, got absolute {rel!s}."
            )
        resolved = (root_resolved / rel).resolve()
        try:
            resolved.relative_to(root_resolved)
        except ValueError:
            raise ValueError(
                f"Corrupt render manifest: relative_path {rel!s} "
                f"escapes the report-run root."
            )
        if not resolved.exists():
            raise FileNotFoundError(
                f"Declared render output {rel!s} is missing from disk."
            )
        if not resolved.is_file():
            raise ValueError(
                f"Declared render output {rel!s} exists but is not a "
                f"regular file."
            )

    return manifest
