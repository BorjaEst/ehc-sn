"""Report-run directory reader and structural verifier.

This module provides the symmetric read counterpart to
:func:`~ehc_sn.reporting.writer.write_report_run_manifest`.

Usage::

    from ehc_sn.reporting import load_report_run, ReportRun

    report = load_report_run(Path("reports/tem_v2/version_12_best"))
    report.manifest.eval_artifacts  # authoritative artifact list
    report.provenance               # creation metadata
    report.metrics                  # lazy-loaded metric records
    report.figures                  # lazy-loaded figure index
    report.rendered                 # lazy-loaded render manifest

The reader is a **strict verifier**: any structural defect raises an error.
It does not parse ``config.resolved.yaml``, metric records, figures, tables,
rendered outputs, or ``source_spec``.

This module must remain free of ``lightning/``, ``eval/``, ``models/``,
``tasks/``, ``adapters/``, and benchmark imports.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from ehc_sn.reporting.loaders import load_eval_artifact_manifest
from ehc_sn.reporting.provenance import ReportRunProvenance
from ehc_sn.reporting.schema import (
    FigureIndex,
    MetricRecord,
    RenderManifest,
    ReportRunManifest,
)


# ---------------------------------------------------------------------------
# Constants (mirrored from writer.py to avoid import coupling)
# ---------------------------------------------------------------------------

_MANIFEST_FILENAME = "report_manifest.json"
_PROVENANCE_FILENAME = "provenance.json"
_SUCCESS_FILENAME = "_SUCCESS"
_REQUIRED_SCHEMA_VERSION = "ehc_sn.reporting.report_run.v1"


# ---------------------------------------------------------------------------
# Public type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportRun:
    """Immutable view of one verified report-run directory.

    Core fields (``root``, ``manifest``, ``provenance``) are populated at
    construction time by :func:`load_report_run`.  Convenience accessors
    (``metrics``, ``figures``, ``rendered``) are lazily loaded via
    ``cached_property``.  ``metrics`` and ``figures`` return empty
    lists/indexes when no data exists; ``rendered`` raises
    ``FileNotFoundError`` if the report has not been rendered.

    .. warning::

        Do **not** add ``slots=True``.  The ``cached_property`` accessors
        require an instance ``__dict__``.

    Fields
    ------
    root:
        Absolute path to the report-run directory.
    manifest:
        Parsed and validated ``report_manifest.json``.
    provenance:
        Parsed and validated ``provenance.json``.
    """

    root: Path
    manifest: ReportRunManifest
    provenance: ReportRunProvenance

    # ------------------------------------------------------------------
    # Lazy-loaded convenience accessors
    # ------------------------------------------------------------------

    @cached_property
    def metrics(self) -> list[MetricRecord]:
        """Metric records from ``metrics_records_json``, or ``[]``."""
        return _load_metric_records(self)

    @cached_property
    def figures(self) -> FigureIndex:
        """Figure index from ``figures_index_json``, or empty index."""
        return _load_figure_index(self)

    @cached_property
    def rendered(self) -> RenderManifest:
        """Render manifest for this report run.

        Raises ``FileNotFoundError`` if the report has not been
        rendered (``rendered/_SUCCESS`` is missing).
        """
        return _load_render_manifest(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_contained(
    root: Path, relative: Path, *, field_name: str
) -> None:
    """Verify *relative* is non-absolute and stays inside *root*.

    Raises ``RuntimeError`` if the path is absolute or ``../`` escapes
    beyond the report-run directory.
    """
    if relative.is_absolute():
        raise RuntimeError(
            f"Corrupt report manifest: {field_name} must be relative, "
            f"got absolute path: {relative}"
        )
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        raise RuntimeError(
            f"Corrupt report manifest: {field_name} escapes report-run "
            f"directory: {relative}"
        )


def _load_json(path: Path) -> dict[str, Any]:
    """Load and return JSON from *path*.  Raises ``FileNotFoundError`` or
    ``ValueError`` on malformed JSON."""
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Malformed JSON in {path}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Lazy-load helpers for ReportRun convenience accessors
# ---------------------------------------------------------------------------


def _load_metric_records(report: ReportRun) -> list[MetricRecord]:
    """Load metric records from *report*, returning ``[]`` when absent."""
    rel_path = report.manifest.metrics_records_json
    if rel_path is None:
        return []

    target = (report.root / rel_path).resolve()
    if not target.exists():
        return []

    raw = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []

    records: list[MetricRecord] = []
    for item in raw:
        records.append(MetricRecord.model_validate(item))
    return records


def _load_figure_index(report: ReportRun) -> FigureIndex:
    """Load figure index from *report*, returning empty index when absent."""
    rel_path = report.manifest.figures_index_json
    if rel_path is None:
        return FigureIndex(entries=[])

    target = (report.root / rel_path).resolve()
    if not target.exists():
        return FigureIndex(entries=[])

    raw = json.loads(target.read_text(encoding="utf-8"))
    return FigureIndex.model_validate(raw)


def _load_render_manifest(report: ReportRun) -> RenderManifest:
    """Load render manifest from *report*.

    Uses a lazy import to avoid a circular dependency with
    :mod:`ehc_sn.reporting.render`.

    Raises ``FileNotFoundError`` if the report has not been rendered.
    """
    from ehc_sn.reporting.render import load_render_manifest

    return load_render_manifest(report)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_report_run(path: Path) -> ReportRun:
    """Load and validate a report-run directory at *path*.

    Validates the directory structure, parses manifest and provenance
    into their typed models, verifies timestamp consistency, checks
    output-path safety, and cross-validates referenced eval artifacts.

    Parameters
    ----------
    path:
        Path to the report-run directory.

    Returns
    -------
    ReportRun
        Immutable view with parsed manifest and provenance.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist, is not a directory, or a required file
        (``report_manifest.json``, ``provenance.json``, ``_SUCCESS``,
        or a referenced eval artifact) is missing.
    ValueError
        If JSON is malformed, pydantic validation fails, or the manifest
        ``schema_version`` is not ``ehc_sn.reporting.report_run.v1``.
    RuntimeError
        If ``manifest.created_at != provenance.created_at``, an output
        path is absolute or escapes the report-run directory, or a
        referenced eval artifact's identity fields do not match the
        manifest reference.
    """
    root = Path(path).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Report-run directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(
            f"Report-run path is not a directory: {root}"
        )

    # ---- Required files ---------------------------------------------------
    manifest_path = root / _MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {_MANIFEST_FILENAME} in report-run directory: {root}"
        )

    provenance_path = root / _PROVENANCE_FILENAME
    if not provenance_path.exists():
        raise FileNotFoundError(
            f"Missing {_PROVENANCE_FILENAME} in report-run directory: {root}"
        )

    success_path = root / _SUCCESS_FILENAME
    if not success_path.exists():
        raise RuntimeError(
            f"Report-run at {root} is missing {_SUCCESS_FILENAME} sentinel "
            f"(possibly an incomplete or corrupted write)."
        )

    # ---- Parse models -----------------------------------------------------
    manifest_raw = _load_json(manifest_path)
    provenance_raw = _load_json(provenance_path)

    try:
        manifest = ReportRunManifest.model_validate(manifest_raw)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse {_MANIFEST_FILENAME} as ReportRunManifest: {exc}"
        ) from exc

    try:
        provenance = ReportRunProvenance.model_validate(provenance_raw)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse {_PROVENANCE_FILENAME} as ReportRunProvenance: {exc}"
        ) from exc

    # ---- Schema version ---------------------------------------------------
    if manifest.schema_version != _REQUIRED_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported report-run schema version: "
            f"{manifest.schema_version!r}. "
            f"Expected {_REQUIRED_SCHEMA_VERSION!r}."
        )

    # ---- created_at equality ----------------------------------------------
    if manifest.created_at != provenance.created_at:
        raise RuntimeError(
            f"{_MANIFEST_FILENAME}.created_at and "
            f"{_PROVENANCE_FILENAME}.created_at differ: "
            f"{manifest.created_at} vs {provenance.created_at}. "
            f"Timestamp inconsistency detected."
        )

    # ---- Safe relative output paths ---------------------------------------
    _validate_contained(root, manifest.provenance_json, field_name="provenance_json")
    _validate_contained(
        root, manifest.resolved_config_yaml, field_name="resolved_config_yaml"
    )
    _validate_contained(root, manifest.figures_dir, field_name="figures_dir")
    _validate_contained(root, manifest.tables_dir, field_name="tables_dir")
    _validate_contained(root, manifest.rendered_dir, field_name="rendered_dir")
    if manifest.metrics_records_json is not None:
        _validate_contained(
            root,
            manifest.metrics_records_json,
            field_name="metrics_records_json",
        )
    if manifest.figures_index_json is not None:
        _validate_contained(
            root,
            manifest.figures_index_json,
            field_name="figures_index_json",
        )

    # ---- Eval artifact cross-validation -----------------------------------
    for ref in manifest.eval_artifacts:
        ref_path = Path(ref.path)
        if not ref_path.exists():
            raise FileNotFoundError(
                f"Referenced eval artifact directory no longer exists: "
                f"{ref_path} (regime_id={ref.regime_id!r}, "
                f"task={ref.task!r})"
            )

        eval_manifest = load_eval_artifact_manifest(ref_path)

        # schema version is already validated inside load_eval_artifact_manifest
        if ref.task != eval_manifest.get("task"):
            raise RuntimeError(
                f"Eval artifact identity mismatch for regime_id={ref.regime_id!r}: "
                f"manifest reference task={ref.task!r} but eval manifest "
                f"task={eval_manifest.get('task')!r}."
            )
        if ref.regime_id != eval_manifest.get("regime_id"):
            raise RuntimeError(
                f"Eval artifact identity mismatch: manifest reference "
                f"regime_id={ref.regime_id!r} but eval manifest "
                f"regime_id={eval_manifest.get('regime_id')!r}."
            )
        if ref.regime_kind != eval_manifest.get("regime_kind"):
            raise RuntimeError(
                f"Eval artifact identity mismatch for regime_id={ref.regime_id!r}: "
                f"manifest reference regime_kind={ref.regime_kind!r} but eval "
                f"manifest regime_kind={eval_manifest.get('regime_kind')!r}."
            )

    return ReportRun(root=root, manifest=manifest, provenance=provenance)
