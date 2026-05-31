"""Report-run manifest writer.

Produces the on-disk directory contract for one completed report run
without executing evaluation, loading checkpoints, rendering figures,
or computing metrics.

Usage::

    from ehc_sn.reporting.writer import write_report_run_manifest

    manifest = write_report_run_manifest(
        spec=report_spec,
        eval_artifacts=selected_artifacts,
        output_dir=Path("reports/tem_v2/version_12_best"),
    )

Directory layout produced::

    <output_dir>/
        report_manifest.json
        config.resolved.yaml
        provenance.json
        figures/
        tables/
        rendered/
        _SUCCESS

This module must remain free of ``lightning/``, ``eval/``, ``models/``,
``tasks/``, ``adapters/``, and benchmark imports.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ehc_sn.reporting.provenance import CodeProvenance, ReportRunProvenance
from ehc_sn.reporting.schema import (
    EvalArtifactReference,
    ReportRunManifest,
    ReportSpec,
)

_SUCCESS_FILENAME = "_SUCCESS"
_MANIFEST_FILENAME = "report_manifest.json"
_CONFIG_FILENAME = "config.resolved.yaml"
_PROVENANCE_FILENAME = "provenance.json"


# =============================================================================
def write_report_run_manifest(
    *,
    spec: ReportSpec,
    eval_artifacts: Sequence[EvalArtifactReference],
    output_dir: Path,
    code_provenance: CodeProvenance | None = None,
    overwrite: bool = False,
    metrics_records_json: Path | None = None,
    figures_index_json: Path | None = None,
) -> ReportRunManifest:
    """Initialize a report-run directory and write its manifest.

    Creates *output_dir* (and subdirectories ``figures/``, ``tables/``,
    ``rendered/``), writes ``config.resolved.yaml``, ``provenance.json``,
    ``report_manifest.json``, and a ``_SUCCESS`` sentinel.

    Parameters
    ----------
    spec:
        Report assembly specification (model family, checkpoint,
        artifact root, render toggles).
    eval_artifacts:
        Selected regime-level eval artifact references to record in
        the manifest.
    output_dir:
        Target directory for the report run.  Created if absent.
    code_provenance:
        Optional version-control metadata for the producing code.
        Stored in ``provenance.json`` under the ``code`` key.
    overwrite:
        If ``True``, remove any existing completed report run at
        *output_dir* before writing.  If ``False`` (default), a
        ``RuntimeError`` is raised when a ``_SUCCESS`` sentinel
        already exists.
    metrics_records_json:
        Optional relative path to a metrics records file within
        *output_dir*.  Stored in the manifest as
        ``metrics_records_json``.  Defaults to ``None``.

    Returns
    -------
    ReportRunManifest
        The manifest written to ``report_manifest.json``.

    Raises
    ------
    RuntimeError
        If *output_dir* already contains a completed report run
        (``_SUCCESS`` present) and *overwrite* is ``False``.
    """
    output_dir = Path(output_dir).resolve()
    success_path = output_dir / _SUCCESS_FILENAME

    if success_path.exists():
        if not overwrite:
            raise RuntimeError(
                f"Report run at {output_dir} already contains "
                f"{_SUCCESS_FILENAME!r} (completed run). "
                "Use overwrite=True to replace."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Subdirectories
    # ------------------------------------------------------------------
    for subdir in ("figures", "tables", "rendered"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # config.resolved.yaml
    # ------------------------------------------------------------------
    config_path = output_dir / _CONFIG_FILENAME
    config_path.write_text(
        yaml.dump(
            spec.model_dump(mode="json"),
            default_flow_style=False,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # provenance.json
    # ------------------------------------------------------------------
    created_at = datetime.now(timezone.utc)
    provenance = ReportRunProvenance(
        model_family=spec.model_family,
        checkpoint=spec.checkpoint,
        created_at=created_at,
        code=code_provenance,
    )
    provenance_path = output_dir / _PROVENANCE_FILENAME
    provenance_path.write_text(
        provenance.model_dump_json(indent=2),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # report_manifest.json
    # ------------------------------------------------------------------
    manifest = ReportRunManifest(
        model_family=spec.model_family,
        checkpoint=spec.checkpoint,
        eval_artifacts=list(eval_artifacts),
        metrics_records_json=metrics_records_json,
        figures_index_json=figures_index_json,
        created_at=created_at,
    )
    manifest_path = output_dir / _MANIFEST_FILENAME
    manifest_path.write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # _SUCCESS sentinel (written last)
    # ------------------------------------------------------------------
    success_path.write_text("", encoding="utf-8")

    return manifest


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
