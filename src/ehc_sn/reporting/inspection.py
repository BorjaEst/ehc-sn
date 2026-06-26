"""Notebook-friendly report inspection API.

Provides :func:`open_report` as the single entry point for notebooks and
scripts to load and inspect a completed :class:`ReportRun`.  This module
does **not** run benchmarks, execute evaluation, render figures, or
mutate any on-disk artifacts.

Usage::

    from ehc_sn.reporting import open_report

    report = open_report("artifacts/reports/tem_v2_best")

    report.summary()           # compact dict
    report.metrics()           # list of MetricRecord
    report.metric_rows()       # list of plain dicts for notebook tables
    report.figures()           # FigureIndex or None
    report.provenance()        # ReportRunProvenance
    report.eval_artifacts()    # list of EvalArtifactReference
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ehc_sn.reporting.provenance import ReportRunProvenance
from ehc_sn.reporting.reader import ReportRun, load_report_run
from ehc_sn.reporting.schema import (
    EvalArtifactReference,
    FigureIndex,
    FigureIndexEntry,
    MetricRecord,
)


# =============================================================================
@dataclass(frozen=True)
class OpenReport:
    """Immutable inspection surface for one completed report run.

    Wraps a verified :class:`~ehc_sn.reporting.reader.ReportRun` and
    exposes notebook-friendly accessors.  All data is read-only and
    derived from the underlying on-disk report artifacts.
    """

    _run: ReportRun

    # -- identity ------------------------------------------------------------

    @property
    def root(self) -> Path:
        """Absolute path to the report-run directory."""
        return self._run.root

    @property
    def model_family(self) -> str:
        """Canonical model-family identifier."""
        return self._run.manifest.model_family

    # -- provenance ----------------------------------------------------------

    def provenance(self) -> ReportRunProvenance:
        """Return the provenance record for this report run."""
        return self._run.provenance

    # -- eval artifacts ------------------------------------------------------

    def eval_artifacts(self) -> list[EvalArtifactReference]:
        """Return the list of regime-level eval artifacts backing this report."""
        return list(self._run.manifest.eval_artifacts)

    # -- metrics -------------------------------------------------------------

    def metrics(self) -> list[MetricRecord]:
        """Return all normalized metric records, or ``[]`` if none exist."""
        return list(self._run.metrics)

    def metric_rows(self) -> list[dict[str, Any]]:
        """Return metric records as plain dictionaries suitable for DataFrame
        construction or notebook table display."""
        return [record.model_dump() for record in self._run.metrics]

    # -- figures -------------------------------------------------------------

    def figures(self) -> FigureIndex | None:
        """Return the figure index if figures were rendered, or ``None``."""
        index = self._run.figures
        if not index.entries:
            return None
        return index

    def figure_entry(
        self,
        figure_id: str,
        *,
        preferred_format: str = "png",
        task: str | None = None,
        regime_id: str | None = None,
    ) -> FigureIndexEntry | None:
        """Return one figure entry matching *figure_id* and optional filters.

        Selects the first entry whose ``figure_id`` matches *figure_id*.
        When *preferred_format* is provided, entries with that format are
        preferred over others.  When *task* or *regime_id* are provided,
        only entries with matching values are considered.

        Args:
            figure_id: The figure identifier (e.g. ``\"prediction_overlay_arena\"``).
            preferred_format: Preferred output format (``\"png\"`` or ``\"pdf\"``).
            task: Optional task filter.
            regime_id: Optional regime-id filter.

        Returns:
            A matching :class:`~ehc_sn.reporting.schema.FigureIndexEntry` or
            ``None`` if no entry matches.
        """
        index = self.figures()
        if index is None:
            return None

        entries = [e for e in index.entries if e.figure_id == figure_id]

        if task is not None:
            entries = [e for e in entries if e.task == task]
        if regime_id is not None:
            entries = [e for e in entries if e.regime_id == regime_id]

        preferred = [e for e in entries if e.format == preferred_format]
        return preferred[0] if preferred else (entries[0] if entries else None)

    # -- summary -------------------------------------------------------------

    def summary(self) -> dict[str, object]:
        """Return a compact inspection dict useful in notebooks.

        The returned dict includes identity, provenance summary, metrics
        preview, and figure availability — all without loading large
        artifacts or traces.
        """
        prov = self._run.provenance
        metrics_list = list(self._run.metrics)
        figures = self.figures()

        metric_count = len(metrics_list)
        metric_preview: list[dict[str, Any]] = []
        for m in metrics_list[:10]:
            metric_preview.append(
                {
                    "task": m.task,
                    "regime_id": m.regime_id,
                    "metric": m.metric,
                    "value": m.value,
                }
            )

        return {
            "root": str(self._run.root),
            "model_family": self._run.manifest.model_family,
            "checkpoint": {
                "path": str(prov.checkpoint.path),
                "sha256": prov.checkpoint.sha256,
            },
            "created_at": prov.created_at.isoformat(),
            "code": {
                "git_commit": prov.code.git_commit if prov.code else None,
                "git_dirty": prov.code.git_dirty if prov.code else None,
            },
            "eval_artifact_count": len(self._run.manifest.eval_artifacts),
            "metric_count": metric_count,
            "metric_preview": metric_preview,
            "has_figures": figures is not None,
            "figure_count": (len(figures.entries) if figures else 0),
        }


# =============================================================================
def open_report(path: str | Path) -> OpenReport:
    """Load a completed report-run directory and return an inspection surface.

    Args:
        path: Path to the report-run directory (must contain
            ``report_manifest.json``, ``provenance.json``, and ``_SUCCESS``).

    Returns:
        :class:`OpenReport` wrapping the verified report run.

    Raises:
        FileNotFoundError: If *path* does not exist or required files are
            missing.
        RuntimeError: If ``_SUCCESS`` is missing (incomplete report).
        ValueError: If the manifest has an unsupported schema version.
    """
    run = load_report_run(Path(path))
    return OpenReport(_run=run)
