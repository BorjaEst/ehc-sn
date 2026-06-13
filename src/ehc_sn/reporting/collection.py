"""Notebook-facing view over multiple OpenReport objects.

``ReportCollection`` is a strict post-hoc composition layer.  It wraps
existing :class:`~ehc_sn.reporting.inspection.OpenReport` instances and
exposes display methods for side-by-side scientific comparison.  It is
**not** a new report abstraction — the authoritative evidence unit
remains each :class:`~ehc_sn.reporting.reader.ReportRun`.

Boundary rule:
    This module must remain free of ``lightning/``, ``eval/``,
    ``models/``, ``tasks/``, ``adapters/``, and benchmark imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ehc_sn.reporting.builder import build_report_run
from ehc_sn.reporting.inspection import OpenReport, open_report
from ehc_sn.reporting.loaders import load_report_spec
from ehc_sn.reporting.schema import ReportSpec

# =============================================================================
# Public type
# =============================================================================


@dataclass
class ReportCollection:
    """Notebook-facing view over multiple per-subject report runs.

    Each report in ``reports`` is an independently assembled
    :class:`OpenReport`.  The collection provides convenience display
    methods for side-by-side comparison in a notebook cell context.

    Fields
    ------
    title:
        Scientific title for the comparison (displayed via
        ``display_header()``).
    question:
        Specific research question addressed by this collection.
    reports:
        One :class:`OpenReport` per subject (model variant, treatment,
        etc.).
    """

    title: str = ""
    question: str = ""
    reports: list[OpenReport] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Display methods — each prints Markdown or tabular content inline
    # in a notebook cell.
    # ------------------------------------------------------------------

    def display_header(self) -> None:
        """Print the scientific title and question as Markdown."""
        self._require_reports()
        self._md(f"# {self.title}")
        self._md("")
        if self.question:
            self._md(f"**Question:** {self.question}")
            self._md("")

    def display_evidence_plan(
        self, primary_metric: str = "To be specified"
    ) -> None:
        """Print a table of subjects, benchmark, task, split, and primary metric.

        Parameters
        ----------
        primary_metric:
            Primary metric name for the comparison.  Displayed in the table
            as the ``Primary Metric`` column.  Defaults to a placeholder.
        """
        self._require_reports()
        self._md("### Evidence Plan")
        self._md("")
        rows = []
        for r in self.reports:
            row = {
                "Subject": r.model_family,
                "Benchmark": self._infer_benchmark(r),
                "Task": self._infer_task(r),
                "Split": "test",
                "Primary Metric": primary_metric,
            }
            rows.append(row)
        self._print_table(rows)

    def display_binding_table(self) -> None:
        """Print a table of subject, model family, checkpoint, and adapter."""
        self._require_reports()
        self._md("### Binding Configuration")
        self._md("")
        rows = []
        for r in self.reports:
            prov = r.provenance()
            row = {
                "Subject": r.model_family,
                "Model Family": r.model_family,
                "Checkpoint": str(prov.checkpoint.path),
                "Adapter Family": self._infer_adapter(r),
            }
            rows.append(row)
        self._print_table(rows)

    def display_primary_metrics(
        self, primary_metric: str | None = None
    ) -> None:
        """Print a DataFrame of subjects × primary metric.

        Parameters
        ----------
        primary_metric:
            Metric name to show.  Each report's ``MetricRecord`` list is
            searched for an exact match on ``metric``.  If no match is
            found for a report that cell shows ``None``.

            When ``None`` (default), a warning is printed and no table is
            rendered.  Callers should always pass the relevant primary
            metric name for their task.
        """
        self._require_reports()
        self._md("### Primary Metrics")
        self._md("")

        if primary_metric is None:
            self._md(
                "> No primary metric specified.  "
                "Call ``display_primary_metrics(primary_metric=...)`` "
                "with the relevant metric name."
            )
            self._md("")
            return

        try:
            import pandas as pd
        except ImportError:
            self._md("*pandas is not installed — cannot render metric table.*")
            return

        data = {}
        for r in self.reports:
            metrics = r.metrics()
            matches = [m for m in metrics if m.metric == primary_metric]
            data[r.model_family] = matches[0].value if matches else None

        df = pd.DataFrame([data], index=[primary_metric]).T
        df.columns = [primary_metric]
        self._display_dataframe(df)

    def display_secondary_metrics(
        self, metric_names: list[str] | None = None
    ) -> None:
        """Print a DataFrame of subjects × secondary metrics.

        Parameters
        ----------
        metric_names:
            Metric columns to include.  When ``None`` (default), shows
            all records whose ``metric`` field is NOT the primary metric
            (caller must have already called ``display_primary_metrics``
            — purely a heuristic fallback).  When an explicit list is
            provided only those columns are rendered; absent metrics
            show ``None``.
        """
        self._require_reports()
        self._md("### Secondary Metrics")
        self._md("")
        try:
            import pandas as pd
        except ImportError:
            self._md("*pandas is not installed — cannot render metric table.*")
            return

        # Group all metric names across all reports if no explicit list.
        if metric_names is None:
            metric_names_set: set[str] = set()
            for r in self.reports:
                for m in r.metrics():
                    metric_names_set.add(m.metric)
            metric_names = sorted(metric_names_set)

        records = []
        for r in self.reports:
            metrics = {m.metric: m.value for m in r.metrics()}
            row = {"Subject": r.model_family}
            for field in metric_names:
                row[field] = metrics.get(field)
            records.append(row)

        df = pd.DataFrame(records).set_index("Subject")
        self._display_dataframe(df)

    def display_figure_availability(self) -> None:
        """Print a table of figure_ids × subjects with available/missing status.

        Reads each report's :class:`~ehc_sn.reporting.schema.FigureIndex` via
        :meth:`OpenReport.figures`.  A figure is ``available`` when
        ``figure_entry(figure_id)`` returns a non-``None`` entry.
        """
        self._require_reports()
        self._md("### Figure Availability")
        self._md("")

        # Collect all figure IDs across all reports.  Preserve insertion
        # order (first report's figure order determines row sequence).
        all_ids: list[str] = []
        seen: set[str] = set()
        for r in self.reports:
            fig_index = r.figures()
            if fig_index is None:
                continue
            for entry in fig_index.entries:
                fid = entry.figure_id
                if fid not in seen:
                    seen.add(fid)
                    all_ids.append(fid)

        if not all_ids:
            self._md("*No figure entries found in any report.*")
            self._md("")
            return

        rows: list[dict[str, object]] = []
        for fid in all_ids:
            row: dict[str, object] = {"Figure": fid}
            for r in self.reports:
                entry = r.figure_entry(fid, preferred_format="png")
                row[r.model_family] = "available" if entry else "missing"
            rows.append(row)

        self._print_table(rows)
        self._md("")

    def display_evidence_status(
        self,
        evidence_axes: list[dict[str, str]],
    ) -> None:
        """Print a table of evidence axes × subjects with per-axis status.

        Each entry in *evidence_axes* is a dict with keys:

        - ``"axis"`` (required): display label for the evidence row.
        - ``"figure_id"`` (optional): figure ID to check availability against
          each report's :class:`~ehc_sn.reporting.schema.FigureIndex`.  When
          provided, the cell value is ``"available"`` or ``"missing"``.
        - ``"status"`` (optional, per-report): a dict mapping model family
          names to explicit status strings (e.g. ``{"HRM v1": "available",
          "HRM v2": "warning"}``).  When both ``figure_id`` and
          ``status`` are omitted, the cell shows ``"manual"``.

        Parameters
        ----------
        evidence_axes:
            List of evidence-axis descriptors.  See above for key semantics.
        """
        self._require_reports()
        self._md("### Evidence Status")
        self._md("")

        if not evidence_axes:
            self._md("*No evidence axes provided.*")
            self._md("")
            return

        model_names = [r.model_family for r in self.reports]
        rows: list[dict[str, object]] = []

        for ax in evidence_axes:
            row: dict[str, object] = {"Evidence axis": ax.get("axis", "—")}
            figure_id = ax.get("figure_id")
            explicit_status: dict[str, str] = ax.get("status", {})

            for r in self.reports:
                mf = r.model_family
                if mf in explicit_status:
                    row[mf] = explicit_status[mf]
                elif figure_id:
                    entry = r.figure_entry(figure_id, preferred_format="png")
                    row[mf] = "available" if entry else "missing"
                else:
                    row[mf] = "manual"
            rows.append(row)

        self._print_table(rows)
        self._md("")

    def display_failure_cases(self) -> None:
        """Placeholder — real failure-case analysis is not yet implemented."""
        self._md("> ⚠️ Not yet implemented.")
        self._md("")

    def display_limitations(self) -> None:
        """Placeholder — real limitations analysis is not yet implemented."""
        self._md("> ⚠️ Not yet implemented.")
        self._md("")

    def display_provenance(self) -> None:
        """Print a provenance table for each subject."""
        self._require_reports()
        self._md("### Provenance")
        self._md("")
        rows = []
        for r in self.reports:
            prov = r.provenance()
            row = {
                "Subject": r.model_family,
                "Git Commit": prov.code.git_commit if prov.code else "—",
                "Git Dirty": str(prov.code.git_dirty) if prov.code else "—",
                "Created At": prov.created_at.isoformat(),
            }
            rows.append(row)
        self._print_table(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _md(text: str) -> None:
        """Print a Markdown line — works in any notebook/console context."""
        print(text)

    @staticmethod
    def _display_dataframe(df: Any) -> None:
        """Display a DataFrame; uses IPython display if available."""
        try:
            from IPython.display import display as ipy_display

            ipy_display(df)
        except ImportError:
            print(df.to_string())

    @staticmethod
    def _print_table(rows: list[dict[str, object]]) -> None:
        """Print a simple Markdown table from a list of dicts."""
        if not rows:
            print("_(Empty table)_")
            return
        headers = list(rows[0].keys())
        # Header row
        print("| " + " | ".join(headers) + " |")
        print("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows:
            print(
                "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |"
            )
        print("")

    def _require_reports(self) -> None:
        """Guard: raise if reports list is empty."""
        if not self.reports:
            raise ValueError(
                "ReportCollection has no reports. "
                "Use build_or_load_report_collection() to populate it."
            )

    @staticmethod
    def _infer_benchmark(report: OpenReport) -> str:
        """Infer benchmark track from the first eval artifact's task."""
        artifacts = report.eval_artifacts()
        if not artifacts:
            return "—"
        task = artifacts[0].task
        if task == "arena":
            return "Arena-Struct"
        if task == "mazehard":
            return "MazeHard-Delib"
        return task

    @staticmethod
    def _infer_task(report: OpenReport) -> str:
        """Infer task family from the first eval artifact."""
        artifacts = report.eval_artifacts()
        return artifacts[0].task if artifacts else "—"

    @staticmethod
    def _infer_adapter(report: OpenReport) -> str:
        """Infer adapter family from model family."""
        mf = report.model_family
        if mf in ("tem-v1", "tem-v2"):
            return "arena/tem"
        if mf == "ehp-v1":
            return "arena/ehp"
        return "—"


# =============================================================================
# Builder
# =============================================================================


def build_or_load_report_collection(
    specs: list[Path | ReportSpec | OpenReport],
    *,
    force_rebuild: bool = False,
    title: str = "",
    question: str = "",
) -> ReportCollection:
    """Build or load a report collection from per-subject spec inputs.

    For each entry in *specs*:

    - If it is an ``OpenReport``, use it directly.
    - If it is a ``ReportSpec``, call ``build_report_run(spec, overwrite=force_rebuild)``
      and wrap the result in ``OpenReport``.
    - If it is a ``Path`` to a YAML file, load it as a ``ReportSpec`` and
      proceed as above.

    Parameters
    ----------
    specs:
        One entry per subject.  Each entry must be a path to a ``ReportSpec``
        YAML, an already-instantiated ``ReportSpec``, or an already-loaded
        ``OpenReport``.
    force_rebuild:
        If ``True``, pass ``overwrite=True`` to ``build_report_run()`` for
        each spec — existing report runs will be replaced.
    title:
        Scientific title for the collection.
    question:
        Research question for the collection.

    Returns
    -------
    ReportCollection
    """
    reports: list[OpenReport] = []

    for spec_input in specs:
        if isinstance(spec_input, OpenReport):
            reports.append(spec_input)
        else:
            if isinstance(spec_input, Path):
                spec = load_report_spec(spec_input)
            elif isinstance(spec_input, ReportSpec):
                spec = spec_input
            else:
                raise TypeError(
                    f"Unsupported spec type: {type(spec_input).__name__}. "
                    f"Expected Path, ReportSpec, or OpenReport."
                )

            build_report_run(spec, overwrite=force_rebuild)
            # Reload from disk as an OpenReport.
            reports.append(open_report(spec.output_dir))

    return ReportCollection(title=title, question=question, reports=reports)


# ---------------------------------------------------------------------------
__all__ = ["ReportCollection", "build_or_load_report_collection"]
