"""Report-run builder — assembles a complete report-run container.

The builder orchestrates existing pieces (discovery, selection, metric
serialization, manifest writing, verification) without executing
evaluation, computing metrics, or rendering figures.

Usage::

    from ehc_sn.reporting import ReportSpec, build_report_run

    spec = ReportSpec(
        model_family="tem-v2",
        checkpoint=CheckpointSpec(path=Path("/ckpt.pt")),
        eval_artifacts_root=Path("/evaluation"),
        regimes=[RegimeSelector(task="arena")],
        output_dir=Path("reports/tem_v2/version_12_best"),
    )
    report = build_report_run(spec)

This module must remain free of ``lightning/``, ``eval/``, ``models/``,
``tasks/``, ``adapters/``, and benchmark imports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from ehc_sn.reporting.figures import write_figure_index
from ehc_sn.reporting.loaders import (
    discover_eval_artifacts,
    load_eval_artifact_manifest,
    select_eval_artifacts,
)
from ehc_sn.reporting.metrics.records import write_metric_records
from ehc_sn.reporting.provenance import CodeProvenance
from ehc_sn.reporting.reader import ReportRun, load_report_run
from ehc_sn.reporting.schema import (
    EvalArtifactReference,
    FigureIndex,
    FigureIndexEntry,
    MetricRecord,
    MetricSuiteSpec,
    ReportSpec,
)
from ehc_sn.reporting.writer import write_report_run_manifest

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_report_run(
    spec: ReportSpec,
    *,
    eval_artifacts: Sequence[EvalArtifactReference] | None = None,
    overwrite: bool = False,
    code_provenance: CodeProvenance | None = None,
    metric_records: Sequence[MetricRecord] | None = None,
    metric_normalizers: Mapping[str, object] | None = None,
    figure_renderers: Mapping[str, object] | None = None,
) -> ReportRun:
    """Assemble a validated report-run container from a :class:`ReportSpec`.

    Discovers eval artifacts from ``spec.eval_artifacts_root`` (or uses
    *eval_artifacts* when provided directly), selects them using
    ``spec.regimes``, optionally writes metric records, optionally
    renders figures (if a figure-renderer registry is injected), writes
    the report-run manifest and provenance, and returns a verified
    :class:`ReportRun`.

    Parameters
    ----------
    spec:
        Report assembly specification.  ``spec.output_dir`` is the
        authoritative output path for the completed report run.
    eval_artifacts:
        Optional pre-resolved eval artifact references.  When provided,
        discovery and regime filtering are skipped and these artifacts
        are used directly.  This is the pipeline handoff: the caller
        (e.g. :func:`build_report_run_from_checkpoint`) produces eval
        artifacts at a known location and passes them in without
        requiring ``spec.eval_artifacts_root`` to point at existing
        artifacts.
    overwrite:
        If ``True``, remove any existing completed report run at
        ``spec.output_dir`` before writing.  Defaults to ``False``.
    code_provenance:
        Optional version-control metadata stored in ``provenance.json``.
    metric_records:
        Explicit metric records to persist.

        - ``None`` (default): records may be derived from
          *metric_normalizers* if provided.  If normalizers are also
          ``None``, no metrics are produced (current behavior).
        - ``[]`` (explicit empty): write no ``metrics.records.json``.
          Normalizers are **not** invoked.
        - non-empty list: write those records.  Normalizers are **not**
          invoked.
    metric_normalizers:
        Optional normalizer registry (e.g. a
        :class:`~ehc_sn.report_metrics.MetricNormalizerRegistry`)
        used to convert eval-artifact summaries into
        :class:`MetricRecord` rows.  Ignored when
        *metric_records* is not ``None``.

        Artifacts whose ``(task, regime_kind)`` has no registered
        normalizer are silently skipped (permissive).
    figure_renderers:
        Optional figure-renderer registry (structurally conforming to
        :class:`~ehc_sn.reporting.figures.ReportFigureRendererRegistry`).
        When provided and ``spec.figures.figures`` is non-empty, the
        renderer is called once per selected eval artifact to produce
        figure files into ``<output_dir>/figures/``.

    Returns
    -------
    ReportRun
        Verified report-run container (validated through
        :func:`load_report_run`).

    Raises
    ------
    RuntimeError
        If ``spec.regimes`` selectors produce no matching eval artifacts
        (only when *eval_artifacts* is not provided and discovery is
        used).
    FileNotFoundError
        If ``spec.eval_artifacts_root`` does not exist (only when
        *eval_artifacts* is not provided and discovery is used).
    KeyError
        If a supported artifact has a missing required metric field.
        The error message includes the artifact path, task, regime_id,
        regime_kind, and the original missing-key message.
    """
    output_dir = spec.output_dir

    if eval_artifacts is not None:
        selected = list(eval_artifacts)
    else:
        artifacts = discover_eval_artifacts(spec.eval_artifacts_root)
        selected = select_eval_artifacts(artifacts, spec.regimes)

        if not selected:
            raise RuntimeError(
                "No eval artifacts matched the report spec selectors. "
                f"Root: {spec.eval_artifacts_root}. "
                f"Selectors: {spec.regimes}."
            )

    metrics_records_json: Path | None = None
    all_records: list[MetricRecord] | None = None
    if metric_records is not None:
        if metric_records:
            metrics_path = write_metric_records(
                metric_records,
                output_dir=output_dir,
            )
            metrics_records_json = metrics_path.relative_to(
                output_dir.resolve()
            )
    elif metric_normalizers is not None:
        all_records = []
        for artifact in selected:
            manifest = load_eval_artifact_manifest(artifact.path)
            summary = dict(manifest.get("summary", {}))
            try:
                normalizer = metric_normalizers.resolve(
                    task=artifact.task,
                    regime_kind=artifact.regime_kind,
                )
            except KeyError:
                # No normalizer registered for this pair; skip silently.
                continue
            try:
                records = normalizer.normalize(
                    artifact_ref=artifact, summary=summary
                )
            except KeyError as exc:
                raise KeyError(
                    "Metric normalization failed for artifact "
                    f"{artifact.path!s} "
                    f"(task={artifact.task!r}, "
                    f"regime_id={artifact.regime_id!r}, "
                    f"regime_kind={artifact.regime_kind!r}): {exc}"
                ) from exc
            all_records.extend(records)
        if all_records:
            metrics_path = write_metric_records(
                all_records,
                output_dir=output_dir,
            )
            metrics_records_json = metrics_path.relative_to(
                output_dir.resolve()
            )

    # ---- Validate metric suites (report-level contract) -----------------
    if spec.metric_suites:
        _validate_metric_suites(
            suites=spec.metric_suites,
            records=all_records if all_records is not None else None,
            selected_artifacts=selected,
            metric_normalizers=metric_normalizers,
            metric_records_explicit=metric_records is not None,
        )

    # ------------------------------------------------------------------ #
    # Figure rendering
    # ------------------------------------------------------------------ #
    figures: FigureIndex | None = None
    if spec.figures.figures and figure_renderers is not None:
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        all_entries: list[FigureIndexEntry] = []
        for artifact in selected:
            idx = figure_renderers.render(
                artifact=artifact,
                output_dir=figures_dir,
                requested_figures=spec.figures.figures,
                formats=spec.figures.formats,
            )
            all_entries.extend(idx.entries)

        if spec.figures.figures:
            produced = {e.figure_id for e in all_entries}
            missing = set(spec.figures.figures) - produced
            if missing:
                raise KeyError(
                    f"Required figures not produced by renderer: "
                    f"{sorted(missing)}. "
                    f"Produced figure IDs: {sorted(produced) if produced else '(none)'}."
                )

        for entry in all_entries:
            if not (output_dir / entry.path).exists():
                raise FileNotFoundError(
                    f"Figure index entry references non-existent file: "
                    f"{entry.path}. "
                    f"Figure {entry.figure_id!r} for regime {entry.regime_id!r}."
                )

        figures = FigureIndex(entries=all_entries)
        index_path = write_figure_index(figures, figures_dir)
    elif spec.figures.figures and figure_renderers is None:
        raise RuntimeError(
            "Report requested figures but no figure renderer was injected. "
            "Provide figure_renderers or set figures.figures to an empty list."
        )

    figures_index_json: Path | None = None
    if figures is not None:
        figures_index_json = index_path.relative_to(output_dir.resolve())

    write_report_run_manifest(
        spec=spec,
        eval_artifacts=selected,
        output_dir=output_dir,
        overwrite=overwrite,
        code_provenance=code_provenance,
        metrics_records_json=metrics_records_json,
        figures_index_json=figures_index_json,
    )

    return load_report_run(output_dir)


# ---------------------------------------------------------------------------
# Metric suite validation (module-private)
# ---------------------------------------------------------------------------


def _validate_metric_suites(
    *,
    suites: tuple[MetricSuiteSpec, ...],
    records: list[MetricRecord] | None,
    selected_artifacts: list[EvalArtifactReference],
    metric_normalizers: object | None,
    metric_records_explicit: bool,
) -> None:
    """Validate that declared metric suites are satisfied by produced records.

    Raises
    ------
    RuntimeError
        If a suite is declared but no normalizer is injected and no
        explicit ``metric_records`` were provided.
    KeyError
        If a required metric is missing from the normalized records.
    """
    # When explicit metric_records is provided, skip suite validation
    # (the caller owns the contract for explicitly supplied records).
    if metric_records_explicit:
        return

    # Group selected artifacts by (task, regime_kind).
    artifact_pairs: dict[tuple[str, str], list[EvalArtifactReference]] = {}
    for a in selected_artifacts:
        key = (a.task, a.regime_kind)
        artifact_pairs.setdefault(key, []).append(a)

    # Build a mapping from regime_id back to (task, regime_kind) so we can
    # match MetricRecords (which carry task + regime_id) to suites.
    regime_id_to_pair: dict[str, tuple[str, str]] = {}
    for a in selected_artifacts:
        regime_id_to_pair[a.regime_id] = (a.task, a.regime_kind)

    # Collect metric names per (task, regime_kind) from produced records.
    record_metrics_by_pair: dict[tuple[str, str], set[str]] = {}
    if records:
        for r in records:
            if r.task is not None and r.regime_id is not None:
                pair = regime_id_to_pair.get(r.regime_id, (r.task, "unknown"))
                record_metrics_by_pair.setdefault(pair, set()).add(r.metric)

    for suite in suites:
        key = (suite.task, suite.regime_kind)

        # No artifacts selected for this pair; nothing to validate.
        if key not in artifact_pairs:
            continue

        # Check that a normalizer was injected.
        if metric_normalizers is None:
            raise RuntimeError(
                f"Report requires metrics for "
                f"(task={suite.task!r}, regime_kind={suite.regime_kind!r}) "
                f"but no metric normalizer was injected. "
                f"Provide metric_normalizers or remove the metric suite."
            )

        # Check that the normalizer can resolve this pair.
        resolve = getattr(metric_normalizers, "resolve", None)
        if resolve is not None:
            try:
                resolve(task=suite.task, regime_kind=suite.regime_kind)
            except KeyError:
                known = _format_known_normalizer_keys(metric_normalizers)
                raise RuntimeError(
                    f"Report requires metrics for "
                    f"(task={suite.task!r}, regime_kind={suite.regime_kind!r}) "
                    f"but no metric normalizer is registered for that pair. "
                    f"Known keys: {known}."
                )

        # Check required metrics appear in produced records.
        produced = record_metrics_by_pair.get(key, set())
        missing = set(suite.required_metrics) - produced
        if missing:
            raise KeyError(
                f"Required metrics missing for "
                f"(task={suite.task!r}, regime_kind={suite.regime_kind!r}): "
                f"{sorted(missing)}. "
                f"Produced metric names: {sorted(produced) if produced else '(none)'}."
            )


def _format_known_normalizer_keys(metric_normalizers: object) -> str:
    """Pretty-print known normalizer keys for error messages."""
    if hasattr(metric_normalizers, "_entries"):
        keys = sorted(
            f"({k.task!r}, {k.regime_kind!r})"
            for k in metric_normalizers._entries
        )
        return "[" + ", ".join(keys) + "]"
    return "(unknown)"


# ---------------------------------------------------------------------------
__all__ = ["build_report_run"]
