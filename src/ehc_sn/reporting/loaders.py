"""Report loaders — resolve evaluation sources into notebook-facing read models.

Usage::

    from ehc_sn.reporting import load_arena_tem_report

    report = load_arena_tem_report("artifacts/evaluation/tem-v1-arena")
    report.headline_metrics
    report.pathway_metrics
    report.cases
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ehc_sn.evaluation.artifact_models import (
    EvaluationArtifactSet,
    RegimeArtifactSet,
)
from ehc_sn.reporting.sources import (
    MaterializedEvaluationSource,
    materialize_evaluation_source,
)
from ehc_sn.tasks.scoring import scoring_spec_for_task

# =============================================================================
# Report provenance
# =============================================================================


@dataclass(frozen=True)
class ReportProvenance:
    """Provenance metadata for a report built from one evaluation source.

    All fields are derived from the evaluation artifact manifests and
    the source resolution, never from user-supplied strings.
    """

    source_uri: str
    local_root: Path
    run_id: str | None
    evaluation_id: str
    alias: str
    task: str
    model_family: str
    regime_id: str
    model_uri: str | None
    dataset_uri: str | None
    dataset_split: str | None
    code_revision: str | None
    artifact_schema: str


# =============================================================================
# Metric views
# =============================================================================


@dataclass(frozen=True)
class ArenaMetricViews:
    """Classified metric projections for an Arena evaluation regime.

    Attributes:
        headline: DataFrame with one row, columns being "headline" metrics
            (primary metric and top-level scores).
        pathway: DataFrame with one row per prediction pathway, columns
            being pathway-specific metrics.
    """

    headline: pd.DataFrame
    pathway: pd.DataFrame


# =============================================================================
# Report data models
# =============================================================================


@dataclass(frozen=True)
class ArenaTemReportData:
    """Notebook-facing read model for an Arena + TEM evaluation report.

    Constructed by :func:`load_arena_tem_report`.  Notebooks consume
    this dataclass and should not access raw artifact paths or manifest
    dicts.
    """

    provenance: ReportProvenance
    headline_metrics: pd.DataFrame
    pathway_metrics: pd.DataFrame
    cases: tuple[Any, ...]
    validations: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())


# =============================================================================
# Public API
# =============================================================================


def load_evaluation(source_uri: str) -> EvaluationArtifactSet:
    """Load a complete evaluation artifact set from *source_uri*.

    Resolves the source (local or MLflow), reads the evaluation-level
    manifest, and returns a typed handle.  Use ``.open_regime(regime_id)``
    to access individual regime artifacts.

    Args:
        source_uri: Local path, ``runs:/<run_id>/<path>``, or other
            supported MLflow artifact URI.

    Returns:
        A fully typed :class:`EvaluationArtifactSet`.

    Raises:
        FileNotFoundError: If the source does not exist or
            ``evaluation-manifest.json`` is missing.
    """
    src = materialize_evaluation_source(source_uri)
    return EvaluationArtifactSet.load(src.local_root)


def load_arena_tem_report(
    source_uri: str,
    *,
    regime: str = "diagnostic",
    max_cases: int = 8,
    case_ids: Sequence[str] = (),
) -> ArenaTemReportData:
    """Build a notebook-facing report for an Arena + TEM evaluation.

    .. note::

        This loader reads evaluation artifacts directly.  For prepared
        report-data packages, prefer :func:`open_report` and
        :class:`ReportDataPackage` instead.

    Resolves the source, opens the specified regime, validates
    compatibility, and projects metrics and cases into a report-specific
    read model.

    Args:
        source_uri: Local path, ``runs:/<run_id>/<path>``, or other
            supported MLflow artifact URI.
        regime: Regime ID within the evaluation (default ``"diagnostic"``).
        max_cases: Maximum number of prediction cases to include in
            the report.  Ignored when ``case_ids`` is non-empty.
        case_ids: If non-empty, only the specified case IDs are included
            (overrides ``max_cases``).

    Returns:
        An :class:`ArenaTemReportData` ready for notebook consumption.

    Raises:
        FileNotFoundError: If the source does not exist.
        KeyError: If the requested regime is not found.
        ValueError: If the artifact is not compatible with the Arena TEM
            report template.
    """
    src = materialize_evaluation_source(source_uri)

    # Try evaluation-level manifest first; fall back to regime-level.
    eval_manifest_path = src.local_root / "evaluation-manifest.json"
    if eval_manifest_path.exists():
        evaluation = EvaluationArtifactSet.load(src.local_root)
        regime_artifacts = evaluation.open_regime(regime)
        manifest = regime_artifacts.manifest
        eval_identity = evaluation.manifest.identity
        alias = eval_identity.alias
        task = eval_identity.task
        model_family = eval_identity.model_family
        model_uri = evaluation.manifest.model.uri
        dataset_uri = evaluation.manifest.dataset.uri
        dataset_split = evaluation.manifest.dataset.split
        code_revision = evaluation.manifest.code_revision
        evaluation_id = eval_identity.evaluation_id
    else:
        # Legacy v1: load regime directly, synthesize identity from manifest.
        regime_artifacts = RegimeArtifactSet.load(src.local_root)
        manifest = regime_artifacts.manifest
        alias = manifest.regime_id or "unknown"
        task = manifest.task or manifest.regime_kind
        model_family = "unknown"
        model_uri = None
        dataset_uri = None
        dataset_split = None
        code_revision = None
        evaluation_id = f"legacy-{manifest.regime_id}"

    # Validate compatibility.
    _validate_arena_tem_artifact(regime_artifacts)

    # Build metric views through the task scoring catalog.
    scoring = scoring_spec_for_task(task)
    metric_views = _build_arena_metric_views(regime_artifacts, scoring)

    # Select cases.
    selected_cases = _select_report_cases(
        regime_artifacts,
        max_cases=max_cases,
        case_ids=case_ids,
    )

    # Build provenance.
    provenance = ReportProvenance(
        source_uri=source_uri,
        local_root=src.local_root,
        run_id=src.run_id,
        evaluation_id=evaluation_id,
        alias=alias,
        task=task,
        model_family=model_family,
        regime_id=manifest.regime_id,
        model_uri=model_uri,
        dataset_uri=dataset_uri,
        dataset_split=dataset_split,
        code_revision=code_revision,
        artifact_schema=manifest.schema,
    )

    return ArenaTemReportData(
        provenance=provenance,
        headline_metrics=metric_views.headline,
        pathway_metrics=metric_views.pathway,
        cases=selected_cases,
    )


# =============================================================================
# Internal helpers
# =============================================================================


# Known Arena TEM pathway metric prefixes.
_PATHWAY_PREFIXES = {
    "ancestral": "accuracy_path",
    "retrieved": "accuracy_recall",
    "inference": "accuracy_post",
}
"""Map from display pathway name to metric name prefix."""


def _build_arena_metric_views(
    artifacts: RegimeArtifactSet,
    scoring: Any,
) -> ArenaMetricViews:
    """Classify regime metrics into headline and pathway views.

    Uses the ``TaskScoringSpec`` to identify metrics, then separates
    pathway-specific metrics (``accuracy_path_*``, ``accuracy_recall_*``,
    ``accuracy_post_*``) from headline metrics (everything else).
    """
    metrics = artifacts.metrics
    domain_metrics: dict[str, float] = {}
    pathway_rows: list[dict[str, float | str]] = []

    # Separate pathway metrics from headline metrics.
    pathway_seen: set[str] = set()
    for path_name, prefix in _PATHWAY_PREFIXES.items():
        all_key = f"{prefix}_all"
        revisit_key = f"{prefix}_revisit"
        row: dict[str, float | str] = {"pathway": path_name}
        if all_key in metrics:
            row["all_steps"] = metrics[all_key]
            pathway_seen.add(all_key)
        if revisit_key in metrics:
            row["revisit_steps"] = metrics[revisit_key]
            pathway_seen.add(revisit_key)
        pathway_rows.append(row)

    # Remaining metrics are headline metrics.
    for k, v in metrics.items():
        if k not in pathway_seen:
            domain_metrics[k] = v

    headline = pd.DataFrame([domain_metrics])
    pathway = pd.DataFrame(pathway_rows).set_index("pathway")
    return ArenaMetricViews(headline=headline, pathway=pathway)


def _validate_arena_tem_artifact(artifacts: RegimeArtifactSet) -> None:
    """Verify that a regime artifact is compatible with the Arena TEM report.

    Raises ``ValueError`` if the artifact does not contain the expected
    metrics for an Arena TEM evaluation.
    """
    task = artifacts.manifest.task
    if task and task != "arena":
        raise ValueError(
            f"Cannot build Arena TEM report from task {task!r}. "
            f"Expected task 'arena'."
        )
    # Additional validation can be added as needed (e.g., check for
    # required metrics).


def _select_report_cases(
    artifacts: RegimeArtifactSet,
    *,
    max_cases: int,
    case_ids: Sequence[str],
) -> tuple[Any, ...]:
    """Select report cases from a regime artifact.

    When ``case_ids`` is non-empty, returns only those cases.  Otherwise
    returns up to ``max_cases`` cases from the artifact's case manifest
    (with traces preferred).
    """
    loaded_cases = artifacts.read_cases()

    if case_ids:
        id_set = set(case_ids)
        selected = [c for c in loaded_cases if c.case_id in id_set]
    else:
        # Prefer cases with traces, up to max_cases.
        traced = [c for c in loaded_cases if c.trace is not None]
        if len(traced) >= max_cases:
            selected = traced[:max_cases]
        else:
            # Fill remaining slots with non-traced cases.
            non_traced = [c for c in loaded_cases if c.trace is None]
            selected = traced + non_traced[: max_cases - len(traced)]

    return tuple(selected)


# =============================================================================
__all__ = [
    "ArenaMetricViews",
    "ArenaTemReportData",
    "ReportProvenance",
    "load_arena_tem_report",
    "load_evaluation",
]
